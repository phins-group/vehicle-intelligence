"""Filesystem catalog and durable private-Hub synchronization jobs for datasets."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import cv2
import numpy as np
from pydantic import ValidationError

from vehicle_intelligence.application.dataset_registry import (
    DatasetHubSyncCommand,
    DetectorDatasetSampleQuery,
)
from vehicle_intelligence.config import DatasetRegistryConfig
from vehicle_intelligence.domain.dataset_registry import (
    DatasetHubSyncJob,
    DatasetHubSyncStatus,
    DatasetRegistryCapabilities,
    DetectorDatasetExport,
    DetectorDatasetSampleAnnotation,
    DetectorDatasetSampleBox,
    DetectorDatasetSampleImage,
    DetectorDatasetSampleKind,
    DetectorDatasetSamplePage,
    DetectorDatasetSamplePreview,
    DetectorDatasetVersion,
)
from vehicle_intelligence.exceptions import (
    DatasetRegistryConflictError,
    DatasetRegistryNotFoundError,
    DatasetRegistryStorageError,
    DatasetRegistryValidationError,
    InvalidCursorError,
)
from vehicle_intelligence.training.config import (
    DetectorDatasetConfig,
    load_training_settings,
)
from vehicle_intelligence.training.dataset import (
    DetectorDatasetBuilder,
    verify_detector_dataset,
)
from vehicle_intelligence.training.domain import DetectorRole, DetectorSample, HubUploadResult
from vehicle_intelligence.training.huggingface import HuggingFacePrivateRegistry
from vehicle_intelligence.training.video_review_source import VIDEO_REVIEW_SOURCE_TYPE
from vehicle_intelligence.training.warehouse_plate_review import (
    WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE,
)

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOB_ID = re.compile(r"^dataset-sync-[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_REVIEW_ONLY_SOURCE_TYPES = {
    VIDEO_REVIEW_SOURCE_TYPE,
    WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE,
}


class _DatasetUploader(Protocol):
    def upload_dataset(
        self,
        directory: Path,
        repo_id: str,
        *,
        revision: str = "main",
        allow_restricted_private: bool = False,
    ) -> HubUploadResult: ...


@dataclass(frozen=True, slots=True)
class _ManifestFile:
    relative_path: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    directory: Path
    version: DetectorDatasetVersion
    files: dict[str, _ManifestFile]
    images_by_sha256: dict[str, _ManifestFile]


@dataclass(frozen=True, slots=True)
class _ExportRecord:
    directory: Path
    export: DetectorDatasetExport
    source_id: str


class FileDatasetRegistryRepository:
    """Catalog immutable plate sources and run checksum-bound Hub sync jobs."""

    def __init__(
        self,
        config: DatasetRegistryConfig,
        *,
        dataset_config: DetectorDatasetConfig | None = None,
        hub_repo_id: str | None = None,
        hub_enabled: bool | None = None,
        uploader: _DatasetUploader | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._sources_root = config.sources_directory.expanduser().resolve()
        self._exports_root = config.exports_directory.expanduser().resolve()
        self._workspace = config.workspace_directory.expanduser().resolve()
        self._dataset_config = dataset_config
        self._hub_repo_id = hub_repo_id
        self._hub_enabled = hub_enabled
        self._uploader = uploader
        self._clock = clock
        self._sources: dict[str, _SourceRecord] = {}
        self._exports: dict[str, _ExportRecord] = {}
        self._jobs: dict[str, DatasetHubSyncJob] = {}
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    async def close(self) -> None:
        return None

    async def capabilities(self) -> DatasetRegistryCapabilities:
        return DatasetRegistryCapabilities(
            enabled=self._config.enabled,
            hub_enabled=bool(self._hub_enabled),
            repo_id=self._hub_repo_id,
            credentials_configured=self._credentials_configured(),
            restricted_private_sync_enabled=self._config.restricted_private_sync_enabled,
        )

    async def list_datasets(self) -> tuple[DetectorDatasetVersion, ...]:
        async with self._write_lock:
            await asyncio.to_thread(self._refresh_catalog_sync)
            versions = [self._effective_version(source) for source in self._sources.values()]
        return tuple(
            sorted(versions, key=lambda item: (item.created_at, item.source_id), reverse=True)
        )

    async def list_samples(
        self,
        query: DetectorDatasetSampleQuery,
    ) -> DetectorDatasetSamplePage:
        if not 1 <= query.limit <= 50:
            raise DatasetRegistryValidationError("dataset sample page limit is invalid")
        if not isinstance(query.kind, DetectorDatasetSampleKind):
            raise DatasetRegistryValidationError("dataset sample kind is invalid")
        lighting = _lighting_filter(query.lighting)
        async with self._write_lock:
            source = self._source(query.source_id)
        return await asyncio.to_thread(
            self._list_samples_sync,
            source,
            replace(query, lighting=lighting),
        )

    async def get_sample_image(
        self,
        source_id: str,
        image_sha256: str,
    ) -> DetectorDatasetSampleImage:
        if not _SHA256.fullmatch(image_sha256):
            raise DatasetRegistryValidationError("dataset sample image id is invalid")
        async with self._write_lock:
            source = self._source(source_id)
        return await asyncio.to_thread(
            self._get_sample_image_sync,
            source,
            image_sha256,
        )

    async def create_sync_job(
        self,
        source_id: str,
        command: DatasetHubSyncCommand,
        requested_by: str,
    ) -> DatasetHubSyncJob:
        async with self._write_lock:
            await asyncio.to_thread(self._refresh_catalog_sync)
            source = self._source(source_id)
            export_id = _identifier(command.export_id, "dataset export id")
            revision = _revision(command.revision)
            self._validate_sync(source.version, command)
            if not requested_by.strip() or len(requested_by) > 128:
                raise DatasetRegistryValidationError("dataset sync actor is invalid")
            exact = [
                job
                for job in self._jobs.values()
                if job.source_id == source_id
                and job.source_manifest_sha256 == source.version.source_manifest_sha256
                and job.export_id == export_id
                and job.repo_id == self._hub_repo_id
                and job.requested_revision == revision
            ]
            completed = [job for job in exact if job.status is DatasetHubSyncStatus.COMPLETED]
            if completed:
                return max(completed, key=lambda item: item.updated_at)
            active = [
                job
                for job in self._jobs.values()
                if job.source_id == source_id
                and job.status
                in {
                    DatasetHubSyncStatus.QUEUED,
                    DatasetHubSyncStatus.PREPARING_EXPORT,
                    DatasetHubSyncStatus.UPLOADING,
                }
            ]
            if active:
                current = max(active, key=lambda item: item.updated_at)
                if current in exact:
                    return current
                raise DatasetRegistryConflictError(
                    "another Hugging Face sync is already active for this source"
                )
            if len(self._jobs) >= self._config.maximum_jobs:
                raise DatasetRegistryValidationError("dataset sync job limit is reached")
            now = _now(self._clock)
            job = DatasetHubSyncJob(
                id=f"dataset-sync-{uuid.uuid4().hex}",
                source_id=source_id,
                source_manifest_sha256=source.version.source_manifest_sha256,
                export_id=export_id,
                repo_id=str(self._hub_repo_id),
                requested_revision=revision,
                status=DatasetHubSyncStatus.QUEUED,
                requested_by=requested_by.strip(),
                restricted_transfer_confirmed=command.restricted_transfer_confirmed,
                created_at=now,
                updated_at=now,
            )
            await asyncio.to_thread(self._write_job, job)
            self._jobs[job.id] = job
            return job

    async def run_sync_job(self, job_id: str) -> None:
        async with self._write_lock:
            job = self._job(job_id)
            if job.status is not DatasetHubSyncStatus.QUEUED:
                return
            preparing = replace(
                job,
                status=DatasetHubSyncStatus.PREPARING_EXPORT,
                updated_at=_now(self._clock),
                error_code=None,
            )
            await asyncio.to_thread(self._write_job, preparing)
            self._jobs[job_id] = preparing
        try:
            export, reused = await asyncio.to_thread(self._prepare_export, preparing)
            uploading = replace(
                preparing,
                status=DatasetHubSyncStatus.UPLOADING,
                updated_at=_now(self._clock),
                export_manifest_sha256=export.export.manifest_sha256,
                reused_export=reused,
            )
            async with self._write_lock:
                await asyncio.to_thread(self._write_job, uploading)
                self._jobs[job_id] = uploading
                self._exports[export.export.export_id] = export
            result = await asyncio.to_thread(self._upload, uploading, export.directory)
        except Exception as exc:
            logger.exception(
                "private dataset Hub synchronization failed",
                extra={"dataset_sync_job_id": job_id, "source_id": preparing.source_id},
            )
            failed = replace(
                self._jobs.get(job_id, preparing),
                status=DatasetHubSyncStatus.FAILED,
                updated_at=_now(self._clock),
                error_code=type(exc).__name__,
            )
            async with self._write_lock:
                await asyncio.to_thread(self._write_job, failed)
                self._jobs[job_id] = failed
            return
        completed = replace(
            uploading,
            status=DatasetHubSyncStatus.COMPLETED,
            updated_at=_now(self._clock),
            hub_commit_sha=result.revision,
            hub_url=result.url,
        )
        async with self._write_lock:
            await asyncio.to_thread(self._write_job, completed)
            self._jobs[job_id] = completed

    async def get_sync_job(self, job_id: str) -> DatasetHubSyncJob:
        return self._job(job_id)

    async def fail_queued_sync_job(self, job_id: str, error_code: str) -> None:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", error_code):
            raise DatasetRegistryValidationError("dataset sync error code is invalid")
        async with self._write_lock:
            job = self._job(job_id)
            if job.status is not DatasetHubSyncStatus.QUEUED:
                return
            failed = replace(
                job,
                status=DatasetHubSyncStatus.FAILED,
                updated_at=_now(self._clock),
                error_code=error_code,
            )
            await asyncio.to_thread(self._write_job, failed)
            self._jobs[job_id] = failed

    def _list_samples_sync(
        self,
        source: _SourceRecord,
        query: DetectorDatasetSampleQuery,
    ) -> DetectorDatasetSamplePage:
        annotations = source.files.get("annotations.jsonl")
        if annotations is None:
            raise DatasetRegistryStorageError("dataset source annotations are not manifested")
        path = _safe_source_child(source.directory, annotations.relative_path)
        _verify_file_shape(path, annotations)
        offset = (
            _decode_sample_cursor(query.cursor, source, query, annotations.size)
            if query.cursor
            else 0
        )
        maximum_line_bytes = (
            self._dataset_config.maximum_annotation_line_bytes
            if self._dataset_config is not None
            else 1_000_000
        )
        items: list[DetectorDatasetSamplePreview] = []
        try:
            with path.open("rb") as stream:
                if offset:
                    stream.seek(offset - 1)
                    if stream.read(1) != b"\n":
                        raise InvalidCursorError("dataset sample cursor is not a line boundary")
                stream.seek(offset)
                while len(items) < query.limit:
                    line = stream.readline(maximum_line_bytes + 1)
                    if not line:
                        break
                    if len(line) > maximum_line_bytes or not line.endswith(b"\n"):
                        raise DatasetRegistryStorageError(
                            "dataset source annotation line is oversized or incomplete"
                        )
                    sample = _parse_sample(line)
                    if _sample_matches(sample, query):
                        items.append(self._sample_preview(source, sample))
                next_offset = stream.tell()
        except OSError as exc:
            raise DatasetRegistryStorageError("cannot read dataset source annotations") from exc
        next_cursor = (
            _encode_sample_cursor(next_offset, source, query)
            if next_offset < annotations.size
            else None
        )
        return DetectorDatasetSamplePage(tuple(items), next_cursor)

    def _sample_preview(
        self,
        source: _SourceRecord,
        sample: DetectorSample,
    ) -> DetectorDatasetSamplePreview:
        image_sha256 = sample.attributes.get("sourceImageSha256")
        if not isinstance(image_sha256, str) or not _SHA256.fullmatch(image_sha256):
            raise DatasetRegistryStorageError("dataset sample image identity is invalid")
        expected = source.files.get(sample.image_path)
        indexed = source.images_by_sha256.get(image_sha256)
        if expected is None or indexed != expected or expected.sha256 != image_sha256:
            raise DatasetRegistryStorageError("dataset sample image is not bound to its manifest")
        data, image = self._verified_image(source, expected)
        height, width = image.shape[:2]
        annotations: list[DetectorDatasetSampleAnnotation] = []
        for annotation in sample.annotations:
            box = annotation.bbox
            if box.x + box.width > width + 1e-6 or box.y + box.height > height + 1e-6:
                raise DatasetRegistryStorageError("dataset sample bbox exceeds image bounds")
            annotations.append(
                DetectorDatasetSampleAnnotation(
                    class_name=annotation.class_name,
                    bbox=DetectorDatasetSampleBox(
                        x=box.x,
                        y=box.y,
                        width=box.width,
                        height=box.height,
                    ),
                )
            )
        del data
        lighting = sample.attributes.get("lighting")
        normalized_lighting = (
            lighting.strip().upper()
            if isinstance(lighting, str) and lighting.strip()
            else "UNKNOWN"
        )
        annotation_status = sample.attributes.get("annotationReviewStatus")
        return DetectorDatasetSamplePreview(
            source_id=source.version.source_id,
            sample_id=sample.sample_id,
            image_sha256=image_sha256,
            camera_id=sample.camera_id,
            group_id=sample.group_id,
            captured_at=sample.captured_at.astimezone(UTC),
            split=sample.split.value if sample.split is not None else None,
            lighting=normalized_lighting,
            annotation_status=(
                annotation_status.strip()
                if isinstance(annotation_status, str) and annotation_status.strip()
                else None
            ),
            negative=not bool(annotations),
            image_width=width,
            image_height=height,
            annotations=tuple(annotations),
        )

    def _get_sample_image_sync(
        self,
        source: _SourceRecord,
        image_sha256: str,
    ) -> DetectorDatasetSampleImage:
        expected = source.images_by_sha256.get(image_sha256)
        if expected is None:
            raise DatasetRegistryNotFoundError("dataset sample image not found")
        self._verified_image(source, expected)
        path = _safe_source_child(source.directory, expected.relative_path)
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return DetectorDatasetSampleImage(str(path), media_type, image_sha256)

    def _verified_image(
        self,
        source: _SourceRecord,
        expected: _ManifestFile,
    ) -> tuple[bytes, np.ndarray[Any, Any]]:
        path = _safe_source_child(source.directory, expected.relative_path)
        _verify_file_shape(path, expected)
        maximum_image_bytes = (
            self._dataset_config.maximum_image_bytes
            if self._dataset_config is not None
            else 20_000_000
        )
        if expected.size <= 0 or expected.size > maximum_image_bytes:
            raise DatasetRegistryStorageError("dataset sample image size exceeds policy")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DatasetRegistryStorageError("cannot read dataset sample image") from exc
        if _sha256(data) != expected.sha256:
            raise DatasetRegistryStorageError("dataset sample image checksum verification failed")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise DatasetRegistryStorageError("dataset sample image cannot be decoded")
        maximum_pixels = (
            self._dataset_config.maximum_image_pixels
            if self._dataset_config is not None
            else 40_000_000
        )
        if image.shape[0] * image.shape[1] > maximum_pixels:
            raise DatasetRegistryStorageError("dataset sample image dimensions exceed policy")
        return data, image

    def _initialize_sync(self) -> None:
        if not self._config.enabled:
            return
        if self._dataset_config is None or self._hub_repo_id is None or self._hub_enabled is None:
            settings = load_training_settings(self._config.training_config)
            target = settings.target(DetectorRole.PLATE)
            self._dataset_config = self._dataset_config or target.dataset
            self._hub_repo_id = self._hub_repo_id or target.hub.dataset_repo
            if self._hub_enabled is None:
                self._hub_enabled = settings.huggingface.enabled
        self._refresh_catalog_sync()
        self._jobs = self._load_jobs()

    def _refresh_catalog_sync(self) -> None:
        self._sources = self._load_sources()
        self._exports = self._load_exports()

    def _load_sources(self) -> dict[str, _SourceRecord]:
        if not self._sources_root.exists():
            return {}
        if not self._sources_root.is_dir() or self._sources_root.is_symlink():
            raise DatasetRegistryStorageError("dataset source catalog root is unsafe")
        roots = sorted(
            path for path in self._sources_root.iterdir() if path.is_dir() and not path.is_symlink()
        )
        if len(roots) > self._config.maximum_sources:
            raise DatasetRegistryStorageError("dataset source catalog exceeds configured limit")
        sources: dict[str, _SourceRecord] = {}
        for root in roots:
            manifest_path = root / "source-manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                continue
            raw, manifest = _read_manifest(manifest_path, "dataset source")
            source_id = manifest.get("sourceId")
            if manifest.get("type") in _REVIEW_ONLY_SOURCE_TYPES:
                if (
                    manifest.get("schemaVersion") != 1
                    or manifest.get("role") != "plate"
                    or not isinstance(source_id, str)
                    or not _IDENTIFIER.fullmatch(source_id)
                    or root.name != source_id
                    or manifest.get("releaseEligible") is not False
                    or manifest.get("distributionEligible") is not False
                    or manifest.get("promotionEligible") is not False
                ):
                    raise DatasetRegistryStorageError(
                        "review-only dataset source manifest is invalid"
                    )
                continue
            if (
                manifest.get("schemaVersion") != 1
                or manifest.get("type") != "FIRST_PARTY_DETECTOR_SOURCE"
                or manifest.get("role") != "plate"
                or not isinstance(source_id, str)
                or not _IDENTIFIER.fullmatch(source_id)
                or root.name != source_id
            ):
                raise DatasetRegistryStorageError("dataset source manifest is invalid")
            parent = manifest.get("parentSource")
            version = DetectorDatasetVersion(
                source_id=source_id,
                source_manifest_sha256=_sha256(raw),
                created_at=_datetime(manifest.get("createdAt"), "dataset source"),
                sample_count=_count(manifest, "sampleCount"),
                annotation_count=_count(manifest, "annotationCount"),
                negative_sample_count=_count(manifest, "negativeSampleCount"),
                review_queue_count=_count(manifest, "reviewQueueCount"),
                release_eligible=manifest.get("releaseEligible") is True,
                distribution_eligible=manifest.get("distributionEligible") is True,
                privacy_classification=str(manifest.get("privacyClassification", "UNKNOWN")),
                parent_source_id=(
                    str(parent["id"])
                    if isinstance(parent, dict) and isinstance(parent.get("id"), str)
                    else None
                ),
            )
            if source_id in sources:
                raise DatasetRegistryStorageError("dataset source id is duplicated")
            files = _manifest_files(manifest)
            images_by_sha256: dict[str, _ManifestFile] = {}
            for item in files.values():
                if not item.relative_path.startswith("images/"):
                    continue
                if item.sha256 in images_by_sha256:
                    raise DatasetRegistryStorageError(
                        "dataset source manifests a duplicate canonical image"
                    )
                images_by_sha256[item.sha256] = item
            sources[source_id] = _SourceRecord(
                root.resolve(),
                version,
                files,
                images_by_sha256,
            )
        return sources

    def _load_exports(self) -> dict[str, _ExportRecord]:
        if not self._exports_root.exists():
            return {}
        if not self._exports_root.is_dir() or self._exports_root.is_symlink():
            raise DatasetRegistryStorageError("dataset export catalog root is unsafe")
        roots = sorted(
            path for path in self._exports_root.iterdir() if path.is_dir() and not path.is_symlink()
        )
        if len(roots) > self._config.maximum_exports:
            raise DatasetRegistryStorageError("dataset export catalog exceeds configured limit")
        exports: dict[str, _ExportRecord] = {}
        for root in roots:
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                continue
            raw, manifest = _read_manifest(manifest_path, "dataset export")
            source = manifest.get("source")
            export_id = manifest.get("exportId")
            if (
                manifest.get("schemaVersion") != 1
                or manifest.get("type") != "DETECTOR_COCO"
                or manifest.get("role") != "plate"
                or not isinstance(export_id, str)
                or not _IDENTIFIER.fullmatch(export_id)
                or root.name != export_id
                or not isinstance(source, dict)
                or not isinstance(source.get("id"), str)
                or not isinstance(source.get("sourceManifestSha256"), str)
                or not _SHA256.fullmatch(source["sourceManifestSha256"])
            ):
                continue
            split_counts = _split_counts(manifest.get("splitCounts"))
            export = DetectorDatasetExport(
                export_id=export_id,
                manifest_sha256=_sha256(raw),
                created_at=_datetime(manifest.get("createdAt"), "dataset export"),
                sample_count=_count(manifest, "sampleCount"),
                annotation_count=_count(manifest, "annotationCount"),
                negative_sample_count=_count(manifest, "negativeSampleCount"),
                split_counts=split_counts,
                release_eligible=manifest.get("releaseEligible") is True,
                distribution_eligible=manifest.get("distributionEligible") is True,
                source_manifest_sha256=source["sourceManifestSha256"],
            )
            exports[export_id] = _ExportRecord(root.resolve(), export, source["id"])
        return exports

    def _load_jobs(self) -> dict[str, DatasetHubSyncJob]:
        directory = self._jobs_directory()
        if not directory.exists():
            return {}
        if not directory.is_dir() or directory.is_symlink():
            raise DatasetRegistryStorageError("dataset sync job directory is unsafe")
        paths = sorted(directory.glob("dataset-sync-*.json"))
        if len(paths) > self._config.maximum_jobs:
            raise DatasetRegistryStorageError("dataset sync job history exceeds configured limit")
        jobs: dict[str, DatasetHubSyncJob] = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise DatasetRegistryStorageError("dataset sync job path is unsafe")
            try:
                job = _job_from_json(json.loads(path.read_bytes()))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise DatasetRegistryStorageError("dataset sync job evidence is invalid") from exc
            if path.stem != job.id:
                raise DatasetRegistryStorageError("dataset sync job filename is invalid")
            if job.status in {
                DatasetHubSyncStatus.QUEUED,
                DatasetHubSyncStatus.PREPARING_EXPORT,
                DatasetHubSyncStatus.UPLOADING,
            }:
                job = replace(
                    job,
                    status=DatasetHubSyncStatus.FAILED,
                    updated_at=_now(self._clock),
                    error_code="INTERRUPTED",
                )
                self._write_job(job)
            jobs[job.id] = job
        return jobs

    def _effective_version(self, source: _SourceRecord) -> DetectorDatasetVersion:
        matching_exports = [
            item.export
            for item in self._exports.values()
            if item.source_id == source.version.source_id
            and item.export.source_manifest_sha256 == source.version.source_manifest_sha256
        ]
        export = (
            max(matching_exports, key=lambda item: (item.created_at, item.export_id))
            if matching_exports
            else None
        )
        jobs = [job for job in self._jobs.values() if job.source_id == source.version.source_id]
        latest_sync = max(jobs, key=lambda item: (item.created_at, item.id)) if jobs else None
        return replace(source.version, export=export, latest_sync=latest_sync)

    def _validate_sync(
        self,
        version: DetectorDatasetVersion,
        command: DatasetHubSyncCommand,
    ) -> None:
        if not self._hub_enabled or self._hub_repo_id is None:
            raise DatasetRegistryValidationError("Hugging Face dataset sync is disabled")
        if not self._credentials_configured():
            raise DatasetRegistryValidationError(
                "Hugging Face credentials are not configured for the API runtime"
            )
        if version.review_queue_count:
            raise DatasetRegistryValidationError(
                "dataset source still has pending review items and cannot be synchronized"
            )
        if not version.release_eligible:
            raise DatasetRegistryValidationError("dataset source is not release-eligible")
        if not version.distribution_eligible:
            if not self._config.restricted_private_sync_enabled:
                raise DatasetRegistryValidationError(
                    "restricted private dataset sync requires an explicit server policy opt-in"
                )
            if not command.restricted_transfer_confirmed:
                raise DatasetRegistryValidationError(
                    "restricted private dataset transfer must be explicitly confirmed"
                )

    def _prepare_export(self, job: DatasetHubSyncJob) -> tuple[_ExportRecord, bool]:
        source = self._source(job.source_id)
        if source.version.source_manifest_sha256 != job.source_manifest_sha256:
            raise DatasetRegistryConflictError("dataset source manifest changed after sync request")
        if self._dataset_config is None:
            raise DatasetRegistryStorageError("dataset export configuration is unavailable")
        export_config = self._dataset_config.model_copy(
            update={
                "source_directory": source.directory,
                "output_directory": self._exports_root,
            }
        )
        result = DetectorDatasetBuilder(export_config).build(job.export_id)
        manifest, digest = verify_detector_dataset(result.directory)
        evidence = manifest.get("source")
        if (
            manifest.get("role") != DetectorRole.PLATE.value
            or manifest.get("classes") != ["license_plate"]
            or not isinstance(evidence, dict)
            or evidence.get("id") != job.source_id
            or evidence.get("sourceManifestSha256") != job.source_manifest_sha256
        ):
            raise DatasetRegistryConflictError(
                "existing immutable export belongs to another source version"
            )
        export = DetectorDatasetExport(
            export_id=job.export_id,
            manifest_sha256=digest,
            created_at=_datetime(manifest.get("createdAt"), "dataset export"),
            sample_count=_count(manifest, "sampleCount"),
            annotation_count=_count(manifest, "annotationCount"),
            negative_sample_count=_count(manifest, "negativeSampleCount"),
            split_counts={
                str(key): int(value) for key, value in dict(manifest.get("splitCounts", {})).items()
            },
            release_eligible=manifest.get("releaseEligible") is True,
            distribution_eligible=manifest.get("distributionEligible") is True,
            source_manifest_sha256=job.source_manifest_sha256,
        )
        return _ExportRecord(result.directory.resolve(), export, job.source_id), result.reused

    def _upload(self, job: DatasetHubSyncJob, directory: Path) -> HubUploadResult:
        uploader = self._uploader or HuggingFacePrivateRegistry(token=_hub_token())
        source = self._source(job.source_id).version
        return uploader.upload_dataset(
            directory,
            job.repo_id,
            revision=job.requested_revision,
            allow_restricted_private=(
                not source.distribution_eligible and job.restricted_transfer_confirmed
            ),
        )

    def _source(self, source_id: str) -> _SourceRecord:
        source = self._sources.get(source_id)
        if source is None:
            raise DatasetRegistryNotFoundError(f"dataset source not found: {source_id}")
        return source

    def _job(self, job_id: str) -> DatasetHubSyncJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise DatasetRegistryNotFoundError(f"dataset sync job not found: {job_id}")
        return job

    def _credentials_configured(self) -> bool:
        return self._uploader is not None or _hub_token() is not None

    def _jobs_directory(self) -> Path:
        root = (self._workspace / "huggingface" / "jobs").resolve()
        if not root.is_relative_to(self._workspace):
            raise DatasetRegistryStorageError("dataset sync workspace path is unsafe")
        return root

    def _write_job(self, job: DatasetHubSyncJob) -> None:
        directory = self._jobs_directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = (directory / f"{job.id}.json").resolve()
        if not path.is_relative_to(directory) or not _JOB_ID.fullmatch(job.id):
            raise DatasetRegistryStorageError("dataset sync job path is unsafe")
        temporary = directory / f".{job.id}.{uuid.uuid4().hex}.tmp"
        payload = _json_bytes(_job_json(job))
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise DatasetRegistryStorageError("cannot persist dataset sync job") from exc


def _job_json(job: DatasetHubSyncJob) -> dict[str, object]:
    value = asdict(job)
    value["status"] = job.status.value
    value["created_at"] = _timestamp(job.created_at)
    value["updated_at"] = _timestamp(job.updated_at)
    value["schema_version"] = 1
    return value


def _job_from_json(value: object) -> DatasetHubSyncJob:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("dataset sync job contract is invalid")
    job_id = value.get("id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("dataset sync job id is invalid")
    source_sha = str(value.get("source_manifest_sha256", ""))
    export_sha = value.get("export_manifest_sha256")
    if not _SHA256.fullmatch(source_sha) or (
        export_sha is not None and not _SHA256.fullmatch(str(export_sha))
    ):
        raise ValueError("dataset sync manifest hash is invalid")
    return DatasetHubSyncJob(
        id=job_id,
        source_id=_identifier(str(value["source_id"]), "dataset source id"),
        source_manifest_sha256=source_sha,
        export_id=_identifier(str(value["export_id"]), "dataset export id"),
        repo_id=str(value["repo_id"]),
        requested_revision=_revision(str(value["requested_revision"])),
        status=DatasetHubSyncStatus(str(value["status"])),
        requested_by=str(value["requested_by"]),
        restricted_transfer_confirmed=bool(value["restricted_transfer_confirmed"]),
        created_at=_datetime(value["created_at"], "dataset sync job"),
        updated_at=_datetime(value["updated_at"], "dataset sync job"),
        export_manifest_sha256=str(export_sha) if export_sha is not None else None,
        hub_commit_sha=(
            str(value["hub_commit_sha"]) if value.get("hub_commit_sha") is not None else None
        ),
        hub_url=str(value["hub_url"]) if value.get("hub_url") is not None else None,
        reused_export=bool(value.get("reused_export", False)),
        error_code=str(value["error_code"]) if value.get("error_code") is not None else None,
    )


def _manifest_files(manifest: dict[str, Any]) -> dict[str, _ManifestFile]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or len(raw_files) > 1_000_100:
        raise DatasetRegistryStorageError("dataset source manifest files are invalid")
    files: dict[str, _ManifestFile] = {}
    for value in raw_files:
        if not isinstance(value, dict):
            raise DatasetRegistryStorageError("dataset source file evidence is invalid")
        relative = value.get("path")
        digest = value.get("sha256")
        size = value.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise DatasetRegistryStorageError("dataset source file evidence is invalid")
        path = PurePosixPath(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise DatasetRegistryStorageError("dataset source manifest path is unsafe")
        normalized = str(path)
        if normalized != relative or normalized in files:
            raise DatasetRegistryStorageError("dataset source manifest path is duplicated")
        files[normalized] = _ManifestFile(normalized, digest, size)
    return files


def _parse_sample(line: bytes) -> DetectorSample:
    try:
        return DetectorSample.model_validate_json(line)
    except ValidationError as exc:
        raise DatasetRegistryStorageError("dataset source sample contract is invalid") from exc


def _sample_matches(
    sample: DetectorSample,
    query: DetectorDatasetSampleQuery,
) -> bool:
    negative = not bool(sample.annotations)
    if query.kind is DetectorDatasetSampleKind.POSITIVE and negative:
        return False
    if query.kind is DetectorDatasetSampleKind.NEGATIVE and not negative:
        return False
    lighting = sample.attributes.get("lighting")
    normalized_lighting = (
        lighting.strip().upper() if isinstance(lighting, str) and lighting.strip() else "UNKNOWN"
    )
    return query.lighting is None or normalized_lighting == query.lighting


def _lighting_filter(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    normalized = value.strip().upper()
    if normalized not in {"DAY", "NIGHT", "UNKNOWN"}:
        raise DatasetRegistryValidationError("dataset sample lighting filter is invalid")
    return normalized


def _encode_sample_cursor(
    offset: int,
    source: _SourceRecord,
    query: DetectorDatasetSampleQuery,
) -> str:
    payload = {
        "schemaVersion": 1,
        "sourceId": source.version.source_id,
        "sourceManifestSha256": source.version.source_manifest_sha256,
        "offset": offset,
        "kind": query.kind.value,
        "lighting": query.lighting,
    }
    return base64.urlsafe_b64encode(_json_bytes(payload).strip()).decode().rstrip("=")


def _decode_sample_cursor(
    cursor: str,
    source: _SourceRecord,
    query: DetectorDatasetSampleQuery,
    maximum_offset: int,
) -> int:
    if not cursor or len(cursor) > 2048:
        raise InvalidCursorError("dataset sample cursor is invalid")
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(cursor + padding))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("dataset sample cursor is invalid") from exc
    offset = value.get("offset") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schemaVersion") != 1
        or value.get("sourceId") != source.version.source_id
        or value.get("sourceManifestSha256") != source.version.source_manifest_sha256
        or value.get("kind") != query.kind.value
        or value.get("lighting") != query.lighting
        or isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset <= 0
        or offset > maximum_offset
    ):
        raise InvalidCursorError("dataset sample cursor does not match source or filters")
    return offset


def _safe_source_child(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DatasetRegistryStorageError("dataset source path is unsafe")
    child = root.joinpath(*path.parts).resolve()
    if not child.is_relative_to(root):
        raise DatasetRegistryStorageError("dataset source path escapes its root")
    return child


def _verify_file_shape(path: Path, expected: _ManifestFile) -> None:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected.size:
            raise DatasetRegistryStorageError("dataset source file size verification failed")
    except OSError as exc:
        raise DatasetRegistryStorageError("cannot inspect dataset source file") from exc


def _read_manifest(path: Path, description: str) -> tuple[bytes, dict[str, Any]]:
    if path.stat().st_size <= 0 or path.stat().st_size > 100_000_000:
        raise DatasetRegistryStorageError(f"{description} manifest is oversized")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetRegistryStorageError(f"{description} manifest is invalid") from exc
    if not isinstance(value, dict):
        raise DatasetRegistryStorageError(f"{description} manifest root is invalid")
    return raw, value


def _identifier(value: str, description: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise DatasetRegistryValidationError(f"{description} is invalid")
    return normalized


def _revision(value: str) -> str:
    normalized = value.strip().strip("/")
    if not _REVISION.fullmatch(normalized) or ".." in normalized.split("/") or "//" in normalized:
        raise DatasetRegistryValidationError("Hugging Face revision is invalid")
    return normalized


def _count(value: dict[str, Any], key: str) -> int:
    count = value.get(key)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise DatasetRegistryStorageError(f"dataset manifest {key} is invalid")
    return count


def _split_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise DatasetRegistryStorageError("dataset export split counts are invalid")
    result: dict[str, int] = {}
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise DatasetRegistryStorageError("dataset export split counts are invalid")
        result[key] = count
    return result


def _datetime(value: object, description: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetRegistryStorageError(f"{description} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise DatasetRegistryStorageError(f"{description} timestamp has no timezone")
    return parsed.astimezone(UTC)


def _now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None:
        raise DatasetRegistryStorageError("dataset registry clock must be timezone-aware")
    return value.astimezone(UTC)


def _hub_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    cached = get_token()
    return cached.strip() if isinstance(cached, str) and cached.strip() else None


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
