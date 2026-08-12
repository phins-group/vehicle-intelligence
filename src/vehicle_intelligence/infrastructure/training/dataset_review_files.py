"""Filesystem review overlay for immutable first-party detector sources."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from vehicle_intelligence.application.dataset_review import DetectorReviewQuery
from vehicle_intelligence.config import DatasetReviewConfig
from vehicle_intelligence.domain.dataset_review import (
    DetectorPromotionJob,
    DetectorPromotionStatus,
    DetectorReviewAction,
    DetectorReviewAnnotation,
    DetectorReviewBox,
    DetectorReviewDecision,
    DetectorReviewImage,
    DetectorReviewItem,
    DetectorReviewPage,
    DetectorReviewSourceSummary,
    DetectorReviewStatus,
)
from vehicle_intelligence.exceptions import (
    DatasetReviewConflictError,
    DatasetReviewNotFoundError,
    DatasetReviewStorageError,
    DatasetReviewValidationError,
    InvalidCursorError,
)
from vehicle_intelligence.training.video_review_source import VIDEO_REVIEW_SOURCE_TYPE

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_ID = re.compile(r"^review-[0-9a-f]{24}$")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SourceState:
    source_id: str
    root: Path
    manifest_sha256: str
    queue_sha256: str
    items: dict[str, DetectorReviewItem]
    source_type: str
    collection_method: str
    rights_status: str
    promotion_eligible: bool
    release_eligible: bool
    distribution_eligible: bool


@dataclass(frozen=True, slots=True)
class _ModelSuggestionState:
    run_id: str
    created_at: datetime
    annotations: tuple[DetectorReviewAnnotation, ...]


class FileDetectorReviewRepository:
    """Persist review revisions outside immutable source directories.

    Each revision is a separate file created with no-overwrite semantics. The
    source manifest and source review queue are never modified.
    """

    def __init__(
        self,
        config: DatasetReviewConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._sources_root = config.sources_directory.expanduser().resolve()
        self._workspace = config.workspace_directory.expanduser().resolve()
        self._promoted_root = config.promoted_sources_directory.expanduser().resolve()
        self._clock = clock
        self._sources: dict[str, _SourceState] = {}
        self._model_suggestions: dict[
            tuple[str, str], _ModelSuggestionState
        ] = {}
        self._decisions: dict[tuple[str, str], DetectorReviewDecision] = {}
        self._jobs: dict[str, DetectorPromotionJob] = {}
        self._dimensions: dict[tuple[str, str], tuple[int, int]] = {}
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if not self._config.enabled:
            return
        await asyncio.to_thread(self._initialize_sync)

    async def close(self) -> None:
        return None

    async def list_sources(self) -> tuple[DetectorReviewSourceSummary, ...]:
        summaries = [self._summary(source) for source in self._sources.values()]
        return tuple(sorted(summaries, key=lambda item: item.source_id))

    async def list_items(self, query: DetectorReviewQuery) -> DetectorReviewPage:
        source = self._source(query.source_id)
        after = _decode_cursor(query.cursor, query) if query.cursor else None
        selected: list[DetectorReviewItem] = []
        has_more = False
        for review_id in sorted(source.items):
            if after is not None and review_id <= after:
                continue
            item = self._effective_item(source.items[review_id])
            if query.status is not None and item.status is not query.status:
                continue
            if query.reason is not None and item.reason != query.reason:
                continue
            if len(selected) == query.limit:
                has_more = True
                break
            selected.append(item)
        next_cursor = (
            _encode_cursor(selected[-1].review_id, query)
            if has_more and selected
            else None
        )
        return DetectorReviewPage(tuple(selected), next_cursor)

    async def get_item(self, source_id: str, review_id: str) -> DetectorReviewItem:
        item = self._item(source_id, review_id)
        width, height = await asyncio.to_thread(self._image_dimensions, item)
        return replace(self._effective_item(item), image_width=width, image_height=height)

    async def get_image(self, source_id: str, review_id: str) -> DetectorReviewImage:
        item = self._item(source_id, review_id)
        path = self._image_path(item)
        await asyncio.to_thread(self._verify_image_bytes, item, path)
        media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return DetectorReviewImage(path, media_type, item.source_image_sha256)

    async def save_decision(
        self,
        source_id: str,
        review_id: str,
        decision: DetectorReviewDecision,
        expected_revision: int,
    ) -> DetectorReviewItem:
        item = self._item(source_id, review_id)
        async with self._write_lock:
            current = self._decisions.get((source_id, review_id))
            current_revision = current.revision if current is not None else 0
            if current_revision != expected_revision:
                raise DatasetReviewConflictError(
                    f"stale detector review revision: expected {expected_revision}, "
                    f"actual {current_revision}"
                )
            if decision.revision != expected_revision + 1:
                raise DatasetReviewValidationError("detector review revision is not sequential")
            await asyncio.to_thread(self._write_decision, item, decision)
            self._decisions[(source_id, review_id)] = decision
        width, height = await asyncio.to_thread(self._image_dimensions, item)
        return replace(
            self._effective_item(item),
            image_width=width,
            image_height=height,
        )

    async def decision_history(
        self,
        source_id: str,
        review_id: str,
    ) -> tuple[DetectorReviewDecision, ...]:
        self._item(source_id, review_id)
        return await asyncio.to_thread(self._load_history, source_id, review_id)

    async def create_promotion_job(
        self,
        source_id: str,
        target_source_id: str,
        requested_by: str,
    ) -> DetectorPromotionJob:
        source = self._source(source_id)
        if not source.promotion_eligible:
            raise DatasetReviewValidationError(
                "detector review source is not eligible for promotion"
            )
        if not _IDENTIFIER.fullmatch(target_source_id):
            raise DatasetReviewValidationError("target source id is not path-safe")
        if target_source_id == source_id:
            raise DatasetReviewValidationError("promotion must create a new source id")
        async with self._write_lock:
            if len(self._sources) >= self._config.maximum_sources:
                raise DatasetReviewValidationError(
                    "detector review source limit must be increased before promotion"
                )
            target = (self._promoted_root / target_source_id).resolve()
            if not target.is_relative_to(self._promoted_root) or target.exists():
                raise DatasetReviewConflictError(
                    "promoted source target already exists or is unsafe"
                )
            if any(
                existing.target_source_id == target_source_id
                and existing.status
                in {DetectorPromotionStatus.QUEUED, DetectorPromotionStatus.RUNNING}
                for existing in self._jobs.values()
            ):
                raise DatasetReviewConflictError(
                    "a promotion for this target is already running"
                )
            decision_revisions = {
                review_id: decision.revision
                for (decision_source_id, review_id), decision in self._decisions.items()
                if decision_source_id == source_id
            }
            if not decision_revisions:
                raise DatasetReviewValidationError(
                    "promotion requires at least one human review decision"
                )
            reviewed = sum(
                self._effective_item(item).status
                in {
                    DetectorReviewStatus.APPROVED,
                    DetectorReviewStatus.CORRECTED,
                    DetectorReviewStatus.NEGATIVE,
                }
                for item in source.items.values()
            )
            pending = sum(
                self._effective_item(item).status
                is DetectorReviewStatus.PENDING_REVIEW
                for item in source.items.values()
            )
            now = _now(self._clock)
            job_id = f"promotion-{uuid.uuid4().hex}"
            snapshot = _promotion_snapshot_bytes(source, decision_revisions)
            job = DetectorPromotionJob(
                id=job_id,
                source_id=source_id,
                target_source_id=target_source_id,
                status=DetectorPromotionStatus.QUEUED,
                created_at=now,
                updated_at=now,
                requested_by=requested_by,
                reviewed_sample_count=reviewed,
                pending_sample_count=pending,
                decision_snapshot_sha256=_sha256(snapshot),
            )
            await asyncio.to_thread(self._write_promotion_snapshot, job_id, snapshot)
            await asyncio.to_thread(self._write_job, job)
            self._jobs[job.id] = job
        return job

    async def run_promotion_job(self, job_id: str) -> None:
        async with self._write_lock:
            job = self._job(job_id)
            if job.status not in {
                DetectorPromotionStatus.QUEUED,
                DetectorPromotionStatus.FAILED,
            }:
                return
            running = replace(
                job,
                status=DetectorPromotionStatus.RUNNING,
                updated_at=_now(self._clock),
                error_code=None,
            )
            await asyncio.to_thread(self._write_job, running)
            self._jobs[job_id] = running
        try:
            decisions = await asyncio.to_thread(
                self._load_promotion_snapshot,
                running,
            )
            result = await asyncio.to_thread(self._promote_sync, running, decisions)
            promoted_source = await asyncio.to_thread(
                self._load_source,
                result[0],
                result[0] / "source-manifest.json",
                result[0] / "REVIEW_QUEUE.jsonl",
            )
        except Exception as exc:  # Job exposes a safe code; logs retain the traceback upstream.
            logger.exception(
                "detector dataset promotion failed",
                extra={
                    "promotion_job_id": job_id,
                    "source_id": running.source_id,
                    "target_source_id": running.target_source_id,
                },
            )
            failed = replace(
                running,
                status=DetectorPromotionStatus.FAILED,
                updated_at=_now(self._clock),
                error_code=type(exc).__name__,
            )
            async with self._write_lock:
                await asyncio.to_thread(self._write_job, failed)
                self._jobs[job_id] = failed
            return
        completed = replace(
            running,
            status=DetectorPromotionStatus.COMPLETED,
            updated_at=_now(self._clock),
            output_directory=str(result[0]),
            manifest_sha256=result[1],
        )
        async with self._write_lock:
            await asyncio.to_thread(self._write_job, completed)
            self._jobs[job_id] = completed
            self._sources[promoted_source.source_id] = promoted_source

    async def get_promotion_job(self, job_id: str) -> DetectorPromotionJob:
        return self._job(job_id)

    def _initialize_sync(self) -> None:
        self._sources = self._load_sources()
        self._model_suggestions = self._load_model_suggestions()
        self._decisions.clear()
        for source in self._sources.values():
            for review_id in source.items:
                history = self._load_history(source.source_id, review_id)
                if history:
                    self._decisions[(source.source_id, review_id)] = history[-1]
        self._jobs = self._load_jobs()

    def _load_sources(self) -> dict[str, _SourceState]:
        if not self._sources_root.exists():
            return {}
        if not self._sources_root.is_dir() or self._sources_root.is_symlink():
            raise DatasetReviewStorageError("detector review sources root is unsafe")
        candidates = [path for path in self._sources_root.iterdir() if path.is_dir()]
        candidates = [path for path in candidates if not path.name.startswith(".")]
        if len(candidates) > self._config.maximum_sources:
            raise DatasetReviewStorageError("detector review source count exceeds configured limit")
        sources: dict[str, _SourceState] = {}
        for root in sorted(candidates):
            manifest_path = root / "source-manifest.json"
            queue_path = root / "REVIEW_QUEUE.jsonl"
            if not manifest_path.is_file() or not queue_path.is_file():
                continue
            source = self._load_source(root, manifest_path, queue_path)
            if source.source_id in sources:
                raise DatasetReviewStorageError("detector review source id is duplicated")
            sources[source.source_id] = source
        return sources

    def _load_source(self, root: Path, manifest_path: Path, queue_path: Path) -> _SourceState:
        try:
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetReviewStorageError("detector source manifest is invalid") from exc
        source_id = manifest.get("sourceId") if isinstance(manifest, dict) else None
        source_type = manifest.get("type") if isinstance(manifest, dict) else None
        if (
            not isinstance(source_id, str)
            or not _IDENTIFIER.fullmatch(source_id)
            or source_type not in {"FIRST_PARTY_DETECTOR_SOURCE", VIDEO_REVIEW_SOURCE_TYPE}
            or manifest.get("role") != "plate"
            or not isinstance(manifest.get("files"), list)
        ):
            raise DatasetReviewStorageError("detector source is not a reviewable plate source")
        if source_type == VIDEO_REVIEW_SOURCE_TYPE and (
            manifest.get("licenseStatus") != "REVIEW_REQUIRED"
            or manifest.get("acceptanceEligible") is not False
            or manifest.get("releaseEligible") is not False
            or manifest.get("distributionEligible") is not False
            or manifest.get("promotionEligible") is not False
        ):
            raise DatasetReviewStorageError("video detector review source eligibility is invalid")
        file_entries = {
            entry.get("path"): entry
            for entry in manifest["files"]
            if isinstance(entry, dict) and isinstance(entry.get("path"), str)
        }
        queue_entry = file_entries.get("REVIEW_QUEUE.jsonl")
        queue_raw = queue_path.read_bytes()
        queue_sha = _sha256(queue_raw)
        if not isinstance(queue_entry, dict) or queue_entry.get("sha256") != queue_sha:
            raise DatasetReviewStorageError("detector review queue checksum does not match source")
        lines = queue_raw.splitlines()
        if len(lines) > self._config.maximum_queue_items_per_source:
            raise DatasetReviewStorageError("detector review queue exceeds configured limit")
        items: dict[str, DetectorReviewItem] = {}
        for line in lines:
            if not line.strip():
                continue
            item = self._parse_queue_item(source_id, root, line, file_entries)
            if item.review_id in items:
                raise DatasetReviewStorageError("detector review queue id is duplicated")
            items[item.review_id] = item
        expected = manifest.get("reviewQueueCount")
        if expected != len(items):
            raise DatasetReviewStorageError("detector review queue count does not match manifest")
        return _SourceState(
            source_id,
            root.resolve(),
            _sha256(manifest_raw),
            queue_sha,
            items,
            str(source_type),
            str(manifest.get("collectionMethod", "UNKNOWN")),
            str(manifest.get("licenseStatus", "UNKNOWN")),
            (
                manifest.get(
                    "promotionEligible",
                    manifest.get("releaseEligible") is True,
                )
                is True
            ),
            manifest.get("releaseEligible") is True,
            manifest.get("distributionEligible") is True,
        )

    def _parse_queue_item(
        self,
        source_id: str,
        root: Path,
        line: bytes,
        files: dict[object, dict[str, Any]],
    ) -> DetectorReviewItem:
        try:
            document = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetReviewStorageError("detector review queue record is invalid") from exc
        review_id = document.get("reviewId") if isinstance(document, dict) else None
        image_path = document.get("imagePath") if isinstance(document, dict) else None
        image_sha = document.get("sourceImageSha256") if isinstance(document, dict) else None
        filename_sha = document.get("sourceFilenameSha256") if isinstance(document, dict) else None
        reason = document.get("reason") if isinstance(document, dict) else None
        if (
            not isinstance(review_id, str)
            or not _REVIEW_ID.fullmatch(review_id)
            or not isinstance(image_path, str)
            or not isinstance(image_sha, str)
            or not _SHA256.fullmatch(image_sha)
            or not isinstance(filename_sha, str)
            or not _SHA256.fullmatch(filename_sha)
            or not isinstance(reason, str)
            or not reason
            or document.get("status") != "PENDING_REVIEW"
        ):
            raise DatasetReviewStorageError("detector review queue contract is invalid")
        entry = files.get(image_path)
        image = _safe_child(root, image_path)
        if (
            not isinstance(entry, dict)
            or entry.get("sha256") != image_sha
            or not image.is_file()
            or image.is_symlink()
        ):
            raise DatasetReviewStorageError("detector review image evidence is invalid")
        raw_suggestions = document.get("suggestions", [])
        if not isinstance(raw_suggestions, list) or len(raw_suggestions) > 16:
            raise DatasetReviewStorageError("detector review suggestions are invalid")
        suggestions = tuple(_annotation_from_json(value) for value in raw_suggestions)
        return DetectorReviewItem(
            source_id=source_id,
            review_id=review_id,
            image_path=image_path,
            source_image_sha256=image_sha,
            source_filename_sha256=filename_sha,
            reason=reason,
            suggestions=suggestions,
        )

    def _summary(self, source: _SourceState) -> DetectorReviewSourceSummary:
        items = [self._effective_item(item) for item in source.items.values()]
        statuses = Counter(item.status.value for item in items)
        reasons = Counter(item.reason for item in items)
        pending = statuses[DetectorReviewStatus.PENDING_REVIEW.value]
        return DetectorReviewSourceSummary(
            source_id=source.source_id,
            source_manifest_sha256=source.manifest_sha256,
            source_type=source.source_type,
            collection_method=source.collection_method,
            rights_status=source.rights_status,
            promotion_eligible=source.promotion_eligible,
            release_eligible=source.release_eligible,
            distribution_eligible=source.distribution_eligible,
            queue_count=len(items),
            status_counts=dict(sorted(statuses.items())),
            reason_counts=dict(sorted(reasons.items())),
            reviewed_count=len(items) - pending,
            pending_count=pending,
        )

    def _source(self, source_id: str) -> _SourceState:
        source = self._sources.get(source_id)
        if source is None:
            raise DatasetReviewNotFoundError(f"detector review source not found: {source_id}")
        return source

    def _item(self, source_id: str, review_id: str) -> DetectorReviewItem:
        source = self._source(source_id)
        item = source.items.get(review_id)
        if item is None:
            raise DatasetReviewNotFoundError(f"detector review item not found: {review_id}")
        return item

    def _job(self, job_id: str) -> DetectorPromotionJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise DatasetReviewNotFoundError(f"detector promotion job not found: {job_id}")
        return job

    def _effective_item(self, item: DetectorReviewItem) -> DetectorReviewItem:
        suggestion = self._model_suggestions.get((item.source_id, item.review_id))
        effective = item
        if suggestion is not None and not item.suggestions:
            effective = replace(
                item,
                reason="MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW",
                suggestions=suggestion.annotations,
            )
        decision = self._decisions.get((item.source_id, item.review_id))
        if decision is None:
            return effective
        return replace(
            effective,
            status=decision.status,
            revision=decision.revision,
            decision=decision,
        )

    def _load_model_suggestions(
        self,
    ) -> dict[tuple[str, str], _ModelSuggestionState]:
        loaded: dict[tuple[str, str], _ModelSuggestionState] = {}
        total = 0
        for source in self._sources.values():
            root = (self._workspace / source.source_id / "model-suggestions").resolve()
            if not root.exists():
                continue
            if not root.is_relative_to(self._workspace) or not root.is_dir() or root.is_symlink():
                raise DatasetReviewStorageError("detector model suggestion path is unsafe")
            for review_directory in sorted(root.iterdir()):
                if (
                    not review_directory.is_dir()
                    or review_directory.is_symlink()
                    or not _REVIEW_ID.fullmatch(review_directory.name)
                    or review_directory.name not in source.items
                ):
                    raise DatasetReviewStorageError(
                        "detector model suggestion review path is invalid"
                    )
                candidates: list[_ModelSuggestionState] = []
                paths = sorted(review_directory.glob("suggestion-*.json"))
                if len(paths) > 64:
                    raise DatasetReviewStorageError(
                        "detector model suggestion history exceeds limit"
                    )
                for path in paths:
                    if path.is_symlink() or path.stat().st_size > 1_000_000:
                        raise DatasetReviewStorageError(
                            "detector model suggestion evidence is unsafe"
                        )
                    try:
                        candidates.append(
                            _model_suggestion_from_json(
                                json.loads(path.read_bytes()),
                                source,
                                source.items[review_directory.name],
                                path.stem,
                            )
                        )
                    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                        raise DatasetReviewStorageError(
                            "detector model suggestion evidence is invalid"
                        ) from exc
                    total += 1
                    if total > self._config.maximum_queue_items_per_source * max(
                        len(self._sources), 1
                    ):
                        raise DatasetReviewStorageError(
                            "detector model suggestion evidence exceeds limit"
                        )
                if candidates:
                    loaded[(source.source_id, review_directory.name)] = max(
                        candidates,
                        key=lambda item: (item.created_at, item.run_id),
                    )
        return loaded

    def _image_path(self, item: DetectorReviewItem) -> Path:
        return _safe_child(self._source(item.source_id).root, item.image_path)

    def _image_dimensions(self, item: DetectorReviewItem) -> tuple[int, int]:
        key = (item.source_id, item.review_id)
        cached = self._dimensions.get(key)
        if cached is not None:
            return cached
        path = self._image_path(item)
        data = self._verify_image_bytes(item, path)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise DatasetReviewStorageError("detector review image cannot be decoded")
        height, width = image.shape[:2]
        if width * height > self._config.maximum_image_pixels:
            raise DatasetReviewStorageError("detector review image dimensions exceed limit")
        self._dimensions[key] = (width, height)
        return width, height

    def _verify_image_bytes(self, item: DetectorReviewItem, path: Path) -> bytes:
        size = path.stat().st_size
        if size <= 0 or size > self._config.maximum_image_bytes:
            raise DatasetReviewStorageError("detector review image size exceeds limit")
        data = path.read_bytes()
        if _sha256(data) != item.source_image_sha256:
            raise DatasetReviewStorageError("detector review image checksum verification failed")
        return data

    def _decision_directory(self, source_id: str, review_id: str) -> Path:
        path = (self._workspace / source_id / "decisions" / review_id).resolve()
        if not path.is_relative_to(self._workspace):
            raise DatasetReviewStorageError("detector review decision path is unsafe")
        return path

    def _write_decision(
        self,
        item: DetectorReviewItem,
        decision: DetectorReviewDecision,
    ) -> None:
        directory = self._decision_directory(item.source_id, item.review_id)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{decision.revision:08d}.json"
        payload = _decision_json(item, self._source(item.source_id), decision)
        _write_exclusive(destination, _json_bytes(payload))

    def _load_history(
        self,
        source_id: str,
        review_id: str,
    ) -> tuple[DetectorReviewDecision, ...]:
        directory = self._decision_directory(source_id, review_id)
        if not directory.exists():
            return ()
        if not directory.is_dir() or directory.is_symlink():
            raise DatasetReviewStorageError("detector review history path is unsafe")
        item = self._source(source_id).items[review_id]
        history: list[DetectorReviewDecision] = []
        for path in sorted(directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")):
            try:
                document = json.loads(path.read_bytes())
                decision = _decision_from_json(document, item, self._source(source_id))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise DatasetReviewStorageError("detector review history is invalid") from exc
            if decision.revision != len(history) + 1:
                raise DatasetReviewStorageError(
                    "detector review history revisions are not contiguous"
                )
            history.append(decision)
        return tuple(history)

    def _jobs_directory(self) -> Path:
        path = (self._workspace / "promotion-jobs").resolve()
        if not path.is_relative_to(self._workspace):
            raise DatasetReviewStorageError("detector promotion job path is unsafe")
        return path

    def _promotion_snapshot_path(self, job_id: str) -> Path:
        if not job_id.startswith("promotion-") or not _IDENTIFIER.fullmatch(job_id):
            raise DatasetReviewStorageError("detector promotion job id is unsafe")
        path = (self._jobs_directory() / f"{job_id}.decisions.json").resolve()
        if not path.is_relative_to(self._jobs_directory()):
            raise DatasetReviewStorageError("detector promotion snapshot path is unsafe")
        return path

    def _write_promotion_snapshot(self, job_id: str, payload: bytes) -> None:
        path = self._promotion_snapshot_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_exclusive(path, payload)

    def _load_promotion_snapshot(
        self,
        job: DetectorPromotionJob,
    ) -> dict[str, DetectorReviewDecision]:
        path = self._promotion_snapshot_path(job.id)
        try:
            payload = path.read_bytes()
            document = json.loads(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetReviewStorageError("detector promotion snapshot is invalid") from exc
        source = self._source(job.source_id)
        if (
            _sha256(payload) != job.decision_snapshot_sha256
            or not isinstance(document, dict)
            or document.get("schemaVersion") != 1
            or document.get("sourceId") != source.source_id
            or document.get("sourceManifestSha256") != source.manifest_sha256
            or document.get("sourceQueueSha256") != source.queue_sha256
            or not isinstance(document.get("decisions"), dict)
        ):
            raise DatasetReviewStorageError("detector promotion snapshot evidence is invalid")
        decisions: dict[str, DetectorReviewDecision] = {}
        for review_id, revision_value in document["decisions"].items():
            if not isinstance(review_id, str) or not _REVIEW_ID.fullmatch(review_id):
                raise DatasetReviewStorageError("detector promotion review id is invalid")
            if isinstance(revision_value, bool) or not isinstance(revision_value, int):
                raise DatasetReviewStorageError("detector promotion revision is invalid")
            history = self._load_history(source.source_id, review_id)
            if revision_value < 1 or revision_value > len(history):
                raise DatasetReviewStorageError("detector promotion revision is unavailable")
            decisions[review_id] = history[revision_value - 1]
        if not decisions:
            raise DatasetReviewStorageError("detector promotion snapshot is empty")
        return decisions

    def _write_job(self, job: DetectorPromotionJob) -> None:
        directory = self._jobs_directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{job.id}.json"
        _atomic_replace(path, _json_bytes(_job_json(job)))

    def _load_jobs(self) -> dict[str, DetectorPromotionJob]:
        directory = self._jobs_directory()
        if not directory.exists():
            return {}
        jobs: dict[str, DetectorPromotionJob] = {}
        for path in sorted(directory.glob("promotion-*.json")):
            if path.name.endswith(".decisions.json"):
                continue
            try:
                job = _job_from_json(json.loads(path.read_bytes()))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise DatasetReviewStorageError("detector promotion job is invalid") from exc
            if job.status in {
                DetectorPromotionStatus.QUEUED,
                DetectorPromotionStatus.RUNNING,
            }:
                job = self._recover_interrupted_job(job)
                self._write_job(job)
            jobs[job.id] = job
        return jobs

    def _recover_interrupted_job(self, job: DetectorPromotionJob) -> DetectorPromotionJob:
        target = self._promoted_root / job.target_source_id
        if target.is_dir():
            try:
                from vehicle_intelligence.training.first_party import (
                    verify_first_party_detector_source,
                )

                manifest, digest = verify_first_party_detector_source(target)
                if manifest.get("sourceId") == job.target_source_id:
                    return replace(
                        job,
                        status=DetectorPromotionStatus.COMPLETED,
                        updated_at=_now(self._clock),
                        output_directory=str(target.resolve()),
                        manifest_sha256=digest,
                        error_code=None,
                    )
            except Exception:
                logger.exception(
                    "interrupted detector promotion target could not be verified",
                    extra={"promotion_job_id": job.id},
                )
        return replace(
            job,
            status=DetectorPromotionStatus.FAILED,
            updated_at=_now(self._clock),
            error_code="PROCESS_RESTARTED",
        )

    def _promote_sync(
        self,
        job: DetectorPromotionJob,
        decisions: dict[str, DetectorReviewDecision],
    ) -> tuple[Path, str]:
        from vehicle_intelligence.training.review_promotion import (
            ReviewedFirstPartySourceBuilder,
        )

        source = self._source(job.source_id)
        result = ReviewedFirstPartySourceBuilder(
            source_directory=source.root,
            output_directory=self._promoted_root / job.target_source_id,
            target_source_id=job.target_source_id,
            decisions=decisions,
            model_suggestions={
                review_id: state.annotations
                for (source_id, review_id), state in self._model_suggestions.items()
                if source_id == source.source_id
            },
            clock=self._clock,
        ).build()
        return result.directory, result.manifest_sha256


def _annotation_from_json(value: object) -> DetectorReviewAnnotation:
    if not isinstance(value, dict):
        raise DatasetReviewStorageError("detector review annotation is not an object")
    bbox = value.get("bbox")
    if not isinstance(bbox, dict):
        raise DatasetReviewStorageError("detector review annotation bbox is invalid")
    attributes = value.get("attributes", {})
    if not isinstance(attributes, dict):
        raise DatasetReviewStorageError("detector review annotation attributes are invalid")
    try:
        return DetectorReviewAnnotation(
            class_name=str(value.get("className", "license_plate")),
            bbox=DetectorReviewBox(
                x=float(bbox["x"]),
                y=float(bbox["y"]),
                width=float(bbox["width"]),
                height=float(bbox["height"]),
            ),
            attributes={str(key): item for key, item in attributes.items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetReviewStorageError("detector review annotation is invalid") from exc


def _annotation_json(annotation: DetectorReviewAnnotation) -> dict[str, object]:
    return {
        "className": annotation.class_name,
        "bbox": asdict(annotation.bbox),
        "attributes": annotation.attributes,
    }


def _model_suggestion_from_json(
    value: object,
    source: _SourceState,
    item: DetectorReviewItem,
    filename_run_id: str,
) -> _ModelSuggestionState:
    if not isinstance(value, dict):
        raise ValueError("model suggestion evidence must be an object")
    run_id = value.get("runId")
    model = value.get("model")
    suggestions = value.get("suggestions")
    if (
        value.get("schemaVersion") != 1
        or value.get("type") != "DETECTOR_MODEL_SUGGESTION"
        or value.get("sourceId") != source.source_id
        or value.get("sourceManifestSha256") != source.manifest_sha256
        or value.get("sourceQueueSha256") != source.queue_sha256
        or value.get("reviewId") != item.review_id
        or value.get("sourceImageSha256") != item.source_image_sha256
        or value.get("status") != "PENDING_HUMAN_REVIEW"
        or not isinstance(run_id, str)
        or not _IDENTIFIER.fullmatch(run_id)
        or run_id != filename_run_id
        or not isinstance(model, dict)
        or not isinstance(model.get("provider"), str)
        or not str(model["provider"]).strip()
        or not isinstance(model.get("name"), str)
        or not str(model["name"]).strip()
        or not isinstance(model.get("version"), str)
        or not str(model["version"]).strip()
        or not isinstance(model.get("sha256"), str)
        or not _SHA256.fullmatch(str(model["sha256"]))
        or not isinstance(suggestions, list)
        or not 1 <= len(suggestions) <= 16
    ):
        raise ValueError("model suggestion evidence binding is invalid")
    created_at = datetime.fromisoformat(str(value["createdAt"]).replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        raise ValueError("model suggestion timestamp has no timezone")
    annotations = tuple(_annotation_from_json(suggestion) for suggestion in suggestions)
    model_hash = str(model["sha256"])
    if any(annotation.attributes.get("modelHash") != model_hash for annotation in annotations):
        raise ValueError("model suggestion annotation metadata is inconsistent")
    return _ModelSuggestionState(run_id, created_at.astimezone(UTC), annotations)


def _decision_json(
    item: DetectorReviewItem,
    source: _SourceState,
    decision: DetectorReviewDecision,
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "sourceId": item.source_id,
        "sourceManifestSha256": source.manifest_sha256,
        "sourceQueueSha256": source.queue_sha256,
        "reviewId": item.review_id,
        "sourceImageSha256": item.source_image_sha256,
        "action": decision.action.value,
        "status": decision.status.value,
        "annotations": [_annotation_json(item) for item in decision.annotations],
        "revision": decision.revision,
        "reviewedBy": decision.reviewed_by,
        "reviewerDisplayName": decision.reviewer_display_name,
        "reviewedAt": _timestamp(decision.reviewed_at),
        "note": decision.note,
    }


def _decision_from_json(
    value: object,
    item: DetectorReviewItem,
    source: _SourceState,
) -> DetectorReviewDecision:
    if not isinstance(value, dict):
        raise ValueError("review decision must be an object")
    if (
        value.get("schemaVersion") != 1
        or value.get("sourceId") != source.source_id
        or value.get("sourceManifestSha256") != source.manifest_sha256
        or value.get("sourceQueueSha256") != source.queue_sha256
        or value.get("reviewId") != item.review_id
        or value.get("sourceImageSha256") != item.source_image_sha256
    ):
        raise ValueError("review decision evidence binding is invalid")
    annotations = value.get("annotations")
    if not isinstance(annotations, list) or len(annotations) > 16:
        raise ValueError("review decision annotations are invalid")
    reviewed_at = datetime.fromisoformat(str(value["reviewedAt"]).replace("Z", "+00:00"))
    if reviewed_at.tzinfo is None:
        raise ValueError("review decision timestamp has no timezone")
    note = value.get("note")
    return DetectorReviewDecision(
        action=DetectorReviewAction(str(value["action"])),
        status=DetectorReviewStatus(str(value["status"])),
        annotations=tuple(_annotation_from_json(annotation) for annotation in annotations),
        revision=int(value["revision"]),
        reviewed_by=str(value["reviewedBy"]),
        reviewer_display_name=str(value["reviewerDisplayName"]),
        reviewed_at=reviewed_at.astimezone(UTC),
        note=str(note) if note is not None else None,
    )


def _job_json(job: DetectorPromotionJob) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "id": job.id,
        "sourceId": job.source_id,
        "targetSourceId": job.target_source_id,
        "status": job.status.value,
        "createdAt": _timestamp(job.created_at),
        "updatedAt": _timestamp(job.updated_at),
        "requestedBy": job.requested_by,
        "reviewedSampleCount": job.reviewed_sample_count,
        "pendingSampleCount": job.pending_sample_count,
        "decisionSnapshotSha256": job.decision_snapshot_sha256,
        "outputDirectory": job.output_directory,
        "manifestSha256": job.manifest_sha256,
        "errorCode": job.error_code,
    }


def _job_from_json(value: object) -> DetectorPromotionJob:
    if not isinstance(value, dict) or value.get("schemaVersion") != 1:
        raise ValueError("promotion job contract is invalid")
    created_at = datetime.fromisoformat(str(value["createdAt"]).replace("Z", "+00:00"))
    updated_at = datetime.fromisoformat(str(value["updatedAt"]).replace("Z", "+00:00"))
    snapshot_sha256 = str(value["decisionSnapshotSha256"])
    if not _SHA256.fullmatch(snapshot_sha256):
        raise ValueError("promotion decision snapshot checksum is invalid")
    return DetectorPromotionJob(
        id=str(value["id"]),
        source_id=str(value["sourceId"]),
        target_source_id=str(value["targetSourceId"]),
        status=DetectorPromotionStatus(str(value["status"])),
        created_at=created_at,
        updated_at=updated_at,
        requested_by=str(value["requestedBy"]),
        reviewed_sample_count=int(value["reviewedSampleCount"]),
        pending_sample_count=int(value["pendingSampleCount"]),
        decision_snapshot_sha256=snapshot_sha256,
        output_directory=(
            str(value["outputDirectory"]) if value.get("outputDirectory") is not None else None
        ),
        manifest_sha256=(
            str(value["manifestSha256"]) if value.get("manifestSha256") is not None else None
        ),
        error_code=str(value["errorCode"]) if value.get("errorCode") is not None else None,
    )


def _promotion_snapshot_bytes(
    source: _SourceState,
    decision_revisions: dict[str, int],
) -> bytes:
    return _json_bytes(
        {
            "schemaVersion": 1,
            "sourceId": source.source_id,
            "sourceManifestSha256": source.manifest_sha256,
            "sourceQueueSha256": source.queue_sha256,
            "decisions": dict(sorted(decision_revisions.items())),
        }
    )


def _encode_cursor(review_id: str, query: DetectorReviewQuery) -> str:
    payload = {
        "source": query.source_id,
        "after": review_id,
        "status": query.status.value if query.status is not None else None,
        "reason": query.reason,
    }
    return base64.urlsafe_b64encode(_json_bytes(payload).strip()).decode().rstrip("=")


def _decode_cursor(cursor: str, query: DetectorReviewQuery) -> str:
    try:
        padding = "=" * (-len(cursor) % 4)
        value = json.loads(base64.urlsafe_b64decode(cursor + padding))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("detector review cursor is invalid") from exc
    expected_status = query.status.value if query.status is not None else None
    if (
        not isinstance(value, dict)
        or value.get("source") != query.source_id
        or value.get("status") != expected_status
        or value.get("reason") != query.reason
        or not isinstance(value.get("after"), str)
        or not _REVIEW_ID.fullmatch(value["after"])
    ):
        raise InvalidCursorError("detector review cursor does not match filters")
    return str(value["after"])


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DatasetReviewStorageError("detector review source path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DatasetReviewStorageError("detector review source path escapes its root")
    return path


def _write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DatasetReviewConflictError("detector review revision already exists") from exc


def _atomic_replace(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None:
        raise DatasetReviewStorageError("detector review clock must be timezone-aware")
    return value.astimezone(UTC)
