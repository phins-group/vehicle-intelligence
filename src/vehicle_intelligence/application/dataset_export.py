"""Atomic, resumable OCR feedback export for offline retraining."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from vehicle_intelligence.application.ports import (
    DatasetSampleRepository,
    MediaObjectReader,
    VehicleEventRepository,
)
from vehicle_intelligence.config import DatasetExportConfig
from vehicle_intelligence.domain import DatasetSample
from vehicle_intelligence.exceptions import DatasetExportError, MediaStorageError

_EXPORT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


@dataclass(frozen=True, slots=True)
class DatasetExportResult:
    export_id: str
    directory: Path | None
    manifest_sha256: str | None
    exported_count: int
    failed_count: int
    split_counts: dict[str, int]
    reused: bool = False


class OCRDatasetExportService:
    def __init__(
        self,
        config: DatasetExportConfig,
        samples: DatasetSampleRepository,
        events: VehicleEventRepository,
        media: MediaObjectReader,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._samples = samples
        self._events = events
        self._media = media
        self._clock = clock
        self._root = config.output_directory.expanduser().resolve()

    async def export(
        self,
        export_id: str,
        limit: int | None = None,
    ) -> DatasetExportResult:
        if not _EXPORT_ID.fullmatch(export_id):
            raise DatasetExportError("dataset export id is not path-safe")
        batch_size = limit or self._config.batch_size
        if not 1 <= batch_size <= self._config.batch_size:
            raise DatasetExportError("dataset export limit exceeds configured batch size")
        now = self._clock()
        if now.tzinfo is None:
            raise DatasetExportError("dataset export clock must be timezone-aware")
        now = now.astimezone(UTC)
        target = self._target(export_id)
        if target.exists():
            return await self._resume_existing(export_id, target, now)

        claimed = await self._samples.claim_for_export(
            export_id,
            batch_size,
            now,
            now - timedelta(seconds=self._config.claim_stale_seconds),
        )
        if not claimed:
            return DatasetExportResult(export_id, None, None, 0, 0, {})

        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)
        temporary = self._root / f".{export_id}.tmp-{uuid.uuid4().hex}"
        records: list[dict[str, Any]] = []
        failed: list[str] = []
        failure_codes: dict[str, list[str]] = {}
        try:
            await asyncio.to_thread(temporary.mkdir, parents=False, exist_ok=False)
            for sample in claimed:
                record, error_code = await self._prepare_sample(temporary, sample)
                if record is None:
                    failed.append(sample.id)
                    failure_codes.setdefault(error_code or "SAMPLE_INVALID", []).append(sample.id)
                    continue
                records.append(record)
            for error_code, sample_ids in failure_codes.items():
                await self._samples.mark_export_failed(
                    tuple(sample_ids),
                    export_id,
                    error_code,
                )
            if not records:
                await asyncio.to_thread(_remove_tree, temporary, self._root)
                return DatasetExportResult(export_id, None, None, 0, len(failed), {})

            records.sort(key=lambda item: str(item["sampleId"]))
            labels = b"".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                for record in records
            )
            labels_path = temporary / "labels.jsonl"
            await asyncio.to_thread(_write_new, labels_path, labels)
            files = [
                {
                    "path": str(record["imagePath"]),
                    "sha256": str(record["imageSha256"]),
                    "size": int(record["imageSize"]),
                }
                for record in records
            ]
            files.append(
                {
                    "path": "labels.jsonl",
                    "sha256": _sha256(labels),
                    "size": len(labels),
                }
            )
            split_counts = dict(Counter(str(record["split"]) for record in records))
            model_counts = Counter(
                _model_key(record.get("prediction", {}).get("model")) for record in records
            )
            manifest = {
                "schemaVersion": 1,
                "exportId": export_id,
                "type": "PLATE_OCR",
                "createdAt": _timestamp(now),
                "sampleIds": [record["sampleId"] for record in records],
                "sampleCount": len(records),
                "failedCount": len(failed),
                "splitStrategy": {
                    "type": "CAMERA_HASH",
                    "seed": self._config.split_seed,
                    "ratios": {
                        "train": self._config.train_ratio,
                        "validation": self._config.validation_ratio,
                        "test": self._config.test_ratio,
                    },
                },
                "splitCounts": split_counts,
                "modelCounts": dict(sorted(model_counts.items())),
                "files": sorted(files, key=lambda item: str(item["path"])),
            }
            manifest_bytes = json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
            await asyncio.to_thread(_write_new, temporary / "manifest.json", manifest_bytes)
            manifest_sha256 = _sha256(manifest_bytes)
            await asyncio.to_thread(_install_directory, temporary, target)
            sample_ids = tuple(str(record["sampleId"]) for record in records)
            persisted = await self._samples.mark_exported(
                sample_ids,
                export_id,
                manifest_sha256,
                now,
            )
            if persisted != len(sample_ids):
                raise DatasetExportError(
                    "dataset artifact is complete but status reconciliation failed"
                )
            return DatasetExportResult(
                export_id=export_id,
                directory=target,
                manifest_sha256=manifest_sha256,
                exported_count=len(records),
                failed_count=len(failed),
                split_counts=split_counts,
            )
        except DatasetExportError:
            await asyncio.to_thread(_remove_tree, temporary, self._root)
            raise
        except Exception as exc:
            await self._samples.mark_export_failed(
                tuple(sample.id for sample in claimed if sample.id not in failed),
                export_id,
                "EXPORT_BUILD_FAILED",
            )
            await asyncio.to_thread(_remove_tree, temporary, self._root)
            raise DatasetExportError("cannot build dataset export") from exc

    async def close(self) -> None:
        try:
            await self._samples.close()
        finally:
            await self._events.close()

    async def _prepare_sample(
        self,
        temporary: Path,
        sample: DatasetSample,
    ) -> tuple[dict[str, Any] | None, str | None]:
        event = await self._events.get(sample.source_event_id)
        if event is None:
            return None, "SOURCE_EVENT_MISSING"
        try:
            source = await self._media.get(sample.image_key, self._config.maximum_image_bytes)
        except MediaStorageError:
            return None, "MEDIA_READ_FAILED"
        if source is None:
            return None, "MEDIA_MISSING"
        image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None, "MEDIA_INVALID"
        if image.shape[0] * image.shape[1] > self._config.maximum_image_pixels:
            return None, "MEDIA_DIMENSIONS_EXCEEDED"
        encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not encoded:
            return None, "MEDIA_ENCODE_FAILED"
        data = bytes(buffer)
        split = self._split(event.camera.id)
        filename = f"{hashlib.sha256(sample.id.encode()).hexdigest()[:24]}.jpg"
        relative = PurePosixPath("images", split, filename)
        destination = temporary.joinpath(*relative.parts)
        await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_write_new, destination, data)
        model = sample.prediction.model
        return (
            {
                "sampleId": sample.id,
                "sourceEventId": sample.source_event_id,
                "cameraId": event.camera.id,
                "split": split,
                "imagePath": str(relative),
                "imageSha256": _sha256(data),
                "imageSize": len(data),
                "label": sample.label,
                "reason": sample.reason.value,
                "reviewRevision": sample.review_revision,
                "prediction": {
                    "raw": sample.prediction.raw,
                    "normalized": sample.prediction.normalized,
                    "confidence": sample.prediction.confidence,
                    "model": (
                        {"name": model.name, "version": model.version, "hash": model.hash}
                        if model is not None
                        else None
                    ),
                },
            },
            None,
        )

    def _split(self, camera_id: str) -> str:
        digest = hashlib.sha256(f"{self._config.split_seed}:{camera_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        if value < self._config.train_ratio:
            return "train"
        if value < self._config.train_ratio + self._config.validation_ratio:
            return "validation"
        return "test"

    def _target(self, export_id: str) -> Path:
        target = (self._root / export_id).resolve()
        if not target.is_relative_to(self._root):
            raise DatasetExportError("dataset export target escapes configured root")
        return target

    async def _resume_existing(
        self,
        export_id: str,
        target: Path,
        now: datetime,
    ) -> DatasetExportResult:
        manifest, digest = await asyncio.to_thread(_verify_manifest, target, export_id)
        sample_ids = tuple(str(item) for item in manifest["sampleIds"])
        persisted = await self._samples.mark_exported(sample_ids, export_id, digest, now)
        if persisted != len(sample_ids):
            raise DatasetExportError("existing dataset artifact cannot reconcile sample states")
        return DatasetExportResult(
            export_id=export_id,
            directory=target,
            manifest_sha256=digest,
            exported_count=len(sample_ids),
            failed_count=int(manifest.get("failedCount", 0)),
            split_counts={
                str(key): int(value) for key, value in manifest.get("splitCounts", {}).items()
            },
            reused=True,
        )


def verify_dataset_export(directory: Path) -> tuple[dict[str, Any], str]:
    """Verify an immutable export before training or evaluation consumes it."""

    resolved = directory.expanduser().resolve()
    if not _EXPORT_ID.fullmatch(resolved.name):
        raise DatasetExportError("dataset export directory name is not a valid export id")
    return _verify_manifest(resolved, resolved.name)


def _verify_manifest(target: Path, export_id: str) -> tuple[dict[str, Any], str]:
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 5_000_000:
        raise DatasetExportError("existing dataset manifest is missing or oversized")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetExportError("existing dataset manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("exportId") != export_id
        or not isinstance(manifest.get("sampleIds"), list)
        or not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) > 5000
    ):
        raise DatasetExportError("existing dataset manifest contract is invalid")
    for item in manifest["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise DatasetExportError("dataset manifest file contract is invalid")
        path = _safe_child(target, item["path"])
        if not path.is_file() or path.stat().st_size != int(item.get("size", -1)):
            raise DatasetExportError("dataset export file size verification failed")
        if _sha256(path.read_bytes()) != item.get("sha256"):
            raise DatasetExportError("dataset export file checksum verification failed")
    return manifest, _sha256(raw)


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise DatasetExportError("dataset manifest contains an unsafe path")
    target = root.joinpath(*posix.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise DatasetExportError("dataset manifest path escapes export directory")
    return target


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _install_directory(temporary: Path, target: Path) -> None:
    if target.exists():
        raise DatasetExportError("dataset export id already exists")
    temporary.replace(target)


def _remove_tree(target: Path, root: Path) -> None:
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise DatasetExportError("refusing to remove an unsafe dataset path")
    if resolved.exists():
        shutil.rmtree(resolved)


def _model_key(model: Any) -> str:
    if not isinstance(model, dict):
        return "unknown"
    return f"{model.get('name', 'unknown')}@{model.get('version', 'unknown')}"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
