"""Generate model suggestions without mutating immutable review sources.

Suggestions are evidence overlays only.  They remain pending human review and
never enter a detector training export until an operator approves or corrects
them and promotes the source.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.application.ports import (
    BatchPlateDetector,
    PlateDetector,
)
from vehicle_intelligence.domain import PlateDetection
from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.first_party import (
    verify_first_party_detector_source,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REVIEW_ID = re.compile(r"^review-[0-9a-f]{24}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ReviewSuggestionModel:
    provider: str
    name: str
    version: str
    sha256: str
    confidence: float
    iou: float
    image_size: int

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.name.strip() or not self.version.strip():
            raise ValueError("review suggestion model identity is required")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("review suggestion model SHA-256 is invalid")
        if not 0 <= self.confidence <= 1 or not 0 <= self.iou <= 1:
            raise ValueError("review suggestion thresholds must be in [0, 1]")
        if self.image_size <= 0:
            raise ValueError("review suggestion image size must be positive")

    def as_json(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "confidence": self.confidence,
            "iou": self.iou,
            "imageSize": self.image_size,
        }


@dataclass(frozen=True, slots=True)
class ReviewSuggestionOptions:
    source_directory: Path
    workspace_directory: Path
    batch_size: int = 4
    maximum_suggestions_per_image: int = 4
    maximum_image_bytes: int = 20_000_000
    maximum_image_pixels: int = 40_000_000
    limit: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 64:
            raise ValueError("review suggestion batch size must be in [1, 64]")
        if not 1 <= self.maximum_suggestions_per_image <= 16:
            raise ValueError("maximum suggestions per image must be in [1, 16]")
        if self.maximum_image_bytes <= 0 or self.maximum_image_pixels <= 0:
            raise ValueError("review suggestion image limits must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("review suggestion limit must be positive")


@dataclass(frozen=True, slots=True)
class ReviewSuggestionResult:
    source_id: str
    source_manifest_sha256: str
    suggestion_run_id: str
    candidates: int
    scanned: int
    suggested_items: int
    suggestion_boxes: int
    no_detection: int
    skipped_human_reviewed: int
    skipped_source_suggestions: int
    reused_evidence: int
    failures: tuple[str, ...]
    workspace_directory: Path


@dataclass(frozen=True, slots=True)
class _Candidate:
    review_id: str
    image_path: str
    image_sha256: str


class DetectorReviewSuggestionGenerator:
    """Run a plate detector over unresolved review images and write overlays."""

    def __init__(
        self,
        detector: PlateDetector,
        model: ReviewSuggestionModel,
        options: ReviewSuggestionOptions,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        progress: Callable[[str, dict[str, object]], None] | None = None,
    ) -> None:
        self._detector = detector
        self._model = model
        self._options = options
        self._clock = clock
        self._progress = progress

    def generate(self) -> ReviewSuggestionResult:
        source = self._options.source_directory.expanduser().resolve()
        workspace = self._options.workspace_directory.expanduser().resolve()
        manifest, manifest_sha256 = verify_first_party_detector_source(source)
        source_id = str(manifest.get("sourceId", ""))
        if not _IDENTIFIER.fullmatch(source_id):
            raise DetectorDatasetError("review suggestion source id is invalid")
        queue_path = source / "REVIEW_QUEUE.jsonl"
        queue_bytes = queue_path.read_bytes()
        queue_sha256 = _sha256(queue_bytes)
        run_id = _run_id(self._model, self._options.maximum_suggestions_per_image)
        started_at = _now(self._clock)
        candidates, skipped_source, skipped_reviewed = self._candidates(
            source_id,
            source,
            workspace,
            queue_bytes,
        )
        if self._options.limit is not None:
            candidates = candidates[: self._options.limit]

        scanned = 0
        suggested_items = 0
        suggestion_boxes = 0
        no_detection = 0
        reused_evidence = 0
        failures: list[str] = []
        self._emit(
            "review_suggestions_started",
            {
                "sourceId": source_id,
                "runId": run_id,
                "candidates": len(candidates),
            },
        )
        for offset in range(0, len(candidates), self._options.batch_size):
            batch_candidates = candidates[offset : offset + self._options.batch_size]
            pending: list[_Candidate] = []
            images: list[NDArray[np.uint8]] = []
            for candidate in batch_candidates:
                evidence_path = _evidence_path(
                    workspace,
                    source_id,
                    candidate.review_id,
                    run_id,
                )
                if evidence_path.is_file():
                    _verify_existing_evidence(
                        evidence_path,
                        source_id=source_id,
                        source_manifest_sha256=manifest_sha256,
                        source_queue_sha256=queue_sha256,
                        candidate=candidate,
                        run_id=run_id,
                    )
                    reused_evidence += 1
                    continue
                try:
                    images.append(self._read_image(source, candidate))
                    pending.append(candidate)
                except DetectorDatasetError:
                    failures.append(candidate.review_id)
            if pending:
                results = self._infer_with_isolation(pending, images, failures)
                for candidate, detections in results:
                    scanned += 1
                    suggestions = _suggestions(
                        detections,
                        self._model,
                        run_id,
                        self._options.maximum_suggestions_per_image,
                    )
                    if not suggestions:
                        no_detection += 1
                        continue
                    evidence = {
                        "schemaVersion": 1,
                        "type": "DETECTOR_MODEL_SUGGESTION",
                        "sourceId": source_id,
                        "sourceManifestSha256": manifest_sha256,
                        "sourceQueueSha256": queue_sha256,
                        "reviewId": candidate.review_id,
                        "sourceImageSha256": candidate.image_sha256,
                        "createdAt": _timestamp(started_at),
                        "runId": run_id,
                        "status": "PENDING_HUMAN_REVIEW",
                        "model": self._model.as_json(),
                        "suggestions": suggestions,
                    }
                    destination = _evidence_path(
                        workspace,
                        source_id,
                        candidate.review_id,
                        run_id,
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _write_new(destination, _json_bytes(evidence))
                    suggested_items += 1
                    suggestion_boxes += len(suggestions)
            self._emit(
                "review_suggestions_progress",
                {
                    "sourceId": source_id,
                    "processed": min(offset + len(batch_candidates), len(candidates)),
                    "candidates": len(candidates),
                    "suggestedItems": suggested_items,
                    "noDetection": no_detection,
                    "failures": len(failures),
                },
            )

        result = ReviewSuggestionResult(
            source_id=source_id,
            source_manifest_sha256=manifest_sha256,
            suggestion_run_id=run_id,
            candidates=len(candidates),
            scanned=scanned,
            suggested_items=suggested_items,
            suggestion_boxes=suggestion_boxes,
            no_detection=no_detection,
            skipped_human_reviewed=skipped_reviewed,
            skipped_source_suggestions=skipped_source,
            reused_evidence=reused_evidence,
            failures=tuple(failures),
            workspace_directory=workspace,
        )
        self._emit(
            "review_suggestions_completed",
            {
                "sourceId": source_id,
                "runId": run_id,
                "suggestedItems": suggested_items,
                "suggestionBoxes": suggestion_boxes,
                "noDetection": no_detection,
                "failures": len(failures),
            },
        )
        return result

    def _candidates(
        self,
        source_id: str,
        source: Path,
        workspace: Path,
        queue_bytes: bytes,
    ) -> tuple[list[_Candidate], int, int]:
        candidates: list[_Candidate] = []
        skipped_source = 0
        skipped_reviewed = 0
        for line in queue_bytes.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DetectorDatasetError("review suggestion queue is invalid") from exc
            candidate = _candidate(record, source)
            suggestions = record.get("suggestions")
            if not isinstance(suggestions, list):
                raise DetectorDatasetError("review suggestion queue annotations are invalid")
            if suggestions:
                skipped_source += 1
                continue
            decision_directory = workspace / source_id / "decisions" / candidate.review_id
            if decision_directory.is_dir() and any(
                decision_directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].json")
            ):
                skipped_reviewed += 1
                continue
            candidates.append(candidate)
        return sorted(candidates, key=lambda item: item.review_id), skipped_source, skipped_reviewed

    def _read_image(self, source: Path, candidate: _Candidate) -> NDArray[np.uint8]:
        path = _safe_child(source, candidate.image_path)
        if not path.is_file() or path.is_symlink():
            raise DetectorDatasetError("review suggestion image is missing or unsafe")
        size = path.stat().st_size
        if size <= 0 or size > self._options.maximum_image_bytes:
            raise DetectorDatasetError("review suggestion image size exceeds limit")
        data = path.read_bytes()
        if _sha256(data) != candidate.image_sha256:
            raise DetectorDatasetError("review suggestion image checksum does not match")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise DetectorDatasetError("review suggestion image cannot be decoded")
        height, width = image.shape[:2]
        if width * height > self._options.maximum_image_pixels:
            raise DetectorDatasetError("review suggestion image dimensions exceed limit")
        return image

    def _infer_with_isolation(
        self,
        candidates: Sequence[_Candidate],
        images: Sequence[NDArray[np.uint8]],
        failures: list[str],
    ) -> list[tuple[_Candidate, list[PlateDetection]]]:
        if isinstance(self._detector, BatchPlateDetector):
            try:
                batches = self._detector.detect_batch(images)
                if len(batches) != len(images):
                    raise DetectorDatasetError("plate detector batch result count does not match")
                return [
                    (candidate, list(detections))
                    for candidate, detections in zip(candidates, batches, strict=True)
                ]
            except Exception:
                # Retry independently so one bad image cannot discard a completed batch.
                pass
        results: list[tuple[_Candidate, list[PlateDetection]]] = []
        for candidate, image in zip(candidates, images, strict=True):
            try:
                results.append((candidate, self._detector.detect(image)))
            except Exception:
                failures.append(candidate.review_id)
        return results

    def _emit(self, event: str, payload: dict[str, object]) -> None:
        if self._progress is not None:
            self._progress(event, payload)


def _candidate(value: object, source: Path) -> _Candidate:
    if not isinstance(value, dict):
        raise DetectorDatasetError("review suggestion queue record is invalid")
    review_id = value.get("reviewId")
    image_path = value.get("imagePath")
    image_sha256 = value.get("sourceImageSha256")
    if (
        not isinstance(review_id, str)
        or not _REVIEW_ID.fullmatch(review_id)
        or not isinstance(image_path, str)
        or not isinstance(image_sha256, str)
        or not _SHA256.fullmatch(image_sha256)
    ):
        raise DetectorDatasetError("review suggestion queue identity is invalid")
    _safe_child(source, image_path)
    return _Candidate(review_id, image_path, image_sha256)


def _suggestions(
    detections: Sequence[PlateDetection],
    model: ReviewSuggestionModel,
    run_id: str,
    maximum: int,
) -> list[dict[str, object]]:
    selected = sorted(detections, key=lambda item: item.confidence, reverse=True)[:maximum]
    return [
        {
            "className": "license_plate",
            "bbox": {
                "x": float(detection.bbox.x1),
                "y": float(detection.bbox.y1),
                "width": float(detection.bbox.width),
                "height": float(detection.bbox.height),
            },
            "attributes": {
                "annotationSource": "MODEL_SUGGESTION",
                "reviewStatus": "PENDING_REVIEW",
                "confidence": round(detection.confidence, 6),
                "modelProvider": model.provider,
                "modelName": model.name,
                "modelVersion": model.version,
                "modelHash": model.sha256,
                "suggestionRunId": run_id,
                "layoutSuggestion": (
                    "SINGLE_LINE"
                    if detection.bbox.width / detection.bbox.height >= 2.5
                    else "TWO_LINE"
                ),
            },
        }
        for detection in selected
    ]


def _run_id(model: ReviewSuggestionModel, maximum: int) -> str:
    payload = {"schemaVersion": 1, "model": model.as_json(), "maximumSuggestions": maximum}
    return f"suggestion-{_sha256(_json_bytes(payload))[:24]}"


def _evidence_path(
    workspace: Path,
    source_id: str,
    review_id: str,
    run_id: str,
) -> Path:
    root = (workspace / source_id / "model-suggestions").resolve()
    path = (root / review_id / f"{run_id}.json").resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("review suggestion evidence path is unsafe")
    return path


def _verify_existing_evidence(
    path: Path,
    *,
    source_id: str,
    source_manifest_sha256: str,
    source_queue_sha256: str,
    candidate: _Candidate,
    run_id: str,
) -> None:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError("existing review suggestion evidence is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schemaVersion") != 1
        or value.get("type") != "DETECTOR_MODEL_SUGGESTION"
        or value.get("sourceId") != source_id
        or value.get("sourceManifestSha256") != source_manifest_sha256
        or value.get("sourceQueueSha256") != source_queue_sha256
        or value.get("reviewId") != candidate.review_id
        or value.get("sourceImageSha256") != candidate.image_sha256
        or value.get("runId") != run_id
        or not isinstance(value.get("suggestions"), list)
        or not value["suggestions"]
    ):
        raise DetectorDatasetError("existing review suggestion evidence binding is invalid")


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DetectorDatasetError("review suggestion image path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("review suggestion image path escapes its source")
    return path


def _write_new(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DetectorDatasetError("review suggestion evidence already exists") from exc


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None:
        raise DetectorDatasetError("review suggestion clock must be timezone-aware")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
