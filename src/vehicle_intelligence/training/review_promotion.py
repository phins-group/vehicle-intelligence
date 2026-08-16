"""Promote reviewed detector labels into a new immutable first-party source."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from vehicle_intelligence.domain.dataset_review import (
    DetectorReviewAnnotation,
    DetectorReviewDecision,
    DetectorReviewStatus,
)
from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.domain import (
    DetectorAnnotation,
    DetectorSample,
    TrainingBoundingBox,
)
from vehicle_intelligence.training.first_party import verify_first_party_detector_source


@dataclass(frozen=True, slots=True)
class ReviewedSourceResult:
    directory: Path
    source_id: str
    manifest_sha256: str
    sample_count: int
    promoted_review_count: int
    remaining_review_count: int
    rejected_count: int
    reused: bool = False


class ReviewedFirstPartySourceBuilder:
    """Create a full source version without mutating its immutable parent."""

    def __init__(
        self,
        *,
        source_directory: Path,
        output_directory: Path,
        target_source_id: str,
        decisions: dict[str, DetectorReviewDecision],
        model_suggestions: dict[str, tuple[DetectorReviewAnnotation, ...]] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._source = source_directory.expanduser().resolve()
        self._target = output_directory.expanduser().resolve()
        self._target_source_id = target_source_id
        self._decisions = dict(decisions)
        self._model_suggestions = dict(model_suggestions or {})
        self._clock = clock

    def build(self) -> ReviewedSourceResult:
        if self._target.exists():
            manifest, digest = verify_first_party_detector_source(self._target)
            if manifest.get("sourceId") != self._target_source_id:
                raise DetectorDatasetError("promoted source target id does not match")
            return _result(self._target, manifest, digest, reused=True)
        parent_manifest, parent_digest = verify_first_party_detector_source(self._source)
        queue = _read_queue(self._source / "REVIEW_QUEUE.jsonl")
        unknown = set(self._decisions) - set(queue)
        if unknown:
            raise DetectorDatasetError("promotion contains decisions outside the source queue")
        now = self._clock()
        if now.tzinfo is None:
            raise DetectorDatasetError("promotion clock must be timezone-aware")
        now = now.astimezone(UTC)
        parent = self._target.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{self._target.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            manifest = self._materialize(
                temporary,
                parent_manifest,
                parent_digest,
                queue,
                now,
            )
            manifest_bytes = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "source-manifest.json", manifest_bytes)
            temporary.replace(self._target)
            verified, digest = verify_first_party_detector_source(self._target)
            return _result(self._target, verified, digest)
        except DetectorDatasetError:
            _remove_temporary(temporary, parent)
            raise
        except Exception as exc:
            _remove_temporary(temporary, parent)
            raise DetectorDatasetError("cannot promote reviewed first-party source") from exc

    def _materialize(
        self,
        temporary: Path,
        parent_manifest: dict[str, Any],
        parent_digest: str,
        queue: dict[str, dict[str, Any]],
        now: datetime,
    ) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        samples = _load_samples(self._source / "annotations.jsonl")
        annotation_lines: list[bytes] = []
        annotation_count = 0
        negative_count = 0
        for sample in samples:
            source_image = _safe_child(self._source, sample.image_path)
            destination = _safe_child(temporary, sample.image_path)
            _copy_new(source_image, destination)
            files.append(_file_entry(destination, temporary))
            annotation_lines.append(
                _json_bytes(sample.model_dump(mode="json", by_alias=True), pretty=False)
            )
            annotation_count += len(sample.annotations)
            negative_count += not bool(sample.annotations)

        pending_lines: list[bytes] = []
        pending_auto_count = 0
        pending_conflict_count = 0
        pending_unlabeled_count = 0
        pending_overlay_count = 0
        decision_lines: list[bytes] = []
        rejected_lines = list((self._source / "REJECTS.jsonl").read_bytes().splitlines())
        promoted_count = 0
        rejected_count = 0
        attribution_rows = _read_attribution(self._source / "ATTRIBUTION.csv")
        source_created_at = _parse_timestamp(str(parent_manifest["createdAt"]))

        for review_id, record in sorted(queue.items()):
            decision = self._decisions.get(review_id)
            source_image = _safe_child(self._source, str(record["imagePath"]))
            if decision is None:
                pending_record = dict(record)
                overlay = self._model_suggestions.get(review_id)
                if overlay and not pending_record.get("suggestions"):
                    pending_record["reason"] = "MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW"
                    pending_record["suggestions"] = [
                        _review_annotation_json(annotation) for annotation in overlay
                    ]
                    pending_overlay_count += 1
                destination = _safe_child(temporary, str(record["imagePath"]))
                _copy_new(source_image, destination)
                files.append(_file_entry(destination, temporary))
                pending_lines.append(_json_bytes(pending_record, pretty=False))
                pending_auto_count += (
                    pending_record.get("reason") == "MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW"
                )
                pending_conflict_count += (
                    pending_record.get("reason") == "AUTO_LABEL_CONFLICT_REQUIRES_HUMAN_REVIEW"
                )
                pending_unlabeled_count += (
                    pending_record.get("reason") == "MISSING_VERIFIED_ANNOTATION"
                )
                continue
            decision_lines.append(_json_bytes(_decision_evidence(record, decision), pretty=False))
            if decision.status is DetectorReviewStatus.REJECTED:
                rejected_count += 1
                rejected_lines.append(
                    _json_bytes(
                        {
                            "schemaVersion": 1,
                            "reviewId": review_id,
                            "sourceImageSha256": record["sourceImageSha256"],
                            "reason": "HUMAN_REJECTED",
                            "reviewRevision": decision.revision,
                            "reviewedBy": decision.reviewed_by,
                            "reviewedAt": _timestamp(decision.reviewed_at),
                            "note": decision.note,
                        },
                        pretty=False,
                    ).rstrip(b"\n")
                )
                continue
            digest = str(record["sourceImageSha256"])
            suffix = (
                ".jpg" if source_image.suffix.lower() == ".jpeg" else source_image.suffix.lower()
            )
            relative = PurePosixPath("images", digest[:2], f"{digest}{suffix}")
            destination = temporary.joinpath(*relative.parts)
            _copy_new(source_image, destination)
            files.append(_file_entry(destination, temporary))
            sample = _reviewed_sample(
                record=record,
                decision=decision,
                image_path=str(relative),
                image_file=destination,
                captured_at=source_created_at,
                parent_source_id=str(parent_manifest["sourceId"]),
                parent_manifest_sha256=parent_digest,
            )
            annotation_lines.append(
                _json_bytes(sample.model_dump(mode="json", by_alias=True), pretty=False)
            )
            annotation_count += len(sample.annotations)
            negative_count += not bool(sample.annotations)
            promoted_count += 1
            attribution_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_dataset": self._target_source_id,
                    "source_revision": "human-review",
                    "license": "PROPRIETARY-FIRST-PARTY",
                    "author": decision.reviewed_by,
                    "landing_url": "",
                }
            )

        evidence_path = temporary / "REVIEW_DECISIONS.jsonl"
        _write_new(evidence_path, b"".join(decision_lines))
        files.append(_file_entry(evidence_path, temporary))
        annotations_path = temporary / "annotations.jsonl"
        _write_new(annotations_path, b"".join(annotation_lines))
        files.append(_file_entry(annotations_path, temporary))
        queue_path = temporary / "REVIEW_QUEUE.jsonl"
        _write_new(queue_path, b"".join(pending_lines))
        files.append(_file_entry(queue_path, temporary))
        duplicates_path = temporary / "DUPLICATES.jsonl"
        _copy_new(self._source / "DUPLICATES.jsonl", duplicates_path)
        files.append(_file_entry(duplicates_path, temporary))
        rejects_path = temporary / "REJECTS.jsonl"
        rejects_payload = b"".join(line.rstrip(b"\n") + b"\n" for line in rejected_lines if line)
        _write_new(rejects_path, rejects_payload)
        files.append(_file_entry(rejects_path, temporary))
        attribution_path = temporary / "ATTRIBUTION.csv"
        _write_new(attribution_path, _attribution(attribution_rows))
        files.append(_file_entry(attribution_path, temporary))
        card_path = temporary / "SOURCE_CARD.md"
        _write_new(
            card_path,
            _source_card(
                target_id=self._target_source_id,
                parent_id=str(parent_manifest["sourceId"]),
                production_count=len(samples) + promoted_count,
                promoted_count=promoted_count,
                pending_count=len(pending_lines),
                rejected_count=rejected_count,
            ).encode(),
        )
        files.append(_file_entry(card_path, temporary))

        statistics = parent_manifest.get("statistics", {})
        return {
            "schemaVersion": 1,
            "type": "FIRST_PARTY_DETECTOR_SOURCE",
            "role": "plate",
            "sourceId": self._target_source_id,
            "ownerNamespace": parent_manifest["ownerNamespace"],
            "founderId": parent_manifest["founderId"],
            "createdAt": _timestamp(now),
            "collectionMethod": "FIRST_PARTY_USER_COLLECTED",
            "rightsAssertion": "USER_CONFIRMED_FIRST_PARTY_COLLECTION",
            "licenseStatus": "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED",
            "privacyClassification": "RESTRICTED_VEHICLE_IDENTIFIER",
            "acceptanceEligible": True,
            "releaseEligible": True,
            "distributionEligible": False,
            "annotationPolicy": "VERIFIED_PARENT_LABELS_PLUS_REVISIONED_HUMAN_REVIEW",
            "sampleCount": len(samples) + promoted_count,
            "annotationCount": annotation_count,
            "negativeSampleCount": negative_count,
            "reviewQueueCount": len(pending_lines),
            "statistics": {
                "inputImageFiles": int(
                    statistics.get("inputImageFiles", len(samples) + len(queue))
                ),
                "uniqueImages": int(statistics.get("uniqueImages", len(samples) + len(queue))),
                "verifiedProductionImages": len(samples) + promoted_count,
                "humanReviewedPromoted": promoted_count,
                "humanRejected": rejected_count,
                "remainingPendingReview": len(pending_lines),
                "autoLabeledPendingReview": pending_auto_count,
                "autoLabelConflictsPendingReview": pending_conflict_count,
                "unlabeledPendingReview": pending_unlabeled_count,
                "modelSuggestionOverlayPendingReview": pending_overlay_count,
                "exactDuplicateFilesExcluded": int(
                    statistics.get("exactDuplicateFilesExcluded", 0)
                ),
                "unsupportedFiles": int(statistics.get("unsupportedFiles", 0)),
            },
            "inputInventorySha256": parent_manifest["inputInventorySha256"],
            "labelReference": {
                "type": "REVIEWED_FIRST_PARTY_SOURCE_PARENT",
                "id": parent_manifest["sourceId"],
                "sha256": parent_digest,
            },
            "parentSource": {
                "id": parent_manifest["sourceId"],
                "manifestSha256": parent_digest,
            },
            "reviewPromotion": {
                "promotedCount": promoted_count,
                "rejectedCount": rejected_count,
                "remainingPendingCount": len(pending_lines),
                "decisionEvidencePath": "REVIEW_DECISIONS.jsonl",
            },
            "files": sorted(files, key=lambda item: item["path"]),
        }


def _reviewed_sample(
    *,
    record: dict[str, Any],
    decision: DetectorReviewDecision,
    image_path: str,
    image_file: Path,
    captured_at: datetime,
    parent_source_id: str,
    parent_manifest_sha256: str,
) -> DetectorSample:
    data = image_file.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise DetectorDatasetError("promoted review image cannot be decoded")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotations = tuple(
        DetectorAnnotation(
            className="license_plate",
            bbox=TrainingBoundingBox(
                x=annotation.bbox.x,
                y=annotation.bbox.y,
                width=annotation.bbox.width,
                height=annotation.bbox.height,
            ),
            attributes={
                **{
                    key: value
                    for key, value in annotation.attributes.items()
                    if isinstance(value, (str, bool, int, float)) or value is None
                },
                "annotationOrigin": "REVISIONED_HUMAN_REVIEW",
                "reviewStatus": decision.status.value,
                "reviewAction": decision.action.value,
                "reviewRevision": decision.revision,
                "reviewedBy": decision.reviewed_by,
                "reviewedAt": _timestamp(decision.reviewed_at),
            },
        )
        for annotation in decision.annotations
    )
    for annotation in annotations:
        box = annotation.bbox
        if box.x + box.width > width or box.y + box.height > height:
            raise DetectorDatasetError("promoted human annotation is outside image bounds")
    digest = str(record["sourceImageSha256"])
    return DetectorSample(
        sampleId=f"phins-first-party-plate-{digest[:24]}",
        imagePath=image_path,
        groupId=f"phins-group:human-reviewed-plate:{digest[:24]}",
        cameraId="first-party-collection",
        capturedAt=captured_at,
        split=None,
        attributes={
            "sourceCollection": "FIRST_PARTY_USER_COLLECTED",
            "sourceLicense": "PROPRIETARY-FIRST-PARTY",
            "sourceImageSha256": digest,
            "sourceFilenameSha256": str(record["sourceFilenameSha256"]),
            "parentSourceId": parent_source_id,
            "parentSourceManifestSha256": parent_manifest_sha256,
            "sourceReviewId": str(record["reviewId"]),
            "sourceReviewReason": str(record["reason"]),
            "annotationOrigin": "REVISIONED_HUMAN_REVIEW",
            "annotationReviewStatus": decision.status.value,
            "reviewRevision": decision.revision,
            "reviewedBy": decision.reviewed_by,
            "reviewedAt": _timestamp(decision.reviewed_at),
            "capturedAtBasis": "PARENT_SOURCE_CREATED_AT_FALLBACK",
            "actualCaptureTimeKnown": False,
            "groupingBasis": "SOURCE_IMAGE_SHA256_NO_SEQUENCE_METADATA",
            "acceptanceEligible": True,
            "releaseEligible": True,
            "distributionEligible": False,
            "negativeSample": not bool(annotations),
            "lighting": "NIGHT" if float(np.mean(gray)) < 70 else "DAY",
            "imageBrightness": round(float(np.mean(gray)), 4),
            "imageContrast": round(float(np.std(gray)), 4),
            "imageSharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
        },
        annotations=annotations,
    )


def _review_annotation_json(annotation: DetectorReviewAnnotation) -> dict[str, object]:
    return {
        "className": annotation.class_name,
        "bbox": {
            "x": annotation.bbox.x,
            "y": annotation.bbox.y,
            "width": annotation.bbox.width,
            "height": annotation.bbox.height,
        },
        "attributes": annotation.attributes,
    }


def _decision_evidence(
    record: dict[str, Any],
    decision: DetectorReviewDecision,
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "reviewId": record["reviewId"],
        "sourceImageSha256": record["sourceImageSha256"],
        "sourceReason": record["reason"],
        "action": decision.action.value,
        "status": decision.status.value,
        "revision": decision.revision,
        "reviewedBy": decision.reviewed_by,
        "reviewerDisplayName": decision.reviewer_display_name,
        "reviewedAt": _timestamp(decision.reviewed_at),
        "note": decision.note,
        "annotations": [
            {
                "className": annotation.class_name,
                "bbox": {
                    "x": annotation.bbox.x,
                    "y": annotation.bbox.y,
                    "width": annotation.bbox.width,
                    "height": annotation.bbox.height,
                },
            }
            for annotation in decision.annotations
        ],
    }


def _load_samples(path: Path) -> list[DetectorSample]:
    samples: list[DetectorSample] = []
    for line in path.read_bytes().splitlines():
        if not line:
            continue
        try:
            samples.append(DetectorSample.model_validate_json(line))
        except ValidationError as exc:
            raise DetectorDatasetError("parent detector samples are invalid") from exc
    return samples


def _read_queue(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_bytes().splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DetectorDatasetError("parent detector review queue is invalid") from exc
        review_id = record.get("reviewId") if isinstance(record, dict) else None
        if not isinstance(review_id, str) or review_id in result:
            raise DetectorDatasetError("parent detector review queue ids are invalid")
        result[review_id] = record
    return result


def _read_attribution(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except (OSError, csv.Error) as exc:
        raise DetectorDatasetError("parent first-party attribution is invalid") from exc


def _attribution(rows: list[dict[str, str]]) -> bytes:
    fields = (
        "sample_id",
        "source_dataset",
        "source_revision",
        "license",
        "author",
        "landing_url",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return stream.getvalue().encode()


def _source_card(
    *,
    target_id: str,
    parent_id: str,
    production_count: int,
    promoted_count: int,
    pending_count: int,
    rejected_count: int,
) -> str:
    return f"""# {target_id}

Immutable first-party Vietnamese license-plate detector source promoted from
`{parent_id}` through revisioned human review.

- Production samples: {production_count}
- Human-reviewed samples promoted: {promoted_count}
- Remaining review items: {pending_count}
- Human-rejected images: {rejected_count}
- Distribution: restricted (vehicle identifiers)

The parent source remains unchanged. `REVIEW_DECISIONS.jsonl` preserves the
human decision evidence included in this source version.
"""


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> ReviewedSourceResult:
    promotion = manifest.get("reviewPromotion", {})
    return ReviewedSourceResult(
        directory=directory,
        source_id=str(manifest["sourceId"]),
        manifest_sha256=digest,
        sample_count=int(manifest["sampleCount"]),
        promoted_review_count=int(promotion.get("promotedCount", 0)),
        remaining_review_count=int(manifest["reviewQueueCount"]),
        rejected_count=int(promotion.get("rejectedCount", 0)),
        reused=reused,
    )


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DetectorDatasetError("promoted source path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("promoted source path escapes its root")
    return path


def _copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    except FileExistsError as exc:
        raise DetectorDatasetError("promoted source image path collision") from exc


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _file_entry(path: Path, root: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
    }


def _json_bytes(value: object, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            value,
            indent=2 if pretty else None,
            sort_keys=True,
            ensure_ascii=False,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise DetectorDatasetError("parent source timestamp has no timezone")
    return parsed.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _remove_temporary(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise DetectorDatasetError("refusing to remove unsafe promotion directory")
    if resolved.exists():
        shutil.rmtree(resolved)
