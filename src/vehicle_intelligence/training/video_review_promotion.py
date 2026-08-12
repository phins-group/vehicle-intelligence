"""Promote fully reviewed, first-party video samples into a production source.

The review-only source remains immutable.  This module creates a new production
source version by combining an existing verified source with all terminal human
decisions from a video review source and an explicit first-party rights
attestation.  Frames remain grouped by source video to prevent adjacent-frame
leakage across train, validation, and test splits.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from vehicle_intelligence.domain.dataset_review import (
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
from vehicle_intelligence.training.video_review_source import (
    verify_video_plate_review_source,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RIGHTS_ASSERTION = "USER_CONFIRMED_FIRST_PARTY_VIDEO_COLLECTION"
_LICENSE_STATUS = "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED"


@dataclass(frozen=True, slots=True)
class AttestedVideoPromotionResult:
    directory: Path
    source_id: str
    manifest_sha256: str
    sample_count: int
    annotation_count: int
    negative_sample_count: int
    promoted_review_count: int
    promoted_positive_count: int
    promoted_negative_count: int
    rejected_count: int
    reused: bool = False


class AttestedVideoReviewPromotionBuilder:
    """Create a production source from a base source and reviewed video queue."""

    def __init__(
        self,
        *,
        base_source_directory: Path,
        review_source_directory: Path,
        output_directory: Path,
        target_source_id: str,
        decisions: dict[str, DetectorReviewDecision],
        rights_holder: str,
        attested_by: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not _IDENTIFIER.fullmatch(target_source_id):
            raise ValueError("attested promotion target source id is not path-safe")
        self._base = base_source_directory.expanduser().resolve()
        self._review = review_source_directory.expanduser().resolve()
        self._target = output_directory.expanduser().resolve()
        self._target_source_id = target_source_id
        self._decisions = dict(decisions)
        self._rights_holder = _required_text(rights_holder, "rights holder")
        self._attested_by = _required_text(attested_by, "attested by")
        self._clock = clock

    def build(self) -> AttestedVideoPromotionResult:
        if self._target.exists():
            manifest, digest = verify_first_party_detector_source(self._target)
            promotion = manifest.get("videoReviewPromotion", {})
            if (
                manifest.get("sourceId") != self._target_source_id
                or not isinstance(promotion, dict)
                or promotion.get("reviewSourceId") != self._review.name
            ):
                raise DetectorDatasetError(
                    "existing attested promotion does not match requested sources"
                )
            return _result(self._target, manifest, digest, reused=True)

        base_manifest, base_digest = verify_first_party_detector_source(self._base)
        review_manifest, review_digest = verify_video_plate_review_source(self._review)
        if self._target in {self._base, self._review}:
            raise DetectorDatasetError("attested promotion output must be a new source")
        if (
            base_manifest.get("ownerNamespace") != review_manifest.get("ownerNamespace")
            or base_manifest.get("founderId") != review_manifest.get("founderId")
        ):
            raise DetectorDatasetError("base and video review ownership metadata do not match")
        queue = _read_jsonl_by_key(self._review / "REVIEW_QUEUE.jsonl", "reviewId")
        provenance = _read_jsonl_by_key(self._review / "PROVENANCE.jsonl", "reviewId")
        if set(queue) != set(provenance):
            raise DetectorDatasetError("video review queue and provenance do not match")
        if set(self._decisions) != set(queue):
            missing = len(set(queue) - set(self._decisions))
            unknown = len(set(self._decisions) - set(queue))
            raise DetectorDatasetError(
                f"attested promotion requires all review decisions; missing={missing}, "
                f"unknown={unknown}"
            )
        _validate_decisions(self._decisions, queue)
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise DetectorDatasetError("attested promotion clock must be timezone-aware")
        now = now.astimezone(UTC)

        parent = self._target.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{self._target.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            manifest = self._materialize(
                temporary=temporary,
                base_manifest=base_manifest,
                base_digest=base_digest,
                review_manifest=review_manifest,
                review_digest=review_digest,
                queue=queue,
                provenance=provenance,
                now=now,
            )
            manifest_raw = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "source-manifest.json", manifest_raw)
            if self._target.exists():
                raise DetectorDatasetError("attested promotion target already exists")
            temporary.replace(self._target)
            verified, digest = verify_first_party_detector_source(self._target)
            return _result(self._target, verified, digest)
        except DetectorDatasetError:
            _remove_temporary(temporary, parent)
            raise
        except Exception as exc:
            _remove_temporary(temporary, parent)
            raise DetectorDatasetError("cannot promote attested video review source") from exc

    def _materialize(
        self,
        *,
        temporary: Path,
        base_manifest: dict[str, Any],
        base_digest: str,
        review_manifest: dict[str, Any],
        review_digest: str,
        queue: dict[str, dict[str, Any]],
        provenance: dict[str, dict[str, Any]],
        now: datetime,
    ) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        annotation_lines: list[bytes] = []
        base_samples = _load_samples(self._base / "annotations.jsonl")
        base_image_hashes: set[str] = set()
        annotation_count = 0
        negative_count = 0

        for sample in base_samples:
            source_image = _safe_child(self._base, sample.image_path)
            image_digest = _sha256_file(source_image)
            if image_digest in base_image_hashes:
                raise DetectorDatasetError("base production source contains duplicate images")
            base_image_hashes.add(image_digest)
            destination = _safe_child(temporary, sample.image_path)
            _copy_new(source_image, destination)
            files.append(_file_entry(destination, temporary))
            annotation_lines.append(
                _json_bytes(sample.model_dump(mode="json", by_alias=True), pretty=False)
            )
            annotation_count += len(sample.annotations)
            negative_count += not bool(sample.annotations)

        attestation_document = {
            "schemaVersion": 1,
            "type": "DATASET_RIGHTS_ATTESTATION",
            "assertion": _RIGHTS_ASSERTION,
            "rightsHolder": self._rights_holder,
            "attestedBy": self._attested_by,
            "attestedAt": _timestamp(now),
            "attestationMethod": "INTERACTIVE_USER_CONFIRMATION",
            "reviewSourceId": review_manifest["sourceId"],
            "reviewSourceManifestSha256": review_digest,
            "sourceVideoSha256": review_manifest["sourceExtraction"]["videoSha256"],
            "scope": {
                "training": True,
                "commercialModelUse": True,
                "rawDatasetDistribution": False,
            },
        }
        attestation_raw = _json_bytes(attestation_document, pretty=True)
        attestation_sha256 = _sha256(attestation_raw)

        decision_lines: list[bytes] = []
        rejected_lines: list[bytes] = []
        video_attribution_rows: list[dict[str, str]] = []
        status_counts: Counter[str] = Counter()
        promoted_count = 0
        promoted_positive_count = 0
        promoted_negative_count = 0
        rejected_count = 0

        for review_id, record in sorted(queue.items()):
            decision = self._decisions[review_id]
            status_counts[decision.status.value] += 1
            decision_lines.append(
                _json_bytes(
                    _decision_evidence(
                        record,
                        decision,
                        review_source_id=str(review_manifest["sourceId"]),
                        review_manifest_sha256=review_digest,
                    ),
                    pretty=False,
                )
            )
            if decision.status is DetectorReviewStatus.REJECTED:
                rejected_count += 1
                rejected_lines.append(
                    _json_bytes(
                        {
                            "schemaVersion": 1,
                            "reviewId": review_id,
                            "sourceImageSha256": record["sourceImageSha256"],
                            "reason": "HUMAN_REJECTED_VIDEO_REVIEW",
                            "reviewRevision": decision.revision,
                            "reviewedBy": decision.reviewed_by,
                            "reviewedAt": _timestamp(decision.reviewed_at),
                            "note": decision.note,
                        },
                        pretty=False,
                    )
                )
                continue

            digest = str(record["sourceImageSha256"])
            if digest in base_image_hashes:
                raise DetectorDatasetError(
                    "reviewed video image duplicates an existing production image"
                )
            base_image_hashes.add(digest)
            source_image = _safe_child(self._review, str(record["imagePath"]))
            suffix = ".jpg" if source_image.suffix.lower() == ".jpeg" else source_image.suffix
            relative = PurePosixPath("images", digest[:2], f"{digest}{suffix.lower()}")
            destination = temporary.joinpath(*relative.parts)
            _copy_new(source_image, destination)
            files.append(_file_entry(destination, temporary))
            sample = _reviewed_video_sample(
                record=record,
                provenance=provenance[review_id],
                decision=decision,
                image_path=str(relative),
                image_file=destination,
                base_source_id=str(base_manifest["sourceId"]),
                base_manifest_sha256=base_digest,
                review_source_id=str(review_manifest["sourceId"]),
                review_manifest_sha256=review_digest,
                rights_holder=self._rights_holder,
                attestation_sha256=attestation_sha256,
            )
            annotation_lines.append(
                _json_bytes(sample.model_dump(mode="json", by_alias=True), pretty=False)
            )
            annotation_count += len(sample.annotations)
            negative_count += not bool(sample.annotations)
            promoted_count += 1
            if sample.annotations:
                promoted_positive_count += 1
            else:
                promoted_negative_count += 1
            video_attribution_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_dataset": str(review_manifest["sourceId"]),
                    "source_revision": "human-review-attested",
                    "license": "PROPRIETARY-FIRST-PARTY",
                    "author": self._rights_holder,
                    "landing_url": "",
                }
            )

        annotations_path = temporary / "annotations.jsonl"
        _write_new(annotations_path, b"".join(annotation_lines))
        files.append(_file_entry(annotations_path, temporary))
        queue_path = temporary / "REVIEW_QUEUE.jsonl"
        _write_new(queue_path, b"")
        files.append(_file_entry(queue_path, temporary))
        decisions_path = temporary / "REVIEW_DECISIONS.jsonl"
        _write_new(decisions_path, b"".join(decision_lines))
        files.append(_file_entry(decisions_path, temporary))

        duplicates_path = temporary / "DUPLICATES.jsonl"
        duplicates_payload = (self._base / "DUPLICATES.jsonl").read_bytes()
        video_duplicates = self._review / "DUPLICATES.jsonl"
        if video_duplicates.is_file():
            duplicates_payload += video_duplicates.read_bytes()
        _write_new(duplicates_path, duplicates_payload)
        files.append(_file_entry(duplicates_path, temporary))

        rejects_path = temporary / "REJECTS.jsonl"
        rejects_payload = (self._base / "REJECTS.jsonl").read_bytes()
        rejects_payload += b"".join(rejected_lines)
        _write_new(rejects_path, rejects_payload)
        files.append(_file_entry(rejects_path, temporary))

        attribution_rows = _read_attribution(self._base / "ATTRIBUTION.csv")
        attribution_rows.extend(video_attribution_rows)
        attribution_path = temporary / "ATTRIBUTION.csv"
        _write_new(attribution_path, _attribution(attribution_rows))
        files.append(_file_entry(attribution_path, temporary))

        attestation_path = temporary / "RIGHTS_ATTESTATION.json"
        _write_new(attestation_path, attestation_raw)
        files.append(_file_entry(attestation_path, temporary))
        review_manifest_path = temporary / "VIDEO_REVIEW_SOURCE_MANIFEST.json"
        review_manifest_raw = (self._review / "source-manifest.json").read_bytes()
        _write_new(review_manifest_path, review_manifest_raw)
        files.append(_file_entry(review_manifest_path, temporary))
        review_provenance_path = temporary / "VIDEO_REVIEW_PROVENANCE.jsonl"
        _copy_new(self._review / "PROVENANCE.jsonl", review_provenance_path)
        files.append(_file_entry(review_provenance_path, temporary))

        source_card_path = temporary / "SOURCE_CARD.md"
        _write_new(
            source_card_path,
            _source_card(
                target_id=self._target_source_id,
                base_id=str(base_manifest["sourceId"]),
                review_id=str(review_manifest["sourceId"]),
                production_count=len(base_samples) + promoted_count,
                promoted_count=promoted_count,
                positive_count=promoted_positive_count,
                negative_count=promoted_negative_count,
                rejected_count=rejected_count,
                rights_holder=self._rights_holder,
            ).encode("utf-8"),
        )
        files.append(_file_entry(source_card_path, temporary))

        base_statistics = base_manifest.get("statistics", {})
        review_statistics = review_manifest.get("statistics", {})
        inventory_sha256 = _sha256(
            "\n".join(
                (
                    str(base_manifest["inputInventorySha256"]),
                    review_digest,
                    _sha256(b"".join(decision_lines)),
                    attestation_sha256,
                )
            ).encode("utf-8")
        )
        return {
            "schemaVersion": 1,
            "type": "FIRST_PARTY_DETECTOR_SOURCE",
            "role": "plate",
            "sourceId": self._target_source_id,
            "ownerNamespace": base_manifest["ownerNamespace"],
            "founderId": base_manifest["founderId"],
            "createdAt": _timestamp(now),
            "collectionMethod": "FIRST_PARTY_USER_COLLECTED_IMAGES_AND_VIDEO",
            "rightsAssertion": _RIGHTS_ASSERTION,
            "licenseStatus": _LICENSE_STATUS,
            "privacyClassification": "RESTRICTED_VEHICLE_IDENTIFIER",
            "acceptanceEligible": True,
            "releaseEligible": True,
            "distributionEligible": False,
            "promotionEligible": True,
            "annotationPolicy": (
                "VERIFIED_BASE_LABELS_PLUS_RIGHTS_ATTESTED_REVISIONED_VIDEO_REVIEW"
            ),
            "sampleCount": len(base_samples) + promoted_count,
            "annotationCount": annotation_count,
            "negativeSampleCount": negative_count,
            "reviewQueueCount": 0,
            "statistics": {
                "inputImageFiles": int(base_statistics.get("inputImageFiles", len(base_samples)))
                + int(review_statistics.get("sourceRecordCount", len(queue))),
                "uniqueImages": len(base_samples) + len(queue),
                "verifiedProductionImages": len(base_samples) + promoted_count,
                "existingProductionImages": len(base_samples),
                "humanReviewedVideoPromoted": promoted_count,
                "humanReviewedVideoPositive": promoted_positive_count,
                "humanReviewedVideoNegative": promoted_negative_count,
                "humanRejectedVideo": rejected_count,
                "humanDecisionStatusCounts": dict(sorted(status_counts.items())),
                "sourceVideoCount": int(review_statistics.get("sourceVideoCount", 0)),
                "videoExactDuplicateImagesMerged": int(
                    review_statistics.get("exactDuplicateImagesMerged", 0)
                ),
                "exactDuplicateFilesExcluded": int(
                    base_statistics.get("exactDuplicateFilesExcluded", 0)
                )
                + int(review_statistics.get("exactDuplicateImagesMerged", 0)),
                "unsupportedFiles": int(base_statistics.get("unsupportedFiles", 0)),
                "remainingPendingReview": 0,
            },
            "inputInventorySha256": inventory_sha256,
            "labelReference": {
                "type": "BASE_SOURCE_PLUS_ATTESTED_VIDEO_REVIEW",
                "id": base_manifest["sourceId"],
                "sha256": base_digest,
            },
            "parentSource": {
                "id": base_manifest["sourceId"],
                "manifestSha256": base_digest,
            },
            "videoReviewPromotion": {
                "reviewSourceId": review_manifest["sourceId"],
                "reviewSourceManifestSha256": review_digest,
                "reviewSourceManifestPath": "VIDEO_REVIEW_SOURCE_MANIFEST.json",
                "reviewProvenancePath": "VIDEO_REVIEW_PROVENANCE.jsonl",
                "decisionEvidencePath": "REVIEW_DECISIONS.jsonl",
                "promotedCount": promoted_count,
                "positiveCount": promoted_positive_count,
                "negativeCount": promoted_negative_count,
                "rejectedCount": rejected_count,
                "remainingPendingCount": 0,
            },
            "rightsAttestation": {
                "path": "RIGHTS_ATTESTATION.json",
                "sha256": attestation_sha256,
                "assertion": _RIGHTS_ASSERTION,
                "rightsHolder": self._rights_holder,
                "attestedBy": self._attested_by,
                "attestedAt": _timestamp(now),
            },
            "files": sorted(files, key=lambda item: item["path"]),
        }


def _reviewed_video_sample(
    *,
    record: dict[str, Any],
    provenance: dict[str, Any],
    decision: DetectorReviewDecision,
    image_path: str,
    image_file: Path,
    base_source_id: str,
    base_manifest_sha256: str,
    review_source_id: str,
    review_manifest_sha256: str,
    rights_holder: str,
    attestation_sha256: str,
) -> DetectorSample:
    data = image_file.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise DetectorDatasetError("promoted video review image cannot be decoded")
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
                "annotationOrigin": "REVISIONED_HUMAN_VIDEO_REVIEW",
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
            raise DetectorDatasetError("promoted video annotation is outside image bounds")

    raw_records = provenance.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise DetectorDatasetError("video review provenance records are invalid")
    video_hashes = sorted(
        {
            str(item["sourceVideoSha256"])
            for item in raw_records
            if isinstance(item, dict)
            and isinstance(item.get("sourceVideoSha256"), str)
            and _SHA256.fullmatch(str(item["sourceVideoSha256"]))
        }
    )
    if not video_hashes:
        raise DetectorDatasetError("video review provenance has no source video hash")
    captured_at = min(
        _parse_timestamp(str(item["capturedAt"]))
        for item in raw_records
        if isinstance(item, dict) and isinstance(item.get("capturedAt"), str)
    )
    camera_ids = sorted(
        {
            str(item["cameraId"])
            for item in raw_records
            if isinstance(item, dict) and isinstance(item.get("cameraId"), str)
        }
    )
    group_basis = _sha256("\n".join(video_hashes).encode("utf-8"))[:32]
    digest = str(record["sourceImageSha256"])
    return DetectorSample(
        sampleId=f"phins-first-party-video-plate-{digest[:24]}",
        imagePath=image_path,
        groupId=f"phins-group:first-party-video:{group_basis}",
        cameraId=camera_ids[0] if len(camera_ids) == 1 else "first-party-video-collection",
        capturedAt=captured_at,
        split=None,
        attributes={
            "sourceCollection": "FIRST_PARTY_USER_COLLECTED_VIDEO",
            "sourceLicense": "PROPRIETARY-FIRST-PARTY",
            "sourceOwner": rights_holder,
            "sourceImageSha256": digest,
            "sourceFilenameSha256": str(record["sourceFilenameSha256"]),
            "sourceVideoSha256": video_hashes[0],
            "sourceVideoCount": len(video_hashes),
            "baseSourceId": base_source_id,
            "baseSourceManifestSha256": base_manifest_sha256,
            "sourceReviewId": str(record["reviewId"]),
            "sourceReviewReason": str(record["reason"]),
            "reviewSourceId": review_source_id,
            "reviewSourceManifestSha256": review_manifest_sha256,
            "rightsAssertion": _RIGHTS_ASSERTION,
            "rightsAttestationSha256": attestation_sha256,
            "annotationOrigin": "REVISIONED_HUMAN_VIDEO_REVIEW",
            "annotationReviewStatus": decision.status.value,
            "reviewRevision": decision.revision,
            "reviewedBy": decision.reviewed_by,
            "reviewedAt": _timestamp(decision.reviewed_at),
            "capturedAtBasis": "VIDEO_FILE_MTIME_PLUS_FRAME_OFFSET",
            "actualCaptureTimeKnown": False,
            "groupingBasis": "SOURCE_VIDEO_SHA256",
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


def _validate_decisions(
    decisions: dict[str, DetectorReviewDecision],
    queue: dict[str, dict[str, Any]],
) -> None:
    for review_id, decision in decisions.items():
        record = queue[review_id]
        if decision.revision < 1 or decision.reviewed_at.tzinfo is None:
            raise DetectorDatasetError("video review decision revision is invalid")
        if decision.status is DetectorReviewStatus.PENDING_REVIEW:
            raise DetectorDatasetError("video review decision is still pending")
        if decision.status in {
            DetectorReviewStatus.APPROVED,
            DetectorReviewStatus.CORRECTED,
        } and not decision.annotations:
            raise DetectorDatasetError("positive video review decision has no annotation")
        if decision.status in {
            DetectorReviewStatus.NEGATIVE,
            DetectorReviewStatus.REJECTED,
        } and decision.annotations:
            raise DetectorDatasetError("negative/rejected video decision contains annotations")
        image_sha = record.get("sourceImageSha256")
        if not isinstance(image_sha, str) or not _SHA256.fullmatch(image_sha):
            raise DetectorDatasetError("video review queue image hash is invalid")


def _decision_evidence(
    record: dict[str, Any],
    decision: DetectorReviewDecision,
    *,
    review_source_id: str,
    review_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "reviewSourceId": review_source_id,
        "reviewSourceManifestSha256": review_manifest_sha256,
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


def _read_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise DetectorDatasetError(f"cannot read review evidence: {path.name}") from exc
    for line in lines:
        if not line:
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DetectorDatasetError(f"review evidence is invalid: {path.name}") from exc
        value = record.get(key) if isinstance(record, dict) else None
        if not isinstance(value, str) or value in result:
            raise DetectorDatasetError(f"review evidence keys are invalid: {path.name}")
        result[value] = record
    return result


def _load_samples(path: Path) -> list[DetectorSample]:
    samples: list[DetectorSample] = []
    for line in path.read_bytes().splitlines():
        if not line:
            continue
        try:
            samples.append(DetectorSample.model_validate_json(line))
        except ValidationError as exc:
            raise DetectorDatasetError("base production samples are invalid") from exc
    return samples


def _read_attribution(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except (OSError, csv.Error) as exc:
        raise DetectorDatasetError("base production attribution is invalid") from exc


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
    writer.writerows(
        {field: row.get(field, "") for field in fields}
        for row in sorted(rows, key=lambda item: item.get("sample_id", ""))
    )
    return stream.getvalue().encode("utf-8")


def _source_card(
    *,
    target_id: str,
    base_id: str,
    review_id: str,
    production_count: int,
    promoted_count: int,
    positive_count: int,
    negative_count: int,
    rejected_count: int,
    rights_holder: str,
) -> str:
    return f"""# {target_id}

Immutable production source combining `{base_id}` with fully reviewed,
first-party video samples from `{review_id}`.

- Rights holder: `{rights_holder}`
- Rights assertion: `{_RIGHTS_ASSERTION}`
- Production samples: {production_count}
- Reviewed video samples promoted: {promoted_count}
- Reviewed video positives: {positive_count}
- Reviewed video negatives: {negative_count}
- Reviewed video rejects: {rejected_count}
- Remaining review items: 0
- Raw dataset distribution: disabled

`RIGHTS_ATTESTATION.json` records the interactive first-party confirmation.
`REVIEW_DECISIONS.jsonl` preserves every human labeling decision. Source-video
SHA-256 grouping prevents frames from the same video crossing dataset splits.
"""


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> AttestedVideoPromotionResult:
    promotion = manifest["videoReviewPromotion"]
    return AttestedVideoPromotionResult(
        directory=directory,
        source_id=str(manifest["sourceId"]),
        manifest_sha256=digest,
        sample_count=int(manifest["sampleCount"]),
        annotation_count=int(manifest["annotationCount"]),
        negative_sample_count=int(manifest["negativeSampleCount"]),
        promoted_review_count=int(promotion["promotedCount"]),
        promoted_positive_count=int(promotion["positiveCount"]),
        promoted_negative_count=int(promotion["negativeCount"]),
        rejected_count=int(promotion["rejectedCount"]),
        reused=reused,
    )


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DetectorDatasetError("attested promotion path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("attested promotion path escapes its root")
    return path


def _copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except FileExistsError as exc:
        raise DetectorDatasetError("attested promotion image path collision") from exc


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _file_entry(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
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
    ).encode("utf-8")


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DetectorDatasetError("video provenance timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise DetectorDatasetError("video provenance timestamp has no timezone")
    return parsed.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128 or any(char in normalized for char in "/\\\0"):
        raise ValueError(f"{label} is invalid")
    return normalized


def _remove_temporary(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if (
        resolved.parent != parent.resolve()
        or not resolved.name.startswith(".")
        or ".tmp-" not in resolved.name
    ):
        raise DetectorDatasetError("refusing to remove unsafe attested promotion directory")
    if resolved.exists():
        shutil.rmtree(resolved)
