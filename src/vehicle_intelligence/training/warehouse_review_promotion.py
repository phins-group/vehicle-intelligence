"""Promote fully reviewed warehouse frames into a production plate source."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

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
from vehicle_intelligence.training.video_review_promotion import (
    _attribution,
    _copy_new,
    _decision_evidence,
    _file_entry,
    _json_bytes,
    _load_samples,
    _read_attribution,
    _read_jsonl_by_key,
    _remove_temporary,
    _required_text,
    _safe_child,
    _sha256,
    _sha256_file,
    _timestamp,
    _write_new,
)
from vehicle_intelligence.training.warehouse_plate_review import (
    verify_warehouse_plate_review_source,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RIGHTS_ASSERTION = "USER_CONFIRMED_FIRST_PARTY_WAREHOUSE_CAMERA_COLLECTION"
_LICENSE_STATUS = "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED"


@dataclass(frozen=True, slots=True)
class AttestedWarehousePromotionResult:
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


@dataclass(slots=True)
class _BaseMaterialization:
    files: list[dict[str, object]]
    annotation_lines: list[bytes]
    image_hashes: set[str]
    sample_count: int
    annotation_count: int
    negative_count: int


@dataclass(slots=True)
class _ReviewMaterialization:
    files: list[dict[str, object]]
    annotation_lines: list[bytes]
    decision_lines: list[bytes]
    rejected_lines: list[bytes]
    attribution_rows: list[dict[str, str]]
    status_counts: Counter[str]
    promoted_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    rejected_count: int = 0
    annotation_count: int = 0


class AttestedWarehouseReviewPromotionBuilder:
    """Merge a fully reviewed warehouse queue with a verified production base."""

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
            raise ValueError("warehouse promotion target source id is not path-safe")
        self._base = base_source_directory.expanduser().resolve()
        self._review = review_source_directory.expanduser().resolve()
        self._target = output_directory.expanduser().resolve()
        self._target_source_id = target_source_id
        self._decisions = dict(decisions)
        self._rights_holder = _required_text(rights_holder, "rights holder")
        self._attested_by = _required_text(attested_by, "attested by")
        self._clock = clock

    def build(self) -> AttestedWarehousePromotionResult:
        if self._target.exists():
            manifest, digest = verify_first_party_detector_source(self._target)
            promotion = manifest.get("warehouseReviewPromotion", {})
            if (
                manifest.get("sourceId") != self._target_source_id
                or not isinstance(promotion, dict)
                or promotion.get("reviewSourceId") != self._review.name
            ):
                raise DetectorDatasetError(
                    "existing warehouse promotion does not match requested sources"
                )
            return _result(self._target, manifest, digest, reused=True)

        base_manifest, base_digest = verify_first_party_detector_source(self._base)
        review_manifest, review_digest = verify_warehouse_plate_review_source(self._review)
        self._validate_inputs(base_manifest, review_manifest)
        queue = _read_jsonl_by_key(self._review / "REVIEW_QUEUE.jsonl", "reviewId")
        provenance = _read_jsonl_by_key(self._review / "PROVENANCE.jsonl", "reviewId")
        _validate_review_evidence(queue, provenance, self._decisions)
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise DetectorDatasetError("warehouse promotion clock must be timezone-aware")

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
                now=now.astimezone(UTC),
            )
            manifest_raw = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "source-manifest.json", manifest_raw)
            if self._target.exists():
                raise DetectorDatasetError("warehouse promotion target already exists")
            temporary.replace(self._target)
            verified, digest = verify_first_party_detector_source(self._target)
            return _result(self._target, verified, digest)
        except DetectorDatasetError:
            _remove_temporary(temporary, parent)
            raise
        except Exception as exc:
            _remove_temporary(temporary, parent)
            raise DetectorDatasetError("cannot promote warehouse review source") from exc

    def _validate_inputs(
        self,
        base_manifest: dict[str, Any],
        review_manifest: dict[str, Any],
    ) -> None:
        if self._target in {self._base, self._review}:
            raise DetectorDatasetError("warehouse promotion output must be a new source")
        if base_manifest.get("ownerNamespace") != review_manifest.get(
            "ownerNamespace"
        ) or base_manifest.get("founderId") != review_manifest.get("founderId"):
            raise DetectorDatasetError("base and warehouse review ownership metadata do not match")

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
        base = self._copy_base(temporary)
        attestation, attestation_raw, attestation_sha256 = self._attestation(
            review_manifest,
            review_digest,
            now,
        )
        reviewed = self._materialize_reviews(
            temporary=temporary,
            base=base,
            base_manifest=base_manifest,
            base_digest=base_digest,
            review_manifest=review_manifest,
            review_digest=review_digest,
            queue=queue,
            provenance=provenance,
            attestation_sha256=attestation_sha256,
        )
        files = [*base.files, *reviewed.files]
        files.extend(
            self._write_evidence(
                temporary=temporary,
                base_manifest=base_manifest,
                review_manifest=review_manifest,
                review_digest=review_digest,
                base=base,
                reviewed=reviewed,
                attestation_raw=attestation_raw,
            )
        )
        return self._production_manifest(
            files=files,
            base_manifest=base_manifest,
            base_digest=base_digest,
            review_manifest=review_manifest,
            review_digest=review_digest,
            base=base,
            reviewed=reviewed,
            attestation=attestation,
            attestation_sha256=attestation_sha256,
            now=now,
        )

    def _copy_base(self, temporary: Path) -> _BaseMaterialization:
        samples = _load_samples(self._base / "annotations.jsonl")
        result = _BaseMaterialization([], [], set(), len(samples), 0, 0)
        for sample in samples:
            source_image = _safe_child(self._base, sample.image_path)
            image_digest = _sha256_file(source_image)
            if image_digest in result.image_hashes:
                raise DetectorDatasetError("base production source contains duplicate images")
            result.image_hashes.add(image_digest)
            destination = _safe_child(temporary, sample.image_path)
            _copy_new(source_image, destination)
            result.files.append(_file_entry(destination, temporary))
            result.annotation_lines.append(
                _json_bytes(sample.model_dump(mode="json", by_alias=True), pretty=False)
            )
            result.annotation_count += len(sample.annotations)
            result.negative_count += not bool(sample.annotations)
        return result

    def _attestation(
        self,
        review_manifest: dict[str, Any],
        review_digest: str,
        now: datetime,
    ) -> tuple[dict[str, Any], bytes, str]:
        source_archive = review_manifest["sourceArchive"]
        document = {
            "schemaVersion": 1,
            "type": "DATASET_RIGHTS_ATTESTATION",
            "assertion": _RIGHTS_ASSERTION,
            "rightsHolder": self._rights_holder,
            "attestedBy": self._attested_by,
            "attestedAt": _timestamp(now),
            "attestationMethod": "INTERACTIVE_USER_CONFIRMATION",
            "reviewSourceId": review_manifest["sourceId"],
            "reviewSourceManifestSha256": review_digest,
            "sourceArchiveSha256": source_archive["sha256"],
            "scope": {
                "training": True,
                "commercialModelUse": True,
                "rawDatasetDistribution": False,
            },
        }
        raw = _json_bytes(document, pretty=True)
        return document, raw, _sha256(raw)

    def _materialize_reviews(
        self,
        *,
        temporary: Path,
        base: _BaseMaterialization,
        base_manifest: dict[str, Any],
        base_digest: str,
        review_manifest: dict[str, Any],
        review_digest: str,
        queue: dict[str, dict[str, Any]],
        provenance: dict[str, dict[str, Any]],
        attestation_sha256: str,
    ) -> _ReviewMaterialization:
        result = _ReviewMaterialization([], [], [], [], [], Counter())
        for review_id, record in sorted(queue.items()):
            decision = self._decisions[review_id]
            result.status_counts[decision.status.value] += 1
            result.decision_lines.append(
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
                result.rejected_count += 1
                result.rejected_lines.append(_rejected_evidence(review_id, record, decision))
                continue
            sample, file_entry = self._materialize_review_sample(
                temporary=temporary,
                base=base,
                base_manifest=base_manifest,
                base_digest=base_digest,
                review_manifest=review_manifest,
                review_digest=review_digest,
                record=record,
                provenance=provenance[review_id],
                decision=decision,
                attestation_sha256=attestation_sha256,
            )
            result.files.append(file_entry)
            result.annotation_lines.append(
                _json_bytes(sample.model_dump(mode="json", by_alias=True), pretty=False)
            )
            result.attribution_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_dataset": str(review_manifest["sourceId"]),
                    "source_revision": "human-review-attested",
                    "license": "PROPRIETARY-FIRST-PARTY",
                    "author": self._rights_holder,
                    "landing_url": "",
                }
            )
            result.promoted_count += 1
            result.annotation_count += len(sample.annotations)
            result.positive_count += bool(sample.annotations)
            result.negative_count += not bool(sample.annotations)
        return result

    def _materialize_review_sample(
        self,
        *,
        temporary: Path,
        base: _BaseMaterialization,
        base_manifest: dict[str, Any],
        base_digest: str,
        review_manifest: dict[str, Any],
        review_digest: str,
        record: dict[str, Any],
        provenance: dict[str, Any],
        decision: DetectorReviewDecision,
        attestation_sha256: str,
    ) -> tuple[DetectorSample, dict[str, object]]:
        digest = str(record["sourceImageSha256"])
        if digest in base.image_hashes:
            raise DetectorDatasetError(
                "reviewed warehouse image duplicates an existing production image"
            )
        base.image_hashes.add(digest)
        source_image = _safe_child(self._review, str(record["imagePath"]))
        if _sha256_file(source_image) != digest:
            raise DetectorDatasetError("warehouse review image checksum changed")
        relative = PurePosixPath("images", digest[:2], f"{digest}.jpg")
        destination = temporary.joinpath(*relative.parts)
        _copy_new(source_image, destination)
        sample = _reviewed_warehouse_sample(
            record=record,
            provenance=provenance,
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
        return sample, _file_entry(destination, temporary)

    def _write_evidence(
        self,
        *,
        temporary: Path,
        base_manifest: dict[str, Any],
        review_manifest: dict[str, Any],
        review_digest: str,
        base: _BaseMaterialization,
        reviewed: _ReviewMaterialization,
        attestation_raw: bytes,
    ) -> list[dict[str, object]]:
        payloads = self._evidence_payloads(
            base_manifest=base_manifest,
            review_manifest=review_manifest,
            review_digest=review_digest,
            base=base,
            reviewed=reviewed,
            attestation_raw=attestation_raw,
        )
        files: list[dict[str, object]] = []
        for name, payload in payloads:
            path = temporary / name
            _write_new(path, payload)
            files.append(_file_entry(path, temporary))
        return files

    def _evidence_payloads(
        self,
        *,
        base_manifest: dict[str, Any],
        review_manifest: dict[str, Any],
        review_digest: str,
        base: _BaseMaterialization,
        reviewed: _ReviewMaterialization,
        attestation_raw: bytes,
    ) -> tuple[tuple[str, bytes], ...]:
        duplicates = (self._base / "DUPLICATES.jsonl").read_bytes()
        duplicates += (self._review / "DUPLICATES.jsonl").read_bytes()
        rejects = (self._base / "REJECTS.jsonl").read_bytes()
        rejects += b"".join(reviewed.rejected_lines)
        attribution_rows = _read_attribution(self._base / "ATTRIBUTION.csv")
        attribution_rows.extend(reviewed.attribution_rows)
        return (
            (
                "annotations.jsonl",
                b"".join([*base.annotation_lines, *reviewed.annotation_lines]),
            ),
            ("REVIEW_QUEUE.jsonl", b""),
            ("REVIEW_DECISIONS.jsonl", b"".join(reviewed.decision_lines)),
            ("DUPLICATES.jsonl", duplicates),
            ("REJECTS.jsonl", rejects),
            ("ATTRIBUTION.csv", _attribution(attribution_rows)),
            ("RIGHTS_ATTESTATION.json", attestation_raw),
            (
                "WAREHOUSE_REVIEW_SOURCE_MANIFEST.json",
                (self._review / "source-manifest.json").read_bytes(),
            ),
            (
                "WAREHOUSE_REVIEW_PROVENANCE.jsonl",
                (self._review / "PROVENANCE.jsonl").read_bytes(),
            ),
            (
                "SOURCE_CARD.md",
                _source_card(
                    target_id=self._target_source_id,
                    base_id=str(base_manifest["sourceId"]),
                    review_id=str(review_manifest["sourceId"]),
                    production_count=base.sample_count + reviewed.promoted_count,
                    promoted_count=reviewed.promoted_count,
                    positive_count=reviewed.positive_count,
                    negative_count=reviewed.negative_count,
                    rejected_count=reviewed.rejected_count,
                    rights_holder=self._rights_holder,
                    review_digest=review_digest,
                ).encode("utf-8"),
            ),
        )

    def _production_manifest(
        self,
        *,
        files: list[dict[str, object]],
        base_manifest: dict[str, Any],
        base_digest: str,
        review_manifest: dict[str, Any],
        review_digest: str,
        base: _BaseMaterialization,
        reviewed: _ReviewMaterialization,
        attestation: dict[str, Any],
        attestation_sha256: str,
        now: datetime,
    ) -> dict[str, Any]:
        base_statistics = base_manifest.get("statistics", {})
        review_statistics = review_manifest.get("statistics", {})
        inventory_sha256 = _sha256(
            "\n".join(
                (
                    str(base_manifest["inputInventorySha256"]),
                    review_digest,
                    _sha256(b"".join(reviewed.decision_lines)),
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
            "collectionMethod": "FIRST_PARTY_IMAGES_VIDEO_AND_WAREHOUSE_CAMERA",
            "rightsAssertion": _RIGHTS_ASSERTION,
            "licenseStatus": _LICENSE_STATUS,
            "privacyClassification": "RESTRICTED_VEHICLE_IDENTIFIER",
            "acceptanceEligible": True,
            "releaseEligible": True,
            "distributionEligible": False,
            "promotionEligible": True,
            "annotationPolicy": (
                "VERIFIED_BASE_LABELS_PLUS_RIGHTS_ATTESTED_HUMAN_WAREHOUSE_REVIEW"
            ),
            "sampleCount": base.sample_count + reviewed.promoted_count,
            "annotationCount": base.annotation_count + reviewed.annotation_count,
            "negativeSampleCount": base.negative_count + reviewed.negative_count,
            "reviewQueueCount": 0,
            "statistics": {
                "inputImageFiles": int(base_statistics.get("inputImageFiles", base.sample_count))
                + int(review_manifest["sourceArchive"]["imageCount"]),
                "uniqueImages": base.sample_count + len(self._decisions),
                "verifiedProductionImages": base.sample_count + reviewed.promoted_count,
                "existingProductionImages": base.sample_count,
                "humanReviewedWarehousePromoted": reviewed.promoted_count,
                "humanReviewedWarehousePositive": reviewed.positive_count,
                "humanReviewedWarehouseNegative": reviewed.negative_count,
                "humanRejectedWarehouse": reviewed.rejected_count,
                "humanDecisionStatusCounts": dict(sorted(reviewed.status_counts.items())),
                "warehouseExactDuplicateFilesExcluded": int(
                    review_statistics.get("exactDuplicateFilesExcluded", 0)
                ),
                "warehouseNearDuplicateImagesExcluded": int(
                    review_statistics.get("perceptualNearDuplicateImagesExcluded", 0)
                ),
                "exactDuplicateFilesExcluded": int(
                    base_statistics.get("exactDuplicateFilesExcluded", 0)
                )
                + int(review_statistics.get("exactDuplicateFilesExcluded", 0)),
                "unsupportedFiles": int(base_statistics.get("unsupportedFiles", 0))
                + int(review_statistics.get("rejectedUniqueImages", 0)),
                "remainingPendingReview": 0,
            },
            "inputInventorySha256": inventory_sha256,
            "labelReference": {
                "type": "BASE_SOURCE_PLUS_ATTESTED_WAREHOUSE_REVIEW",
                "id": base_manifest["sourceId"],
                "sha256": base_digest,
            },
            "parentSource": {
                "id": base_manifest["sourceId"],
                "manifestSha256": base_digest,
            },
            "warehouseReviewPromotion": {
                "reviewSourceId": review_manifest["sourceId"],
                "reviewSourceManifestSha256": review_digest,
                "reviewSourceManifestPath": "WAREHOUSE_REVIEW_SOURCE_MANIFEST.json",
                "reviewProvenancePath": "WAREHOUSE_REVIEW_PROVENANCE.jsonl",
                "decisionEvidencePath": "REVIEW_DECISIONS.jsonl",
                "promotedCount": reviewed.promoted_count,
                "positiveCount": reviewed.positive_count,
                "negativeCount": reviewed.negative_count,
                "rejectedCount": reviewed.rejected_count,
                "remainingPendingCount": 0,
            },
            "rightsAttestation": {
                "path": "RIGHTS_ATTESTATION.json",
                "sha256": attestation_sha256,
                "assertion": _RIGHTS_ASSERTION,
                "rightsHolder": self._rights_holder,
                "attestedBy": self._attested_by,
                "attestedAt": attestation["attestedAt"],
            },
            "files": sorted(files, key=lambda item: str(item["path"])),
        }


def _validate_review_evidence(
    queue: dict[str, dict[str, Any]],
    provenance: dict[str, dict[str, Any]],
    decisions: dict[str, DetectorReviewDecision],
) -> None:
    if set(queue) != set(provenance):
        raise DetectorDatasetError("warehouse review queue and provenance do not match")
    if set(decisions) != set(queue):
        missing = len(set(queue) - set(decisions))
        unknown = len(set(decisions) - set(queue))
        raise DetectorDatasetError(
            f"warehouse promotion requires all review decisions; missing={missing}, "
            f"unknown={unknown}"
        )
    for review_id, decision in decisions.items():
        _validate_decision(queue[review_id], decision)


def _validate_decision(record: dict[str, Any], decision: DetectorReviewDecision) -> None:
    if decision.revision < 1 or decision.reviewed_at.tzinfo is None:
        raise DetectorDatasetError("warehouse review decision revision is invalid")
    if decision.status is DetectorReviewStatus.PENDING_REVIEW:
        raise DetectorDatasetError("warehouse review decision is still pending")
    if (
        decision.status
        in {
            DetectorReviewStatus.APPROVED,
            DetectorReviewStatus.CORRECTED,
        }
        and not decision.annotations
    ):
        raise DetectorDatasetError("positive warehouse review decision has no annotation")
    if (
        decision.status
        in {
            DetectorReviewStatus.NEGATIVE,
            DetectorReviewStatus.REJECTED,
        }
        and decision.annotations
    ):
        raise DetectorDatasetError("negative/rejected warehouse decision contains annotations")
    image_sha = record.get("sourceImageSha256")
    if not isinstance(image_sha, str) or not _SHA256.fullmatch(image_sha):
        raise DetectorDatasetError("warehouse review queue image hash is invalid")


def _reviewed_warehouse_sample(
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
    image = cv2.imread(str(image_file))
    if image is None or image.size == 0:
        raise DetectorDatasetError("promoted warehouse image cannot be decoded")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    annotations = _reviewed_annotations(decision, width, height)
    source = _canonical_provenance(provenance)
    digest = str(record["sourceImageSha256"])
    return DetectorSample(
        sampleId=f"phins-first-party-warehouse-plate-{digest[:24]}",
        imagePath=image_path,
        groupId=str(source["groupId"]),
        cameraId=str(source["cameraId"]),
        capturedAt=_parse_timestamp(str(source["capturedAt"])),
        split=None,
        attributes={
            "sourceCollection": "FIRST_PARTY_WAREHOUSE_CAMERA",
            "sourceLicense": "PROPRIETARY-FIRST-PARTY",
            "sourceOwner": rights_holder,
            "sourceImageSha256": digest,
            "sourceFilenameSha256": str(record["sourceFilenameSha256"]),
            "sourceRawImageSha256": str(source["sourceRawImageSha256"]),
            "sourceArchiveSha256": str(source["sourceArchiveSha256"]),
            "cameraView": str(source["cameraView"]),
            "baseSourceId": base_source_id,
            "baseSourceManifestSha256": base_manifest_sha256,
            "sourceReviewId": str(record["reviewId"]),
            "sourceReviewReason": str(record["reason"]),
            "reviewSourceId": review_source_id,
            "reviewSourceManifestSha256": review_manifest_sha256,
            "rightsAssertion": _RIGHTS_ASSERTION,
            "rightsAttestationSha256": attestation_sha256,
            "annotationOrigin": "REVISIONED_HUMAN_WAREHOUSE_REVIEW",
            "annotationReviewStatus": decision.status.value,
            "reviewRevision": decision.revision,
            "reviewedBy": decision.reviewed_by,
            "reviewedAt": _timestamp(decision.reviewed_at),
            "capturedAtBasis": "WAREHOUSE_ARCHIVE_FILENAME_TIMESTAMP",
            "actualCaptureTimeKnown": False,
            "groupingBasis": "WAREHOUSE_TRANSACTION_ID",
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


def _reviewed_annotations(
    decision: DetectorReviewDecision,
    width: int,
    height: int,
) -> tuple[DetectorAnnotation, ...]:
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
                "annotationOrigin": "REVISIONED_HUMAN_WAREHOUSE_REVIEW",
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
            raise DetectorDatasetError("promoted warehouse annotation is outside image bounds")
    return annotations


def _canonical_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    records = provenance.get("records") if isinstance(provenance, dict) else None
    if not isinstance(records, list) or not records:
        raise DetectorDatasetError("warehouse review provenance records are invalid")
    selected = min(
        (item for item in records if isinstance(item, dict)),
        key=lambda item: str(item.get("archiveMember", "")),
        default=None,
    )
    if (
        selected is None
        or not isinstance(selected.get("groupId"), str)
        or not isinstance(selected.get("cameraId"), str)
        or not isinstance(selected.get("capturedAt"), str)
        or not isinstance(selected.get("cameraView"), str)
        or not isinstance(selected.get("sourceRawImageSha256"), str)
        or not _SHA256.fullmatch(str(selected["sourceRawImageSha256"]))
        or not isinstance(selected.get("sourceArchiveSha256"), str)
        or not _SHA256.fullmatch(str(selected["sourceArchiveSha256"]))
    ):
        raise DetectorDatasetError("warehouse review provenance record is invalid")
    return selected


def _rejected_evidence(
    review_id: str,
    record: dict[str, Any],
    decision: DetectorReviewDecision,
) -> bytes:
    return _json_bytes(
        {
            "schemaVersion": 1,
            "reviewId": review_id,
            "sourceImageSha256": record["sourceImageSha256"],
            "reason": "HUMAN_REJECTED_WAREHOUSE_REVIEW",
            "reviewRevision": decision.revision,
            "reviewedBy": decision.reviewed_by,
            "reviewedAt": _timestamp(decision.reviewed_at),
            "note": decision.note,
        },
        pretty=False,
    )


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
    review_digest: str,
) -> str:
    return f"""# {target_id}

Immutable production source combining `{base_id}` with fully reviewed,
rights-attested warehouse camera samples from `{review_id}`.

- Rights holder: `{rights_holder}`
- Rights assertion: `{_RIGHTS_ASSERTION}`
- Review source manifest: `{review_digest}`
- Production samples: {production_count}
- Reviewed warehouse samples promoted: {promoted_count}
- Reviewed warehouse positives: {positive_count}
- Reviewed warehouse negatives: {negative_count}
- Reviewed warehouse rejects: {rejected_count}
- Remaining review items: 0
- Raw dataset distribution: disabled

`RIGHTS_ATTESTATION.json` records the explicit first-party confirmation and
`REVIEW_DECISIONS.jsonl` preserves each terminal human decision. Warehouse
transaction grouping prevents related camera frames crossing dataset splits.
"""


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> AttestedWarehousePromotionResult:
    promotion = manifest["warehouseReviewPromotion"]
    return AttestedWarehousePromotionResult(
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


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DetectorDatasetError("warehouse provenance timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise DetectorDatasetError("warehouse provenance timestamp has no timezone")
    return parsed.astimezone(UTC)
