"""Import warehouse vehicle captures into an immutable bootstrap source.

The archive contains camera frames with a blue detector rectangle burned into
the pixels.  This importer recovers that rectangle, removes it from the image,
deduplicates both exact bytes and conservative perceptual matches, and admits
only high-confidence vehicle-class suggestions.  Everything uncertain remains
outside ``annotations.jsonl`` and is described in an auditable review log.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.bootstrap.domain import (
    AcquiredDetectorSample,
    BootstrapSourceInfo,
)
from vehicle_intelligence.training.bootstrap.writer import (
    BootstrapSourceWriter,
    verify_bootstrap_source,
)
from vehicle_intelligence.training.domain import (
    DetectorAnnotation,
    DetectorRole,
    DetectorSample,
    TrainingBoundingBox,
)

_ARCHIVE_IMAGE = re.compile(
    r"^Cameras/(?P<view>front|rear)_image_"
    r"(?P<group>txn_[0-9a-f]+|[0-9a-f-]{36})_"
    r"(?P<timestamp>[0-9]{13})_(?P<nonce>[A-Za-z0-9]+)"
    r"(?P<suffix>\.(?:jpg|jpeg|png))$"
)
_VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_MAX_NESTED_BOX_DELTA = 16
_ATTRIBUTION_FIELDS = (
    "sample_id",
    "source_dataset",
    "source_revision",
    "license",
    "author",
    "landing_url",
)


@dataclass(frozen=True, slots=True)
class VehicleClassPrediction:
    class_name: str
    confidence: float
    class_margin: float
    area_ratio: float
    center_distance: float


class VehicleCropClassifier(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_sha256(self) -> str: ...

    def classify(self, image: NDArray[np.uint8]) -> VehicleClassPrediction | None: ...


@dataclass(frozen=True, slots=True)
class WarehouseVehicleImportOptions:
    accepted_classes: tuple[str, ...] = ("truck",)
    classification_confidence: float = 0.65
    minimum_class_margin: float = 0.20
    minimum_target_area_ratio: float = 0.75
    maximum_target_center_distance: float = 0.22
    minimum_brightness: float = 25.0
    minimum_contrast: float = 20.0
    minimum_sharpness: float = 150.0
    maximum_phash_distance: int = 8
    maximum_dhash_distance: int = 10
    maximum_thumbnail_mae: float = 34.0
    minimum_thumbnail_correlation: float = 0.80
    minimum_edge_dice: float = 0.85
    maximum_members: int = 10_000
    maximum_member_bytes: int = 20_000_000
    maximum_total_bytes: int = 5_000_000_000
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if not self.accepted_classes or set(self.accepted_classes) - set(
            _VEHICLE_CLASS_IDS.values()
        ):
            raise ValueError("accepted warehouse vehicle classes are invalid")
        unit_values = (
            self.classification_confidence,
            self.minimum_class_margin,
            self.minimum_target_area_ratio,
            self.maximum_target_center_distance,
            self.minimum_thumbnail_correlation,
            self.minimum_edge_dice,
        )
        if any(not 0 <= value <= 1 for value in unit_values):
            raise ValueError("warehouse import probability thresholds must be within [0, 1]")
        if (
            self.maximum_phash_distance < 0
            or self.maximum_dhash_distance < 0
            or self.maximum_thumbnail_mae < 0
            or self.maximum_members < 1
            or self.maximum_member_bytes < 1
            or self.maximum_total_bytes < self.maximum_member_bytes
            or not 1 <= self.jpeg_quality <= 100
        ):
            raise ValueError("warehouse import bounds are invalid")


@dataclass(frozen=True, slots=True)
class WarehouseVehicleSourceResult:
    source_id: str
    directory: Path
    manifest_sha256: str
    archive_sha256: str
    base_sample_count: int
    appended_sample_count: int
    combined_sample_count: int
    exact_duplicate_files_excluded: int
    near_duplicate_images_excluded: int
    review_queue_count: int
    reject_count: int
    reused: bool = False


@dataclass(frozen=True, slots=True)
class _ArchiveMember:
    name: str
    view: str
    group_id: str
    timestamp_ms: int
    size: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    raw_sha256: str
    member: _ArchiveMember
    width: int
    height: int
    bbox: tuple[int, int, int, int]
    bbox_coverage: float
    brightness: float
    contrast: float
    sharpness: float
    phash: int
    dhash: int
    normalized_gray: NDArray[np.uint8]
    normalized_edges: NDArray[np.uint8]
    prediction: VehicleClassPrediction | None


@dataclass(frozen=True, slots=True)
class _RejectedImage:
    raw_sha256: str
    member: _ArchiveMember
    reason: str


@dataclass(frozen=True, slots=True)
class _ArchiveScan:
    member_count: int
    image_count: int
    declared_bytes: int
    names_by_digest: dict[str, tuple[_ArchiveMember, ...]]
    candidates: tuple[_Candidate, ...]
    rejects: tuple[_RejectedImage, ...]


# Shared immutable scan contracts used by the plate-review adapter.  The
# vehicle importer remains the owner of archive parsing and overlay recovery so
# both dataset roles apply exactly the same safety and deduplication policy.
WarehouseArchiveMember = _ArchiveMember
WarehouseImageCandidate = _Candidate
WarehouseRejectedImage = _RejectedImage
WarehouseArchiveScan = _ArchiveScan


class OpenCvYoloVehicleCropClassifier:
    """Small OpenCV-DNN classifier for a standard raw-output COCO YOLO model."""

    def __init__(self, model_path: Path, *, image_size: int = 640) -> None:
        path = model_path.expanduser().resolve()
        if path.suffix.lower() != ".onnx" or not path.is_file():
            raise DetectorDatasetError("warehouse classifier must be a local ONNX file")
        self._model_sha256 = _sha256_file(path)
        self._model_name = path.name
        self._image_size = image_size
        try:
            self._network = cv2.dnn.readNetFromONNX(str(path))
        except cv2.error as exc:
            raise DetectorDatasetError("cannot load warehouse vehicle classifier") from exc

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def classify(self, image: NDArray[np.uint8]) -> VehicleClassPrediction | None:
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise DetectorDatasetError("warehouse classifier received an invalid crop")
        tensor, scale, left, top = self._preprocess(image)
        try:
            self._network.setInput(tensor)
            predictions = np.asarray(self._network.forward(), dtype=np.float32)
        except cv2.error as exc:
            raise DetectorDatasetError("warehouse vehicle classification failed") from exc
        while predictions.ndim > 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise DetectorDatasetError("warehouse classifier output shape is unsupported")
        if predictions.shape[0] == 84:
            predictions = predictions.T
        if predictions.shape[1] != 84:
            raise DetectorDatasetError("warehouse classifier must expose 80 COCO classes")
        return self._select_target(predictions, image.shape, scale, left, top)

    def _preprocess(self, image: NDArray[np.uint8]) -> tuple[NDArray[np.float32], float, int, int]:
        height, width = image.shape[:2]
        scale = min(self._image_size / width, self._image_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (self._image_size - resized_width) // 2
        right = self._image_size - resized_width - left
        top = (self._image_size - resized_height) // 2
        bottom = self._image_size - resized_height - top
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1 / 255.0,
            size=(self._image_size, self._image_size),
            swapRB=True,
            crop=False,
        )
        return np.asarray(tensor, dtype=np.float32), scale, left, top

    @staticmethod
    def _select_target(
        predictions: NDArray[np.float32],
        image_shape: tuple[int, ...],
        scale: float,
        left: int,
        top: int,
    ) -> VehicleClassPrediction | None:
        height, width = image_shape[:2]
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        candidates: list[tuple[float, VehicleClassPrediction]] = []
        for index in np.where(
            (confidences >= 0.03) & np.isin(class_ids, tuple(_VEHICLE_CLASS_IDS))
        )[0]:
            center_x, center_y, box_width, box_height = predictions[index, :4]
            x1 = max(0.0, (float(center_x - box_width / 2) - left) / scale)
            y1 = max(0.0, (float(center_y - box_height / 2) - top) / scale)
            x2 = min(float(width), (float(center_x + box_width / 2) - left) / scale)
            y2 = min(float(height), (float(center_y + box_height / 2) - top) / scale)
            if x2 <= x1 or y2 <= y1:
                continue
            area_ratio = (x2 - x1) * (y2 - y1) / (width * height)
            normalized_x = (x1 + x2) / (2 * width)
            normalized_y = (y1 + y2) / (2 * height)
            center_distance = math.hypot(normalized_x - 0.5, normalized_y - 0.5)
            vehicle_scores = sorted(
                float(scores[index, class_id]) for class_id in _VEHICLE_CLASS_IDS
            )
            confidence = float(confidences[index])
            prediction = VehicleClassPrediction(
                class_name=_VEHICLE_CLASS_IDS[int(class_ids[index])],
                confidence=confidence,
                class_margin=vehicle_scores[-1] - vehicle_scores[-2],
                area_ratio=area_ratio,
                center_distance=center_distance,
            )
            target_score = (
                confidence * min(1.0, area_ratio / 0.35) * max(0.0, 1.0 - center_distance)
            )
            candidates.append((target_score, prediction))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None


class WarehouseVehicleSourceBuilder:
    def __init__(
        self,
        *,
        archive_path: Path,
        base_source_directory: Path,
        output_directory: Path,
        model_path: Path,
        source_id: str,
        owner_namespace: str,
        founder_id: str,
        options: WarehouseVehicleImportOptions | None = None,
        classifier: VehicleCropClassifier | None = None,
    ) -> None:
        self._archive = archive_path.expanduser().resolve()
        self._base_source = base_source_directory.expanduser().resolve()
        self._target = output_directory.expanduser().resolve()
        self._source_id = _identifier(source_id, "warehouse source id")
        self._owner_namespace = _identifier(owner_namespace, "owner namespace")
        self._founder_id = _identifier(founder_id, "founder id")
        self._options = options or WarehouseVehicleImportOptions()
        self._classifier = classifier or OpenCvYoloVehicleCropClassifier(model_path)

    def build(self) -> WarehouseVehicleSourceResult:
        if self._target.exists():
            return _existing_result(self._target, self._source_id)
        if not self._archive.is_file():
            raise DetectorDatasetError(f"warehouse image archive is missing: {self._archive}")
        if self._target == self._base_source or self._target.is_relative_to(self._base_source):
            raise DetectorDatasetError("warehouse combined source cannot overwrite its base source")

        base_samples, base_manifest_sha256 = _load_base_source(self._base_source)
        archive_sha256 = _sha256_file(self._archive)
        scan = scan_warehouse_archive(
            self._archive,
            self._options,
            classifier=self._classifier,
        )
        eligible = [candidate for candidate in scan.candidates if self._is_eligible(candidate)]
        selected, near_duplicates = deduplicate_warehouse_candidates(
            eligible,
            self._options,
        )
        appended, cleaned_duplicates = self._materialize_samples(selected, archive_sha256)
        eligible_digests = {candidate.raw_sha256 for candidate in eligible}
        review = [
            candidate
            for candidate in scan.candidates
            if candidate.raw_sha256 not in eligible_digests
        ]
        duplicate_records = _exact_duplicate_records(scan.names_by_digest)
        duplicate_records.extend(near_duplicates)
        duplicate_records.extend(cleaned_duplicates)

        report = self._report(
            archive_sha256=archive_sha256,
            base_manifest_sha256=base_manifest_sha256,
            base_sample_count=len(base_samples),
            appended_sample_count=len(appended),
            scan=scan,
            review_count=len(review),
            near_duplicate_count=len(near_duplicates),
            cleaned_duplicate_count=len(cleaned_duplicates),
        )
        source = BootstrapSourceInfo(
            source_id=self._source_id,
            dataset_url=f"urn:sha256:{archive_sha256}",
            revision=archive_sha256,
            annotation_license="MIXED-SEE-ATTRIBUTION",
            image_license="MIXED-SEE-ATTRIBUTION",
            license_review_status="REVIEW_REQUIRED",
            acceptance_eligible=False,
        )
        evidence = {
            "DUPLICATES.jsonl": _json_lines(duplicate_records),
            "REJECTS.jsonl": _json_lines(self._reject_records(scan)),
            "REVIEW_QUEUE.jsonl": _json_lines(self._review_records(review)),
            "INGESTION_REPORT.json": _json_bytes(report, pretty=True),
            "SOURCE_CARD.md": self._source_card(report).encode(),
        }
        result = BootstrapSourceWriter(DetectorRole.VEHICLE, self._target).write(
            source,
            [*base_samples, *appended],
            evidence_files=evidence,
        )
        return WarehouseVehicleSourceResult(
            source_id=self._source_id,
            directory=result.directory,
            manifest_sha256=result.manifest_sha256,
            archive_sha256=archive_sha256,
            base_sample_count=len(base_samples),
            appended_sample_count=len(appended),
            combined_sample_count=result.sample_count,
            exact_duplicate_files_excluded=sum(
                len(names) - 1 for names in scan.names_by_digest.values()
            ),
            near_duplicate_images_excluded=len(near_duplicates) + len(cleaned_duplicates),
            review_queue_count=len(review),
            reject_count=len(scan.rejects),
        )

    def _is_eligible(self, candidate: _Candidate) -> bool:
        prediction = candidate.prediction
        return bool(
            prediction is not None
            and prediction.class_name in self._options.accepted_classes
            and prediction.confidence >= self._options.classification_confidence
            and prediction.class_margin >= self._options.minimum_class_margin
            and prediction.area_ratio >= self._options.minimum_target_area_ratio
            and prediction.center_distance <= self._options.maximum_target_center_distance
            and candidate.brightness >= self._options.minimum_brightness
            and candidate.contrast >= self._options.minimum_contrast
            and candidate.sharpness >= self._options.minimum_sharpness
        )

    def _materialize_samples(
        self,
        selected: list[_Candidate],
        archive_sha256: str,
    ) -> tuple[list[AcquiredDetectorSample], list[dict[str, Any]]]:
        by_name = {candidate.member.name: candidate for candidate in selected}
        samples: list[AcquiredDetectorSample] = []
        cleaned_digests: dict[str, _Candidate] = {}
        duplicates: list[dict[str, Any]] = []
        try:
            with tarfile.open(self._archive, mode="r:gz") as archive:
                for tar_info in archive:
                    candidate = by_name.get(tar_info.name)
                    if candidate is None:
                        continue
                    stream = archive.extractfile(tar_info)
                    if stream is None:
                        raise DetectorDatasetError("selected warehouse image cannot be read")
                    data = stream.read(self._options.maximum_member_bytes + 1)
                    if _sha256(data) != candidate.raw_sha256:
                        raise DetectorDatasetError("warehouse archive changed during ingestion")
                    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.size == 0:
                        raise DetectorDatasetError("selected warehouse image cannot be decoded")
                    cleaned = _remove_blue_overlay(image, candidate.bbox)
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        cleaned,
                        [cv2.IMWRITE_JPEG_QUALITY, self._options.jpeg_quality],
                    )
                    if not ok:
                        raise DetectorDatasetError("cleaned warehouse image cannot be encoded")
                    image_bytes = encoded.tobytes()
                    cleaned_sha256 = _sha256(image_bytes)
                    previous = cleaned_digests.get(cleaned_sha256)
                    if previous is not None:
                        duplicates.append(
                            {
                                "reason": "POST_CLEAN_EXACT_DUPLICATE",
                                "cleanedSha256": cleaned_sha256,
                                "keptRawSha256": previous.raw_sha256,
                                "excludedRawSha256": candidate.raw_sha256,
                                "keptArchiveMember": previous.member.name,
                                "excludedArchiveMember": candidate.member.name,
                            }
                        )
                        continue
                    cleaned_digests[cleaned_sha256] = candidate
                    samples.append(
                        self._sample(candidate, cleaned_sha256, image_bytes, archive_sha256)
                    )
        except DetectorDatasetError:
            raise
        except (OSError, tarfile.TarError, cv2.error) as exc:
            raise DetectorDatasetError("cannot materialize selected warehouse images") from exc
        if len(samples) + len(duplicates) != len(selected):
            raise DetectorDatasetError("selected warehouse images are missing from the archive")
        return samples, duplicates

    def _sample(
        self,
        candidate: _Candidate,
        cleaned_sha256: str,
        image_bytes: bytes,
        archive_sha256: str,
    ) -> AcquiredDetectorSample:
        prediction = candidate.prediction
        if prediction is None:
            raise DetectorDatasetError("eligible warehouse sample is missing a prediction")
        x, y, width, height = candidate.bbox
        relative = f"images/warehouse/{cleaned_sha256[:2]}/{cleaned_sha256}.jpg"
        sample = DetectorSample(
            sampleId=f"phins-warehouse-{prediction.class_name}-{cleaned_sha256[:24]}",
            imagePath=relative,
            groupId=f"phins-warehouse:{candidate.member.group_id}",
            cameraId=f"warehouse-{candidate.member.view}-camera",
            capturedAt=datetime.fromtimestamp(candidate.member.timestamp_ms / 1000, tz=UTC),
            split=None,
            attributes={
                "acceptanceEligible": False,
                "bootstrapOnly": True,
                "distributionEligible": False,
                "licenseReviewStatus": "REVIEW_REQUIRED",
                "sourceDataset": self._source_id,
                "archiveSha256": archive_sha256,
                "sourceImageSha256": candidate.raw_sha256,
                "cleanedImageSha256": cleaned_sha256,
                "cameraView": candidate.member.view,
                "bboxRecoveryMethod": "BURNED_IN_BLUE_RECTANGLE",
                "bboxLineCoverage": round(candidate.bbox_coverage, 6),
                "imageBrightness": round(candidate.brightness, 4),
                "imageContrast": round(candidate.contrast, 4),
                "imageSharpness": round(candidate.sharpness, 4),
                "annotationOrigin": "RECOVERED_BBOX_AND_HIGH_CONFIDENCE_MODEL_CLASS",
                "annotationReviewStatus": "MODEL_ASSISTED_UNREVIEWED",
                "classificationModel": self._classifier.model_name,
                "classificationModelSha256": self._classifier.model_sha256,
                "classificationConfidence": round(prediction.confidence, 6),
                "classificationMargin": round(prediction.class_margin, 6),
                "targetAreaRatio": round(prediction.area_ratio, 6),
                "targetCenterDistance": round(prediction.center_distance, 6),
            },
            annotations=(
                DetectorAnnotation(
                    className=prediction.class_name,
                    bbox=TrainingBoundingBox(x=x, y=y, width=width, height=height),
                    attributes={
                        "bboxOrigin": "BURNED_IN_BLUE_RECTANGLE",
                        "classOrigin": "HIGH_CONFIDENCE_COCO_MODEL",
                        "humanReviewed": False,
                        "confidence": round(prediction.confidence, 6),
                        "classMargin": round(prediction.class_margin, 6),
                    },
                ),
            ),
        )
        return AcquiredDetectorSample(
            sample=sample,
            image_bytes=image_bytes,
            attribution={
                "sample_id": sample.sample_id,
                "source_dataset": self._source_id,
                "source_revision": archive_sha256,
                "license": "RIGHTS_REVIEW_REQUIRED",
                "author": self._founder_id,
                "landing_url": "",
            },
        )

    def _review_records(self, candidates: list[_Candidate]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item.raw_sha256):
            prediction = candidate.prediction
            records.append(
                {
                    "schemaVersion": 1,
                    "reviewId": f"warehouse-review-{candidate.raw_sha256[:24]}",
                    "status": "PENDING_REVIEW",
                    "reason": self._review_reason(candidate),
                    "archiveMember": candidate.member.name,
                    "sourceImageSha256": candidate.raw_sha256,
                    "cameraView": candidate.member.view,
                    "groupId": candidate.member.group_id,
                    "recoveredBbox": list(candidate.bbox),
                    "suggestion": (
                        {
                            "className": prediction.class_name,
                            "confidence": round(prediction.confidence, 6),
                            "classMargin": round(prediction.class_margin, 6),
                            "areaRatio": round(prediction.area_ratio, 6),
                            "centerDistance": round(prediction.center_distance, 6),
                        }
                        if prediction is not None
                        else None
                    ),
                }
            )
        return records

    def _review_reason(self, candidate: _Candidate) -> str:
        prediction = candidate.prediction
        if prediction is None:
            return "NO_SUPPORTED_VEHICLE_DETECTION"
        if prediction.class_name not in self._options.accepted_classes:
            return "CLASS_REQUIRES_HUMAN_REVIEW"
        if prediction.confidence < self._options.classification_confidence:
            return "LOW_CLASSIFICATION_CONFIDENCE"
        if prediction.class_margin < self._options.minimum_class_margin:
            return "AMBIGUOUS_CLASS_MARGIN"
        if (
            prediction.area_ratio < self._options.minimum_target_area_ratio
            or prediction.center_distance > self._options.maximum_target_center_distance
        ):
            return "TARGET_DETECTION_GEOMETRY_REJECTED"
        return "IMAGE_QUALITY_REQUIRES_REVIEW"

    @staticmethod
    def _reject_records(scan: _ArchiveScan) -> list[dict[str, Any]]:
        return [
            {
                "schemaVersion": 1,
                "reason": rejected.reason,
                "archiveMember": rejected.member.name,
                "sourceImageSha256": rejected.raw_sha256,
                "size": rejected.member.size,
            }
            for rejected in sorted(scan.rejects, key=lambda item: item.raw_sha256)
        ]

    def _report(
        self,
        *,
        archive_sha256: str,
        base_manifest_sha256: str,
        base_sample_count: int,
        appended_sample_count: int,
        scan: _ArchiveScan,
        review_count: int,
        near_duplicate_count: int,
        cleaned_duplicate_count: int,
    ) -> dict[str, Any]:
        exact_duplicates = sum(len(names) - 1 for names in scan.names_by_digest.values())
        return {
            "schemaVersion": 1,
            "type": "WAREHOUSE_VEHICLE_ARCHIVE_INGESTION",
            "sourceId": self._source_id,
            "ownerNamespace": self._owner_namespace,
            "founderId": self._founder_id,
            "archive": {
                "sha256": archive_sha256,
                "memberCount": scan.member_count,
                "imageCount": scan.image_count,
                "declaredImageBytes": scan.declared_bytes,
            },
            "baseSource": {
                "directoryName": self._base_source.name,
                "manifestSha256": base_manifest_sha256,
                "sampleCount": base_sample_count,
            },
            "model": {
                "name": self._classifier.model_name,
                "sha256": self._classifier.model_sha256,
            },
            "policy": {
                "acceptedClasses": list(self._options.accepted_classes),
                "classificationConfidence": self._options.classification_confidence,
                "minimumClassMargin": self._options.minimum_class_margin,
                "minimumTargetAreaRatio": self._options.minimum_target_area_ratio,
                "maximumTargetCenterDistance": self._options.maximum_target_center_distance,
                "nearDuplicate": {
                    "maximumPhashDistance": self._options.maximum_phash_distance,
                    "maximumDhashDistance": self._options.maximum_dhash_distance,
                    "maximumThumbnailMae": self._options.maximum_thumbnail_mae,
                    "minimumThumbnailCorrelation": self._options.minimum_thumbnail_correlation,
                    "minimumEdgeDice": self._options.minimum_edge_dice,
                    "sameCameraViewOnly": True,
                },
            },
            "statistics": {
                "uniqueRawImages": len(scan.names_by_digest),
                "recoverableVehicleBoxes": len(scan.candidates),
                "exactDuplicateFilesExcluded": exact_duplicates,
                "perceptualNearDuplicateImagesExcluded": near_duplicate_count,
                "postCleanExactDuplicateImagesExcluded": cleaned_duplicate_count,
                "rejectedUniqueImages": len(scan.rejects),
                "pendingReviewUniqueImages": review_count,
                "appendedWarehouseSamples": appended_sample_count,
                "combinedSamples": base_sample_count + appended_sample_count,
            },
            "eligibility": {
                "acceptanceEligible": False,
                "releaseEligible": False,
                "distributionEligible": False,
                "rightsReviewRequired": True,
                "humanClassReviewComplete": False,
            },
        }

    def _source_card(self, report: dict[str, Any]) -> str:
        stats = report["statistics"]
        perceptual_duplicates = (
            stats["perceptualNearDuplicateImagesExcluded"]
            + stats["postCleanExactDuplicateImagesExcluded"]
        )
        return f"""# {self._source_id}

Immutable bootstrap vehicle source combining the existing vehicle bootstrap
corpus with conservatively selected warehouse camera captures.

- Warehouse samples appended: {stats["appendedWarehouseSamples"]}
- Exact duplicate files excluded: {stats["exactDuplicateFilesExcluded"]}
- Perceptual/post-clean duplicates excluded: {perceptual_duplicates}
- Pending manual review: {stats["pendingReviewUniqueImages"]}
- Rejected unique images: {stats["rejectedUniqueImages"]}
- Automatically accepted classes: `{", ".join(self._options.accepted_classes)}`
- Rights review: required
- Acceptance/release eligibility: disabled

Blue detector rectangles were recovered as bounding boxes and inpainted from
accepted images.  COCO `car`, `bus`, and `motorcycle` suggestions are retained
only in `REVIEW_QUEUE.jsonl` because visual sampling showed warehouse-truck
taxonomy confusion.  See `INGESTION_REPORT.json` for the fixed thresholds and
model checksum.
"""


def scan_warehouse_archive(
    archive_path: Path,
    options: WarehouseVehicleImportOptions,
    *,
    classifier: VehicleCropClassifier | None = None,
) -> _ArchiveScan:
    """Read a warehouse archive once per unique image and recover cleanable frames."""

    archive_path = archive_path.expanduser().resolve()
    names_by_digest: dict[str, list[_ArchiveMember]] = defaultdict(list)
    candidates_by_digest: dict[str, _Candidate] = {}
    rejects_by_digest: dict[str, _RejectedImage] = {}
    member_count = 0
    image_count = 0
    declared_bytes = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for tar_info in archive:
                member_count += 1
                if member_count > options.maximum_members:
                    raise DetectorDatasetError("warehouse archive exceeds the member limit")
                _validate_tar_path(tar_info.name)
                if tar_info.isdir():
                    continue
                if not tar_info.isfile():
                    raise DetectorDatasetError("warehouse archive contains a link or device")
                member = _parse_archive_member(tar_info.name, tar_info.size)
                if tar_info.size <= 0 or tar_info.size > options.maximum_member_bytes:
                    raise DetectorDatasetError("warehouse archive image size is invalid")
                declared_bytes += tar_info.size
                if declared_bytes > options.maximum_total_bytes:
                    raise DetectorDatasetError("warehouse archive exceeds the byte limit")
                stream = archive.extractfile(tar_info)
                if stream is None:
                    raise DetectorDatasetError("warehouse archive image cannot be read")
                data = stream.read(options.maximum_member_bytes + 1)
                if len(data) != tar_info.size:
                    raise DetectorDatasetError("warehouse archive member size changed while read")
                image_count += 1
                raw_sha256 = _sha256(data)
                names_by_digest[raw_sha256].append(member)
                if raw_sha256 in candidates_by_digest or raw_sha256 in rejects_by_digest:
                    continue
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    rejects_by_digest[raw_sha256] = _RejectedImage(
                        raw_sha256, member, "IMAGE_DECODE_FAILED"
                    )
                    continue
                bbox_result = _find_blue_bbox(image)
                if bbox_result is None:
                    rejects_by_digest[raw_sha256] = _RejectedImage(
                        raw_sha256, member, "BURNED_IN_BOUNDING_BOX_NOT_RECOVERED"
                    )
                    continue
                bbox, coverage = bbox_result
                cleaned = _remove_blue_overlay(image, bbox)
                crop = _target_crop(cleaned, bbox)
                if crop.size == 0:
                    rejects_by_digest[raw_sha256] = _RejectedImage(
                        raw_sha256, member, "RECOVERED_BOUNDING_BOX_IS_EMPTY"
                    )
                    continue
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                normalized_gray = cv2.equalizeHist(
                    cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
                )
                candidates_by_digest[raw_sha256] = _Candidate(
                    raw_sha256=raw_sha256,
                    member=member,
                    width=image.shape[1],
                    height=image.shape[0],
                    bbox=bbox,
                    bbox_coverage=coverage,
                    brightness=float(np.mean(gray)),
                    contrast=float(np.std(gray)),
                    sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    phash=_phash(normalized_gray),
                    dhash=_dhash(normalized_gray),
                    normalized_gray=normalized_gray,
                    normalized_edges=cv2.Canny(normalized_gray, 60, 140),
                    prediction=classifier.classify(crop) if classifier is not None else None,
                )
    except DetectorDatasetError:
        raise
    except (OSError, tarfile.TarError, cv2.error) as exc:
        raise DetectorDatasetError("cannot scan warehouse image archive") from exc

    canonical_names = {
        digest: tuple(sorted(names, key=lambda item: item.name))
        for digest, names in names_by_digest.items()
    }
    candidates = tuple(
        replace(candidate, member=canonical_names[digest][0])
        for digest, candidate in sorted(candidates_by_digest.items())
    )
    rejects = tuple(
        replace(rejected, member=canonical_names[digest][0])
        for digest, rejected in sorted(rejects_by_digest.items())
    )
    if not candidates:
        raise DetectorDatasetError("warehouse archive contains no recoverable vehicle boxes")
    return _ArchiveScan(
        member_count=member_count,
        image_count=image_count,
        declared_bytes=declared_bytes,
        names_by_digest=canonical_names,
        candidates=candidates,
        rejects=rejects,
    )


def deduplicate_warehouse_candidates(
    candidates: list[_Candidate],
    options: WarehouseVehicleImportOptions,
) -> tuple[list[_Candidate], list[dict[str, Any]]]:
    """Collapse conservative perceptual matches while retaining the best frame."""

    parent = list(range(len(candidates)))
    adjacency: dict[int, set[int]] = defaultdict(set)
    direct_metrics: dict[tuple[int, int], dict[str, int | float]] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if candidates[left].member.view != candidates[right].member.view:
                continue
            metrics = _near_duplicate_metrics(candidates[left], candidates[right], options)
            if metrics is None:
                continue
            direct_metrics[(left, right)] = metrics
            adjacency[left].add(right)
            adjacency[right].add(left)
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        components[find(index)].append(index)
    keep: set[int] = set()
    records: list[dict[str, Any]] = []
    for component in components.values():
        keeper = min(component, key=lambda index: _quality_key(candidates[index]))
        keep.add(keeper)
        for index in sorted(component):
            if index == keeper:
                continue
            matches = [
                (
                    _direct_match_key(direct_metrics[tuple(sorted((index, neighbor)))]),
                    neighbor,
                    direct_metrics[tuple(sorted((index, neighbor)))],
                )
                for neighbor in adjacency[index]
            ]
            _, matched_index, metrics = min(matches)
            records.append(
                {
                    "reason": "PERCEPTUAL_NEAR_DUPLICATE",
                    "keptRawSha256": candidates[keeper].raw_sha256,
                    "excludedRawSha256": candidates[index].raw_sha256,
                    "keptArchiveMember": candidates[keeper].member.name,
                    "excludedArchiveMember": candidates[index].member.name,
                    "directMatchRawSha256": candidates[matched_index].raw_sha256,
                    "directMatchArchiveMember": candidates[matched_index].member.name,
                    "keeperIsDirectMatch": keeper in adjacency[index],
                    **metrics,
                }
            )
    selected = [candidate for index, candidate in enumerate(candidates) if index in keep]
    return selected, records


def clean_warehouse_image(
    image: NDArray[np.uint8], bbox: tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    """Remove the recovered detector overlay from a warehouse frame."""

    return _remove_blue_overlay(image, bbox)


def exact_warehouse_duplicate_records(
    names_by_digest: dict[str, tuple[_ArchiveMember, ...]],
) -> list[dict[str, Any]]:
    """Return deterministic evidence for byte-identical archive members."""

    return _exact_duplicate_records(names_by_digest)


def _load_base_source(root: Path) -> tuple[list[AcquiredDetectorSample], str]:
    manifest, manifest_sha256 = verify_bootstrap_source(root)
    if manifest.get("role") != DetectorRole.VEHICLE.value:
        raise DetectorDatasetError("warehouse base source must contain vehicle samples")
    attribution_path = root / "ATTRIBUTION.csv"
    try:
        rows = list(csv.DictReader(io.StringIO(attribution_path.read_text())))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DetectorDatasetError("warehouse base-source attribution is invalid") from exc
    attribution = {row.get("sample_id", ""): row for row in rows}
    samples: list[AcquiredDetectorSample] = []
    for line in (root / "annotations.jsonl").read_bytes().splitlines():
        if not line.strip():
            continue
        sample = DetectorSample.model_validate_json(line)
        row = attribution.get(sample.sample_id)
        if row is None or any(field not in row for field in _ATTRIBUTION_FIELDS):
            raise DetectorDatasetError("warehouse base-source attribution is incomplete")
        image_path = _safe_child(root, sample.image_path)
        samples.append(
            AcquiredDetectorSample(
                sample=sample,
                image_bytes=image_path.read_bytes(),
                attribution={field: str(row[field]) for field in _ATTRIBUTION_FIELDS},
            )
        )
    if len(samples) != int(manifest["sampleCount"]):
        raise DetectorDatasetError("warehouse base-source sample count changed")
    return samples, manifest_sha256


def _find_blue_bbox(
    image: NDArray[np.uint8],
) -> tuple[tuple[int, int, int, int], float] | None:
    blue = _blue_mask(image)
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height
    choices: list[tuple[float, int, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if width < 35 or height < 25 or area >= image_area * 0.85:
            continue
        perimeter = max(1.0, 2.0 * (width + height))
        contour_coverage = min(1.0, cv2.arcLength(contour, True) / perimeter)
        rectangularity = abs(cv2.contourArea(contour)) / area
        if contour_coverage < 0.28 or rectangularity < 0.45:
            continue
        choices.append((contour_coverage * rectangularity, area, (x, y, width, height)))
    if not choices:
        return None
    score, _, inner = max(choices, key=lambda item: (item[0], item[1]))
    inner_x, inner_y, inner_width, inner_height = inner
    nearby_outer = [
        choice
        for choice in choices
        if choice[2][0] <= inner_x
        and choice[2][1] <= inner_y
        and choice[2][0] + choice[2][2] >= inner_x + inner_width
        and choice[2][1] + choice[2][3] >= inner_y + inner_height
        and choice[2][2] <= inner_width + _MAX_NESTED_BOX_DELTA
        and choice[2][3] <= inner_height + _MAX_NESTED_BOX_DELTA
    ]
    outer_score, _, bbox = max(nearby_outer, key=lambda item: item[1])
    return bbox, min(score, outer_score)


def _blue_mask(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    blue, green, red = cv2.split(image)
    selected = (
        (blue >= 155)
        & (green <= 75)
        & (red <= 75)
        & (blue.astype(np.int16) >= green.astype(np.int16) + 90)
        & (blue.astype(np.int16) >= red.astype(np.int16) + 90)
    )
    return selected.astype(np.uint8) * 255


def _remove_blue_overlay(
    image: NDArray[np.uint8], bbox: tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    x, y, width, height = bbox
    broad_perimeter = np.zeros(image.shape[:2], dtype=np.uint8)
    endpoint = (
        min(image.shape[1] - 1, x + width - 1),
        min(image.shape[0] - 1, y + height - 1),
    )
    # JPEG antialiasing can turn one burned-in rectangle into nested contours.
    # ``_find_blue_bbox`` deliberately returns the enclosing contour, so search
    # the full permitted nesting distance for strict-blue pixels before
    # dilating them to include their compressed fringe.
    broad_thickness = max(
        _MAX_NESTED_BOX_DELTA + 5,
        round(min(image.shape[:2]) * 0.008),
    )
    cv2.rectangle(
        broad_perimeter,
        (x, y),
        endpoint,
        255,
        broad_thickness,
    )
    blue_fringe = cv2.bitwise_and(_blue_mask(image), broad_perimeter)
    blue_fringe = cv2.dilate(
        blue_fringe,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    )
    antialiased_line = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(
        antialiased_line,
        (x, y),
        endpoint,
        255,
        max(3, round(min(image.shape[:2]) * 0.004)),
    )
    mask = cv2.bitwise_or(blue_fringe, antialiased_line)
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA) if np.any(mask) else image.copy()


def _target_crop(image: NDArray[np.uint8], bbox: tuple[int, int, int, int]) -> NDArray[np.uint8]:
    x, y, width, height = bbox
    padding = max(2, round(min(width, height) * 0.01))
    x1 = max(0, x + padding)
    y1 = max(0, y + padding)
    x2 = min(image.shape[1], x + width - padding)
    y2 = min(image.shape[0], y + height - padding)
    return image[y1:y2, x1:x2]


def _phash(normalized_gray: NDArray[np.uint8]) -> int:
    gray = cv2.resize(normalized_gray, (32, 32), interpolation=cv2.INTER_AREA)
    transformed = cv2.dct(gray.astype(np.float32))[:8, :8]
    bits = transformed > np.median(transformed)
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")


def _dhash(normalized_gray: NDArray[np.uint8]) -> int:
    resized = cv2.resize(normalized_gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")


def _near_duplicate_metrics(
    left: _Candidate,
    right: _Candidate,
    options: WarehouseVehicleImportOptions,
) -> dict[str, int | float] | None:
    if (left.phash ^ right.phash).bit_count() > options.maximum_phash_distance:
        return None
    if (left.dhash ^ right.dhash).bit_count() > options.maximum_dhash_distance:
        return None
    metrics = _thumbnail_metrics(left, right)
    accepted = bool(
        metrics["thumbnailMae"] <= options.maximum_thumbnail_mae
        and metrics["thumbnailCorrelation"] >= options.minimum_thumbnail_correlation
        and metrics["edgeDice"] >= options.minimum_edge_dice
    )
    return metrics if accepted else None


def _thumbnail_metrics(left: _Candidate, right: _Candidate) -> dict[str, int | float]:
    left_pixels = left.normalized_gray.astype(np.float32)
    right_pixels = right.normalized_gray.astype(np.float32)
    mae = float(np.mean(np.abs(left_pixels - right_pixels)))
    left_gray = left_pixels.ravel()
    right_gray = right_pixels.ravel()
    correlation = (
        float(np.corrcoef(left_gray, right_gray)[0, 1])
        if float(np.std(left_gray)) > 0 and float(np.std(right_gray)) > 0
        else 0.0
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    left_edges = cv2.dilate(left.normalized_edges, kernel)
    right_edges = cv2.dilate(right.normalized_edges, kernel)
    edge_intersection = int(np.count_nonzero((left_edges > 0) & (right_edges > 0)))
    edge_total = int(np.count_nonzero(left_edges) + np.count_nonzero(right_edges))
    edge_dice = 2 * edge_intersection / max(1, edge_total)
    return {
        "phashDistance": (left.phash ^ right.phash).bit_count(),
        "dhashDistance": (left.dhash ^ right.dhash).bit_count(),
        "thumbnailMae": round(mae, 6),
        "thumbnailCorrelation": round(correlation, 6),
        "edgeDice": round(edge_dice, 6),
    }


def _quality_key(candidate: _Candidate) -> tuple[float, float, float, str]:
    confidence = candidate.prediction.confidence if candidate.prediction is not None else 0.0
    return (
        -confidence,
        -candidate.sharpness,
        abs(candidate.brightness - 110.0),
        candidate.raw_sha256,
    )


def _direct_match_key(metrics: dict[str, int | float]) -> tuple[float, float, float, int]:
    return (
        -float(metrics["thumbnailCorrelation"]),
        -float(metrics["edgeDice"]),
        float(metrics["thumbnailMae"]),
        int(metrics["phashDistance"]) + int(metrics["dhashDistance"]),
    )


def _exact_duplicate_records(
    names_by_digest: dict[str, tuple[_ArchiveMember, ...]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for digest, members in sorted(names_by_digest.items()):
        if len(members) < 2:
            continue
        for duplicate in members[1:]:
            records.append(
                {
                    "reason": "EXACT_SHA256_DUPLICATE",
                    "rawSha256": digest,
                    "keptArchiveMember": members[0].name,
                    "excludedArchiveMember": duplicate.name,
                }
            )
    return records


def _parse_archive_member(name: str, size: int) -> _ArchiveMember:
    match = _ARCHIVE_IMAGE.fullmatch(name)
    if match is None:
        raise DetectorDatasetError("warehouse archive contains an unexpected regular file")
    return _ArchiveMember(
        name=name,
        view=match.group("view"),
        group_id=match.group("group"),
        timestamp_ms=int(match.group("timestamp")),
        size=size,
    )


def _validate_tar_path(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\x00" in name:
        raise DetectorDatasetError("warehouse archive member path is unsafe")


def _safe_child(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise DetectorDatasetError("warehouse source image path is unsafe")
    return path


def _existing_result(directory: Path, source_id: str) -> WarehouseVehicleSourceResult:
    _, manifest_sha256 = verify_bootstrap_source(directory)
    try:
        report = json.loads((directory / "INGESTION_REPORT.json").read_bytes())
        statistics = report["statistics"]
        archive_sha256 = str(report["archive"]["sha256"])
        base_count = int(report["baseSource"]["sampleCount"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError("warehouse ingestion report is invalid") from exc
    appended = int(statistics["appendedWarehouseSamples"])
    return WarehouseVehicleSourceResult(
        source_id=source_id,
        directory=directory,
        manifest_sha256=manifest_sha256,
        archive_sha256=archive_sha256,
        base_sample_count=base_count,
        appended_sample_count=appended,
        combined_sample_count=int(statistics["combinedSamples"]),
        exact_duplicate_files_excluded=int(statistics["exactDuplicateFilesExcluded"]),
        near_duplicate_images_excluded=int(
            statistics["perceptualNearDuplicateImagesExcluded"]
            + statistics["postCleanExactDuplicateImagesExcluded"]
        ),
        review_queue_count=int(statistics["pendingReviewUniqueImages"]),
        reject_count=int(statistics["rejectedUniqueImages"]),
        reused=True,
    )


def _identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > 128
        or any(character in normalized for character in "/\\\x00")
    ):
        raise ValueError(f"{label} is invalid")
    return normalized


def _json_lines(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(record, pretty=False) for record in records)


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
