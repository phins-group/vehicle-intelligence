"""Extract reviewable vehicle and plate detector samples from local videos.

The extractor deliberately writes model suggestions rather than canonical ground
truth.  A human review step must promote ``annotations.auto.jsonl`` before the
normal detector dataset builder is allowed to consume the images.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.application.ports import PlateDetector, VehicleDetector
from vehicle_intelligence.domain import BoundingBox, Detection, PlateDetection
from vehicle_intelligence.exceptions import DetectorDatasetError

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4"})


@dataclass(frozen=True, slots=True)
class VideoExtractionOptions:
    input_directory: Path
    output_directory: Path
    sample_interval_seconds: float = 1.0
    detector_frame_max_edge: int = 1920
    plate_context_max_edge: int = 1920
    vehicle_crop_max_edge: int = 1280
    jpeg_quality: int = 92
    batch_size: int = 8
    maximum_vehicles_per_frame: int = 24
    maximum_plate_contexts_per_frame: int = 12
    minimum_vehicle_width: int = 40
    minimum_vehicle_height: int = 40
    minimum_plate_width: int = 12
    minimum_plate_height: int = 6

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0:
            raise ValueError("sample interval must be a positive finite number")
        if not 320 <= self.detector_frame_max_edge <= 8192:
            raise ValueError("detector frame maximum edge must be in [320, 8192]")
        if not 320 <= self.plate_context_max_edge <= 8192:
            raise ValueError("plate context maximum edge must be in [320, 8192]")
        if not 128 <= self.vehicle_crop_max_edge <= 8192:
            raise ValueError("vehicle crop maximum edge must be in [128, 8192]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        if not 1 <= self.batch_size <= 64:
            raise ValueError("batch size must be in [1, 64]")
        if not 1 <= self.maximum_vehicles_per_frame <= 256:
            raise ValueError("maximum vehicles per frame must be in [1, 256]")
        if not 1 <= self.maximum_plate_contexts_per_frame <= 256:
            raise ValueError("maximum plate contexts per frame must be in [1, 256]")
        if min(
            self.minimum_vehicle_width,
            self.minimum_vehicle_height,
            self.minimum_plate_width,
            self.minimum_plate_height,
        ) <= 0:
            raise ValueError("minimum crop dimensions must be positive")


@dataclass(frozen=True, slots=True)
class VideoExtractionResult:
    output_directory: Path
    manifest_path: Path
    videos_discovered: int
    videos_processed: int
    sampled_frames: int
    vehicle_training_images: int
    vehicle_crop_images: int
    plate_training_images: int
    plate_crop_images: int
    vehicle_class_counts: dict[str, int]
    failed_videos: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _SampledFrame:
    image: NDArray[np.uint8]
    frame_index: int
    offset_seconds: float


@dataclass(slots=True)
class _Counters:
    sampled_frames: int = 0
    vehicle_training_images: int = 0
    vehicle_crop_images: int = 0
    plate_training_images: int = 0
    plate_crop_images: int = 0
    vehicle_class_counts: Counter[str] = field(default_factory=Counter)

    def add(self, other: _Counters) -> None:
        self.sampled_frames += other.sampled_frames
        self.vehicle_training_images += other.vehicle_training_images
        self.vehicle_crop_images += other.vehicle_crop_images
        self.plate_training_images += other.plate_training_images
        self.plate_crop_images += other.plate_crop_images
        self.vehicle_class_counts.update(other.vehicle_class_counts)


ProgressCallback = Callable[[str, dict[str, Any]], None]


class VideoTrainingImageExtractor:
    """Create a provenance-preserving, human-review staging set from videos."""

    def __init__(
        self,
        vehicle_detector: VehicleDetector,
        plate_detector: PlateDetector,
        options: VideoExtractionOptions,
        *,
        owner_namespace: str,
        founder_id: str,
        model_evidence: dict[str, dict[str, Any]] | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self._vehicle_detector = vehicle_detector
        self._plate_detector = plate_detector
        self._options = options
        self._owner_namespace = _required_identifier(owner_namespace, "owner namespace")
        self._founder_id = _required_identifier(founder_id, "founder id")
        self._model_evidence = model_evidence or {}
        self._progress = progress

    def extract(self) -> VideoExtractionResult:
        source_root = self._options.input_directory.expanduser().resolve()
        output_root = self._options.output_directory.expanduser().resolve()
        if not source_root.is_dir():
            raise DetectorDatasetError(f"video input directory is missing: {source_root}")
        if output_root == source_root or output_root.is_relative_to(source_root):
            raise DetectorDatasetError(
                "video extraction output cannot be inside the input directory"
            )
        videos = _discover_videos(source_root)
        if not videos:
            raise DetectorDatasetError(f"no supported video files found below {source_root}")

        paths = _prepare_output(output_root)
        counters = _Counters()
        sources: list[dict[str, Any]] = []
        failures: list[str] = []
        created_at = datetime.now(UTC)
        self._emit("extraction_started", videos=len(videos), output=str(output_root))

        try:
            with (
                paths["vehicle_annotations"].open("x", encoding="utf-8") as vehicle_stream,
                paths["plate_annotations"].open("x", encoding="utf-8") as plate_stream,
            ):
                for index, video in enumerate(videos, start=1):
                    relative_name = str(video.relative_to(source_root).as_posix())
                    self._emit(
                        "video_started",
                        index=index,
                        total=len(videos),
                        video=relative_name,
                    )
                    try:
                        digest = _sha256_file(video)
                        source_id = _source_id(relative_name, digest)
                        source_counters, metadata = self._extract_video(
                            video,
                            relative_name,
                            digest,
                            source_id,
                            paths,
                            vehicle_stream,
                            plate_stream,
                        )
                    except (DetectorDatasetError, OSError, cv2.error) as exc:
                        failures.append(relative_name)
                        sources.append(
                            {
                                "path": relative_name,
                                "status": "FAILED",
                                "error": type(exc).__name__,
                            }
                        )
                        logger.exception(
                            "video_training_extraction_failed",
                            extra={"video": relative_name},
                        )
                        self._emit("video_failed", video=relative_name, error=str(exc))
                        continue
                    counters.add(source_counters)
                    sources.append(metadata)
                    self._emit(
                        "video_completed",
                        video=relative_name,
                        sampledFrames=source_counters.sampled_frames,
                        vehicleImages=source_counters.vehicle_training_images,
                        plateImages=source_counters.plate_training_images,
                    )
        except FileExistsError as exc:
            raise DetectorDatasetError(
                "video extraction output already contains generated files"
            ) from exc

        manifest = self._manifest(
            source_root=source_root,
            created_at=created_at,
            sources=sources,
            counters=counters,
            failures=failures,
        )
        manifest_path = output_root / "manifest.json"
        _write_new(manifest_path, _json_bytes(manifest, pretty=True))
        _write_new(output_root / "README.md", _readme(manifest).encode())
        self._emit(
            "extraction_completed",
            videosProcessed=len(videos) - len(failures),
            vehicleImages=counters.vehicle_training_images,
            plateImages=counters.plate_training_images,
            failures=len(failures),
        )
        return VideoExtractionResult(
            output_directory=output_root,
            manifest_path=manifest_path,
            videos_discovered=len(videos),
            videos_processed=len(videos) - len(failures),
            sampled_frames=counters.sampled_frames,
            vehicle_training_images=counters.vehicle_training_images,
            vehicle_crop_images=counters.vehicle_crop_images,
            plate_training_images=counters.plate_training_images,
            plate_crop_images=counters.plate_crop_images,
            vehicle_class_counts=dict(sorted(counters.vehicle_class_counts.items())),
            failed_videos=tuple(failures),
        )

    def _extract_video(
        self,
        video: Path,
        relative_name: str,
        digest: str,
        source_id: str,
        paths: dict[str, Path],
        vehicle_stream: Any,
        plate_stream: Any,
    ) -> tuple[_Counters, dict[str, Any]]:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            raise DetectorDatasetError(f"cannot open video: {relative_name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            capture.release()
            raise DetectorDatasetError(f"video metadata is invalid: {relative_name}")
        duration = frame_count / fps if frame_count > 0 else None
        file_time = datetime.fromtimestamp(video.stat().st_mtime, tz=UTC)
        counters = _Counters()
        try:
            batch: list[tuple[_SampledFrame, NDArray[np.uint8], float]] = []
            for sampled in _sample_frames(
                capture,
                fps=fps,
                frame_count=frame_count,
                interval=self._options.sample_interval_seconds,
            ):
                inference_image, scale = _resize_max_edge(
                    sampled.image,
                    self._options.detector_frame_max_edge,
                )
                batch.append((sampled, inference_image, scale))
                if len(batch) >= self._options.batch_size:
                    self._process_batch(
                        batch,
                        source_id,
                        relative_name,
                        digest,
                        file_time,
                        paths,
                        vehicle_stream,
                        plate_stream,
                        counters,
                    )
                    batch.clear()
            if batch:
                self._process_batch(
                    batch,
                    source_id,
                    relative_name,
                    digest,
                    file_time,
                    paths,
                    vehicle_stream,
                    plate_stream,
                    counters,
                )
        finally:
            capture.release()

        return counters, {
            "sourceId": source_id,
            "path": relative_name,
            "sha256": digest,
            "size": video.stat().st_size,
            "status": "PROCESSED",
            "licenseReviewStatus": "REVIEW_REQUIRED",
            "releaseEligible": False,
            "distributionEligible": False,
            "video": {
                "width": width,
                "height": height,
                "fps": round(fps, 6),
                "frameCount": frame_count if frame_count > 0 else None,
                "durationSeconds": round(duration, 6) if duration is not None else None,
            },
            "statistics": _counter_payload(counters),
        }

    def _process_batch(
        self,
        batch: Sequence[tuple[_SampledFrame, NDArray[np.uint8], float]],
        source_id: str,
        source_name: str,
        source_digest: str,
        file_time: datetime,
        paths: dict[str, Path],
        vehicle_stream: Any,
        plate_stream: Any,
        counters: _Counters,
    ) -> None:
        inference_images = [item[1] for item in batch]
        detect_batch = getattr(self._vehicle_detector, "detect_batch", None)
        if callable(detect_batch):
            detections_by_frame = list(detect_batch(inference_images))
        else:
            detections_by_frame = [
                self._vehicle_detector.detect(image) for image in inference_images
            ]
        if len(detections_by_frame) != len(batch):
            raise DetectorDatasetError("vehicle detector batch result count is invalid")
        for (sampled, inference_image, scale), detections in zip(
            batch, detections_by_frame, strict=True
        ):
            counters.sampled_frames += 1
            self._process_frame(
                sampled,
                inference_image,
                scale,
                detections,
                source_id,
                source_name,
                source_digest,
                file_time,
                paths,
                vehicle_stream,
                plate_stream,
                counters,
            )

    def _process_frame(
        self,
        sampled: _SampledFrame,
        inference_image: NDArray[np.uint8],
        inference_scale: float,
        detections: Sequence[Detection],
        source_id: str,
        source_name: str,
        source_digest: str,
        file_time: datetime,
        paths: dict[str, Path],
        vehicle_stream: Any,
        plate_stream: Any,
        counters: _Counters,
    ) -> None:
        valid = [
            detection
            for detection in detections
            if detection.bbox.width >= self._options.minimum_vehicle_width
            and detection.bbox.height >= self._options.minimum_vehicle_height
        ]
        valid.sort(key=lambda item: (item.confidence, item.bbox.area), reverse=True)
        valid = valid[: self._options.maximum_vehicles_per_frame]
        if not valid:
            return

        frame_token = f"{source_id}-f{sampled.frame_index:09d}"
        captured_at = file_time + timedelta(seconds=sampled.offset_seconds)
        frame_name = f"phins-vehicle-{frame_token}.jpg"
        _write_jpeg_new(paths["vehicle_images"] / frame_name, inference_image, self._options)
        annotations: list[dict[str, Any]] = []
        plate_contexts: list[tuple[int, Detection, NDArray[np.uint8]]] = []
        original_height, original_width = sampled.image.shape[:2]

        for detection_index, detection in enumerate(valid):
            original_box = _box_to_original(
                detection.bbox,
                inference_scale,
                original_width,
                original_height,
            )
            original_box = _expand_box(original_box, 0.025, original_width, original_height)
            vehicle_crop = _crop(sampled.image, original_box)
            if vehicle_crop is None:
                continue
            crop_image, _ = _resize_max_edge(
                vehicle_crop,
                self._options.vehicle_crop_max_edge,
            )
            crop_name = (
                f"phins-vehicle-{frame_token}-d{detection_index:02d}-{detection.class_name}.jpg"
            )
            crop_relative = PurePosixPath("crops", detection.class_name, crop_name)
            _write_jpeg_new(
                paths["vehicle_root"].joinpath(*crop_relative.parts),
                crop_image,
                self._options,
            )
            counters.vehicle_crop_images += 1
            counters.vehicle_class_counts[detection.class_name] += 1
            annotations.append(
                {
                    "className": detection.class_name,
                    "bbox": _bbox_json(detection.bbox),
                    "attributes": {
                        "annotationSource": "MODEL_SUGGESTION",
                        "reviewStatus": "PENDING_REVIEW",
                        "confidence": round(detection.confidence, 6),
                        "rawClassId": detection.class_id,
                        "modelName": detection.model.name,
                        "modelVersion": detection.model.version,
                        "modelHash": detection.model.hash,
                        "cropPath": str(crop_relative),
                    },
                }
            )
            plate_contexts.append((detection_index, detection, vehicle_crop))

        if not annotations:
            return
        counters.vehicle_training_images += 1
        vehicle_record = _sample_record(
            sample_id=f"phins-vehicle-{frame_token}",
            image_path=str(PurePosixPath("images", frame_name)),
            source_id=source_id,
            source_name=source_name,
            source_digest=source_digest,
            captured_at=captured_at,
            sampled=sampled,
            image=inference_image,
            owner_namespace=self._owner_namespace,
            founder_id=self._founder_id,
            annotations=annotations,
        )
        _append_json_line(vehicle_stream, vehicle_record)

        plate_contexts.sort(
            key=lambda item: (item[1].bbox.area, item[1].confidence), reverse=True
        )
        prepared_contexts: list[tuple[int, Detection, NDArray[np.uint8], float]] = []
        for detection_index, vehicle_detection, vehicle_crop in plate_contexts[
            : self._options.maximum_plate_contexts_per_frame
        ]:
            context_image, context_scale = _resize_max_edge(
                vehicle_crop,
                self._options.plate_context_max_edge,
            )
            prepared_contexts.append(
                (detection_index, vehicle_detection, context_image, context_scale)
            )
        detect_plate_batch = getattr(self._plate_detector, "detect_batch", None)
        context_images = [item[2] for item in prepared_contexts]
        if callable(detect_plate_batch):
            plate_detection_sets = list(detect_plate_batch(context_images))
        else:
            plate_detection_sets = [
                self._plate_detector.detect(image) for image in context_images
            ]
        if len(plate_detection_sets) != len(prepared_contexts):
            raise DetectorDatasetError("plate detector batch result count is invalid")
        for (
            detection_index,
            vehicle_detection,
            context_image,
            context_scale,
        ), plate_detections in zip(
            prepared_contexts,
            plate_detection_sets,
            strict=True,
        ):
            valid_plates = [
                item
                for item in plate_detections
                if item.bbox.width >= self._options.minimum_plate_width
                and item.bbox.height >= self._options.minimum_plate_height
            ]
            valid_plates.sort(key=lambda item: item.confidence, reverse=True)
            if not valid_plates:
                continue
            context_token = f"{frame_token}-v{detection_index:02d}"
            context_name = f"phins-plate-{context_token}.jpg"
            _write_jpeg_new(
                paths["plate_images"] / context_name,
                context_image,
                self._options,
            )
            plate_annotations: list[dict[str, Any]] = []
            context_height, context_width = context_image.shape[:2]
            for plate_index, plate in enumerate(valid_plates):
                crop_box = _expand_box(
                    plate.bbox,
                    0.08,
                    context_width,
                    context_height,
                )
                plate_crop = _crop(context_image, crop_box)
                if plate_crop is None:
                    continue
                crop_name = f"phins-plate-{context_token}-p{plate_index:02d}.jpg"
                crop_relative = PurePosixPath("crops", crop_name)
                _write_jpeg_new(
                    paths["plate_root"].joinpath(*crop_relative.parts),
                    plate_crop,
                    self._options,
                )
                counters.plate_crop_images += 1
                polygon = _scaled_polygon(plate, 1.0, context_width, context_height)
                annotation: dict[str, Any] = {
                    "className": "license_plate",
                    "bbox": _bbox_json(plate.bbox),
                    "attributes": {
                        "annotationSource": "MODEL_SUGGESTION",
                        "reviewStatus": "PENDING_REVIEW",
                        "confidence": round(plate.confidence, 6),
                        "modelName": plate.model.name,
                        "modelVersion": plate.model.version,
                        "modelHash": plate.model.hash,
                        "vehicleClassSuggestion": vehicle_detection.class_name,
                        "vehicleConfidence": round(vehicle_detection.confidence, 6),
                        "layoutSuggestion": _plate_layout(plate.bbox),
                        "cropPath": str(crop_relative),
                    },
                }
                if polygon:
                    annotation["polygon"] = polygon
                plate_annotations.append(annotation)
            if not plate_annotations:
                continue
            counters.plate_training_images += 1
            plate_record = _sample_record(
                sample_id=f"phins-plate-{context_token}",
                image_path=str(PurePosixPath("images", context_name)),
                source_id=source_id,
                source_name=source_name,
                source_digest=source_digest,
                captured_at=captured_at,
                sampled=sampled,
                image=context_image,
                owner_namespace=self._owner_namespace,
                founder_id=self._founder_id,
                annotations=plate_annotations,
                extra_attributes={
                    "vehicleClassSuggestion": vehicle_detection.class_name,
                    "vehicleDetectionConfidence": round(vehicle_detection.confidence, 6),
                    "sourceContextScale": round(context_scale, 8),
                },
            )
            _append_json_line(plate_stream, plate_record)

    def _manifest(
        self,
        *,
        source_root: Path,
        created_at: datetime,
        sources: list[dict[str, Any]],
        counters: _Counters,
        failures: list[str],
    ) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "type": "VIDEO_DETECTOR_SAMPLE_EXTRACTION",
            "ownerNamespace": self._owner_namespace,
            "founderId": self._founder_id,
            "createdAt": _timestamp(created_at),
            "status": "COMPLETE" if not failures else "COMPLETE_WITH_ERRORS",
            "sourceDirectoryName": source_root.name,
            "licenseReviewStatus": "REVIEW_REQUIRED",
            "acceptanceEligible": False,
            "releaseEligible": False,
            "distributionEligible": False,
            "reviewStatus": "PENDING_REVIEW",
            "annotationPolicy": (
                "MODEL_SUGGESTIONS_ONLY_REVIEW_BEFORE_PROMOTION_TO_ANNOTATIONS_JSONL"
            ),
            "models": self._model_evidence,
            "configuration": {
                "sampleIntervalSeconds": self._options.sample_interval_seconds,
                "detectorFrameMaxEdge": self._options.detector_frame_max_edge,
                "plateContextMaxEdge": self._options.plate_context_max_edge,
                "vehicleCropMaxEdge": self._options.vehicle_crop_max_edge,
                "jpegQuality": self._options.jpeg_quality,
                "batchSize": self._options.batch_size,
                "maximumVehiclesPerFrame": self._options.maximum_vehicles_per_frame,
                "maximumPlateContextsPerFrame": (
                    self._options.maximum_plate_contexts_per_frame
                ),
            },
            "statistics": _counter_payload(counters),
            "failedVideos": failures,
            "sources": sources,
        }

    def _emit(self, event: str, **payload: Any) -> None:
        if self._progress is not None:
            self._progress(event, payload)


def _prepare_output(output_root: Path) -> dict[str, Path]:
    if output_root.exists() and output_root.is_symlink():
        raise DetectorDatasetError("video extraction output cannot be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    existing_files = [
        path for path in output_root.rglob("*") if path.is_file() or path.is_symlink()
    ]
    if existing_files:
        raise DetectorDatasetError(
            f"video extraction output is not empty: {output_root}"
        )
    vehicle_root = output_root / "vehicle"
    plate_root = output_root / "plate"
    vehicle_images = vehicle_root / "images"
    plate_images = plate_root / "images"
    for directory in (
        vehicle_images,
        plate_images,
        plate_root / "crops",
        *(
            vehicle_root / "crops" / class_name
            for class_name in ("car", "motorcycle", "bus", "truck")
        ),
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "vehicle_root": vehicle_root,
        "plate_root": plate_root,
        "vehicle_images": vehicle_images,
        "plate_images": plate_images,
        "vehicle_annotations": vehicle_root / "annotations.auto.jsonl",
        "plate_annotations": plate_root / "annotations.auto.jsonl",
    }


def _discover_videos(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in _VIDEO_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(root)).casefold(),
    )


def _sample_frames(
    capture: cv2.VideoCapture,
    *,
    fps: float,
    frame_count: int,
    interval: float,
) -> Iterator[_SampledFrame]:
    duration = frame_count / fps if frame_count > 0 else None
    offset = 0.0
    previous_frame_index = -1
    while duration is None or offset < duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
        ok, image = capture.read()
        if not ok or image is None or image.size == 0:
            break
        frame_index = max(int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1, 0)
        if frame_index <= previous_frame_index:
            if duration is None:
                break
            offset += interval
            continue
        previous_frame_index = frame_index
        yield _SampledFrame(image=image, frame_index=frame_index, offset_seconds=offset)
        offset += interval


def _resize_max_edge(
    image: NDArray[np.uint8],
    maximum_edge: int,
) -> tuple[NDArray[np.uint8], float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= maximum_edge:
        return image, 1.0
    scale = maximum_edge / longest
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _box_to_original(
    bbox: BoundingBox,
    scale: float,
    width: int,
    height: int,
) -> BoundingBox:
    if scale <= 0:
        raise DetectorDatasetError("detector frame scale is invalid")
    x1 = min(max(math.floor(bbox.x1 / scale), 0), width - 1)
    y1 = min(max(math.floor(bbox.y1 / scale), 0), height - 1)
    x2 = min(max(math.ceil(bbox.x2 / scale), x1 + 1), width)
    y2 = min(max(math.ceil(bbox.y2 / scale), y1 + 1), height)
    return BoundingBox(x1, y1, x2, y2)


def _expand_box(bbox: BoundingBox, ratio: float, width: int, height: int) -> BoundingBox:
    dx = round(bbox.width * ratio)
    dy = round(bbox.height * ratio)
    x1 = max(bbox.x1 - dx, 0)
    y1 = max(bbox.y1 - dy, 0)
    x2 = min(bbox.x2 + dx, width)
    y2 = min(bbox.y2 + dy, height)
    return BoundingBox(x1, y1, x2, y2)


def _crop(image: NDArray[np.uint8], bbox: BoundingBox) -> NDArray[np.uint8] | None:
    crop = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop)


def _sample_record(
    *,
    sample_id: str,
    image_path: str,
    source_id: str,
    source_name: str,
    source_digest: str,
    captured_at: datetime,
    sampled: _SampledFrame,
    image: NDArray[np.uint8],
    owner_namespace: str,
    founder_id: str,
    annotations: list[dict[str, Any]],
    extra_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    attributes: dict[str, Any] = {
        "ownerNamespace": owner_namespace,
        "founderId": founder_id,
        "annotationSource": "MODEL_SUGGESTION",
        "reviewStatus": "PENDING_REVIEW",
        "acceptanceEligible": False,
        "releaseEligible": False,
        "licenseReviewStatus": "REVIEW_REQUIRED",
        "sourceVideo": source_name,
        "sourceVideoSha256": source_digest,
        "sourceFrameIndex": sampled.frame_index,
        "sourceOffsetSeconds": round(sampled.offset_seconds, 6),
        "capturedAtBasis": "FILE_MTIME_PLUS_VIDEO_OFFSET",
        "lighting": "NIGHT" if brightness < 70 else "DAY",
        "imageBrightness": round(brightness, 4),
        "imageContrast": round(contrast, 4),
        "imageSharpness": round(sharpness, 4),
    }
    if extra_attributes:
        attributes.update(extra_attributes)
    return {
        "sampleId": sample_id,
        "imagePath": image_path,
        "groupId": f"phins-group:video:{source_id}",
        "cameraId": f"video-{source_id}",
        "capturedAt": _timestamp(captured_at),
        "split": None,
        "attributes": attributes,
        "annotations": annotations,
    }


def _bbox_json(bbox: BoundingBox) -> dict[str, int]:
    return {"x": bbox.x1, "y": bbox.y1, "width": bbox.width, "height": bbox.height}


def _scaled_polygon(
    detection: PlateDetection,
    scale: float,
    width: int,
    height: int,
) -> list[dict[str, float]]:
    if detection.corners is None:
        return []
    return [
        {
            "x": min(max(round(point.x * scale, 4), 0.0), float(width)),
            "y": min(max(round(point.y * scale, 4), 0.0), float(height)),
        }
        for point in detection.corners
    ]


def _plate_layout(bbox: BoundingBox) -> str:
    return "SINGLE_LINE" if bbox.width / bbox.height >= 2.5 else "TWO_LINE"


def _write_jpeg_new(
    path: Path,
    image: NDArray[np.uint8],
    options: VideoExtractionOptions,
) -> None:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, options.jpeg_quality],
    )
    if not ok:
        raise DetectorDatasetError(f"cannot encode extracted image: {path.name}")
    _write_new(path, encoded.tobytes())


def _append_json_line(stream: Any, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")


def _counter_payload(counters: _Counters) -> dict[str, Any]:
    return {
        "sampledFrames": counters.sampled_frames,
        "vehicleTrainingImages": counters.vehicle_training_images,
        "vehicleCropImages": counters.vehicle_crop_images,
        "plateTrainingImages": counters.plate_training_images,
        "plateCropImages": counters.plate_crop_images,
        "vehicleClassCounts": dict(sorted(counters.vehicle_class_counts.items())),
    }


def _source_id(relative_name: str, digest: str) -> str:
    name_hash = hashlib.sha256(relative_name.encode()).hexdigest()[:8]
    return f"{name_hash}-{digest[:12]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    separators = None if pretty else (",", ":")
    return (
        json.dumps(
            value,
            indent=2 if pretty else None,
            sort_keys=True,
            ensure_ascii=False,
            separators=separators,
        )
        + "\n"
    ).encode()


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _required_identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128 or any(char in normalized for char in "/\\\0"):
        raise ValueError(f"{label} is invalid")
    return normalized


def _readme(manifest: dict[str, Any]) -> str:
    statistics = manifest["statistics"]
    return f"""# PHINS video extraction staging set

This directory contains detector **model suggestions**, not reviewed ground truth.

- Owner namespace: `{manifest['ownerNamespace']}`
- Founder/steward: `{manifest['founderId']}`
- Source videos: {len(manifest['sources'])}
- Vehicle training images: {statistics['vehicleTrainingImages']}
- Vehicle crops: {statistics['vehicleCropImages']}
- Plate training images: {statistics['plateTrainingImages']}
- Plate crops: {statistics['plateCropImages']}

## Layout

```text
vehicle/images/                 full traffic frames for vehicle detector review
vehicle/crops/<class>/          vehicle-type crop previews
vehicle/annotations.auto.jsonl  suggested vehicle boxes
plate/images/                   vehicle contexts for plate detector review
plate/crops/                    plate crop previews
plate/annotations.auto.jsonl    suggested license-plate boxes
manifest.json                   source hashes, policy, and extraction statistics
```

Every annotation has `reviewStatus=PENDING_REVIEW`. Review/correct the boxes and
classes before promoting a copy to canonical `annotations.jsonl`. Do not use the
auto-suggestion files as release evidence.

The source-video license and distribution rights were not supplied to the
extractor. The entire staging set is therefore `REVIEW_REQUIRED`,
`releaseEligible=false`, and `distributionEligible=false` until provenance and
commercial-use rights are recorded.
"""
