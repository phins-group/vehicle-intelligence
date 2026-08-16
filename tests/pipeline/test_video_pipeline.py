from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.finalization import VehicleEventFinalizer
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.pipeline import VideoVehiclePipeline
from vehicle_intelligence.application.ports import ImageVariant
from vehicle_intelligence.application.quality import PlateQualityEvaluator
from vehicle_intelligence.application.selection import BestFrameSelector
from vehicle_intelligence.application.voting import PlateCandidateAggregator
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.domain import (
    BoundingBox,
    Detection,
    Direction,
    ModelMetadata,
    OCRResult,
    PlateDetection,
    Point,
    TrackedDetection,
    VideoFrame,
)
from vehicle_intelligence.infrastructure.messaging.direct import RepositoryEventPublisher
from vehicle_intelligence.infrastructure.persistence.jsonl import JsonlVehicleEventRepository
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage
from vehicle_intelligence.infrastructure.vision.opencv import OpenCVVideoSource


class SequenceVehicleDetector:
    def __init__(self) -> None:
        self._index = 0
        self._model = ModelMetadata("deterministic-vehicle", "test-1")

    def detect(self, image):
        del image
        y1 = 10 + self._index * 10
        self._index += 1
        return [Detection(BoundingBox(20, y1, 180, y1 + 100), 0.96, 2, "car", self._model)]


class SingleFrameSource:
    source_id = "batch-plate-source"
    source_fps = 5.0
    decoded_frames = 1

    def __init__(self, camera_id: str) -> None:
        self._camera_id = camera_id
        self.closed = False

    def frames(self):
        image = np.full((180, 200, 3), 10, dtype=np.uint8)
        image[:, 100:] = 200
        yield VideoFrame(
            camera_id=self._camera_id,
            frame_id=0,
            timestamp=datetime(2026, 8, 8, tzinfo=UTC),
            image=image,
        )

    def close(self):
        self.closed = True


class MultiVehicleDetector:
    def __init__(self) -> None:
        self._model = ModelMetadata("multi-vehicle", "test-1")

    def detect(self, image):
        del image
        return [
            Detection(BoundingBox(10, 20, 90, 140), 0.96, 2, "car", self._model),
            Detection(BoundingBox(110, 30, 190, 150), 0.94, 7, "truck", self._model),
        ]


class StableTestTracker:
    def update(self, detections, image):
        del image
        return [TrackedDetection(12, detection) for detection in detections]

    def reset(self):
        return None


class EnumeratingTracker:
    def update(self, detections, image):
        del image
        track_ids = (101, 202)
        return [
            TrackedDetection(track_id, detection)
            for track_id, detection in zip(track_ids, detections, strict=True)
        ]

    def reset(self):
        return None


class FixedPlateDetector:
    def __init__(self) -> None:
        self._model = ModelMetadata("deterministic-plate", "test-1")

    def detect(self, image):
        del image
        return [PlateDetection(BoundingBox(40, 45, 120, 75), 0.95, self._model)]


class MappingScalarPlateDetector:
    def __init__(self) -> None:
        self._model = ModelMetadata("mapping-plate", "test-1")
        self.scalar_calls = 0
        self.input_markers = []

    def detect(self, image):
        self.scalar_calls += 1
        marker = int(image[0, 0, 0])
        self.input_markers.append(marker)
        return self._detections(marker)

    def _detections(self, marker):
        if marker < 100:
            bbox = BoundingBox(5, 10, 35, 30)
            confidence = 0.91
        else:
            bbox = BoundingBox(15, 20, 55, 40)
            confidence = 0.92
        return [PlateDetection(bbox, confidence, self._model)]


class MappingBatchPlateDetector(MappingScalarPlateDetector):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def detect_batch(self, images):
        self.batch_calls += 1
        markers = [int(image[0, 0, 0]) for image in images]
        self.input_markers.extend(markers)
        return [self._detections(marker) for marker in markers]


class OriginalOnlyPreprocessor:
    def variants(self, image, quality, detection):
        del quality, detection
        return [ImageVariant("original", image)]


class ThreeVariantPreprocessor:
    def variants(self, image, quality, detection):
        del quality, detection
        originals = [image.copy() for _ in range(3)]
        for marker, variant in enumerate(originals):
            variant[0, 0, 0] = marker
        return [
            ImageVariant("invalid-high-confidence", originals[0]),
            ImageVariant("valid-high-confidence", originals[1]),
            ImageVariant("slower-unused", originals[2]),
        ]


class SequenceOCR:
    def __init__(self) -> None:
        self._index = 0
        self._values = [
            ("51H12345", 0.94),
            ("51H12345", 0.91),
            ("51H1234S", 0.73),
            ("51H12345", 0.89),
            ("51H12345", 0.92),
        ]
        self._model = ModelMetadata("deterministic-ocr", "test-1")

    def recognize(self, image):
        del image
        text, confidence = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return OCRResult(text=text, confidence=confidence, model=self._model)


class CountingVariantOCR:
    def __init__(self) -> None:
        self.calls = 0
        self._model = ModelMetadata("counting-ocr", "test-1")

    def recognize(self, image):
        self.calls += 1
        if int(image[0, 0, 0]) == 0:
            return OCRResult(text="NOT-A-PLATE", confidence=0.99, model=self._model)
        return OCRResult(text="51H12345", confidence=0.97, model=self._model)


class TestImageEncoder:
    def encode_jpeg(self, image):
        success, buffer = cv2.imencode(".jpg", image)
        assert success
        return buffer.tobytes()


class CapturingLivePreview:
    def __init__(self) -> None:
        self.frames = []

    async def report(self, image, metadata):
        del image
        self.frames.append(metadata)
        return True


class DisabledMediaSelector:
    def vehicle_score(self, *args, **kwargs):
        raise AssertionError("vehicle image scoring must be skipped when media is disabled")

    def plate_score(self, *args, **kwargs):
        raise AssertionError("plate image scoring must be skipped when media is disabled")


class CountingPlateSelector(BestFrameSelector):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.plate_calls = 0

    def plate_score(self, quality, ocr_confidence, detector_confidence):
        del quality, ocr_confidence, detector_confidence
        self.plate_calls += 1
        return min(self.plate_calls / 100, 1.0)


def make_video(path: Path, frame_count: int = 5) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (200, 180),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    for _ in range(frame_count):
        frame = rng.integers(0, 256, size=(180, 200, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


async def run_multi_vehicle_case(tmp_path, plate_detector, name):
    base = load_settings()
    camera = base.camera.model_copy(
        update={"id": f"batch-camera-{name}", "name": f"Batch Camera {name}", "fps_limit": 5.0}
    )
    storage = base.storage.model_copy(
        update={
            "output_directory": tmp_path / name,
            "snapshots": False,
            "vehicle_crops": False,
            "plate_crops": False,
        }
    )
    settings = base.model_copy(update={"camera": camera, "storage": storage})
    repository = JsonlVehicleEventRepository(storage.output_directory / "events.jsonl")
    normalizer = VietnamPlateNormalizer()
    direction = DirectionEstimator(None, Direction.ENTER, "BOTH")
    finalizer = VehicleEventFinalizer(
        camera,
        settings.events,
        storage,
        PlateCandidateAggregator(settings.voting, normalizer),
        direction,
        LocalMediaStorage(storage.output_directory),
        TestImageEncoder(),
        RepositoryEventPublisher(repository),
    )
    live_preview = CapturingLivePreview()
    pipeline = VideoVehiclePipeline(
        settings,
        SingleFrameSource(camera.id),
        MultiVehicleDetector(),
        EnumeratingTracker(),
        plate_detector,
        PlateQualityEvaluator(settings.vision.plate_quality),
        OriginalOnlyPreprocessor(),
        SequenceOCR(),
        normalizer,
        BestFrameSelector(settings.vision.snapshot_selection),
        finalizer,
        direction,
        live_preview=live_preview,
    )
    await repository.ensure_indexes()
    return await pipeline.run(), live_preview


def assert_multi_vehicle_plate_mapping(live_preview):
    assert len(live_preview.frames) == 1
    vehicles = {
        int(vehicle.track_id.rsplit(":", 1)[-1]): vehicle
        for vehicle in live_preview.frames[0].vehicles
    }
    assert set(vehicles) == {101, 202}
    assert vehicles[101].plate is not None
    assert vehicles[202].plate is not None
    assert vehicles[101].plate.bbox.as_xyxy() == (15, 30, 45, 50)
    assert vehicles[202].plate.bbox.as_xyxy() == (125, 50, 165, 70)


async def test_vehicle_tracks_use_one_plate_batch_and_preserve_result_mapping(tmp_path) -> None:
    detector = MappingBatchPlateDetector()

    result, live_preview = await run_multi_vehicle_case(tmp_path, detector, "batch")

    assert detector.batch_calls == 1
    assert detector.scalar_calls == 0
    assert detector.input_markers == [10, 200]
    assert result.stats.plate_inference_calls == 1
    assert result.stats.plate_detections == 2
    assert_multi_vehicle_plate_mapping(live_preview)


async def test_vehicle_tracks_fall_back_to_scalar_plate_detection(tmp_path) -> None:
    detector = MappingScalarPlateDetector()

    result, live_preview = await run_multi_vehicle_case(tmp_path, detector, "scalar")

    assert detector.scalar_calls == 2
    assert detector.input_markers == [10, 200]
    assert result.stats.plate_inference_calls == 2
    assert result.stats.plate_detections == 2
    assert_multi_vehicle_plate_mapping(live_preview)


async def test_video_to_one_persisted_vehicle_event(tmp_path) -> None:
    video_path = tmp_path / "sample.avi"
    make_video(video_path)
    base = load_settings()
    camera = base.camera.model_copy(
        update={
            "crossing_line": ((200.0, 75.0), (0.0, 75.0)),
            "crossing_positive_to_negative": "ENTER",
            "direction": "BOTH",
            "fps_limit": 5.0,
        }
    )
    plate_quality = base.vision.plate_quality.model_copy(update={"minimum": 0.20})
    vision = base.vision.model_copy(update={"plate_quality": plate_quality})
    storage = base.storage.model_copy(update={"output_directory": tmp_path / "output"})
    settings = base.model_copy(update={"camera": camera, "vision": vision, "storage": storage})
    source = OpenCVVideoSource(
        video_path,
        camera.id,
        camera.fps_limit,
        datetime(2026, 8, 8, tzinfo=UTC),
    )
    repository = JsonlVehicleEventRepository(storage.output_directory / "events.jsonl")
    normalizer = VietnamPlateNormalizer()
    direction = DirectionEstimator(
        (Point(200, 75), Point(0, 75)),
        Direction.ENTER,
        "BOTH",
    )
    finalizer = VehicleEventFinalizer(
        camera,
        settings.events,
        storage,
        PlateCandidateAggregator(settings.voting, normalizer),
        direction,
        LocalMediaStorage(storage.output_directory),
        TestImageEncoder(),
        RepositoryEventPublisher(repository),
    )
    live_preview = CapturingLivePreview()
    pipeline = VideoVehiclePipeline(
        settings,
        source,
        SequenceVehicleDetector(),
        StableTestTracker(),
        FixedPlateDetector(),
        PlateQualityEvaluator(plate_quality),
        OriginalOnlyPreprocessor(),
        SequenceOCR(),
        normalizer,
        BestFrameSelector(settings.vision.snapshot_selection),
        finalizer,
        direction,
        live_preview=live_preview,
    )
    await repository.ensure_indexes()

    result = await pipeline.run()

    assert len(result.events) == 1
    event = result.events[0]
    assert event.track_id.startswith(f"{camera.id}:video-")
    assert event.plate is not None
    assert event.plate.normalized == "51H-123.45"
    assert event.plate.observation_count == 3
    assert event.direction is Direction.ENTER
    assert event.event_type.value == "VEHICLE_ENTER"
    assert result.stats.finalized_tracks == 1
    assert result.stats.ocr_requests == 3
    assert len((storage.output_directory / "events.jsonl").read_text().splitlines()) == 1
    assert (storage.output_directory / event.media.snapshot_key).is_file()
    assert (storage.output_directory / event.media.vehicle_crop_key).is_file()
    assert (storage.output_directory / event.media.plate_crop_key).is_file()
    assert len(live_preview.frames) == 5
    latest_live = live_preview.frames[-1]
    assert latest_live.source_width == 200
    assert len(latest_live.vehicles) == 1
    assert latest_live.vehicles[0].track_id.endswith(":12")
    assert latest_live.vehicles[0].plate is not None
    assert latest_live.vehicles[0].plate.text == "51H-123.45"
    assert latest_live.vehicles[0].plate.bbox.as_xyxy() == (60, 95, 140, 125)


async def test_plate_only_video_tracks_full_frame_plates_without_vehicle_inference(
    tmp_path,
) -> None:
    video_path = tmp_path / "plate-only.avi"
    make_video(video_path)
    base = load_settings()
    camera = base.camera.model_copy(update={"fps_limit": 5.0})
    plate_quality = base.vision.plate_quality.model_copy(update={"minimum": 0.20})
    vision = base.vision.model_copy(update={"plate_only": True, "plate_quality": plate_quality})
    storage = base.storage.model_copy(update={"output_directory": tmp_path / "plate-output"})
    settings = base.model_copy(update={"camera": camera, "vision": vision, "storage": storage})
    source = OpenCVVideoSource(
        video_path,
        camera.id,
        camera.fps_limit,
        datetime(2026, 8, 8, tzinfo=UTC),
    )
    repository = JsonlVehicleEventRepository(storage.output_directory / "events.jsonl")
    normalizer = VietnamPlateNormalizer()
    direction = DirectionEstimator(None, Direction.ENTER, "BOTH")
    finalizer = VehicleEventFinalizer(
        camera,
        settings.events,
        storage,
        PlateCandidateAggregator(settings.voting, normalizer),
        direction,
        LocalMediaStorage(storage.output_directory),
        TestImageEncoder(),
        RepositoryEventPublisher(repository),
    )
    live_preview = CapturingLivePreview()
    pipeline = VideoVehiclePipeline(
        settings,
        source,
        None,
        StableTestTracker(),
        FixedPlateDetector(),
        PlateQualityEvaluator(plate_quality),
        OriginalOnlyPreprocessor(),
        SequenceOCR(),
        normalizer,
        BestFrameSelector(settings.vision.snapshot_selection),
        finalizer,
        direction,
        live_preview=live_preview,
    )
    await repository.ensure_indexes()

    result = await pipeline.run()

    assert len(result.events) == 1
    event = result.events[0]
    assert event.plate is not None
    assert event.plate.normalized == "51H-123.45"
    assert event.plate.observation_count == 3
    assert event.vehicle.type == "unknown"
    assert event.vehicle.confidence == 0
    assert event.ai.vehicle_detector is None
    assert event.ai.plate_detector == ModelMetadata("deterministic-plate", "test-1")
    assert event.media.snapshot_key is not None
    assert event.media.vehicle_crop_key is None
    assert event.media.plate_crop_key is not None
    assert result.stats.vehicle_inference_calls == 0
    assert result.stats.vehicle_detections == 0
    assert result.stats.plate_inference_calls == 5
    assert result.stats.plate_detections == 5
    assert result.stats.ocr_requests == 3
    assert result.stats.finalized_tracks == 1
    assert len(live_preview.frames) == 5
    latest = live_preview.frames[-1].vehicles[0]
    assert latest.vehicle_type == "unknown"
    assert latest.plate is not None
    assert latest.plate.bbox.as_xyxy() == (40, 45, 120, 75)


async def test_disabled_media_skips_candidate_scoring_and_image_retention(tmp_path) -> None:
    video_path = tmp_path / "no-media.avi"
    make_video(video_path)
    base = load_settings()
    camera = base.camera.model_copy(update={"fps_limit": 5.0})
    plate_quality = base.vision.plate_quality.model_copy(update={"minimum": 0.20})
    vision = base.vision.model_copy(update={"plate_quality": plate_quality})
    storage = base.storage.model_copy(
        update={
            "output_directory": tmp_path / "no-media-output",
            "snapshots": False,
            "vehicle_crops": False,
            "plate_crops": False,
        }
    )
    settings = base.model_copy(update={"camera": camera, "vision": vision, "storage": storage})
    source = OpenCVVideoSource(
        video_path,
        camera.id,
        camera.fps_limit,
        datetime(2026, 8, 8, tzinfo=UTC),
    )
    repository = JsonlVehicleEventRepository(storage.output_directory / "events.jsonl")
    normalizer = VietnamPlateNormalizer()
    direction = DirectionEstimator(None, Direction.ENTER, "BOTH")
    finalizer = VehicleEventFinalizer(
        camera,
        settings.events,
        storage,
        PlateCandidateAggregator(settings.voting, normalizer),
        direction,
        LocalMediaStorage(storage.output_directory),
        TestImageEncoder(),
        RepositoryEventPublisher(repository),
    )
    pipeline = VideoVehiclePipeline(
        settings,
        source,
        SequenceVehicleDetector(),
        StableTestTracker(),
        FixedPlateDetector(),
        PlateQualityEvaluator(plate_quality),
        OriginalOnlyPreprocessor(),
        SequenceOCR(),
        normalizer,
        DisabledMediaSelector(),
        finalizer,
        direction,
    )
    await repository.ensure_indexes()

    result = await pipeline.run()

    assert len(result.events) == 1
    assert result.events[0].media.snapshot_key is None
    assert result.events[0].media.vehicle_crop_key is None
    assert result.events[0].media.plate_crop_key is None
    assert not list(storage.output_directory.rglob("*.jpg"))


async def test_ocr_load_controls_reduce_requests_without_changing_plate_result(
    tmp_path,
) -> None:
    video_path = tmp_path / "ocr-load.avi"
    make_video(video_path, frame_count=8)
    base = load_settings()
    camera = base.camera.model_copy(update={"fps_limit": 5.0})
    plate_quality = base.vision.plate_quality.model_copy(update={"minimum": 0.20})

    async def run_case(name, ocr_updates):
        ocr_config = base.vision.ocr.model_copy(update=ocr_updates)
        vision = base.vision.model_copy(
            update={
                "plate_only": True,
                "plate_quality": plate_quality,
                "ocr": ocr_config,
            }
        )
        storage = base.storage.model_copy(
            update={"output_directory": tmp_path / f"ocr-{name}-output"}
        )
        settings = base.model_copy(update={"camera": camera, "vision": vision, "storage": storage})
        source = OpenCVVideoSource(
            video_path,
            camera.id,
            camera.fps_limit,
            datetime(2026, 8, 8, tzinfo=UTC),
        )
        repository = JsonlVehicleEventRepository(storage.output_directory / "events.jsonl")
        normalizer = VietnamPlateNormalizer()
        direction = DirectionEstimator(None, Direction.ENTER, "BOTH")
        finalizer = VehicleEventFinalizer(
            camera,
            settings.events,
            storage,
            PlateCandidateAggregator(settings.voting, normalizer),
            direction,
            LocalMediaStorage(storage.output_directory),
            TestImageEncoder(),
            RepositoryEventPublisher(repository),
        )
        ocr = CountingVariantOCR()
        selector = CountingPlateSelector(settings.vision.snapshot_selection)
        pipeline = VideoVehiclePipeline(
            settings,
            source,
            None,
            StableTestTracker(),
            FixedPlateDetector(),
            PlateQualityEvaluator(plate_quality),
            ThreeVariantPreprocessor(),
            ocr,
            normalizer,
            selector,
            finalizer,
            direction,
        )
        await repository.ensure_indexes()
        return await pipeline.run(), ocr, selector

    legacy_result, legacy_ocr, legacy_selector = await run_case(
        "legacy",
        {
            "track_frame_interval": 1,
            "variant_early_stop_confidence": None,
            "consensus_stop_min_observations": None,
        },
    )
    result, ocr, selector = await run_case(
        "optimized",
        {
            "track_frame_interval": 2,
            "variant_early_stop_confidence": 0.95,
            "consensus_stop_min_observations": 3,
            "consensus_stop_min_confidence": 0.90,
        },
    )

    assert len(legacy_result.events) == 1
    assert len(result.events) == 1
    legacy_event = legacy_result.events[0]
    event = result.events[0]
    assert legacy_event.plate is not None
    assert event.plate is not None
    assert legacy_event.plate.normalized == "51H-123.45"
    assert event.plate.normalized == "51H-123.45"
    assert legacy_event.plate.observation_count == 8
    assert event.plate.observation_count == 3
    assert legacy_result.stats.ocr_requests == 24
    assert legacy_ocr.calls == 24
    assert legacy_selector.plate_calls == 8
    assert result.stats.ocr_requests == 6
    assert ocr.calls == 6
    assert selector.plate_calls == 8
