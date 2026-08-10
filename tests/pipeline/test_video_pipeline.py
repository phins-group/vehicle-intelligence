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


class StableTestTracker:
    def update(self, detections, image):
        del image
        return [TrackedDetection(12, detection) for detection in detections]

    def reset(self):
        return None


class FixedPlateDetector:
    def __init__(self) -> None:
        self._model = ModelMetadata("deterministic-plate", "test-1")

    def detect(self, image):
        del image
        return [PlateDetection(BoundingBox(40, 45, 120, 75), 0.95, self._model)]


class OriginalOnlyPreprocessor:
    def variants(self, image, quality, detection):
        del quality, detection
        return [ImageVariant("original", image)]


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


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (200, 180),
    )
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    for _ in range(5):
        frame = rng.integers(0, 256, size=(180, 200, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


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
    assert event.plate.observation_count == 5
    assert event.direction is Direction.ENTER
    assert event.event_type.value == "VEHICLE_ENTER"
    assert result.stats.finalized_tracks == 1
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
    vision = base.vision.model_copy(
        update={"plate_only": True, "plate_quality": plate_quality}
    )
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
    assert event.plate.observation_count == 5
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
    assert result.stats.finalized_tracks == 1
    assert len(live_preview.frames) == 5
    latest = live_preview.frames[-1].vehicles[0]
    assert latest.vehicle_type == "unknown"
    assert latest.plate is not None
    assert latest.plate.bbox.as_xyxy() == (40, 45, 120, 75)
