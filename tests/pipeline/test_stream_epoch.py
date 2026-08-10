import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.finalization import VehicleEventFinalizer
from vehicle_intelligence.application.health import CameraHealthReporter
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.pipeline import VideoVehiclePipeline
from vehicle_intelligence.application.ports import StreamHeartbeat
from vehicle_intelligence.application.quality import PlateQualityEvaluator
from vehicle_intelligence.application.selection import BestFrameSelector
from vehicle_intelligence.application.voting import PlateCandidateAggregator
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.domain import (
    BoundingBox,
    CameraHealth,
    CameraStatus,
    Detection,
    Direction,
    ModelMetadata,
    TrackedDetection,
    VideoFrame,
)
from vehicle_intelligence.infrastructure.messaging.direct import RepositoryEventPublisher
from vehicle_intelligence.infrastructure.persistence.camera_memory import (
    InMemoryCameraHealthRepository,
)
from vehicle_intelligence.infrastructure.persistence.memory import InMemoryVehicleEventRepository
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage
from vehicle_intelligence.infrastructure.vision.opencv import OpenCVImageEncoder


class EpochSource:
    source_id = "rtsp-test"
    source_fps = 5.0

    def __init__(self) -> None:
        self.closed = False

    def frames(self):
        started = datetime(2026, 8, 9, tzinfo=UTC)
        for frame_id, epoch in enumerate((0, 0, 1, 1)):
            yield VideoFrame(
                camera_id="gate-epoch",
                frame_id=frame_id,
                timestamp=started + timedelta(seconds=frame_id / 5),
                image=np.full((120, 160, 3), 100 + frame_id, dtype=np.uint8),
                stream_epoch=epoch,
            )

    def close(self) -> None:
        self.closed = True

    @property
    def health(self) -> CameraHealth:
        return CameraHealth(
            camera_id="gate-epoch",
            status=CameraStatus.STOPPED if self.closed else CameraStatus.ONLINE,
            source_fps=self.source_fps,
            decode_fps=self.source_fps,
            queue_size=0,
            dropped_frames=0,
            reconnect_count=0,
            connection_failures=0,
            stream_epoch=1,
            last_frame_at=datetime(2026, 8, 9, tzinfo=UTC),
            updated_at=datetime(2026, 8, 9, tzinfo=UTC),
        )


class HeartbeatThenFailureSource:
    source_id = "rtsp-timeout"
    source_fps = 5.0

    def __init__(self) -> None:
        self.closed = False

    def frames(self):
        started = datetime(2026, 8, 9, tzinfo=UTC)
        yield VideoFrame(
            camera_id="gate-timeout",
            frame_id=0,
            timestamp=started,
            image=np.full((120, 160, 3), 100, dtype=np.uint8),
        )
        yield StreamHeartbeat(started + timedelta(seconds=3), 0)
        raise RuntimeError("source failed after timeout heartbeat")

    def close(self) -> None:
        self.closed = True


class FixedVehicleDetector:
    def __init__(self) -> None:
        self.model = ModelMetadata("vehicle-test", "1")

    def detect(self, image):
        del image
        return [Detection(BoundingBox(20, 20, 140, 100), 0.9, 2, "car", self.model)]


class CancellingVehicleDetector(FixedVehicleDetector):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        if self.calls == 2:
            raise asyncio.CancelledError
        return super().detect(image)


class ReusedIdTracker:
    def __init__(self) -> None:
        self.reset_count = 0

    def update(self, detections, image):
        del image
        return [TrackedDetection(1, detection) for detection in detections]

    def reset(self) -> None:
        self.reset_count += 1


class NoPlateDetector:
    def detect(self, image):
        del image
        return []


class NoopPreprocessor:
    def variants(self, image, quality, detection):
        del image, quality, detection
        return []


class NoopOCR:
    def recognize(self, image):
        raise AssertionError(f"OCR must not run: {image.shape}")


@pytest.mark.parametrize(("retain_events", "expected_retained"), ((True, 2), (False, 0)))
async def test_reconnect_epoch_finalizes_old_track_and_allows_reused_tracker_id(
    tmp_path, retain_events: bool, expected_retained: int
) -> None:
    base = load_settings()
    camera = base.camera.model_copy(update={"id": "gate-epoch", "name": "Epoch Gate"})
    storage = base.storage.model_copy(update={"output_directory": tmp_path})
    settings = base.model_copy(update={"camera": camera, "storage": storage})
    source = EpochSource()
    tracker = ReusedIdTracker()
    repository = InMemoryVehicleEventRepository()
    observed = []
    health_repository = InMemoryCameraHealthRepository()
    health_reporter = CameraHealthReporter(health_repository, 60)
    await health_reporter.initialize()
    direction = DirectionEstimator(None, Direction.ENTER, "BOTH")
    normalizer = VietnamPlateNormalizer()
    finalizer = VehicleEventFinalizer(
        camera,
        settings.events,
        storage,
        PlateCandidateAggregator(settings.voting, normalizer),
        direction,
        LocalMediaStorage(tmp_path),
        OpenCVImageEncoder(),
        RepositoryEventPublisher(repository),
    )
    pipeline = VideoVehiclePipeline(
        settings,
        source,
        FixedVehicleDetector(),
        tracker,
        NoPlateDetector(),
        PlateQualityEvaluator(settings.vision.plate_quality),
        NoopPreprocessor(),
        NoopOCR(),
        normalizer,
        BestFrameSelector(settings.vision.snapshot_selection),
        finalizer,
        direction,
        retain_events=retain_events,
        event_observer=observed.append,
        health_reporter=health_reporter,
    )

    result = await pipeline.run()

    assert len(result.events) == expected_retained
    assert {event.track_id for event in observed} == {
        "gate-epoch:rtsp-test:1",
        "gate-epoch:rtsp-test-e1:1",
    }
    assert all(event.status.value == "NO_PLATE" for event in observed)
    assert tracker.reset_count == 2
    assert source.closed
    health = await health_repository.get("gate-epoch")
    assert health is not None
    assert health.status is CameraStatus.STOPPED
    assert health.sampled_frames == 4
    assert health.vehicle_detections == 4
    assert health.events_created == 2
    assert health.track_count == 0
    assert health.inference_fps > 0
    assert health.vehicle_inference_latency_ms > 0
    await health_reporter.close()


async def test_cancellation_finalizes_active_track_before_shutdown(tmp_path) -> None:
    base = load_settings()
    camera = base.camera.model_copy(update={"id": "gate-cancel", "name": "Cancel Gate"})
    storage = base.storage.model_copy(update={"output_directory": tmp_path})
    settings = base.model_copy(update={"camera": camera, "storage": storage})
    source = EpochSource()
    tracker = ReusedIdTracker()
    repository = InMemoryVehicleEventRepository()
    direction = DirectionEstimator(None, Direction.ENTER, "BOTH")
    normalizer = VietnamPlateNormalizer()
    observed = []
    finalizer = VehicleEventFinalizer(
        camera,
        settings.events,
        storage,
        PlateCandidateAggregator(settings.voting, normalizer),
        direction,
        LocalMediaStorage(tmp_path),
        OpenCVImageEncoder(),
        RepositoryEventPublisher(repository),
    )
    pipeline = VideoVehiclePipeline(
        settings,
        source,
        CancellingVehicleDetector(),
        tracker,
        NoPlateDetector(),
        PlateQualityEvaluator(settings.vision.plate_quality),
        NoopPreprocessor(),
        NoopOCR(),
        normalizer,
        BestFrameSelector(settings.vision.snapshot_selection),
        finalizer,
        direction,
        retain_events=False,
        event_observer=observed.append,
    )

    with pytest.raises(asyncio.CancelledError):
        await pipeline.run()

    assert len(observed) == 1
    assert observed[0].track_id == "gate-cancel:rtsp-test:1"
    assert source.closed
    assert tracker.reset_count == 1


async def test_idle_heartbeat_finalizes_track_before_source_recovers(tmp_path) -> None:
    base = load_settings()
    camera = base.camera.model_copy(update={"id": "gate-timeout", "name": "Timeout Gate"})
    storage = base.storage.model_copy(update={"output_directory": tmp_path})
    settings = base.model_copy(update={"camera": camera, "storage": storage})
    source = HeartbeatThenFailureSource()
    tracker = ReusedIdTracker()
    repository = InMemoryVehicleEventRepository()
    direction = DirectionEstimator(None, Direction.ENTER, "BOTH")
    normalizer = VietnamPlateNormalizer()
    observed = []
    finalizer = VehicleEventFinalizer(
        camera,
        settings.events,
        storage,
        PlateCandidateAggregator(settings.voting, normalizer),
        direction,
        LocalMediaStorage(tmp_path),
        OpenCVImageEncoder(),
        RepositoryEventPublisher(repository),
    )
    pipeline = VideoVehiclePipeline(
        settings,
        source,
        FixedVehicleDetector(),
        tracker,
        NoPlateDetector(),
        PlateQualityEvaluator(settings.vision.plate_quality),
        NoopPreprocessor(),
        NoopOCR(),
        normalizer,
        BestFrameSelector(settings.vision.snapshot_selection),
        finalizer,
        direction,
        retain_events=False,
        event_observer=observed.append,
    )

    with pytest.raises(RuntimeError, match="after timeout heartbeat"):
        await pipeline.run()

    assert len(observed) == 1
    assert observed[0].track_id == "gate-timeout:rtsp-timeout:1"
    assert source.closed
