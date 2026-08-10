from datetime import UTC, datetime

import numpy as np
import pytest

from vehicle_intelligence.application.gpu_scheduler import (
    FairInferenceCoordinator,
    FairLatestFrameScheduler,
    SchedulerCapacityError,
)
from vehicle_intelligence.config import GPUSchedulerConfig
from vehicle_intelligence.domain import VideoFrame


class Clock:
    def __init__(self, value: float = 0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class BatchDetector:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def detect(self, _image):
        raise AssertionError("batch-capable detector should not use scalar inference")

    def detect_batch(self, images):
        self.batch_sizes.append(len(images))
        return [[] for _ in images]


def frame(camera: str, frame_id: int) -> VideoFrame:
    return VideoFrame(
        camera_id=camera,
        frame_id=frame_id,
        timestamp=datetime(2026, 8, 10, tzinfo=UTC),
        image=np.zeros((16, 32, 3), dtype=np.uint8),
    )


def test_round_robin_prevents_one_busy_camera_from_dominating() -> None:
    scheduler = FairLatestFrameScheduler(
        GPUSchedulerConfig(
            maximum_cameras=3,
            maximum_batch_size=2,
            per_camera_queue_size=3,
            maximum_frame_age_ms=1000,
        )
    )
    scheduler.submit(frame("a", 0), now_monotonic=0)
    scheduler.submit(frame("a", 1), now_monotonic=0)
    scheduler.submit(frame("b", 0), now_monotonic=0)
    scheduler.submit(frame("c", 0), now_monotonic=0)

    first = scheduler.pop_batch(now_monotonic=0)
    second = scheduler.pop_batch(now_monotonic=0)

    assert [(item.frame.camera_id, item.frame.frame_id) for item in first] == [
        ("a", 0),
        ("b", 0),
    ]
    assert [(item.frame.camera_id, item.frame.frame_id) for item in second] == [
        ("c", 0),
        ("a", 1),
    ]


def test_queue_prefers_latest_and_drops_stale_frames() -> None:
    scheduler = FairLatestFrameScheduler(
        GPUSchedulerConfig(
            maximum_cameras=2,
            maximum_batch_size=2,
            per_camera_queue_size=1,
            maximum_frame_age_ms=100,
        )
    )
    assert scheduler.submit(frame("a", 0), now_monotonic=0) == 0
    assert scheduler.submit(frame("a", 1), now_monotonic=0.01) == 1
    scheduler.submit(frame("b", 0), now_monotonic=0)

    batch = scheduler.pop_batch(now_monotonic=0.2)
    snapshot = scheduler.snapshot()

    assert batch == ()
    assert snapshot.dropped_oldest == 1
    assert snapshot.dropped_stale == 2
    assert snapshot.pending == 0


def test_capacity_is_explicit_and_unregister_is_bounded() -> None:
    scheduler = FairLatestFrameScheduler(
        GPUSchedulerConfig(maximum_cameras=1, maximum_batch_size=1)
    )
    scheduler.submit(frame("a", 0))
    with pytest.raises(SchedulerCapacityError, match="capacity"):
        scheduler.submit(frame("b", 0))
    assert scheduler.unregister("a") == 1
    scheduler.submit(frame("b", 0))


def test_coordinator_uses_batch_provider_and_preserves_frame_pairing() -> None:
    clock = Clock(10)
    scheduler = FairLatestFrameScheduler(
        GPUSchedulerConfig(maximum_cameras=2, maximum_batch_size=2),
        clock=clock,
    )
    detector = BatchDetector()
    scheduler.submit(frame("a", 0), now_monotonic=10)
    scheduler.submit(frame("b", 0), now_monotonic=10)
    clock.value = 10.025

    results = FairInferenceCoordinator(scheduler, detector, clock=clock).run_once()

    assert detector.batch_sizes == [2]
    assert [item.frame.camera_id for item in results] == ["a", "b"]
    assert all(item.end_to_end_latency_ms == pytest.approx(25) for item in results)
