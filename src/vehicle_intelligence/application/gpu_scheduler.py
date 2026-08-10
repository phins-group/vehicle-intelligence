"""Thread-safe latest-frame, round-robin scheduler for shared inference devices."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Lock

from vehicle_intelligence.application.ports import BatchVehicleDetector, VehicleDetector
from vehicle_intelligence.config import GPUSchedulerConfig
from vehicle_intelligence.domain import Detection, VideoFrame


@dataclass(frozen=True, slots=True)
class ScheduledFrame:
    frame: VideoFrame
    submitted_monotonic: float


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    submitted: int
    emitted: int
    dropped_oldest: int
    dropped_stale: int
    pending: int
    cameras: int
    emitted_per_camera: dict[str, int]


@dataclass(frozen=True, slots=True)
class ScheduledDetectionResult:
    frame: VideoFrame
    detections: tuple[Detection, ...]
    end_to_end_latency_ms: float


class SchedulerCapacityError(RuntimeError):
    pass


class FairLatestFrameScheduler:
    """Bounded per-camera queues with a ready-camera round robin."""

    def __init__(
        self,
        config: GPUSchedulerConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._clock = clock
        self._queues: dict[str, deque[ScheduledFrame]] = {}
        self._ready: deque[str] = deque()
        self._ready_set: set[str] = set()
        self._condition = Condition(Lock())
        self._submitted = 0
        self._emitted = 0
        self._dropped_oldest = 0
        self._dropped_stale = 0
        self._emitted_per_camera: dict[str, int] = defaultdict(int)

    def submit(self, frame: VideoFrame, *, now_monotonic: float | None = None) -> int:
        now = self._now() if now_monotonic is None else now_monotonic
        with self._condition:
            queue = self._queues.get(frame.camera_id)
            if queue is None:
                if len(self._queues) >= self._config.maximum_cameras:
                    raise SchedulerCapacityError(
                        f"GPU scheduler camera capacity reached: {self._config.maximum_cameras}"
                    )
                queue = deque()
                self._queues[frame.camera_id] = queue
            dropped = 0
            while len(queue) >= self._config.per_camera_queue_size:
                queue.popleft()
                dropped += 1
            queue.append(ScheduledFrame(frame=frame, submitted_monotonic=now))
            self._submitted += 1
            self._dropped_oldest += dropped
            self._mark_ready(frame.camera_id)
            self._condition.notify()
            return dropped

    def pop_batch(self, *, now_monotonic: float | None = None) -> tuple[ScheduledFrame, ...]:
        now = self._now() if now_monotonic is None else now_monotonic
        with self._condition:
            return self._pop_locked(now)

    def wait_batch(self, timeout_seconds: float | None = None) -> tuple[ScheduledFrame, ...]:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("scheduler wait timeout cannot be negative")
        deadline = None if timeout_seconds is None else self._now() + timeout_seconds
        with self._condition:
            while not self._ready:
                remaining = None if deadline is None else deadline - self._now()
                if remaining is not None and remaining <= 0:
                    return ()
                self._condition.wait(remaining)
            batch_deadline = self._now() + self._config.batch_wait_ms / 1000
            if deadline is not None:
                batch_deadline = min(batch_deadline, deadline)
            while self._pending_locked() < self._config.maximum_batch_size:
                remaining = batch_deadline - self._now()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._pop_locked(self._now())

    def unregister(self, camera_id: str) -> int:
        with self._condition:
            queue = self._queues.pop(camera_id, None)
            self._ready_set.discard(camera_id)
            self._ready = deque(item for item in self._ready if item != camera_id)
            return len(queue) if queue is not None else 0

    def snapshot(self) -> SchedulerSnapshot:
        with self._condition:
            return SchedulerSnapshot(
                submitted=self._submitted,
                emitted=self._emitted,
                dropped_oldest=self._dropped_oldest,
                dropped_stale=self._dropped_stale,
                pending=self._pending_locked(),
                cameras=len(self._queues),
                emitted_per_camera=dict(self._emitted_per_camera),
            )

    def _pop_locked(self, now: float) -> tuple[ScheduledFrame, ...]:
        batch: list[ScheduledFrame] = []
        while self._ready and len(batch) < self._config.maximum_batch_size:
            camera_id = self._ready.popleft()
            self._ready_set.discard(camera_id)
            queue = self._queues.get(camera_id)
            if queue is None:
                continue
            while queue and self._is_stale(queue[0], now):
                queue.popleft()
                self._dropped_stale += 1
            if queue:
                item = queue.popleft()
                batch.append(item)
                self._emitted += 1
                self._emitted_per_camera[camera_id] += 1
            if queue:
                self._mark_ready(camera_id)
        return tuple(batch)

    def _mark_ready(self, camera_id: str) -> None:
        if camera_id not in self._ready_set:
            self._ready.append(camera_id)
            self._ready_set.add(camera_id)

    def _is_stale(self, item: ScheduledFrame, now: float) -> bool:
        return (now - item.submitted_monotonic) * 1000 > self._config.maximum_frame_age_ms

    def _pending_locked(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    def _now(self) -> float:
        return float(self._clock())


class FairInferenceCoordinator:
    """Drain scheduled frames and use real provider batching when available."""

    def __init__(
        self,
        scheduler: FairLatestFrameScheduler,
        detector: VehicleDetector,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._scheduler = scheduler
        self._detector = detector
        self._clock = clock

    def run_once(self, *, wait_seconds: float = 0) -> tuple[ScheduledDetectionResult, ...]:
        batch = (
            self._scheduler.wait_batch(wait_seconds)
            if wait_seconds > 0
            else self._scheduler.pop_batch()
        )
        if not batch:
            return ()
        images = [item.frame.image for item in batch]
        if isinstance(self._detector, BatchVehicleDetector):
            detection_sets = self._detector.detect_batch(images)
        else:
            detection_sets = [self._detector.detect(image) for image in images]
        if len(detection_sets) != len(batch):
            raise RuntimeError("batch detector result count does not match input count")
        completed = float(self._clock())
        return tuple(
            ScheduledDetectionResult(
                frame=item.frame,
                detections=tuple(detections),
                end_to_end_latency_ms=max(0, (completed - item.submitted_monotonic) * 1000),
            )
            for item, detections in zip(batch, detection_sets, strict=True)
        )
