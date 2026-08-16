from dataclasses import replace
from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.supervisor import CameraSupervisor
from vehicle_intelligence.domain import (
    Camera,
    CameraDirection,
    CameraStatus,
    SecretUri,
)
from vehicle_intelligence.exceptions import CameraWorkerError
from vehicle_intelligence.infrastructure.persistence.camera_memory import (
    InMemoryCameraHealthRepository,
    InMemoryCameraRepository,
)


class FakeHandle:
    def __init__(self) -> None:
        self.is_running = True
        self.code = None
        self.stop_count = 0

    @property
    def running(self) -> bool:
        return self.is_running

    @property
    def return_code(self):
        return self.code

    async def stop(self) -> None:
        self.stop_count += 1
        self.is_running = False
        self.code = 0

    def crash(self) -> None:
        self.is_running = False
        self.code = 2


class FakeLauncher:
    def __init__(self) -> None:
        self.handles: dict[str, list[FakeHandle]] = {}

    async def start(self, camera):
        handle = FakeHandle()
        self.handles.setdefault(camera.id, []).append(handle)
        return handle


class FakeInferenceLauncher:
    def __init__(self) -> None:
        self.handles: list[FakeHandle] = []

    async def start(self):
        handle = FakeHandle()
        self.handles.append(handle)
        return handle


class AdvancingInferenceLauncher:
    def __init__(self, now: list[float]) -> None:
        self._now = now
        self.handles: list[FakeHandle] = []
        self.attempts = 0

    async def start(self):
        self.attempts += 1
        self._now[0] += 100
        if self.attempts == 1:
            raise RuntimeError("slow startup failure")
        handle = FakeHandle()
        self.handles.append(handle)
        return handle


def camera(camera_id: str) -> Camera:
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    return Camera(
        id=camera_id,
        name=camera_id,
        rtsp_url=SecretUri(f"rtsp://user:secret@{camera_id}.example/live"),
        fps_limit=6,
        direction=CameraDirection.BOTH,
        enabled=True,
        vehicle_confidence=0.4,
        plate_confidence=0.45,
        created_at=timestamp,
        updated_at=timestamp,
    )


async def test_supervisor_isolates_crash_restarts_and_reconciles_config() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    launcher = FakeLauncher()
    now = [0.0]
    wall_time = datetime(2026, 8, 9, tzinfo=UTC)
    supervisor = CameraSupervisor(
        cameras,
        health,
        launcher,
        reconcile_interval_seconds=1,
        restart_backoff_seconds=5,
        monotonic_clock=lambda: now[0],
        wall_clock=lambda: wall_time,
    )
    first = camera("gate-01")
    second = camera("gate-02")
    await cameras.create(first)
    await cameras.create(second)
    await supervisor.initialize()

    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ("gate-01", "gate-02")

    launcher.handles["gate-01"][0].crash()
    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ("gate-02",)
    assert launcher.handles["gate-02"][0].running
    assert (await health.get("gate-01")).status is CameraStatus.OFFLINE

    now[0] = 6
    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ("gate-01", "gate-02")
    assert len(launcher.handles["gate-01"]) == 2

    updated_second = replace(second, revision=2, name="Gate 02 updated")
    assert await cameras.replace(updated_second, 1)
    await supervisor.reconcile_once()
    assert len(launcher.handles["gate-02"]) == 2
    assert launcher.handles["gate-02"][0].stop_count == 1

    disabled_second = replace(updated_second, revision=3, enabled=False)
    assert await cameras.replace(disabled_second, 2)
    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ("gate-01",)
    assert (await health.get("gate-02")).status is CameraStatus.STOPPED

    assert await cameras.delete("gate-01")
    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ()
    assert await health.get("gate-01") is None
    assert supervisor.stats.worker_crashes == 1
    assert supervisor.stats.workers_restarted >= 2
    await supervisor.close()


async def test_supervisor_caps_active_workers_and_start_burst() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    launcher = FakeLauncher()
    for camera_id in ("gate-01", "gate-02", "gate-03"):
        await cameras.create(camera(camera_id))
    supervisor = CameraSupervisor(
        cameras,
        health,
        launcher,
        reconcile_interval_seconds=1,
        restart_backoff_seconds=2,
        maximum_active_workers=2,
        maximum_starts_per_reconcile=1,
    )

    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ("gate-01",)
    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ("gate-01", "gate-02")
    await supervisor.reconcile_once()
    assert supervisor.active_camera_ids == ("gate-01", "gate-02")
    assert "gate-03" not in launcher.handles
    assert supervisor.stats.peak_active_workers == 2
    assert supervisor.stats.workers_capacity_deferred >= 3
    await supervisor.stop_all()
    await supervisor.close()


class FailingLauncher:
    def __init__(self) -> None:
        self.attempts = 0

    async def start(self, _camera):
        self.attempts += 1
        raise RuntimeError("simulated start failure")


async def test_supervisor_uses_capped_exponential_restart_backoff() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    launcher = FailingLauncher()
    now = [0.0]
    await cameras.create(camera("gate-01"))
    supervisor = CameraSupervisor(
        cameras,
        health,
        launcher,
        reconcile_interval_seconds=1,
        restart_backoff_seconds=2,
        restart_backoff_max_seconds=8,
        monotonic_clock=lambda: now[0],
    )

    await supervisor.reconcile_once()
    assert launcher.attempts == 1
    now[0] = 1.9
    await supervisor.reconcile_once()
    assert launcher.attempts == 1
    now[0] = 2
    await supervisor.reconcile_once()
    assert launcher.attempts == 2
    now[0] = 5.9
    await supervisor.reconcile_once()
    assert launcher.attempts == 2
    now[0] = 6
    await supervisor.reconcile_once()
    assert launcher.attempts == 3
    now[0] = 14
    await supervisor.reconcile_once()
    assert launcher.attempts == 4
    assert supervisor.stats.maximum_backoff_seconds_observed == pytest.approx(8)
    assert (await health.get("gate-01")).status is CameraStatus.OFFLINE
    await supervisor.close()


async def test_supervisor_restarts_shared_inference_before_camera_workers() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    workers = FakeLauncher()
    inference = FakeInferenceLauncher()
    now = [0.0]
    await cameras.create(camera("gate-01"))
    supervisor = CameraSupervisor(
        cameras,
        health,
        workers,
        reconcile_interval_seconds=1,
        restart_backoff_seconds=2,
        monotonic_clock=lambda: now[0],
        inference_service_launcher=inference,
    )

    await supervisor.initialize()
    await supervisor.reconcile_once()
    assert len(inference.handles) == 1
    assert len(workers.handles["gate-01"]) == 1

    inference.handles[0].crash()
    with pytest.raises(Exception, match="backoff"):
        await supervisor.reconcile_once()
    assert workers.handles["gate-01"][0].stop_count == 1
    assert len(inference.handles) == 1

    now[0] = 2
    await supervisor.reconcile_once()

    assert len(inference.handles) == 2
    assert inference.handles[1].running
    assert len(workers.handles["gate-01"]) == 2
    assert workers.handles["gate-01"][1].running
    assert supervisor.stats.inference_service_crashes == 1
    assert supervisor.stats.inference_services_restarted == 1

    await supervisor.stop_all()
    await supervisor.close()
    assert inference.handles[1].stop_count == 1
    assert supervisor.stats.inference_services_stopped == 1


async def test_inference_backoff_and_stability_start_after_slow_launcher_finishes() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    workers = FakeLauncher()
    now = [0.0]
    inference = AdvancingInferenceLauncher(now)
    supervisor = CameraSupervisor(
        cameras,
        health,
        workers,
        reconcile_interval_seconds=1,
        restart_backoff_seconds=5,
        restart_stability_seconds=60,
        monotonic_clock=lambda: now[0],
        inference_service_launcher=inference,
    )

    with pytest.raises(CameraWorkerError, match="cannot start"):
        await supervisor.initialize()
    assert now[0] == 100

    now[0] = 104
    with pytest.raises(CameraWorkerError, match="backoff"):
        await supervisor.initialize()
    assert inference.attempts == 1

    now[0] = 105
    await supervisor.initialize()
    assert now[0] == 205
    assert inference.attempts == 2

    now[0] = 264
    await supervisor.reconcile_once()
    inference.handles[0].crash()
    with pytest.raises(CameraWorkerError, match="backoff"):
        await supervisor.reconcile_once()

    now[0] = 269
    with pytest.raises(CameraWorkerError, match="backoff"):
        await supervisor.reconcile_once()
    assert inference.attempts == 2

    now[0] = 274
    await supervisor.reconcile_once()
    assert inference.attempts == 3
    await supervisor.close()
