"""Config-driven multi-camera worker reconciliation and failure isolation."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from vehicle_intelligence.application.ports import (
    CameraHealthRepository,
    CameraRepository,
    CameraWorkerHandle,
    CameraWorkerLauncher,
)
from vehicle_intelligence.domain import CameraHealth, CameraStatus
from vehicle_intelligence.exceptions import CameraWorkerError, PersistenceError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraSupervisorStats:
    workers_started: int = 0
    workers_stopped: int = 0
    workers_restarted: int = 0
    worker_start_failures: int = 0
    worker_crashes: int = 0
    workers_capacity_deferred: int = 0
    peak_active_workers: int = 0
    maximum_backoff_seconds_observed: float = 0.0


@dataclass(slots=True)
class _ManagedWorker:
    revision: int
    handle: CameraWorkerHandle
    started_at: float


class CameraSupervisor:
    def __init__(
        self,
        cameras: CameraRepository,
        health: CameraHealthRepository,
        launcher: CameraWorkerLauncher,
        reconcile_interval_seconds: float,
        restart_backoff_seconds: float,
        restart_backoff_max_seconds: float = 120.0,
        restart_stability_seconds: float = 60.0,
        maximum_active_workers: int = 32,
        maximum_starts_per_reconcile: int = 4,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if (
            reconcile_interval_seconds <= 0
            or restart_backoff_seconds <= 0
            or restart_backoff_max_seconds < restart_backoff_seconds
            or restart_stability_seconds <= 0
            or maximum_active_workers < 1
            or maximum_starts_per_reconcile < 1
        ):
            raise ValueError("camera supervisor intervals must be positive")
        self._cameras = cameras
        self._health = health
        self._launcher = launcher
        self._reconcile_interval = reconcile_interval_seconds
        self._restart_backoff = restart_backoff_seconds
        self._restart_backoff_max = restart_backoff_max_seconds
        self._restart_stability = restart_stability_seconds
        self._maximum_active_workers = maximum_active_workers
        self._maximum_starts_per_reconcile = maximum_starts_per_reconcile
        self._monotonic = monotonic_clock
        self._wall_clock = wall_clock
        self._active: dict[str, _ManagedWorker] = {}
        self._restart_after: dict[str, float] = {}
        self._failure_counts: dict[str, int] = {}
        self.stats = CameraSupervisorStats()

    @property
    def active_camera_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._active))

    async def initialize(self) -> None:
        await self._cameras.ensure_indexes()
        await self._health.ensure_indexes()

    async def run(self, stop_event: asyncio.Event) -> CameraSupervisorStats:
        try:
            while not stop_event.is_set():
                try:
                    await self.initialize()
                    break
                except PersistenceError:
                    logger.exception("camera supervisor initialization failed; retrying")
                    await self._wait(stop_event)
            while not stop_event.is_set():
                try:
                    await self.reconcile_once()
                except (PersistenceError, CameraWorkerError):
                    logger.exception("camera supervisor reconciliation failed; retrying")
                await self._wait(stop_event)
        finally:
            await self.stop_all()
            await self.close()
        return self.stats

    async def reconcile_once(self) -> None:
        desired = {camera.id: camera for camera in await self._cameras.list(enabled_only=True)}
        now = self._monotonic()
        for camera_id in set(self._restart_after) - set(desired):
            del self._restart_after[camera_id]
            self._failure_counts.pop(camera_id, None)

        for camera_id, managed in list(self._active.items()):
            camera = desired.get(camera_id)
            if camera is None or camera.revision != managed.revision:
                await managed.handle.stop()
                del self._active[camera_id]
                self.stats.workers_stopped += 1
                persisted = await self._cameras.get(camera_id)
                if persisted is None:
                    await self._health.delete(camera_id)
                    self._restart_after.pop(camera_id, None)
                    self._failure_counts.pop(camera_id, None)
                else:
                    await self._write_status(camera_id, CameraStatus.STOPPED)
                    if persisted.enabled:
                        self._restart_after[camera_id] = 0
                        self._failure_counts.pop(camera_id, None)
                continue
            if not managed.handle.running:
                del self._active[camera_id]
                self._schedule_failure(camera_id, now)
                self.stats.worker_crashes += 1
                await self._write_status(camera_id, CameraStatus.OFFLINE)
                logger.warning(
                    "camera worker exited",
                    extra={
                        "camera_id": camera_id,
                        "return_code": managed.handle.return_code,
                    },
                )
                continue

            if now - managed.started_at >= self._restart_stability:
                self._failure_counts.pop(camera_id, None)

        starts_remaining = min(
            self._maximum_starts_per_reconcile,
            max(0, self._maximum_active_workers - len(self._active)),
        )
        for camera_id in sorted(desired):
            camera = desired[camera_id]
            if camera_id in self._active or self._restart_after.get(camera_id, 0) > now:
                continue
            if starts_remaining <= 0:
                self.stats.workers_capacity_deferred += 1
                continue
            was_started = camera_id in self._restart_after
            try:
                handle = await self._launcher.start(camera)
            except Exception:
                self._schedule_failure(camera_id, now)
                self.stats.worker_start_failures += 1
                await self._write_status(camera_id, CameraStatus.OFFLINE)
                logger.exception(
                    "camera worker start failed",
                    extra={"camera_id": camera_id},
                )
                continue
            self._active[camera_id] = _ManagedWorker(camera.revision, handle, now)
            self._restart_after.pop(camera_id, None)
            starts_remaining -= 1
            self.stats.workers_started += 1
            self.stats.peak_active_workers = max(
                self.stats.peak_active_workers,
                len(self._active),
            )
            if was_started:
                self.stats.workers_restarted += 1
            await self._write_status(camera_id, CameraStatus.CONNECTING)
            logger.info("camera worker started", extra={"camera_id": camera_id})

    def _schedule_failure(self, camera_id: str, now: float) -> None:
        failures = self._failure_counts.get(camera_id, 0) + 1
        self._failure_counts[camera_id] = failures
        delay = min(
            self._restart_backoff * (2 ** min(failures - 1, 20)),
            self._restart_backoff_max,
        )
        self._restart_after[camera_id] = now + delay
        self.stats.maximum_backoff_seconds_observed = max(
            self.stats.maximum_backoff_seconds_observed,
            delay,
        )

    async def stop_all(self) -> None:
        async def stop_one(camera_id: str, managed: _ManagedWorker) -> None:
            try:
                await managed.handle.stop()
                self.stats.workers_stopped += 1
                await self._write_status(camera_id, CameraStatus.STOPPED)
            except Exception:
                logger.exception(
                    "camera worker stop failed",
                    extra={"camera_id": camera_id},
                )

        active, self._active = self._active, {}
        await asyncio.gather(
            *(stop_one(camera_id, managed) for camera_id, managed in active.items())
        )

    async def close(self) -> None:
        try:
            await self._cameras.close()
        finally:
            await self._health.close()

    async def _write_status(self, camera_id: str, status: CameraStatus) -> None:
        current = await self._health.get(camera_id)
        now = self._wall_clock().astimezone(UTC)
        health = (
            replace(current, status=status, updated_at=now)
            if current is not None
            else CameraHealth(
                camera_id=camera_id,
                status=status,
                source_fps=0,
                decode_fps=0,
                queue_size=0,
                dropped_frames=0,
                reconnect_count=0,
                connection_failures=0,
                stream_epoch=0,
                last_frame_at=None,
                updated_at=now,
            )
        )
        await self._health.save(health)

    async def _wait(self, stop_event: asyncio.Event) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=self._reconcile_interval)
