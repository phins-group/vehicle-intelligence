"""Throttled latest-state camera-health persistence."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from vehicle_intelligence.application.ports import CameraHealthRepository
from vehicle_intelligence.domain import CameraHealth
from vehicle_intelligence.exceptions import PersistenceError

logger = logging.getLogger(__name__)
_CLOSED = object()


class CameraHealthReporter:
    def __init__(
        self,
        repository: CameraHealthRepository,
        minimum_interval_seconds: float,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if minimum_interval_seconds <= 0:
            raise ValueError("camera health publish interval must be positive")
        self._repository = repository
        self._interval = minimum_interval_seconds
        self._monotonic = monotonic_clock
        self._last_attempt: float | None = None
        self._queue: asyncio.Queue[CameraHealth | object] = asyncio.Queue(maxsize=1)
        self._task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        try:
            await self._repository.ensure_indexes()
        except PersistenceError:
            logger.exception("camera health repository initialization failed")
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="camera-health-reporter")

    async def report(self, health: CameraHealth, *, force: bool = False) -> bool:
        now = self._monotonic()
        if (
            not force
            and self._last_attempt is not None
            and now - self._last_attempt < self._interval
        ):
            return False
        self._last_attempt = now
        if force:
            await self._queue.join()
            return await self._save(health)
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="camera-health-reporter")
        if self._queue.full():
            self._queue.get_nowait()
            self._queue.task_done()
        self._queue.put_nowait(health)
        await asyncio.sleep(0)
        return True

    async def close(self) -> None:
        if self._task is not None:
            await self._queue.join()
            await self._queue.put(_CLOSED)
            await self._task
            self._task = None
        await self._repository.close()

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is _CLOSED:
                    return
                await self._save(item)  # type: ignore[arg-type]
            finally:
                self._queue.task_done()

    async def _save(self, health: CameraHealth) -> bool:
        try:
            await self._repository.save(health)
            return True
        except PersistenceError:
            logger.exception(
                "camera health persistence failed",
                extra={"camera_id": health.camera_id, "camera_status": health.status.value},
            )
            return False
        except Exception:
            logger.exception(
                "unexpected camera health persistence failure",
                extra={"camera_id": health.camera_id, "camera_status": health.status.value},
            )
            return False
