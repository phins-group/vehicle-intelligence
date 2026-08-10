"""Throttled latest-state camera-health persistence."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from vehicle_intelligence.application.ports import CameraHealthRepository
from vehicle_intelligence.domain import CameraHealth
from vehicle_intelligence.exceptions import PersistenceError

logger = logging.getLogger(__name__)


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

    async def initialize(self) -> None:
        try:
            await self._repository.ensure_indexes()
        except PersistenceError:
            logger.exception("camera health repository initialization failed")

    async def report(self, health: CameraHealth, *, force: bool = False) -> bool:
        now = self._monotonic()
        if (
            not force
            and self._last_attempt is not None
            and now - self._last_attempt < self._interval
        ):
            return False
        self._last_attempt = now
        try:
            await self._repository.save(health)
            return True
        except PersistenceError:
            logger.exception(
                "camera health persistence failed",
                extra={"camera_id": health.camera_id, "camera_status": health.status.value},
            )
            return False

    async def close(self) -> None:
        await self._repository.close()
