"""Throttled optional edge preview publication isolated from the vision result path."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.application.ports import (
    LiveFramePublisher,
    LivePreviewEncoder,
)
from vehicle_intelligence.config import LiveMonitorConfig
from vehicle_intelligence.domain import LiveFrameMetadata, LiveFramePacket
from vehicle_intelligence.exceptions import EventBusError, EventContractError, MediaStorageError

logger = logging.getLogger(__name__)
_CLOSED = object()


@dataclass(slots=True)
class LivePreviewReporterStats:
    published_frames: int = 0
    throttled_frames: int = 0
    oversized_frames: int = 0
    encode_failures: int = 0
    publish_failures: int = 0
    stale_frames_dropped: int = 0


class LivePreviewReporter:
    def __init__(
        self,
        config: LiveMonitorConfig,
        encoder: LivePreviewEncoder,
        publisher: LiveFramePublisher,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._encoder = encoder
        self._publisher = publisher
        self._monotonic = monotonic_clock
        self._last_attempt: float | None = None
        self._queue: asyncio.Queue[
            tuple[NDArray[np.uint8], LiveFrameMetadata] | object
        ] = asyncio.Queue(maxsize=1)
        self._task: asyncio.Task[None] | None = None
        self.stats = LivePreviewReporterStats()

    async def initialize(self) -> None:
        try:
            await asyncio.wait_for(
                self._publisher.initialize(),
                timeout=self._config.publish_timeout_seconds,
            )
        except (EventBusError, TimeoutError):
            self.stats.publish_failures += 1
            logger.warning("live_preview_publisher_initialization_failed", exc_info=True)
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="live-preview-publisher")

    async def report(
        self,
        image: NDArray[np.uint8],
        metadata: LiveFrameMetadata,
    ) -> bool:
        now = self._monotonic()
        minimum_interval = 1.0 / self._config.preview_fps
        if self._last_attempt is not None and now - self._last_attempt < minimum_interval:
            self.stats.throttled_frames += 1
            return False
        self._last_attempt = now
        if self._queue.full():
            self._queue.get_nowait()
            self.stats.stale_frames_dropped += 1
        self._queue.put_nowait((image.copy(), metadata))
        await asyncio.sleep(0)
        return True

    async def close(self) -> None:
        if self._task is not None:
            await self._queue.put(_CLOSED)
            timeout = self._config.publish_timeout_seconds + 1
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
            except TimeoutError:
                self._task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._task
            self._task = None
        await self._publisher.close()

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is _CLOSED:
                return
            image, metadata = item  # type: ignore[misc]
            await self._encode_and_publish(image, metadata)

    async def _encode_and_publish(
        self,
        image: NDArray[np.uint8],
        metadata: LiveFrameMetadata,
    ) -> None:
        try:
            preview = await asyncio.to_thread(
                self._encoder.encode,
                image,
                self._config.preview_max_width,
                self._config.jpeg_quality,
            )
        except MediaStorageError:
            self.stats.encode_failures += 1
            logger.warning(
                "live_preview_encode_failed",
                exc_info=True,
                extra={"camera_id": metadata.camera_id},
            )
            return
        if len(preview.jpeg) > int(self._config.maximum_payload_bytes * 0.70):
            self.stats.oversized_frames += 1
            logger.warning(
                "live_preview_jpeg_too_large",
                extra={"camera_id": metadata.camera_id, "jpeg_bytes": len(preview.jpeg)},
            )
            return
        packet = LiveFramePacket(
            metadata=metadata,
            jpeg=preview.jpeg,
            preview_width=preview.width,
            preview_height=preview.height,
        )
        try:
            await asyncio.wait_for(
                self._publisher.publish(packet),
                timeout=self._config.publish_timeout_seconds,
            )
        except (EventBusError, EventContractError, TimeoutError):
            self.stats.publish_failures += 1
            logger.warning(
                "live_preview_publish_failed",
                exc_info=True,
                extra={"camera_id": metadata.camera_id},
            )
            return
        self.stats.published_frames += 1
