"""Bounded off-request-thread RTSP connection test adapter."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime

from vehicle_intelligence.application.ports import CameraConnectionTestResult
from vehicle_intelligence.config import RTSPConfig
from vehicle_intelligence.domain import Camera
from vehicle_intelligence.infrastructure.vision.rtsp import CaptureFactory, open_opencv_capture


class OpenCVCameraConnectionTester:
    def __init__(
        self,
        config: RTSPConfig,
        capture_factory: CaptureFactory = open_opencv_capture,
        maximum_concurrency: int = 4,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if maximum_concurrency < 1:
            raise ValueError("camera connection-test concurrency must be positive")
        self._config = config
        self._capture_factory = capture_factory
        self._monotonic = monotonic_clock
        self._wall_clock = wall_clock
        self._semaphore = asyncio.Semaphore(maximum_concurrency)

    async def test(self, camera: Camera) -> CameraConnectionTestResult:
        async with self._semaphore:
            return await asyncio.to_thread(self._test_sync, camera)

    def _test_sync(self, camera: Camera) -> CameraConnectionTestResult:
        started = self._monotonic()
        capture = None
        connected = False
        error_code: str | None = None
        try:
            capture = self._capture_factory(camera.rtsp_url.reveal(), self._config)
            if not capture.isOpened():
                error_code = "OPEN_FAILED"
            else:
                success, image = capture.read()
                connected = bool(success and image is not None and image.size > 0)
                if not connected:
                    error_code = "FRAME_READ_FAILED"
        except Exception:
            error_code = "CONNECTION_ERROR"
        finally:
            if capture is not None:
                capture.release()
        return CameraConnectionTestResult(
            connected=connected,
            latency_ms=max((self._monotonic() - started) * 1000, 0),
            tested_at=self._wall_clock().astimezone(UTC),
            error_code=error_code,
        )
