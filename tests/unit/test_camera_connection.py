from datetime import UTC, datetime

import numpy as np

from vehicle_intelligence.config import RTSPConfig
from vehicle_intelligence.domain import Camera, CameraDirection, SecretUri
from vehicle_intelligence.infrastructure.vision.connection_test import (
    OpenCVCameraConnectionTester,
)


class FakeCapture:
    def __init__(self, opened: bool, readable: bool) -> None:
        self._opened = opened
        self._readable = readable
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        image = np.ones((4, 4, 3), dtype=np.uint8) if self._readable else None
        return self._readable, image

    def get(self, property_id: int) -> float:
        del property_id
        return 0

    def release(self) -> None:
        self.released = True


def configured_camera() -> Camera:
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    return Camera(
        id="gate-test",
        name="Test Gate",
        rtsp_url=SecretUri("rtsp://admin:secret@camera.example/live"),
        fps_limit=6,
        direction=CameraDirection.BOTH,
        enabled=True,
        vehicle_confidence=0.4,
        plate_confidence=0.45,
        created_at=timestamp,
        updated_at=timestamp,
    )


async def test_connection_tester_reads_one_frame_and_releases_capture() -> None:
    capture = FakeCapture(True, True)
    tester = OpenCVCameraConnectionTester(
        RTSPConfig(),
        capture_factory=lambda _url, _config: capture,
    )

    result = await tester.test(configured_camera())

    assert result.connected
    assert result.error_code is None
    assert capture.released


async def test_connection_tester_returns_safe_error_code() -> None:
    capture = FakeCapture(False, False)
    tester = OpenCVCameraConnectionTester(
        RTSPConfig(),
        capture_factory=lambda _url, _config: capture,
    )

    result = await tester.test(configured_camera())

    assert not result.connected
    assert result.error_code == "OPEN_FAILED"
    assert "secret" not in repr(result)
