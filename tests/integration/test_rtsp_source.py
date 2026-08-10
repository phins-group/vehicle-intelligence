from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np
import pytest
from pydantic import SecretStr

from vehicle_intelligence.application.ports import StreamHeartbeat
from vehicle_intelligence.config import RTSPConfig
from vehicle_intelligence.domain import CameraStatus
from vehicle_intelligence.exceptions import VideoSourceError
from vehicle_intelligence.infrastructure.vision.rtsp import OpenCVRTSPSource


class FakeCapture:
    def __init__(
        self,
        frames: list[np.ndarray],
        disconnect: threading.Event | None = None,
        fps: float = 6.0,
    ) -> None:
        self._frames = deque(frames)
        self._disconnect = disconnect or threading.Event()
        self._released = threading.Event()
        self._fps = fps

    def isOpened(self) -> bool:
        return not self._released.is_set()

    def read(self):
        if self._frames:
            return True, self._frames.popleft()
        while not self._disconnect.is_set() and not self._released.is_set():
            time.sleep(0.001)
        return False, None

    def get(self, property_id: int) -> float:
        return self._fps if property_id == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self._released.set()


class ExplodingCapture(FakeCapture):
    def get(self, property_id: int) -> float:
        del property_id
        raise RuntimeError("unexpected decoder failure")


def rtsp_config(queue_size: int = 3) -> RTSPConfig:
    return RTSPConfig(
        queue_size=queue_size,
        reconnect_initial_seconds=0.01,
        reconnect_max_seconds=0.02,
        open_timeout_ms=100,
        read_timeout_ms=100,
        consumer_wait_seconds=0.01,
        shutdown_join_seconds=1.0,
    )


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


def test_rtsp_source_keeps_latest_frame_and_counts_drops() -> None:
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(5)]
    capture = FakeCapture(frames)
    source = OpenCVRTSPSource(
        SecretStr("rtsp://user:secret@example.invalid/stream"),
        "gate-01",
        6.0,
        rtsp_config(queue_size=3),
        capture_factory=lambda _url, _config: capture,
    )
    source.start()
    wait_until(lambda: source.health.dropped_frames >= 2)

    iterator = source.frames()
    latest = next(iterator)

    assert latest.frame_id == 4
    assert int(latest.image[0, 0, 0]) == 4
    assert source.health.dropped_frames == 4
    assert source.health.queue_size == 0
    source.close()
    assert source.health.status is CameraStatus.STOPPED


def test_rtsp_reconnect_increments_epoch_without_exposing_credentials(caplog) -> None:
    first_disconnect = threading.Event()
    first = FakeCapture([np.zeros((8, 8, 3), dtype=np.uint8)], first_disconnect)
    second = FakeCapture([np.ones((8, 8, 3), dtype=np.uint8)])
    captures = deque((first, second))

    def factory(_url, _config):
        return captures.popleft() if captures else second

    source = OpenCVRTSPSource(
        SecretStr("rtsp://admin:top-secret@example.invalid/live"),
        "gate-02",
        6.0,
        rtsp_config(queue_size=3),
        capture_factory=factory,
    )
    iterator = source.frames()
    first_frame = next(iterator)
    assert first_frame.stream_epoch == 0

    first_disconnect.set()
    wait_until(lambda: source.health.stream_epoch == 1 and source.health.queue_size > 0)
    second_frame = next(iterator)

    assert second_frame.stream_epoch == 1
    assert source.health.reconnect_count == 1
    assert source.health.connection_failures >= 1
    assert "top-secret" not in source.source_id
    assert "top-secret" not in caplog.text
    source.close()


def test_rtsp_source_emits_image_free_heartbeat_while_idle() -> None:
    source = OpenCVRTSPSource(
        SecretStr("rtsp://camera.invalid/idle"),
        "gate-idle",
        6.0,
        rtsp_config(),
        capture_factory=lambda _url, _config: FakeCapture([]),
    )

    source_item = next(source.frames())

    assert isinstance(source_item, StreamHeartbeat)
    assert source_item.stream_epoch == 0
    source.close()


def test_unexpected_decoder_failure_stops_source_without_leaking_secret(caplog) -> None:
    source = OpenCVRTSPSource(
        SecretStr("rtsp://operator:decoder-secret@camera.invalid/live"),
        "gate-crash",
        6.0,
        rtsp_config(),
        capture_factory=lambda _url, _config: ExplodingCapture([]),
    )

    with pytest.raises(VideoSourceError, match="gate-crash"):
        next(source.frames())

    assert source.health.status is CameraStatus.STOPPED
    assert "camera_decoder_crashed" in caplog.text
    assert "decoder-secret" not in caplog.text
    source.close()
