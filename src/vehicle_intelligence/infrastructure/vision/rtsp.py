"""Bounded latest-frame RTSP source with reconnect and credential-safe health."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import SecretStr

from vehicle_intelligence.application.ports import StreamHeartbeat
from vehicle_intelligence.config import RTSPConfig
from vehicle_intelligence.domain import CameraHealth, CameraStatus, VideoFrame
from vehicle_intelligence.exceptions import VideoSourceError

logger = logging.getLogger(__name__)


class Capture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def get(self, property_id: int) -> float: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[str, RTSPConfig], Capture]


def open_opencv_capture(url: str, config: RTSPConfig) -> Capture:
    parameters: list[int] = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        parameters.extend((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.open_timeout_ms))
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        parameters.extend((cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.read_timeout_ms))
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, parameters)


class OpenCVRTSPSource:
    def __init__(
        self,
        url: SecretStr,
        camera_id: str,
        fps_limit: float,
        config: RTSPConfig,
        capture_factory: CaptureFactory = open_opencv_capture,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if fps_limit <= 0:
            raise ValueError("RTSP fps_limit must be positive")
        self._url = url
        self._camera_id = camera_id
        self._fps_limit = fps_limit
        self._config = config
        self._capture_factory = capture_factory
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic_clock
        self._source_id = f"rtsp-{self._wall_clock():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self._queue: deque[VideoFrame] = deque()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Capture | None = None
        self._status = CameraStatus.CONNECTING
        self._source_fps = fps_limit
        self._decode_fps = 0.0
        self._decoded_frames = 0
        self._dropped_frames = 0
        self._reconnect_count = 0
        self._connection_failures = 0
        self._stream_epoch = 0
        self._last_frame_at: datetime | None = None
        self._updated_at = self._wall_clock()
        self._frame_id = 0
        self._terminal_error: Exception | None = None

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_fps(self) -> float:
        with self._condition:
            return self._source_fps

    @property
    def decoded_frames(self) -> int:
        with self._condition:
            return self._decoded_frames

    @property
    def health(self) -> CameraHealth:
        with self._condition:
            return CameraHealth(
                camera_id=self._camera_id,
                status=self._status,
                source_fps=self._source_fps,
                decode_fps=self._decode_fps,
                queue_size=len(self._queue),
                dropped_frames=self._dropped_frames,
                reconnect_count=self._reconnect_count,
                connection_failures=self._connection_failures,
                stream_epoch=self._stream_epoch,
                last_frame_at=self._last_frame_at,
                updated_at=self._updated_at,
                decoded_frames=self._decoded_frames,
            )

    def frames(self) -> Iterator[VideoFrame | StreamHeartbeat]:
        self.start()
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: bool(self._queue) or self._stop.is_set(),
                    timeout=self._config.consumer_wait_seconds,
                )
                if self._stop.is_set():
                    if self._terminal_error is not None:
                        raise VideoSourceError(
                            f"RTSP decoder failed for camera {self._camera_id}"
                        ) from self._terminal_error
                    return
                if self._queue:
                    source_item: VideoFrame | StreamHeartbeat = self._queue.pop()
                    stale = len(self._queue)
                    if stale:
                        self._queue.clear()
                        self._dropped_frames += stale
                        self._updated_at = self._wall_clock()
                else:
                    source_item = StreamHeartbeat(
                        timestamp=self._wall_clock(),
                        stream_epoch=self._stream_epoch,
                    )
            yield source_item

    def close(self) -> None:
        self.request_stop()
        with self._condition:
            capture = self._capture
        if capture is not None:
            capture.release()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._config.shutdown_join_seconds)
        with self._condition:
            self._capture = None
            self._status = CameraStatus.STOPPED
            self._updated_at = self._wall_clock()
            self._condition.notify_all()

    def request_stop(self) -> None:
        """Request a non-blocking stop; safe to call from a signal handler."""

        self._stop.set()
        with self._condition:
            self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run_decoder,
                name=f"rtsp-decoder-{self._camera_id}",
                daemon=True,
            )
            self._thread.start()

    def _run_decoder(self) -> None:
        try:
            self._decode_loop()
        except Exception as exc:
            with self._condition:
                self._terminal_error = exc
            logger.exception(
                "camera_decoder_crashed",
                extra={"camera_id": self._camera_id},
            )
        finally:
            with self._condition:
                capture = self._capture
                self._capture = None
            if capture is not None:
                capture.release()
            self._stop.set()
            self._set_status(CameraStatus.STOPPED)

    def _decode_loop(self) -> None:
        backoff = self._config.reconnect_initial_seconds
        connected_once = False
        while not self._stop.is_set():
            self._set_status(CameraStatus.CONNECTING)
            capture = self._open_capture()
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                self._connection_lost()
                backoff = self._wait_before_reconnect(backoff)
                continue
            with self._condition:
                self._capture = capture
                if connected_once:
                    self._stream_epoch += 1
                    self._reconnect_count += 1
                connected_once = True
                reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
                self._source_fps = reported_fps if reported_fps > 0 else self._fps_limit
                self._status = CameraStatus.ONLINE
                self._updated_at = self._wall_clock()
            logger.info(
                "camera_online",
                extra={"camera_id": self._camera_id, "stream_epoch": self._stream_epoch},
            )
            backoff = self._config.reconnect_initial_seconds
            self._read_connection(capture)
            capture.release()
            with self._condition:
                if self._capture is capture:
                    self._capture = None
            if not self._stop.is_set():
                self._connection_lost()
                backoff = self._wait_before_reconnect(backoff)

    def _open_capture(self) -> Capture | None:
        try:
            return self._capture_factory(self._url.get_secret_value(), self._config)
        except Exception:
            logger.warning(
                "camera_connection_error",
                extra={"camera_id": self._camera_id},
            )
            return None

    def _read_connection(self, capture: Capture) -> None:
        decoded_in_epoch = 0
        next_sample_seconds = 0.0
        rate_started = self._monotonic()
        rate_frames = 0
        while not self._stop.is_set():
            try:
                success, image = capture.read()
            except Exception:
                success, image = False, None
            if not success:
                return
            if image is None or image.size == 0:
                continue
            timestamp = self._wall_clock()
            monotonic_now = self._monotonic()
            decoded_in_epoch += 1
            rate_frames += 1
            self._update_decode_rate(monotonic_now, rate_started, rate_frames)
            with self._condition:
                source_fps = self._source_fps
                stream_epoch = self._stream_epoch
                frame_id = self._frame_id
                self._frame_id += 1
                self._decoded_frames += 1
                self._last_frame_at = timestamp
                self._updated_at = timestamp
            media_seconds = (decoded_in_epoch - 1) / max(source_fps, 1e-9)
            if media_seconds + (0.5 / max(source_fps, 1e-9)) < next_sample_seconds:
                continue
            next_sample_seconds += 1.0 / self._fps_limit
            frame = VideoFrame(
                camera_id=self._camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                image=np.ascontiguousarray(image),
                stream_epoch=stream_epoch,
            )
            with self._condition:
                if len(self._queue) >= self._config.queue_size:
                    self._queue.popleft()
                    self._dropped_frames += 1
                self._queue.append(frame)
                self._condition.notify()
            if monotonic_now - rate_started >= 1.0:
                rate_started = monotonic_now
                rate_frames = 0

    def _update_decode_rate(self, now: float, started: float, frames: int) -> None:
        elapsed = now - started
        if elapsed < 0.1:
            return
        with self._condition:
            self._decode_fps = frames / elapsed

    def _connection_lost(self) -> None:
        with self._condition:
            self._status = CameraStatus.OFFLINE
            self._connection_failures += 1
            self._updated_at = self._wall_clock()
        logger.warning(
            "camera_offline",
            extra={
                "camera_id": self._camera_id,
                "connection_failures": self._connection_failures,
            },
        )

    def _wait_before_reconnect(self, delay: float) -> float:
        self._stop.wait(delay)
        return min(delay * 2, self._config.reconnect_max_seconds)

    def _set_status(self, status: CameraStatus) -> None:
        with self._condition:
            self._status = status
            self._updated_at = self._wall_clock()
            self._condition.notify_all()
