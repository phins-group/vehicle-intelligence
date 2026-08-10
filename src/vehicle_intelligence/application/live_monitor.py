"""Bounded latest-frame live-monitor state and broker recovery."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from vehicle_intelligence.application.ports import LiveFrameSubscriber
from vehicle_intelligence.config import LiveMonitorConfig
from vehicle_intelligence.domain import LiveFramePacket
from vehicle_intelligence.exceptions import EventBusError, EventContractError

logger = logging.getLogger(__name__)


class LiveMonitorSourceState(StrEnum):
    STARTING = "STARTING"
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    STOPPED = "STOPPED"


class LiveCameraStatus(StrEnum):
    DISABLED = "DISABLED"
    WAITING = "WAITING"
    LIVE = "LIVE"
    STALE = "STALE"
    OFFLINE = "OFFLINE"


@dataclass(frozen=True, slots=True)
class BufferedLiveFrame:
    sequence: int
    received_at: datetime
    packet: LiveFramePacket


@dataclass(frozen=True, slots=True)
class LiveCameraSnapshot:
    camera_id: str
    status: LiveCameraStatus
    source_state: LiveMonitorSourceState
    latest: BufferedLiveFrame | None


@dataclass(frozen=True, slots=True)
class LiveMonitorStats:
    source_state: LiveMonitorSourceState
    cameras_buffered: int
    frames_received: int
    frames_evicted: int
    reconnect_count: int
    source_failures: int
    invalid_messages: int
    last_frame_at: datetime | None


class LiveMonitorService:
    def __init__(
        self,
        config: LiveMonitorConfig,
        source: LiveFrameSubscriber | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._source = source
        self._clock = clock
        self._buffers: OrderedDict[str, deque[BufferedLiveFrame]] = OrderedDict()
        self._sequence = 0
        self._frames_received = 0
        self._frames_evicted = 0
        self._reconnect_count = 0
        self._source_failures = 0
        self._invalid_messages = 0
        self._last_frame_at: datetime | None = None
        self._source_state = LiveMonitorSourceState.STARTING
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def stats(self) -> LiveMonitorStats:
        return LiveMonitorStats(
            source_state=self._source_state,
            cameras_buffered=len(self._buffers),
            frames_received=self._frames_received,
            frames_evicted=self._frames_evicted,
            reconnect_count=self._reconnect_count,
            source_failures=self._source_failures,
            invalid_messages=self._invalid_messages,
            last_frame_at=self._last_frame_at,
        )

    async def initialize(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        if self._source is None:
            self._source_state = LiveMonitorSourceState.ONLINE
            return
        self._task = asyncio.create_task(self._run_source(), name="live-monitor-source")
        await asyncio.sleep(0)

    def ingest(self, packet: LiveFramePacket) -> BufferedLiveFrame:
        now = self._now()
        camera_id = packet.metadata.camera_id
        frames = self._buffers.get(camera_id)
        if frames is None:
            if len(self._buffers) >= self._config.maximum_cameras:
                _evicted_camera, evicted = self._buffers.popitem(last=False)
                self._frames_evicted += len(evicted)
            frames = deque(maxlen=self._config.frame_buffer_size)
            self._buffers[camera_id] = frames
        else:
            self._buffers.move_to_end(camera_id)
        if len(frames) == frames.maxlen:
            self._frames_evicted += 1
        self._sequence += 1
        buffered = BufferedLiveFrame(self._sequence, now, packet)
        frames.append(buffered)
        self._frames_received += 1
        self._last_frame_at = now
        return buffered

    def snapshot(self, camera_id: str, *, enabled: bool = True) -> LiveCameraSnapshot:
        if not enabled:
            return LiveCameraSnapshot(
                camera_id,
                LiveCameraStatus.DISABLED,
                self._source_state,
                None,
            )
        frames = self._buffers.get(camera_id)
        latest = frames[-1] if frames else None
        if latest is None:
            status = (
                LiveCameraStatus.OFFLINE
                if self._source_state in {
                    LiveMonitorSourceState.OFFLINE,
                    LiveMonitorSourceState.STOPPED,
                }
                else LiveCameraStatus.WAITING
            )
        else:
            age = (self._now() - latest.received_at).total_seconds()
            status = (
                LiveCameraStatus.STALE
                if age > self._config.stale_after_seconds
                else LiveCameraStatus.LIVE
            )
        return LiveCameraSnapshot(camera_id, status, self._source_state, latest)

    def frame(self, camera_id: str, sequence: int) -> BufferedLiveFrame | None:
        frames = self._buffers.get(camera_id)
        if frames is None:
            return None
        return next((item for item in frames if item.sequence == sequence), None)

    async def close(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            timeout = self._config.broker_poll_seconds + 1
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
            except TimeoutError:
                self._task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._task
            self._task = None
        try:
            if self._source is not None:
                await self._source.close()
        finally:
            self._buffers.clear()
            self._source_state = LiveMonitorSourceState.STOPPED

    async def _run_source(self) -> None:
        if self._source is None:
            return
        delay = self._config.reconnect_initial_seconds
        try:
            while not self._stop_event.is_set():
                try:
                    self._source_state = LiveMonitorSourceState.STARTING
                    await self._source.connect()
                    self._source_state = LiveMonitorSourceState.ONLINE
                    while not self._stop_event.is_set():
                        try:
                            packet = await self._source.receive(
                                self._config.broker_poll_seconds
                            )
                        except EventContractError:
                            self._invalid_messages += 1
                            logger.warning("invalid live-preview packet ignored")
                            continue
                        delay = self._config.reconnect_initial_seconds
                        if packet is not None:
                            self.ingest(packet)
                except EventBusError:
                    self._source_failures += 1
                    self._source_state = LiveMonitorSourceState.OFFLINE
                    logger.warning("live-preview source unavailable", exc_info=True)
                finally:
                    await self._source.disconnect()
                if self._stop_event.is_set():
                    break
                self._reconnect_count += 1
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                delay = min(delay * 2, self._config.reconnect_max_seconds)
        finally:
            self._source_state = LiveMonitorSourceState.STOPPED

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("live monitor clock must be timezone-aware")
        return value.astimezone(UTC)
