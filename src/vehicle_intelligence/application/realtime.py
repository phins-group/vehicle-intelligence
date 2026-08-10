"""Bounded in-process realtime fan-out and recoverable broker supervision."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from vehicle_intelligence.application.ports import RealtimeEventSubscriber
from vehicle_intelligence.config import RealtimeConfig
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import EventBusError, EventContractError

logger = logging.getLogger(__name__)
_CLOSED = object()


class RealtimeSubscriptionClosed(Exception):
    """The local subscription was closed during API shutdown or disconnect."""


class RealtimeSourceState(StrEnum):
    STARTING = "STARTING"
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class RealtimeGap:
    reason: str
    dropped_events: int = 0
    last_available_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class RealtimeDelivery:
    event: VehicleEvent | None = None
    gap: RealtimeGap | None = None

    def __post_init__(self) -> None:
        if (self.event is None) == (self.gap is None):
            raise ValueError("realtime delivery requires exactly one event or gap")


@dataclass(frozen=True, slots=True)
class RealtimeHubStats:
    subscribers: int = 0
    events_received: int = 0
    events_distributed: int = 0
    duplicate_events: int = 0
    client_events_dropped: int = 0
    last_event_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RealtimeServiceStats:
    source_state: RealtimeSourceState
    reconnect_count: int
    source_failures: int
    invalid_messages: int
    last_source_event_at: datetime | None
    hub: RealtimeHubStats


class RealtimeSubscription:
    def __init__(self, queue_size: int) -> None:
        self.id = f"rts_{uuid4().hex}"
        self._queue: asyncio.Queue[VehicleEvent | object] = asyncio.Queue(queue_size)
        self._pending_gap: RealtimeGap | None = None
        self._closed = False

    def offer(self, event: VehicleEvent) -> tuple[bool, int]:
        if self._closed:
            return False, 0
        dropped = 0
        if self._queue.full():
            self._queue.get_nowait()
            dropped = 1
            self.mark_gap("slow_consumer", dropped, event.id)
        self._queue.put_nowait(event)
        return True, dropped

    def mark_gap(
        self,
        reason: str,
        dropped_events: int,
        last_available_event_id: str | None,
    ) -> None:
        existing = self._pending_gap
        self._pending_gap = RealtimeGap(
            reason=reason if existing is None else existing.reason,
            dropped_events=dropped_events + (existing.dropped_events if existing else 0),
            last_available_event_id=last_available_event_id,
        )

    async def receive(self, timeout_seconds: float) -> RealtimeDelivery | None:
        if timeout_seconds <= 0:
            raise ValueError("realtime receive timeout must be positive")
        if self._pending_gap is not None:
            gap, self._pending_gap = self._pending_gap, None
            return RealtimeDelivery(gap=gap)
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
        except TimeoutError:
            return None
        if item is _CLOSED:
            raise RealtimeSubscriptionClosed
        return RealtimeDelivery(event=item)  # type: ignore[arg-type]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(_CLOSED)


class RealtimeHub:
    def __init__(self, config: RealtimeConfig) -> None:
        self._queue_size = config.client_queue_size
        self._replay_size = config.replay_size
        self._history: deque[VehicleEvent] = deque()
        self._history_ids: set[str] = set()
        self._subscriptions: dict[str, RealtimeSubscription] = {}
        self._stats = RealtimeHubStats()

    @property
    def stats(self) -> RealtimeHubStats:
        return replace(self._stats, subscribers=len(self._subscriptions))

    def subscribe(self, last_event_id: str | None = None) -> RealtimeSubscription:
        subscription = RealtimeSubscription(self._queue_size)
        self._subscriptions[subscription.id] = subscription
        if last_event_id is None:
            return subscription

        history = tuple(self._history)
        replay_from = next(
            (index for index, event in enumerate(history) if event.id == last_event_id),
            None,
        )
        if replay_from is None:
            subscription.mark_gap(
                "replay_unavailable",
                0,
                history[-1].id if history else None,
            )
            return subscription
        for event in history[replay_from + 1 :]:
            accepted, dropped = subscription.offer(event)
            if accepted and dropped:
                self._stats = replace(
                    self._stats,
                    client_events_dropped=self._stats.client_events_dropped + dropped,
                )
        return subscription

    def unsubscribe(self, subscription: RealtimeSubscription) -> None:
        current = self._subscriptions.pop(subscription.id, None)
        if current is not None:
            current.close()

    def publish(self, event: VehicleEvent) -> bool:
        now = datetime.now(UTC)
        self._stats = replace(
            self._stats,
            events_received=self._stats.events_received + 1,
            last_event_at=now,
        )
        if event.id in self._history_ids:
            self._stats = replace(
                self._stats,
                duplicate_events=self._stats.duplicate_events + 1,
            )
            return False

        if self._replay_size > 0:
            if len(self._history) >= self._replay_size:
                expired = self._history.popleft()
                self._history_ids.discard(expired.id)
            self._history.append(event)
            self._history_ids.add(event.id)

        distributed = 0
        dropped = 0
        for subscription in tuple(self._subscriptions.values()):
            accepted, client_dropped = subscription.offer(event)
            distributed += int(accepted)
            dropped += client_dropped
        self._stats = replace(
            self._stats,
            events_distributed=self._stats.events_distributed + distributed,
            client_events_dropped=self._stats.client_events_dropped + dropped,
        )
        return True

    def close(self) -> None:
        for subscription in tuple(self._subscriptions.values()):
            subscription.close()
        self._subscriptions.clear()


class RealtimeEventService:
    def __init__(
        self,
        config: RealtimeConfig,
        source: RealtimeEventSubscriber | None = None,
    ) -> None:
        self._config = config
        self._source = source
        self._hub = RealtimeHub(config)
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._source_state = RealtimeSourceState.STARTING
        self._reconnect_count = 0
        self._source_failures = 0
        self._invalid_messages = 0
        self._last_source_event_at: datetime | None = None

    @property
    def stats(self) -> RealtimeServiceStats:
        return RealtimeServiceStats(
            source_state=self._source_state,
            reconnect_count=self._reconnect_count,
            source_failures=self._source_failures,
            invalid_messages=self._invalid_messages,
            last_source_event_at=self._last_source_event_at,
            hub=self._hub.stats,
        )

    async def initialize(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        if self._source is None:
            self._source_state = RealtimeSourceState.ONLINE
            return
        self._task = asyncio.create_task(self._run_source(), name="realtime-event-source")
        await asyncio.sleep(0)

    def subscribe(self, last_event_id: str | None = None) -> RealtimeSubscription:
        return self._hub.subscribe(last_event_id)

    def unsubscribe(self, subscription: RealtimeSubscription) -> None:
        self._hub.unsubscribe(subscription)

    async def publish(self, event: VehicleEvent) -> bool:
        return self._hub.publish(event)

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
            self._hub.close()
            self._source_state = RealtimeSourceState.STOPPED

    async def _run_source(self) -> None:
        if self._source is None:
            return
        delay = self._config.reconnect_initial_seconds
        try:
            while not self._stop_event.is_set():
                try:
                    self._source_state = RealtimeSourceState.STARTING
                    await self._source.connect()
                    self._source_state = RealtimeSourceState.ONLINE
                    while not self._stop_event.is_set():
                        try:
                            event = await self._source.receive(
                                self._config.broker_poll_seconds
                            )
                        except EventContractError:
                            self._invalid_messages += 1
                            delay = self._config.reconnect_initial_seconds
                            logger.warning("invalid realtime event ignored")
                            continue
                        delay = self._config.reconnect_initial_seconds
                        if event is not None:
                            self._hub.publish(event)
                            self._last_source_event_at = datetime.now(UTC)
                except EventBusError:
                    self._source_failures += 1
                    self._source_state = RealtimeSourceState.OFFLINE
                    logger.exception("realtime event source unavailable; reconnecting")
                finally:
                    await self._source.disconnect()

                if self._stop_event.is_set():
                    break
                self._reconnect_count += 1
                await self._wait_or_stop(delay)
                delay = min(delay * 2, self._config.reconnect_max_seconds)
        finally:
            self._source_state = RealtimeSourceState.STOPPED

    async def _wait_or_stop(self, timeout: float) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(self._stop_event.wait(), timeout=timeout)
