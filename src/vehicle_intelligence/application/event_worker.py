"""Idempotent event-consumer application service."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass, replace

from vehicle_intelligence.application.ports import (
    BrokerMessage,
    EventMessageConsumer,
    RealtimeEventPublisher,
    VehicleEventCodec,
    VehicleEventPostProcessor,
    VehicleEventRepository,
)
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import (
    EventBusError,
    EventContractError,
    PersistenceError,
    VehicleIntelligenceError,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EventWorkerStats:
    messages_read: int = 0
    messages_reclaimed: int = 0
    events_persisted: int = 0
    duplicate_events: int = 0
    invalid_messages: int = 0
    persistence_failures: int = 0
    policy_failures: int = 0
    matched_rules: int = 0
    actions_succeeded: int = 0
    actions_skipped: int = 0
    realtime_published: int = 0
    realtime_failures: int = 0


class VehicleEventWorker:
    def __init__(
        self,
        consumer: EventMessageConsumer,
        repository: VehicleEventRepository,
        codec: VehicleEventCodec,
        retry_delay_seconds: float = 1.0,
        post_processor: VehicleEventPostProcessor | None = None,
        realtime_publisher: RealtimeEventPublisher | None = None,
    ) -> None:
        if retry_delay_seconds <= 0:
            raise ValueError("event-worker retry delay must be positive")
        self._consumer = consumer
        self._repository = repository
        self._codec = codec
        self._retry_delay_seconds = retry_delay_seconds
        self._post_processor = post_processor
        self._realtime_publisher = realtime_publisher
        self._realtime_initialized = False
        self._stats = EventWorkerStats()

    @property
    def stats(self) -> EventWorkerStats:
        return replace(self._stats)

    async def initialize(self) -> None:
        await self._consumer.initialize()
        await self._repository.ensure_indexes()
        if self._post_processor is not None:
            await self._post_processor.initialize()
        await self._initialize_realtime()

    async def run(self, stop_event: asyncio.Event) -> EventWorkerStats:
        try:
            while not stop_event.is_set():
                try:
                    await self.initialize()
                    break
                except (EventBusError, PersistenceError):
                    logger.exception("event worker initialization failed; worker will retry")
                    await self._wait_or_stop(stop_event)
            while not stop_event.is_set():
                try:
                    await self.run_once()
                except EventBusError:
                    logger.exception("event bus unavailable; worker will retry")
                    await self._wait_or_stop(stop_event)
                    if not stop_event.is_set():
                        try:
                            await self._consumer.initialize()
                        except EventBusError:
                            logger.exception("event consumer recovery failed; worker will retry")
        finally:
            await self.close()
        return self.stats

    async def run_once(self) -> int:
        messages = await self._consumer.reclaim_stale()
        reclaimed = bool(messages)
        if not messages:
            messages = await self._consumer.read_new()
        if reclaimed:
            self._stats.messages_reclaimed += len(messages)
        else:
            self._stats.messages_read += len(messages)
        for message in messages:
            await self._process(message)
        return len(messages)

    async def close(self) -> None:
        try:
            if self._realtime_publisher is not None:
                await self._realtime_publisher.close()
        finally:
            try:
                if self._post_processor is not None:
                    await self._post_processor.close()
            finally:
                try:
                    await self._repository.close()
                finally:
                    await self._consumer.close()

    async def _process(self, message: BrokerMessage) -> None:
        try:
            event = self._codec.decode(message.payload)
        except EventContractError as exc:
            await self._consumer.dead_letter(message, str(exc))
            self._stats.invalid_messages += 1
            logger.warning(
                "invalid event moved to dead-letter stream",
                extra={"stream_message_id": message.message_id},
            )
            return

        try:
            created = await self._repository.save(event)
        except PersistenceError:
            self._stats.persistence_failures += 1
            logger.exception(
                "event persistence failed; message remains pending",
                extra={
                    "stream_message_id": message.message_id,
                    "event_id": event.id,
                    "camera_id": event.camera.id,
                    "track_id": event.track_id,
                },
            )
            return

        if created:
            self._stats.events_persisted += 1
            logger.info(
                "vehicle event persisted",
                extra={
                    "stream_message_id": message.message_id,
                    "event_id": event.id,
                    "camera_id": event.camera.id,
                    "track_id": event.track_id,
                },
            )
        else:
            self._stats.duplicate_events += 1
            logger.info(
                "duplicate vehicle event detected",
                extra={
                    "stream_message_id": message.message_id,
                    "event_id": event.id,
                },
            )

        if self._post_processor is not None:
            try:
                result = await self._post_processor.process(event)
            except VehicleIntelligenceError:
                self._stats.policy_failures += 1
                logger.exception(
                    "vehicle-event policy processing failed; message remains pending",
                    extra={
                        "stream_message_id": message.message_id,
                        "event_id": event.id,
                        "camera_id": event.camera.id,
                        "track_id": event.track_id,
                    },
                )
                return
            self._stats.matched_rules += result.matched_rules
            self._stats.actions_succeeded += result.actions_succeeded
            self._stats.actions_skipped += result.actions_skipped

        await self._publish_realtime(event)
        await self._consumer.acknowledge(message.message_id)

    async def _initialize_realtime(self) -> bool:
        if self._realtime_publisher is None or self._realtime_initialized:
            return self._realtime_initialized
        try:
            await self._realtime_publisher.initialize()
        except EventBusError:
            self._stats.realtime_failures += 1
            logger.exception("realtime publisher unavailable; durable processing continues")
            return False
        self._realtime_initialized = True
        return True

    async def _publish_realtime(self, event: VehicleEvent) -> None:
        if self._realtime_publisher is None:
            return
        if not await self._initialize_realtime():
            return
        try:
            await self._realtime_publisher.publish(event)
            self._stats.realtime_published += 1
        except EventBusError:
            self._realtime_initialized = False
            self._stats.realtime_failures += 1
            logger.exception(
                "realtime publish failed; event remains recoverable through REST",
                extra={"event_id": event.id},
            )

    async def _wait_or_stop(self, stop_event: asyncio.Event) -> None:
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=self._retry_delay_seconds)
