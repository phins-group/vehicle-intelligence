import asyncio
from dataclasses import replace

from vehicle_intelligence.application.event_worker import VehicleEventWorker
from vehicle_intelligence.application.ports import BrokerMessage, EventPolicyResult
from vehicle_intelligence.domain import CameraSnapshot
from vehicle_intelligence.exceptions import ActionExecutionError, EventBusError, PersistenceError
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)


class FakeConsumer:
    def __init__(self) -> None:
        self.new: list[BrokerMessage] = []
        self.reclaimed: list[BrokerMessage] = []
        self.acknowledged: list[str] = []
        self.acknowledged_batches: list[tuple[str, ...]] = []
        self.dead_letters: list[tuple[BrokerMessage, str]] = []
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def read_new(self) -> list[BrokerMessage]:
        messages, self.new = self.new, []
        return messages

    async def reclaim_stale(self) -> list[BrokerMessage]:
        messages, self.reclaimed = self.reclaimed, []
        return messages

    async def acknowledge(self, message_id: str) -> None:
        await self.acknowledge_many((message_id,))

    async def acknowledge_many(self, message_ids: tuple[str, ...]) -> None:
        self.acknowledged_batches.append(message_ids)
        self.acknowledged.extend(message_ids)

    async def dead_letter(self, message: BrokerMessage, reason: str) -> None:
        self.dead_letters.append((message, reason))

    async def close(self) -> None:
        self.closed = True


class FlakyRepository(InMemoryVehicleEventRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_save = True

    async def save(self, event) -> bool:
        if self.fail_next_save:
            self.fail_next_save = False
            raise PersistenceError("temporary MongoDB failure")
        return await super().save(event)


class FlakyPolicyProcessor:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def initialize(self) -> None:
        return None

    async def process(self, _event) -> EventPolicyResult:
        self.calls += 1
        if self.calls == 1:
            raise ActionExecutionError("temporary action failure")
        return EventPolicyResult(matched_rules=1, actions_succeeded=1)

    async def close(self) -> None:
        self.closed = True


class FakeRealtimePublisher:
    def __init__(self, fail_publish: bool = False) -> None:
        self.fail_publish = fail_publish
        self.events = []
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def publish(self, event) -> None:
        if self.fail_publish:
            raise EventBusError("realtime unavailable")
        self.events.append(event)

    async def close(self) -> None:
        self.closed = True


class ConcurrentRepository(InMemoryVehicleEventRepository):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.peak = 0
        self.camera_order: dict[str, list[str]] = {}

    async def save(self, event) -> bool:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.camera_order.setdefault(event.camera.id, []).append(event.id)
        try:
            await asyncio.sleep(0.01)
            return await super().save(event)
        finally:
            self.active -= 1


async def test_worker_retries_pending_message_and_acknowledges_duplicate(sample_event) -> None:
    codec = JsonEventEnvelopeCodec()
    message = BrokerMessage("1-0", codec.encode(sample_event))
    consumer = FakeConsumer()
    consumer.new.append(message)
    repository = FlakyRepository()
    now = [0.0]
    worker = VehicleEventWorker(consumer, repository, codec, monotonic_clock=lambda: now[0])
    await worker.initialize()

    assert await worker.run_once() == 1
    assert consumer.acknowledged == []
    assert worker.stats.persistence_failures == 1

    consumer.reclaimed.append(message)
    now[0] = 5.0
    assert await worker.run_once() == 1
    assert consumer.acknowledged == ["1-0"]
    assert worker.stats.messages_reclaimed == 1
    assert worker.stats.events_persisted == 1

    consumer.new.append(BrokerMessage("2-0", codec.encode(sample_event)))
    assert await worker.run_once() == 1
    assert consumer.acknowledged == ["1-0", "2-0"]
    assert worker.stats.duplicate_events == 1
    await worker.close()
    assert consumer.initialized
    assert consumer.closed


async def test_worker_dead_letters_invalid_contract_without_persistence() -> None:
    consumer = FakeConsumer()
    message = BrokerMessage("3-0", "not-json")
    consumer.new.append(message)
    repository = InMemoryVehicleEventRepository()
    worker = VehicleEventWorker(consumer, repository, JsonEventEnvelopeCodec())
    await worker.initialize()

    assert await worker.run_once() == 1

    assert consumer.acknowledged == []
    assert len(consumer.dead_letters) == 1
    assert consumer.dead_letters[0][0] == message
    assert worker.stats.invalid_messages == 1
    await worker.close()


async def test_worker_reprocesses_persisted_event_until_policy_actions_complete(
    sample_event,
) -> None:
    codec = JsonEventEnvelopeCodec()
    message = BrokerMessage("4-0", codec.encode(sample_event))
    consumer = FakeConsumer()
    consumer.new.append(message)
    repository = InMemoryVehicleEventRepository()
    processor = FlakyPolicyProcessor()
    now = [0.0]
    worker = VehicleEventWorker(
        consumer,
        repository,
        codec,
        post_processor=processor,
        monotonic_clock=lambda: now[0],
    )
    await worker.initialize()

    assert await worker.run_once() == 1
    assert consumer.acknowledged == []
    assert worker.stats.events_persisted == 1
    assert worker.stats.policy_failures == 1

    consumer.reclaimed.append(message)
    now[0] = 5.0
    assert await worker.run_once() == 1
    assert consumer.acknowledged == ["4-0"]
    assert worker.stats.duplicate_events == 1
    assert worker.stats.matched_rules == 1
    assert worker.stats.actions_succeeded == 1
    await worker.close()
    assert processor.closed


async def test_worker_publishes_realtime_after_durable_processing(sample_event) -> None:
    codec = JsonEventEnvelopeCodec()
    consumer = FakeConsumer()
    consumer.new.append(BrokerMessage("5-0", codec.encode(sample_event)))
    publisher = FakeRealtimePublisher()
    worker = VehicleEventWorker(
        consumer,
        InMemoryVehicleEventRepository(),
        codec,
        realtime_publisher=publisher,
    )
    await worker.initialize()

    assert await worker.run_once() == 1
    assert publisher.events == [sample_event]
    assert consumer.acknowledged == ["5-0"]
    assert worker.stats.realtime_published == 1
    await worker.close()
    assert publisher.closed


async def test_realtime_failure_does_not_block_durable_ack(sample_event) -> None:
    codec = JsonEventEnvelopeCodec()
    consumer = FakeConsumer()
    consumer.new.append(BrokerMessage("6-0", codec.encode(sample_event)))
    publisher = FakeRealtimePublisher(fail_publish=True)
    repository = InMemoryVehicleEventRepository()
    worker = VehicleEventWorker(
        consumer,
        repository,
        codec,
        realtime_publisher=publisher,
    )
    await worker.initialize()

    assert await worker.run_once() == 1
    assert await repository.get(sample_event.id) == sample_event
    assert consumer.acknowledged == ["6-0"]
    assert worker.stats.realtime_failures == 1
    assert worker.stats.realtime_published == 0
    await worker.close()


async def test_worker_batches_ack_with_bounded_per_camera_processing(sample_event) -> None:
    codec = JsonEventEnvelopeCodec()
    consumer = FakeConsumer()
    expected: dict[str, list[str]] = {}
    for sequence in range(2):
        for camera_number in range(4):
            camera_id = f"gate-{camera_number}"
            event_id = f"evt-{camera_number}-{sequence}"
            event = replace(
                sample_event,
                id=event_id,
                camera=CameraSnapshot(id=camera_id, name=camera_id, zone="test"),
            )
            expected.setdefault(camera_id, []).append(event_id)
            consumer.new.append(BrokerMessage(f"{sequence}-{camera_number}", codec.encode(event)))
    repository = ConcurrentRepository()
    worker = VehicleEventWorker(
        consumer,
        repository,
        codec,
        maximum_concurrency=2,
    )
    await worker.initialize()

    assert await worker.run_once() == 8

    assert repository.peak == 2
    assert repository.camera_order == expected
    assert len(consumer.acknowledged_batches) == 1
    assert set(consumer.acknowledged_batches[0]) == {
        f"{sequence}-{camera_number}" for sequence in range(2) for camera_number in range(4)
    }
    await worker.close()
