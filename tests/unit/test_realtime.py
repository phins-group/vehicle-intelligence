import asyncio
from dataclasses import replace

from vehicle_intelligence.application.realtime import (
    RealtimeEventService,
    RealtimeHub,
    RealtimeSourceState,
)
from vehicle_intelligence.config import RealtimeConfig
from vehicle_intelligence.exceptions import EventBusError, EventContractError


def event_with_id(sample_event, event_id: str):
    return replace(sample_event, id=event_id)


async def test_hub_bounds_slow_client_queue_and_reports_gap(sample_event) -> None:
    hub = RealtimeHub(RealtimeConfig(client_queue_size=2, replay_size=3))
    subscription = hub.subscribe()

    assert hub.publish(event_with_id(sample_event, "evt-1"))
    assert hub.publish(event_with_id(sample_event, "evt-2"))
    assert hub.publish(event_with_id(sample_event, "evt-3"))
    assert not hub.publish(event_with_id(sample_event, "evt-3"))

    gap = await subscription.receive(0.1)
    second = await subscription.receive(0.1)
    third = await subscription.receive(0.1)

    assert gap is not None and gap.gap is not None
    assert gap.gap.reason == "slow_consumer"
    assert gap.gap.dropped_events == 1
    assert second is not None and second.event is not None
    assert second.event.id == "evt-2"
    assert third is not None and third.event is not None
    assert third.event.id == "evt-3"
    assert hub.stats.client_events_dropped == 1
    assert hub.stats.duplicate_events == 1


async def test_hub_replays_after_known_id_and_signals_missing_history(sample_event) -> None:
    hub = RealtimeHub(RealtimeConfig(client_queue_size=4, replay_size=4))
    hub.publish(event_with_id(sample_event, "evt-replay-1"))
    hub.publish(event_with_id(sample_event, "evt-replay-2"))

    replay = hub.subscribe("evt-replay-1")
    delivery = await replay.receive(0.1)
    assert delivery is not None and delivery.event is not None
    assert delivery.event.id == "evt-replay-2"

    unavailable = hub.subscribe("evt-no-longer-buffered")
    gap = await unavailable.receive(0.1)
    assert gap is not None and gap.gap is not None
    assert gap.gap.reason == "replay_unavailable"
    assert gap.gap.last_available_event_id == "evt-replay-2"


class RecoveringSource:
    def __init__(self, event) -> None:
        self.event = event
        self.connect_calls = 0
        self.receive_calls = 0
        self.closed = False

    async def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_calls == 1:
            raise EventBusError("temporary Redis outage")

    async def receive(self, timeout_seconds: float):
        self.receive_calls += 1
        if self.receive_calls == 1:
            raise EventContractError("bad realtime payload")
        if self.receive_calls == 2:
            return self.event
        await asyncio.sleep(timeout_seconds)
        return None

    async def disconnect(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


async def test_realtime_service_recovers_source_and_ignores_bad_contract(sample_event) -> None:
    source = RecoveringSource(sample_event)
    config = RealtimeConfig(
        enabled=True,
        broker_poll_seconds=0.01,
        reconnect_initial_seconds=0.01,
        reconnect_max_seconds=0.02,
    )
    service = RealtimeEventService(config, source)
    subscription = service.subscribe()
    await service.initialize()

    delivery = await subscription.receive(1)
    stats = service.stats

    assert delivery is not None and delivery.event == sample_event
    assert stats.source_state is RealtimeSourceState.ONLINE
    assert stats.reconnect_count == 1
    assert stats.source_failures == 1
    assert stats.invalid_messages == 1
    await service.close()
    assert source.closed
    assert service.stats.source_state is RealtimeSourceState.STOPPED
