import asyncio
import os
import uuid

import pytest

from vehicle_intelligence.application.realtime import (
    RealtimeEventService,
    RealtimeSourceState,
)
from vehicle_intelligence.config import RealtimeConfig, RedisConfig
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.realtime_redis import (
    RedisRealtimeEventPublisher,
    RedisRealtimeEventSubscriber,
)


@pytest.mark.skipif(not os.getenv("TEST_REDIS_URL"), reason="TEST_REDIS_URL is not configured")
async def test_redis_realtime_pubsub_fans_out_to_bounded_api_hub(sample_event) -> None:
    suffix = uuid.uuid4().hex
    redis_config = RedisConfig(url=os.environ["TEST_REDIS_URL"])
    realtime_config = RealtimeConfig(
        enabled=True,
        redis_channel=f"vehicle.events.realtime.test.{suffix}",
        broker_poll_seconds=0.05,
        reconnect_initial_seconds=0.05,
        reconnect_max_seconds=0.1,
    )
    codec = JsonEventEnvelopeCodec()
    publisher = RedisRealtimeEventPublisher(redis_config, realtime_config, codec)
    source = RedisRealtimeEventSubscriber(redis_config, realtime_config, codec)
    service = RealtimeEventService(realtime_config, source)
    first = service.subscribe()
    second = service.subscribe()
    try:
        await service.initialize()
        for _ in range(40):
            if service.stats.source_state is RealtimeSourceState.ONLINE:
                break
            await asyncio.sleep(0.025)
        assert service.stats.source_state is RealtimeSourceState.ONLINE

        await publisher.initialize()
        await publisher.publish(sample_event)
        first_delivery, second_delivery = await asyncio.gather(
            first.receive(2),
            second.receive(2),
        )

        assert first_delivery is not None and first_delivery.event == sample_event
        assert second_delivery is not None and second_delivery.event == sample_event
        assert service.stats.hub.events_distributed == 2
        assert service.stats.hub.subscribers == 2
    finally:
        await publisher.close()
        await service.close()


@pytest.mark.skipif(not os.getenv("TEST_REDIS_URL"), reason="TEST_REDIS_URL is not configured")
async def test_realtime_service_reconnects_after_live_subscription_disconnect(
    sample_event,
) -> None:
    suffix = uuid.uuid4().hex
    redis_config = RedisConfig(url=os.environ["TEST_REDIS_URL"])
    realtime_config = RealtimeConfig(
        enabled=True,
        redis_channel=f"vehicle.events.realtime.reconnect.{suffix}",
        broker_poll_seconds=0.05,
        reconnect_initial_seconds=0.05,
        reconnect_max_seconds=0.1,
    )
    codec = JsonEventEnvelopeCodec()
    publisher = RedisRealtimeEventPublisher(redis_config, realtime_config, codec)
    source = RedisRealtimeEventSubscriber(redis_config, realtime_config, codec)
    service = RealtimeEventService(realtime_config, source)
    subscription = service.subscribe()
    try:
        await service.initialize()
        await publisher.initialize()
        for _ in range(80):
            if service.stats.source_state is RealtimeSourceState.ONLINE:
                break
            await asyncio.sleep(0.025)
        assert service.stats.source_state is RealtimeSourceState.ONLINE

        await source.disconnect()
        for _ in range(120):
            if (
                service.stats.reconnect_count >= 1
                and service.stats.source_state is RealtimeSourceState.ONLINE
            ):
                break
            await asyncio.sleep(0.025)
        assert service.stats.reconnect_count >= 1
        assert service.stats.source_state is RealtimeSourceState.ONLINE

        await publisher.publish(sample_event)
        delivery = await subscription.receive(2)
        assert delivery is not None and delivery.event == sample_event
    finally:
        await publisher.close()
        await service.close()
