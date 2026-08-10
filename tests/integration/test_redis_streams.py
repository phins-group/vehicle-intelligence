import asyncio
import os
import uuid
from dataclasses import replace

import pytest
from redis.asyncio import Redis

from vehicle_intelligence.application.event_worker import VehicleEventWorker
from vehicle_intelligence.config import MongoConfig, RedisConfig
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.redis_streams import (
    EVENT_PAYLOAD_FIELD,
    RedisStreamEventConsumer,
    RedisStreamEventPublisher,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository


@pytest.mark.skipif(not os.getenv("TEST_REDIS_URL"), reason="TEST_REDIS_URL is not configured")
async def test_redis_stream_publish_ack_reclaim_and_dead_letter(sample_event) -> None:
    suffix = uuid.uuid4().hex
    stream = f"vehicle.events.test.{suffix}"
    dead_letter = f"vehicle.events.dlq.test.{suffix}"
    config = RedisConfig(
        url=os.environ["TEST_REDIS_URL"],
        stream=stream,
        dead_letter_stream=dead_letter,
        consumer_group=f"event-processors-{suffix}",
        max_length=100,
        dead_letter_max_length=10,
        batch_size=10,
        block_ms=50,
        claim_idle_ms=1000,
    )
    codec = JsonEventEnvelopeCodec()
    publisher = RedisStreamEventPublisher(config, codec)
    first = RedisStreamEventConsumer(config, "consumer-a")
    second = RedisStreamEventConsumer(config, "consumer-b")
    admin = Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    try:
        await first.initialize()
        await publisher.initialize()
        assert await publisher.publish(sample_event)

        messages = await first.read_new()
        assert len(messages) == 1
        assert codec.decode(messages[0].payload) == sample_event
        await first.acknowledge(messages[0].message_id)
        assert await admin.xlen(stream) == 0

        assert await publisher.publish(sample_event)
        pending = await first.read_new()
        assert len(pending) == 1
        await asyncio.sleep(1.05)
        await second.initialize()
        reclaimed = await second.reclaim_stale()
        assert [message.message_id for message in reclaimed] == [pending[0].message_id]
        await second.acknowledge(reclaimed[0].message_id)

        await admin.xadd(stream, {EVENT_PAYLOAD_FIELD: "not-json"})
        invalid = await second.read_new()
        assert len(invalid) == 1
        await second.dead_letter(invalid[0], "invalid contract")
        assert await admin.xlen(dead_letter) == 1
        pending_summary = await admin.xpending(stream, config.consumer_group)
        assert pending_summary["pending"] == 0
    finally:
        await admin.delete(stream, dead_letter)
        await admin.aclose()
        await first.close()
        await second.close()
        await publisher.close()


@pytest.mark.skipif(
    not os.getenv("TEST_REDIS_URL") or not os.getenv("TEST_MONGODB_URI"),
    reason="TEST_REDIS_URL and TEST_MONGODB_URI are not configured",
)
async def test_event_worker_persists_duplicate_delivery_once(sample_event) -> None:
    suffix = uuid.uuid4().hex
    stream = f"vehicle.events.worker-test.{suffix}"
    dead_letter = f"vehicle.events.worker-test.dlq.{suffix}"
    redis_config = RedisConfig(
        url=os.environ["TEST_REDIS_URL"],
        stream=stream,
        dead_letter_stream=dead_letter,
        consumer_group=f"event-processors-{suffix}",
        max_length=100,
        dead_letter_max_length=10,
        batch_size=10,
        block_ms=50,
        claim_idle_ms=1000,
    )
    mongo_config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    codec = JsonEventEnvelopeCodec()
    publisher = RedisStreamEventPublisher(redis_config, codec)
    consumer = RedisStreamEventConsumer(redis_config, "worker-integration")
    repository = MongoVehicleEventRepository(mongo_config)
    worker = VehicleEventWorker(consumer, repository, codec)
    admin = Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    event = replace(
        sample_event,
        id=f"evt_worker_{suffix}",
        track_id=f"gate-01:worker-test:{suffix}",
    )
    try:
        await publisher.initialize()
        await worker.initialize()
        assert await publisher.publish(event)
        assert await publisher.publish(event)

        assert await worker.run_once() == 2

        assert await repository.get(event.id) == event
        assert worker.stats.events_persisted == 1
        assert worker.stats.duplicate_events == 1
        pending_summary = await admin.xpending(stream, redis_config.consumer_group)
        assert pending_summary["pending"] == 0
        assert await admin.xlen(stream) == 0
    finally:
        await repository._collection.delete_one({"_id": event.id})
        await worker.close()
        await publisher.close()
        await admin.delete(stream, dead_letter)
        await admin.aclose()
