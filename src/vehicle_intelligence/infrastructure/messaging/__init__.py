"""Event-publishing and consumption adapters."""

from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.direct import RepositoryEventPublisher
from vehicle_intelligence.infrastructure.messaging.realtime_redis import (
    RedisRealtimeEventPublisher,
    RedisRealtimeEventSubscriber,
)
from vehicle_intelligence.infrastructure.messaging.redis_streams import (
    RedisStreamEventConsumer,
    RedisStreamEventPublisher,
)

__all__ = [
    "JsonEventEnvelopeCodec",
    "RedisStreamEventConsumer",
    "RedisStreamEventPublisher",
    "RedisRealtimeEventPublisher",
    "RedisRealtimeEventSubscriber",
    "RepositoryEventPublisher",
]
