"""Best-effort Redis Pub/Sub adapters for realtime API fan-out."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from redis.exceptions import RedisError

from vehicle_intelligence.application.ports import VehicleEventCodec
from vehicle_intelligence.config import RealtimeConfig, RedisConfig
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import EventBusError
from vehicle_intelligence.infrastructure.messaging.redis_connection import create_redis_client


class RedisRealtimeEventPublisher:
    def __init__(
        self,
        redis_config: RedisConfig,
        realtime_config: RealtimeConfig,
        codec: VehicleEventCodec,
        client: Any | None = None,
    ) -> None:
        self._channel = realtime_config.redis_channel
        self._codec = codec
        self._client = client or create_redis_client(redis_config)
        self._owns_client = client is None

    async def initialize(self) -> None:
        try:
            await self._client.ping()
        except RedisError as exc:
            raise EventBusError("cannot connect realtime Redis publisher") from exc

    async def publish(self, event: VehicleEvent) -> None:
        try:
            await self._client.publish(self._channel, self._codec.encode(event))
        except RedisError as exc:
            raise EventBusError(f"cannot publish realtime event: {event.id}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RedisRealtimeEventSubscriber:
    def __init__(
        self,
        redis_config: RedisConfig,
        realtime_config: RealtimeConfig,
        codec: VehicleEventCodec,
        client: Any | None = None,
    ) -> None:
        self._channel = realtime_config.redis_channel
        self._codec = codec
        self._client = client or create_redis_client(redis_config)
        self._owns_client = client is None
        self._pubsub: Any | None = None

    async def connect(self) -> None:
        await self.disconnect()
        pubsub: Any | None = None
        try:
            await self._client.ping()
            pubsub = self._client.pubsub(ignore_subscribe_messages=True)
            await pubsub.subscribe(self._channel)
            self._pubsub = pubsub
        except RedisError as exc:
            if pubsub is not None:
                with suppress(RedisError):
                    await pubsub.aclose()
            raise EventBusError("cannot subscribe to realtime Redis channel") from exc

    async def receive(self, timeout_seconds: float) -> VehicleEvent | None:
        if self._pubsub is None:
            raise EventBusError("realtime Redis subscriber is not connected")
        try:
            message = await self._pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=timeout_seconds,
            )
        except RedisError as exc:
            raise EventBusError("cannot receive realtime Redis event") from exc
        if message is None or message.get("type") != "message":
            return None
        return self._codec.decode(str(message.get("data", "")))

    async def disconnect(self) -> None:
        pubsub, self._pubsub = self._pubsub, None
        if pubsub is None:
            return
        with suppress(RedisError):
            await pubsub.unsubscribe(self._channel)
        with suppress(RedisError):
            await pubsub.aclose()

    async def close(self) -> None:
        await self.disconnect()
        if self._owns_client:
            await self._client.aclose()
