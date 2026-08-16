"""Best-effort Redis Pub/Sub transport for bounded live preview packets."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from redis.exceptions import RedisError

from vehicle_intelligence.application.ports import LiveFrameCodec
from vehicle_intelligence.config import LiveMonitorConfig, RedisConfig
from vehicle_intelligence.domain import LiveFramePacket
from vehicle_intelligence.exceptions import EventBusError
from vehicle_intelligence.infrastructure.messaging.redis_connection import create_redis_client


class RedisLiveFramePublisher:
    def __init__(
        self,
        redis_config: RedisConfig,
        live_config: LiveMonitorConfig,
        codec: LiveFrameCodec,
        client: Any | None = None,
    ) -> None:
        self._channel = live_config.redis_channel
        self._codec = codec
        self._client = client or create_redis_client(redis_config)
        self._owns_client = client is None

    async def initialize(self) -> None:
        try:
            await self._client.ping()
        except RedisError as exc:
            raise EventBusError("cannot connect live-preview Redis publisher") from exc

    async def publish(self, packet: LiveFramePacket) -> None:
        payload = self._codec.encode(packet)
        try:
            await self._client.publish(self._channel, payload)
        except RedisError as exc:
            raise EventBusError(
                f"cannot publish live preview for camera: {packet.metadata.camera_id}"
            ) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RedisLiveFrameSubscriber:
    def __init__(
        self,
        redis_config: RedisConfig,
        live_config: LiveMonitorConfig,
        codec: LiveFrameCodec,
        client: Any | None = None,
    ) -> None:
        self._channel = live_config.redis_channel
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
            raise EventBusError("cannot subscribe to live-preview Redis channel") from exc

    async def receive(self, timeout_seconds: float) -> LiveFramePacket | None:
        if self._pubsub is None:
            raise EventBusError("live-preview Redis subscriber is not connected")
        try:
            message = await self._pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=timeout_seconds,
            )
        except RedisError as exc:
            raise EventBusError("cannot receive live-preview Redis packet") from exc
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
