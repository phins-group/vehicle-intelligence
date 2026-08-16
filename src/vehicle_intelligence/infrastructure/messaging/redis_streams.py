"""Redis Streams adapters with consumer-group recovery and a dead-letter stream."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from redis.exceptions import RedisError, ResponseError

from vehicle_intelligence.application.ports import (
    BrokerMessage,
    VehicleEventCodec,
)
from vehicle_intelligence.config import RedisConfig
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import EventBusError
from vehicle_intelligence.infrastructure.messaging.redis_connection import create_redis_client

EVENT_PAYLOAD_FIELD = "event"


class RedisStreamEventPublisher:
    def __init__(
        self,
        config: RedisConfig,
        codec: VehicleEventCodec,
        client: Any | None = None,
    ) -> None:
        self._config = config
        self._codec = codec
        self._client = client or create_redis_client(config)
        self._owns_client = client is None

    async def initialize(self) -> None:
        try:
            await self._client.ping()
        except RedisError as exc:
            raise EventBusError("cannot connect to Redis event bus") from exc

    async def publish(self, event: VehicleEvent) -> bool:
        payload = self._codec.encode(event)
        try:
            await self._client.xadd(
                self._config.stream,
                {EVENT_PAYLOAD_FIELD: payload},
                maxlen=self._config.max_length,
                approximate=True,
            )
            return True
        except RedisError as exc:
            raise EventBusError(f"cannot publish vehicle event: {event.id}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RedisStreamEventConsumer:
    def __init__(
        self,
        config: RedisConfig,
        consumer_name: str,
        client: Any | None = None,
    ) -> None:
        if not consumer_name.strip():
            raise ValueError("Redis consumer name cannot be empty")
        self._config = config
        self._consumer_name = consumer_name.strip()
        self._client = client or create_redis_client(config)
        self._owns_client = client is None
        self._claim_cursor = "0-0"

    async def initialize(self) -> None:
        try:
            await self._client.ping()
            try:
                await self._client.xgroup_create(
                    self._config.stream,
                    self._config.consumer_group,
                    id="0-0",
                    mkstream=True,
                )
            except ResponseError as exc:
                if "BUSYGROUP" not in str(exc):
                    raise
        except RedisError as exc:
            raise EventBusError("cannot initialize Redis event consumer group") from exc

    async def read_new(self) -> list[BrokerMessage]:
        try:
            streams = await self._client.xreadgroup(
                self._config.consumer_group,
                self._consumer_name,
                {self._config.stream: ">"},
                count=self._config.batch_size,
                block=self._config.block_ms,
            )
            return self._flatten_streams(streams)
        except RedisError as exc:
            raise EventBusError("cannot read new Redis Stream events") from exc

    async def reclaim_stale(self) -> list[BrokerMessage]:
        try:
            result = await self._client.xautoclaim(
                self._config.stream,
                self._config.consumer_group,
                self._consumer_name,
                min_idle_time=self._config.claim_idle_ms,
                start_id=self._claim_cursor,
                count=self._config.batch_size,
            )
            self._claim_cursor = str(result[0])
            return self._messages(result[1])
        except RedisError as exc:
            raise EventBusError("cannot reclaim stale Redis Stream events") from exc

    async def acknowledge(self, message_id: str) -> None:
        await self.acknowledge_many((message_id,))

    async def acknowledge_many(self, message_ids: tuple[str, ...]) -> None:
        unique_ids = tuple(dict.fromkeys(message_ids))
        if not unique_ids:
            return
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.xack(
                    self._config.stream,
                    self._config.consumer_group,
                    *unique_ids,
                )
                if self._config.delete_after_ack:
                    pipe.xdel(self._config.stream, *unique_ids)
                await pipe.execute()
        except RedisError as exc:
            raise EventBusError(
                f"cannot acknowledge Redis Stream event batch: {len(unique_ids)} messages"
            ) from exc

    async def dead_letter(self, message: BrokerMessage, reason: str) -> None:
        fields = {
            "sourceStream": self._config.stream,
            "sourceMessageId": message.message_id,
            EVENT_PAYLOAD_FIELD: message.payload,
            "reason": reason[:2000],
            "failedAt": datetime.now(UTC).isoformat(),
        }
        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.xadd(
                    self._config.dead_letter_stream,
                    fields,
                    maxlen=self._config.dead_letter_max_length,
                    approximate=True,
                )
                pipe.xack(
                    self._config.stream,
                    self._config.consumer_group,
                    message.message_id,
                )
                if self._config.delete_after_ack:
                    pipe.xdel(self._config.stream, message.message_id)
                await pipe.execute()
        except RedisError as exc:
            raise EventBusError(
                f"cannot dead-letter Redis Stream event: {message.message_id}"
            ) from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @classmethod
    def _flatten_streams(cls, streams: Any) -> list[BrokerMessage]:
        messages: list[BrokerMessage] = []
        for _stream_name, stream_messages in streams or []:
            messages.extend(cls._messages(stream_messages))
        return messages

    @staticmethod
    def _messages(messages: Any) -> list[BrokerMessage]:
        result: list[BrokerMessage] = []
        for message_id, fields in messages or []:
            payload = fields.get(EVENT_PAYLOAD_FIELD, "")
            result.append(BrokerMessage(str(message_id), str(payload)))
        return result
