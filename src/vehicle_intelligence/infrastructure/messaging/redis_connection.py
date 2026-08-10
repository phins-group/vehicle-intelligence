"""Shared Redis client construction without leaking credentials to logs."""

from redis.asyncio import Redis

from vehicle_intelligence.config import RedisConfig


def create_redis_client(config: RedisConfig) -> Redis:
    connect_timeout = config.connection_timeout_ms / 1000
    command_timeout = max(connect_timeout, config.block_ms / 1000 + 1)
    return Redis.from_url(
        config.url.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=connect_timeout,
        socket_timeout=command_timeout,
        health_check_interval=30,
    )
