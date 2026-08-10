from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.live_monitor import (
    LiveCameraStatus,
    LiveMonitorService,
    LiveMonitorSourceState,
)
from vehicle_intelligence.config import LiveMonitorConfig, RedisConfig
from vehicle_intelligence.domain import LiveFrameMetadata, LiveFramePacket
from vehicle_intelligence.infrastructure.messaging.live_codec import JsonLiveFrameCodec
from vehicle_intelligence.infrastructure.messaging.live_redis import (
    RedisLiveFramePublisher,
    RedisLiveFrameSubscriber,
)


@pytest.mark.skipif(not os.getenv("TEST_REDIS_URL"), reason="TEST_REDIS_URL is not configured")
async def test_redis_live_preview_reaches_bounded_api_state() -> None:
    suffix = uuid.uuid4().hex
    redis_config = RedisConfig(url=os.environ["TEST_REDIS_URL"])
    live_config = LiveMonitorConfig(
        enabled=True,
        redis_channel=f"vehicle.live.frames.test.{suffix}",
        broker_poll_seconds=0.05,
        reconnect_initial_seconds=0.05,
        reconnect_max_seconds=0.1,
    )
    codec = JsonLiveFrameCodec(live_config.maximum_payload_bytes)
    publisher = RedisLiveFramePublisher(redis_config, live_config, codec)
    source = RedisLiveFrameSubscriber(redis_config, live_config, codec)
    service = LiveMonitorService(live_config, source)
    packet = LiveFramePacket(
        metadata=LiveFrameMetadata(
            camera_id="gate-redis-live",
            frame_id=9,
            stream_epoch=3,
            captured_at=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
            source_width=320,
            source_height=180,
        ),
        jpeg=b"\xff\xd8redis-preview\xff\xd9",
        preview_width=320,
        preview_height=180,
    )
    try:
        await service.initialize()
        for _ in range(40):
            if service.stats.source_state is LiveMonitorSourceState.ONLINE:
                break
            await asyncio.sleep(0.025)
        assert service.stats.source_state is LiveMonitorSourceState.ONLINE
        await publisher.initialize()
        await publisher.publish(packet)
        for _ in range(40):
            if service.stats.frames_received:
                break
            await asyncio.sleep(0.025)

        snapshot = service.snapshot("gate-redis-live")
        assert snapshot.status is LiveCameraStatus.LIVE
        assert snapshot.latest is not None
        assert snapshot.latest.packet == packet
        assert service.stats.frames_received == 1
    finally:
        await publisher.close()
        await service.close()

