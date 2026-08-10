from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from vehicle_intelligence.application.live_monitor import (
    LiveCameraStatus,
    LiveMonitorService,
)
from vehicle_intelligence.application.live_preview import LivePreviewReporter
from vehicle_intelligence.application.ports import EncodedLivePreview
from vehicle_intelligence.config import LiveMonitorConfig
from vehicle_intelligence.domain import (
    BoundingBox,
    Direction,
    LiveFrameMetadata,
    LiveFramePacket,
    LivePlateOverlay,
    LiveVehicleOverlay,
    Point,
)
from vehicle_intelligence.exceptions import EventContractError
from vehicle_intelligence.infrastructure.messaging.live_codec import JsonLiveFrameCodec
from vehicle_intelligence.infrastructure.vision.opencv import OpenCVLivePreviewEncoder


def live_packet(
    camera_id: str = "gate-live",
    frame_id: int = 42,
    jpeg: bytes = b"\xff\xd8preview\xff\xd9",
) -> LiveFramePacket:
    plate = LivePlateOverlay(
        bbox=BoundingBox(60, 70, 120, 90),
        detection_confidence=0.91,
        quality_score=0.82,
        text="51H-123.45",
        ocr_confidence=0.93,
    )
    vehicle = LiveVehicleOverlay(
        track_id=f"{camera_id}:session:12",
        bbox=BoundingBox(20, 20, 180, 140),
        confidence=0.96,
        vehicle_type="car",
        direction=Direction.ENTER,
        plate=plate,
    )
    return LiveFramePacket(
        metadata=LiveFrameMetadata(
            camera_id=camera_id,
            frame_id=frame_id,
            stream_epoch=2,
            captured_at=datetime(2026, 8, 9, 12, 30, tzinfo=UTC),
            source_width=1920,
            source_height=1080,
            vehicles=(vehicle,),
            vehicle_roi=(Point(0, 200), Point(1920, 200), Point(1920, 1080)),
            crossing_line=(Point(0, 600), Point(1920, 600)),
        ),
        jpeg=jpeg,
        preview_width=960,
        preview_height=540,
    )


def test_live_frame_codec_round_trip_and_payload_limit() -> None:
    packet = live_packet()
    codec = JsonLiveFrameCodec(100_000)

    payload = codec.encode(packet)
    decoded = codec.decode(payload)

    assert decoded == packet
    assert decoded.metadata.vehicles[0].plate is not None
    assert decoded.metadata.vehicles[0].plate.text == "51H-123.45"
    with pytest.raises(EventContractError, match="exceeds configured limit"):
        JsonLiveFrameCodec(100).encode(packet)
    with pytest.raises(EventContractError, match="invalid live frame payload"):
        codec.decode('{"schemaVersion":1,"jpegBase64":"not-base64"}')


def test_live_monitor_bounds_frames_cameras_and_marks_stale() -> None:
    now = [datetime(2026, 8, 9, 12, 30, tzinfo=UTC)]
    config = LiveMonitorConfig(
        frame_buffer_size=2,
        maximum_cameras=1,
        stale_after_seconds=2,
    )
    service = LiveMonitorService(config, clock=lambda: now[0])
    first = service.ingest(live_packet(frame_id=1))
    second = service.ingest(live_packet(frame_id=2))
    third = service.ingest(live_packet(frame_id=3))

    assert service.frame("gate-live", first.sequence) is None
    assert service.frame("gate-live", second.sequence) is not None
    assert service.snapshot("gate-live").status is LiveCameraStatus.LIVE
    now[0] += timedelta(seconds=3)
    assert service.snapshot("gate-live").status is LiveCameraStatus.STALE
    assert service.snapshot("gate-live", enabled=False).status is LiveCameraStatus.DISABLED

    service.ingest(live_packet(camera_id="gate-other", frame_id=1))
    assert service.snapshot("gate-live").latest is None
    assert service.snapshot("gate-other").latest is not None
    assert service.stats.frames_evicted == 3
    assert third.sequence == 3


class FakeLiveEncoder:
    def encode(self, image, maximum_width, jpeg_quality):
        del image, maximum_width, jpeg_quality
        return EncodedLivePreview(b"preview", 320, 180)


class FakeLivePublisher:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.packets = []

    async def initialize(self) -> None:
        self.initialized = True

    async def publish(self, packet) -> None:
        self.packets.append(packet)

    async def close(self) -> None:
        self.closed = True


async def test_live_preview_reporter_throttles_without_blocking_pipeline() -> None:
    monotonic = [10.0]
    publisher = FakeLivePublisher()
    reporter = LivePreviewReporter(
        LiveMonitorConfig(preview_fps=2, maximum_payload_bytes=32_768),
        FakeLiveEncoder(),
        publisher,
        monotonic_clock=lambda: monotonic[0],
    )
    metadata = live_packet().metadata
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    await reporter.initialize()

    assert await reporter.report(image, metadata)
    monotonic[0] += 0.2
    assert not await reporter.report(image, metadata)
    monotonic[0] += 0.4
    assert await reporter.report(image, metadata)
    await reporter.close()

    assert publisher.initialized and publisher.closed
    assert len(publisher.packets) == 2
    assert reporter.stats.published_frames == 2
    assert reporter.stats.throttled_frames == 1


def test_opencv_live_preview_encoder_resizes_and_emits_jpeg() -> None:
    image = np.zeros((600, 1200, 3), dtype=np.uint8)

    preview = OpenCVLivePreviewEncoder().encode(image, 600, 70)

    assert (preview.width, preview.height) == (600, 300)
    assert preview.jpeg.startswith(b"\xff\xd8")

