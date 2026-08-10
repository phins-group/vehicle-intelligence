from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlsplit

from vehicle_intelligence.domain.enums import CameraDirection, CameraStatus, Direction
from vehicle_intelligence.domain.geometry import Point


@dataclass(frozen=True, slots=True, repr=False)
class SecretUri:
    """A URI value that cannot reveal itself through repr/str by accident."""

    _value: str = field(repr=False)

    def __post_init__(self) -> None:
        if len(self._value) > 2048 or any(ord(char) < 32 for char in self._value):
            raise ValueError("RTSP URL is invalid")
        try:
            parsed = urlsplit(self._value)
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("RTSP URL is invalid") from exc
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            raise ValueError("RTSP URL must use rtsp:// or rtsps:// with a host")

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretUri('[REDACTED]')"

    def __str__(self) -> str:
        return "[REDACTED]"


@dataclass(frozen=True, slots=True)
class Camera:
    id: str
    name: str
    rtsp_url: SecretUri = field(repr=False)
    fps_limit: float
    direction: CameraDirection
    enabled: bool
    vehicle_confidence: float
    plate_confidence: float
    created_at: datetime
    updated_at: datetime
    location: str | None = None
    zone: str | None = None
    roi: tuple[Point, ...] | None = None
    crossing_line: tuple[Point, Point] | None = None
    crossing_positive_to_negative: Direction = Direction.ENTER
    finalize_on_crossing: bool = False
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip() or any(char in self.id for char in "/\\\0"):
            raise ValueError("camera id must be a non-empty path-safe value")
        if not self.name.strip():
            raise ValueError("camera name is required")
        if self.fps_limit <= 0:
            raise ValueError("camera FPS limit must be positive")
        if not 0 <= self.vehicle_confidence <= 1:
            raise ValueError("vehicle confidence must be in [0, 1]")
        if not 0 <= self.plate_confidence <= 1:
            raise ValueError("plate confidence must be in [0, 1]")
        if self.roi is not None and len(self.roi) < 3:
            raise ValueError("camera ROI requires at least three points")
        if self.crossing_positive_to_negative not in {Direction.ENTER, Direction.EXIT}:
            raise ValueError("line-crossing direction must be ENTER or EXIT")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("camera timestamps must be timezone-aware")
        if self.schema_version < 1 or self.revision < 1:
            raise ValueError("camera schema version and revision must be positive")


@dataclass(frozen=True, slots=True)
class CameraHealth:
    camera_id: str
    status: CameraStatus
    source_fps: float
    decode_fps: float
    queue_size: int
    dropped_frames: int
    reconnect_count: int
    connection_failures: int
    stream_epoch: int
    last_frame_at: datetime | None
    updated_at: datetime
    decoded_frames: int = 0
    sampled_frames: int = 0
    vehicle_detections: int = 0
    plate_detections: int = 0
    ocr_requests: int = 0
    ocr_success: int = 0
    events_created: int = 0
    track_count: int = 0
    inference_fps: float = 0.0
    vehicle_inference_latency_ms: float = 0.0
    plate_inference_latency_ms: float = 0.0
    ocr_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.source_fps < 0 or self.decode_fps < 0:
            raise ValueError("camera FPS metrics cannot be negative")
        counters = (
            self.queue_size,
            self.dropped_frames,
            self.reconnect_count,
            self.connection_failures,
            self.decoded_frames,
            self.sampled_frames,
            self.vehicle_detections,
            self.plate_detections,
            self.ocr_requests,
            self.ocr_success,
            self.events_created,
            self.track_count,
        )
        if any(value < 0 for value in counters):
            raise ValueError("camera counters cannot be negative")
        if self.stream_epoch < 0:
            raise ValueError("stream epoch cannot be negative")
        rates = (
            self.inference_fps,
            self.vehicle_inference_latency_ms,
            self.plate_inference_latency_ms,
            self.ocr_latency_ms,
        )
        if any(value < 0 for value in rates):
            raise ValueError("camera inference metrics cannot be negative")
        if self.updated_at.tzinfo is None:
            raise ValueError("camera health timestamp must be timezone-aware")
        if self.last_frame_at is not None and self.last_frame_at.tzinfo is None:
            raise ValueError("last frame timestamp must be timezone-aware")
