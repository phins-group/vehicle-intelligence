from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vehicle_intelligence.domain.enums import Direction
from vehicle_intelligence.domain.geometry import BoundingBox, Point


def _confidence(value: float | None, name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class LivePlateOverlay:
    bbox: BoundingBox
    detection_confidence: float
    quality_score: float | None = None
    text: str | None = None
    ocr_confidence: float | None = None

    def __post_init__(self) -> None:
        _confidence(self.detection_confidence, "plate detection confidence")
        _confidence(self.quality_score, "plate quality score")
        _confidence(self.ocr_confidence, "plate OCR confidence")
        if self.text is not None and (not self.text.strip() or len(self.text) > 32):
            raise ValueError("live plate text is invalid")


@dataclass(frozen=True, slots=True)
class LiveVehicleOverlay:
    track_id: str
    bbox: BoundingBox
    confidence: float
    vehicle_type: str
    direction: Direction
    plate: LivePlateOverlay | None = None

    def __post_init__(self) -> None:
        if not self.track_id.strip() or len(self.track_id) > 256:
            raise ValueError("live track id is invalid")
        if not self.vehicle_type.strip() or len(self.vehicle_type) > 64:
            raise ValueError("live vehicle type is invalid")
        _confidence(self.confidence, "vehicle confidence")


@dataclass(frozen=True, slots=True)
class LiveFrameMetadata:
    camera_id: str
    frame_id: int
    stream_epoch: int
    captured_at: datetime
    source_width: int
    source_height: int
    vehicles: tuple[LiveVehicleOverlay, ...] = field(default_factory=tuple)
    vehicle_roi: tuple[Point, ...] | None = None
    crossing_line: tuple[Point, Point] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.camera_id.strip() or len(self.camera_id) > 128:
            raise ValueError("live camera id is invalid")
        if self.frame_id < 0 or self.stream_epoch < 0:
            raise ValueError("live frame and stream epoch must be non-negative")
        if self.captured_at.tzinfo is None:
            raise ValueError("live frame timestamp must be timezone-aware")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("live source dimensions must be positive")
        if self.vehicle_roi is not None and len(self.vehicle_roi) < 3:
            raise ValueError("live vehicle ROI requires at least three points")
        if self.schema_version != 1:
            raise ValueError("unsupported live frame schema version")


@dataclass(frozen=True, slots=True)
class LiveFramePacket:
    metadata: LiveFrameMetadata
    jpeg: bytes
    preview_width: int
    preview_height: int

    def __post_init__(self) -> None:
        if not self.jpeg:
            raise ValueError("live preview JPEG cannot be empty")
        if self.preview_width <= 0 or self.preview_height <= 0:
            raise ValueError("live preview dimensions must be positive")
