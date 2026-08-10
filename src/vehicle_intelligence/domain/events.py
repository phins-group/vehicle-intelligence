from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vehicle_intelligence.domain.detection import ModelMetadata
from vehicle_intelligence.domain.enums import Direction, EventStatus, EventType
from vehicle_intelligence.domain.plate import CharacterCorrection


@dataclass(frozen=True, slots=True)
class CameraSnapshot:
    id: str
    name: str
    zone: str | None = None


@dataclass(frozen=True, slots=True)
class PlateReview:
    normalized: str
    revision: int
    reviewed_at: datetime
    reviewed_by: str
    reviewer_display_name: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.normalized.strip():
            raise ValueError("reviewed plate is required")
        if self.revision < 1:
            raise ValueError("plate review revision must be positive")
        if self.reviewed_at.tzinfo is None:
            raise ValueError("plate review timestamp must be timezone-aware")
        if not self.reviewed_by.strip() or not self.reviewer_display_name.strip():
            raise ValueError("plate reviewer identity is required")
        if self.note is not None and len(self.note) > 500:
            raise ValueError("plate review note is too long")


@dataclass(frozen=True, slots=True)
class PlateEvidence:
    raw: str
    normalized: str
    confidence: float
    observation_count: int
    corrections: tuple[CharacterCorrection, ...] = field(default_factory=tuple)
    review: PlateReview | None = None
    partial: bool = False

    @property
    def final_normalized(self) -> str:
        return self.review.normalized if self.review is not None else self.normalized

    @property
    def review_revision(self) -> int:
        return self.review.revision if self.review is not None else 0


@dataclass(frozen=True, slots=True)
class VehicleEvidence:
    type: str
    confidence: float
    color: str | None = None


@dataclass(frozen=True, slots=True)
class MediaReferences:
    snapshot_key: str | None = None
    vehicle_crop_key: str | None = None
    plate_crop_key: str | None = None
    clip_key: str | None = None


@dataclass(frozen=True, slots=True)
class AITrace:
    vehicle_detector: ModelMetadata | None
    plate_detector: ModelMetadata | None
    ocr: ModelMetadata | None
    config_version: str | None = None


@dataclass(frozen=True, slots=True)
class VehicleEvent:
    id: str
    schema_version: int
    camera: CameraSnapshot
    track_id: str
    event_type: EventType
    occurred_at: datetime
    created_at: datetime
    direction: Direction
    status: EventStatus
    vehicle: VehicleEvidence
    plate: PlateEvidence | None
    media: MediaReferences
    ai: AITrace
    vehicle_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.track_id:
            raise ValueError("event id and track id are required")
        if self.schema_version < 1:
            raise ValueError("event schema version must be positive")
        if self.occurred_at.tzinfo is None or self.created_at.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")
        if not 0 <= self.vehicle.confidence <= 1:
            raise ValueError("vehicle confidence must be in [0, 1]")
        if self.plate is not None and not 0 <= self.plate.confidence <= 1:
            raise ValueError("plate confidence must be in [0, 1]")
        if self.plate is not None and self.plate.review is not None and self.schema_version < 2:
            raise ValueError("reviewed vehicle events require schema version 2 or newer")
