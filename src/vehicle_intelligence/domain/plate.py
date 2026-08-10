from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vehicle_intelligence.domain.detection import ModelMetadata


@dataclass(frozen=True, slots=True)
class CharacterCorrection:
    position: int
    from_character: str
    to_character: str
    confidence: float

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("correction position cannot be negative")
        if len(self.from_character) != 1 or len(self.to_character) != 1:
            raise ValueError("correction characters must have length one")
        if not 0 <= self.confidence <= 1:
            raise ValueError("correction confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlateQuality:
    sharpness: float
    brightness: float
    contrast: float
    resolution_score: float
    angle_score: float
    detector_score: float
    total_score: float
    eligible: bool

    def __post_init__(self) -> None:
        values = (
            self.sharpness,
            self.brightness,
            self.contrast,
            self.resolution_score,
            self.angle_score,
            self.detector_score,
            self.total_score,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("plate quality scores must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlateNormalization:
    raw: str
    cleaned: str
    compact: str | None
    normalized: str | None
    valid: bool
    corrections: tuple[CharacterCorrection, ...] = field(default_factory=tuple)
    partial: bool = False


@dataclass(frozen=True, slots=True)
class PlateObservation:
    frame_id: int
    timestamp: datetime
    raw_text: str
    normalized_text: str | None
    compact_text: str | None
    ocr_confidence: float
    detection_confidence: float
    quality_score: float
    corrections: tuple[CharacterCorrection, ...]
    plate_model: ModelMetadata
    ocr_model: ModelMetadata
    partial: bool = False

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id cannot be negative")
        if self.timestamp.tzinfo is None:
            raise ValueError("observation timestamp must be timezone-aware")
        values = (self.ocr_confidence, self.detection_confidence, self.quality_score)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("observation scores must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    raw_text: str
    normalized_text: str
    compact_text: str
    confidence: float
    observation_count: int
    corrections: tuple[CharacterCorrection, ...] = field(default_factory=tuple)
    partial: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("candidate confidence must be in [0, 1]")
        if self.observation_count < 1:
            raise ValueError("candidate must have at least one observation")
