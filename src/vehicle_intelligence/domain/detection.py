from __future__ import annotations

from dataclasses import dataclass, field

from vehicle_intelligence.domain.geometry import BoundingBox, Point


def _validate_confidence(value: float, field_name: str = "confidence") -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    name: str
    version: str
    hash: str | None = None


@dataclass(frozen=True, slots=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    model: ModelMetadata

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)
        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")
        if not self.class_name:
            raise ValueError("class_name is required")


@dataclass(frozen=True, slots=True)
class PlateDetection:
    bbox: BoundingBox
    confidence: float
    model: ModelMetadata
    corners: tuple[Point, Point, Point, Point] | None = None

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    track_id: int
    detection: Detection

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id cannot be negative")


@dataclass(frozen=True, slots=True)
class OCRCharacter:
    text: str
    confidence: float
    position: int

    def __post_init__(self) -> None:
        if len(self.text) != 1:
            raise ValueError("OCR character text must contain exactly one character")
        _validate_confidence(self.confidence)
        if self.position < 0:
            raise ValueError("OCR character position cannot be negative")


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    confidence: float
    model: ModelMetadata
    characters: tuple[OCRCharacter, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_confidence(self.confidence)
