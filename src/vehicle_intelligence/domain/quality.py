"""Immutable model-quality reporting values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vehicle_intelligence.domain.detection import ModelMetadata


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    event_count: int
    readable_plate_count: int
    confirmed_count: int
    needs_review_count: int
    no_plate_count: int
    unreadable_count: int
    reviewed_count: int
    corrected_count: int
    ocr_success_rate: float
    unknown_plate_rate: float
    human_correction_rate: float
    average_plate_confidence: float | None


@dataclass(frozen=True, slots=True)
class ModelQualitySlice:
    model: ModelMetadata | None
    metrics: QualityMetrics


@dataclass(frozen=True, slots=True)
class DailyQualityPoint:
    day: str
    metrics: QualityMetrics


@dataclass(frozen=True, slots=True)
class DatasetFeedbackMetrics:
    total: int = 0
    ready: int = 0
    exporting: int = 0
    exported: int = 0
    export_failed: int = 0
    corrections: int = 0
    confirmations: int = 0


@dataclass(frozen=True, slots=True)
class ModelQualityReport:
    from_time: datetime
    to_time: datetime
    generated_at: datetime
    totals: QualityMetrics
    models: tuple[ModelQualitySlice, ...]
    daily: tuple[DailyQualityPoint, ...]
    feedback: DatasetFeedbackMetrics
    truncated: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        timestamps = (self.from_time, self.to_time, self.generated_at)
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("quality report timestamps must be timezone-aware")
        if self.to_time <= self.from_time:
            raise ValueError("quality report range must be positive")
