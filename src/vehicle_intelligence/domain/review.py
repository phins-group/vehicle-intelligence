from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vehicle_intelligence.domain.detection import ModelMetadata
from vehicle_intelligence.domain.enums import (
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
)


@dataclass(frozen=True, slots=True)
class OCRDatasetPrediction:
    raw: str
    normalized: str
    confidence: float
    model: ModelMetadata | None = None

    def __post_init__(self) -> None:
        if not self.raw.strip() or not self.normalized.strip():
            raise ValueError("dataset OCR prediction is required")
        if not 0 <= self.confidence <= 1:
            raise ValueError("dataset OCR confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class DatasetSample:
    id: str
    sample_type: DatasetSampleType
    status: DatasetSampleStatus
    source_event_id: str
    image_key: str
    prediction: OCRDatasetPrediction
    label: str
    reason: DatasetSampleReason
    review_revision: int
    reviewed_by: str
    reviewer_display_name: str
    reviewed_at: datetime
    created_at: datetime
    export_id: str | None = None
    export_attempts: int = 0
    export_claimed_at: datetime | None = None
    exported_at: datetime | None = None
    export_manifest_sha256: str | None = None
    export_error_code: str | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        required = (
            self.id,
            self.source_event_id,
            self.image_key,
            self.label,
            self.reviewed_by,
            self.reviewer_display_name,
        )
        if any(not value.strip() for value in required):
            raise ValueError("dataset sample identifiers and label are required")
        if self.review_revision < 1:
            raise ValueError("dataset sample review revision must be positive")
        if self.reviewed_at.tzinfo is None or self.created_at.tzinfo is None:
            raise ValueError("dataset sample timestamps must be timezone-aware")
        optional_timestamps = (self.export_claimed_at, self.exported_at)
        if any(value is not None and value.tzinfo is None for value in optional_timestamps):
            raise ValueError("dataset export timestamps must be timezone-aware")
        if self.export_attempts < 0:
            raise ValueError("dataset export attempts cannot be negative")
        if self.export_id is not None and not self.export_id.strip():
            raise ValueError("dataset export id cannot be blank")
        if self.export_manifest_sha256 is not None and (
            len(self.export_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.export_manifest_sha256)
        ):
            raise ValueError("dataset export manifest hash must be lowercase SHA-256")
        if self.schema_version < 1:
            raise ValueError("dataset sample schema version must be positive")
