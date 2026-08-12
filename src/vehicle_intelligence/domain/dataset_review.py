"""Human review contracts for immutable detector-dataset source queues."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path


class DetectorReviewStatus(StrEnum):
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    CORRECTED = "CORRECTED"
    NEGATIVE = "NEGATIVE"
    REJECTED = "REJECTED"


class DetectorReviewAction(StrEnum):
    APPROVE = "APPROVE"
    CORRECT = "CORRECT"
    MARK_NEGATIVE = "MARK_NEGATIVE"
    REJECT = "REJECT"


class DetectorPromotionStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class DetectorReviewBox:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("detector review bounding box values must be finite")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("detector review bounding box is invalid")


@dataclass(frozen=True, slots=True)
class DetectorReviewAnnotation:
    bbox: DetectorReviewBox
    class_name: str = "license_plate"
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.class_name != "license_plate":
            raise ValueError("detector review supports only license_plate annotations")


@dataclass(frozen=True, slots=True)
class DetectorReviewDecision:
    action: DetectorReviewAction
    status: DetectorReviewStatus
    annotations: tuple[DetectorReviewAnnotation, ...]
    revision: int
    reviewed_by: str
    reviewer_display_name: str
    reviewed_at: datetime
    note: str | None = None


@dataclass(frozen=True, slots=True)
class DetectorReviewItem:
    source_id: str
    review_id: str
    image_path: str
    source_image_sha256: str
    source_filename_sha256: str
    reason: str
    suggestions: tuple[DetectorReviewAnnotation, ...]
    status: DetectorReviewStatus = DetectorReviewStatus.PENDING_REVIEW
    revision: int = 0
    decision: DetectorReviewDecision | None = None
    image_width: int | None = None
    image_height: int | None = None


@dataclass(frozen=True, slots=True)
class DetectorReviewPage:
    items: tuple[DetectorReviewItem, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class DetectorReviewSourceSummary:
    source_id: str
    source_manifest_sha256: str
    source_type: str
    collection_method: str
    rights_status: str
    promotion_eligible: bool
    release_eligible: bool
    distribution_eligible: bool
    queue_count: int
    status_counts: dict[str, int]
    reason_counts: dict[str, int]
    reviewed_count: int
    pending_count: int


@dataclass(frozen=True, slots=True)
class DetectorReviewImage:
    path: Path
    media_type: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DetectorPromotionJob:
    id: str
    source_id: str
    target_source_id: str
    status: DetectorPromotionStatus
    created_at: datetime
    updated_at: datetime
    requested_by: str
    reviewed_sample_count: int
    pending_sample_count: int
    decision_snapshot_sha256: str
    output_directory: str | None = None
    manifest_sha256: str | None = None
    error_code: str | None = None
