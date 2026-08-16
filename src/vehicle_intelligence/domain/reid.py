from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vehicle_intelligence.domain.enums import IdentityReviewAction, ReIDVerdict


@dataclass(frozen=True, slots=True)
class ReIDSignals:
    plate: float | None
    embedding: float | None
    vehicle_type: float | None
    color: float | None
    travel_time: float

    def __post_init__(self) -> None:
        values = (
            self.plate,
            self.embedding,
            self.vehicle_type,
            self.color,
            self.travel_time,
        )
        if any(value is not None and not 0 <= value <= 1 for value in values):
            raise ValueError("ReID signal scores must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ReIDScore:
    source_fingerprint_id: str
    candidate_fingerprint_id: str
    source_vehicle_id: str
    candidate_vehicle_id: str
    score: float
    verdict: ReIDVerdict
    signals: ReIDSignals
    scoring_version: str
    topology_edge_id: str

    def __post_init__(self) -> None:
        identifiers = (
            self.source_fingerprint_id,
            self.candidate_fingerprint_id,
            self.source_vehicle_id,
            self.candidate_vehicle_id,
            self.scoring_version,
            self.topology_edge_id,
        )
        if any(not value.strip() for value in identifiers) or not 0 <= self.score <= 1:
            raise ValueError("ReID score is invalid")


@dataclass(frozen=True, slots=True)
class IdentityReviewer:
    id: str
    display_name: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.display_name.strip():
            raise ValueError("identity reviewer is required")


@dataclass(frozen=True, slots=True)
class IdentityMergeReview:
    id: str
    source_vehicle_id: str
    target_vehicle_id: str
    expected_source_revision: int
    expected_target_revision: int
    reviewer: IdentityReviewer
    reviewed_at: datetime
    reason: str
    source_fingerprint_id: str | None = None
    target_fingerprint_id: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if any(
            not value.strip() for value in (self.id, self.source_vehicle_id, self.target_vehicle_id)
        ):
            raise ValueError("identity merge identifiers are required")
        if self.source_vehicle_id == self.target_vehicle_id:
            raise ValueError("cannot merge an identity into itself")
        if self.expected_source_revision < 1 or self.expected_target_revision < 1:
            raise ValueError("identity merge revisions must be positive")
        if self.reviewed_at.tzinfo is None or not self.reason.strip():
            raise ValueError("identity merge review evidence is required")
        if self.score is not None and not 0 <= self.score <= 1:
            raise ValueError("identity merge score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class IdentitySplitReview:
    id: str
    source_vehicle_id: str
    new_vehicle_id: str
    fingerprint_ids: tuple[str, ...]
    expected_source_revision: int
    reviewer: IdentityReviewer
    reviewed_at: datetime
    reason: str

    def __post_init__(self) -> None:
        identifiers = (self.id, self.source_vehicle_id, self.new_vehicle_id)
        if any(not value.strip() for value in identifiers):
            raise ValueError("identity split identifiers are required")
        if self.source_vehicle_id == self.new_vehicle_id:
            raise ValueError("split identity ID must be new")
        if (
            not self.fingerprint_ids
            or len(self.fingerprint_ids) != len(set(self.fingerprint_ids))
            or any(not value.strip() for value in self.fingerprint_ids)
        ):
            raise ValueError("identity split fingerprints must be non-empty and unique")
        if len(self.fingerprint_ids) > 1000:
            raise ValueError("identity split is bounded to 1000 fingerprints")
        if self.expected_source_revision < 1:
            raise ValueError("identity split revision must be positive")
        if self.reviewed_at.tzinfo is None or not self.reason.strip():
            raise ValueError("identity split review evidence is required")


@dataclass(frozen=True, slots=True)
class IdentityReviewResult:
    review_id: str
    action: IdentityReviewAction
    source_vehicle_id: str
    result_vehicle_id: str
    moved_fingerprints: int
    moved_events: int
    reviewed_at: datetime
    idempotent: bool = False

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.review_id, self.source_vehicle_id, self.result_vehicle_id)
        ):
            raise ValueError("identity review result identifiers are required")
        if self.moved_fingerprints < 1 or self.moved_events < 0:
            raise ValueError("identity review result counters are invalid")
        if self.reviewed_at.tzinfo is None:
            raise ValueError("identity review result timestamp must be timezone-aware")
