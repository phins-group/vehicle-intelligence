"""ReID score and human identity-review API contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_intelligence.application.reid import MergeIdentities, SplitIdentity
from vehicle_intelligence.domain import IdentityReviewResult, ReIDScore


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ReIDSignalsPublic(APIModel):
    plate: float | None
    embedding: float | None
    vehicle_type: float | None = Field(alias="vehicleType")
    color: float | None
    travel_time: float = Field(alias="travelTime")


class ReIDScorePublic(APIModel):
    source_fingerprint_id: str = Field(alias="sourceFingerprintId")
    candidate_fingerprint_id: str = Field(alias="candidateFingerprintId")
    source_vehicle_id: str = Field(alias="sourceVehicleId")
    candidate_vehicle_id: str = Field(alias="candidateVehicleId")
    score: float
    verdict: str
    signals: ReIDSignalsPublic
    scoring_version: str = Field(alias="scoringVersion")
    topology_edge_id: str = Field(alias="topologyEdgeId")

    @classmethod
    def from_domain(cls, value: ReIDScore) -> ReIDScorePublic:
        return cls(
            sourceFingerprintId=value.source_fingerprint_id,
            candidateFingerprintId=value.candidate_fingerprint_id,
            sourceVehicleId=value.source_vehicle_id,
            candidateVehicleId=value.candidate_vehicle_id,
            score=value.score,
            verdict=value.verdict.value,
            signals=ReIDSignalsPublic(
                plate=value.signals.plate,
                embedding=value.signals.embedding,
                vehicleType=value.signals.vehicle_type,
                color=value.signals.color,
                travelTime=value.signals.travel_time,
            ),
            scoringVersion=value.scoring_version,
            topologyEdgeId=value.topology_edge_id,
        )


class ReIDScoreListPublic(APIModel):
    source_fingerprint_id: str = Field(alias="sourceFingerprintId")
    items: list[ReIDScorePublic]


class MergeIdentitiesRequest(APIModel):
    review_id: str = Field(alias="reviewId", min_length=8, max_length=128)
    source_vehicle_id: str = Field(alias="sourceVehicleId", min_length=1, max_length=128)
    target_vehicle_id: str = Field(alias="targetVehicleId", min_length=1, max_length=128)
    expected_source_revision: int = Field(alias="expectedSourceRevision", ge=1)
    expected_target_revision: int = Field(alias="expectedTargetRevision", ge=1)
    reason: str = Field(min_length=3, max_length=1000)
    source_fingerprint_id: str | None = Field(
        default=None, alias="sourceFingerprintId", max_length=128
    )
    target_fingerprint_id: str | None = Field(
        default=None, alias="targetFingerprintId", max_length=128
    )

    @model_validator(mode="after")
    def validate_fingerprint_pair(self) -> MergeIdentitiesRequest:
        if (self.source_fingerprint_id is None) != (self.target_fingerprint_id is None):
            raise ValueError("both merge fingerprints must be supplied together")
        return self

    def to_command(self) -> MergeIdentities:
        return MergeIdentities(
            review_id=self.review_id,
            source_vehicle_id=self.source_vehicle_id,
            target_vehicle_id=self.target_vehicle_id,
            expected_source_revision=self.expected_source_revision,
            expected_target_revision=self.expected_target_revision,
            reason=self.reason,
            source_fingerprint_id=self.source_fingerprint_id,
            target_fingerprint_id=self.target_fingerprint_id,
        )


class SplitIdentityRequest(APIModel):
    review_id: str = Field(alias="reviewId", min_length=8, max_length=128)
    source_vehicle_id: str = Field(alias="sourceVehicleId", min_length=1, max_length=128)
    expected_source_revision: int = Field(alias="expectedSourceRevision", ge=1)
    fingerprint_ids: list[str] = Field(
        alias="fingerprintIds", min_length=1, max_length=1000
    )
    reason: str = Field(min_length=3, max_length=1000)

    @field_validator("fingerprint_ids")
    @classmethod
    def validate_fingerprints(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("split fingerprint IDs must be non-empty and unique")
        return sorted(normalized)

    def to_command(self) -> SplitIdentity:
        return SplitIdentity(
            review_id=self.review_id,
            source_vehicle_id=self.source_vehicle_id,
            expected_source_revision=self.expected_source_revision,
            fingerprint_ids=tuple(self.fingerprint_ids),
            reason=self.reason,
        )


class IdentityReviewResultPublic(APIModel):
    review_id: str = Field(alias="reviewId")
    action: str
    source_vehicle_id: str = Field(alias="sourceVehicleId")
    result_vehicle_id: str = Field(alias="resultVehicleId")
    moved_fingerprints: int = Field(alias="movedFingerprints")
    moved_events: int = Field(alias="movedEvents")
    reviewed_at: datetime = Field(alias="reviewedAt")
    idempotent: bool

    @classmethod
    def from_domain(cls, value: IdentityReviewResult) -> IdentityReviewResultPublic:
        return cls(
            reviewId=value.review_id,
            action=value.action.value,
            sourceVehicleId=value.source_vehicle_id,
            resultVehicleId=value.result_vehicle_id,
            movedFingerprints=value.moved_fingerprints,
            movedEvents=value.moved_events,
            reviewedAt=value.reviewed_at,
            idempotent=value.idempotent,
        )
