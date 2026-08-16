from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vehicle_intelligence.domain.dataset_review import DetectorReviewAction


class DetectorReviewBoxRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class DetectorReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    action: DetectorReviewAction
    expected_revision: int = Field(alias="expectedRevision", ge=0)
    annotations: list[DetectorReviewBoxRequest] = Field(default_factory=list, max_length=16)
    note: str | None = Field(default=None, max_length=1000)


class DetectorPromotionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    target_source_id: str = Field(alias="targetSourceId", min_length=1, max_length=128)

    @field_validator("target_source_id")
    @classmethod
    def normalize_target_source_id(cls, value: str) -> str:
        return value.strip()
