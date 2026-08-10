from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlateReviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    text: str = Field(min_length=4, max_length=32)
    expected_revision: int = Field(alias="expectedRevision", ge=0)
    note: str | None = Field(default=None, max_length=500)

    @field_validator("text")
    @classmethod
    def plate_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("plate is required")
        return value


class PlateReviewResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    event: dict[str, object]
    changed: bool
    feedback_reason: str = Field(alias="feedbackReason")
    dataset_sample_id: str | None = Field(alias="datasetSampleId")
