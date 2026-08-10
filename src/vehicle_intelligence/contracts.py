"""Versioned cross-process contract models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventEnvelope(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    schema_version: int = Field(alias="schemaVersion", ge=1)
    occurred_at: datetime = Field(alias="occurredAt")
    source: str = Field(min_length=1)
    correlation_id: str | None = Field(default=None, alias="correlationId")
    data: dict[str, Any]

    @field_validator("occurred_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event envelope occurredAt must be timezone-aware")
        return value
