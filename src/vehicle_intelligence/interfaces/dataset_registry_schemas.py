from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetHubSyncRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    export_id: str = Field(alias="exportId", min_length=1, max_length=128)
    revision: str = Field(default="main", min_length=1, max_length=128)
    confirm_restricted_private_transfer: bool = Field(
        default=False,
        alias="confirmRestrictedPrivateTransfer",
    )

    @field_validator("export_id", "revision")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()

