from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StartModelTrainingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source_id: str = Field(alias="sourceId", min_length=1, max_length=128)
    model_name: str = Field(alias="modelName", min_length=1, max_length=128)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=128)
    epochs: int = Field(ge=1, le=10_000)
    batch_size: int = Field(alias="batchSize", ge=1, le=1024)
    workers: int = Field(ge=0, le=128)
    snapshot_epoch: int = Field(alias="snapshotEpoch", ge=1, le=10_000)
    confirm_dataset_rights: bool = Field(alias="confirmDatasetRights")
    confirm_compute_cost: bool = Field(alias="confirmComputeCost")
    confirm_restricted_data: bool = Field(default=False, alias="confirmRestrictedData")

    @field_validator("source_id", "model_name", "model_version")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()
