"""Canonical offline-training contracts shared by dataset and backend adapters."""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DetectorRole(StrEnum):
    VEHICLE = "vehicle"
    PLATE = "plate"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


type AttributeValue = str | bool | int | float | None


class TrainingBoundingBox(BaseModel):
    """COCO-style pixel box in ``x, y, width, height`` order."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_finite(self) -> TrainingBoundingBox:
        if not all(math.isfinite(value) for value in self.as_xywh()):
            raise ValueError("bounding box values must be finite")
        return self

    def as_xywh(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height


class TrainingPoint(BaseModel):
    """One pixel-space point retained from a polygon/quadrilateral annotation."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0)
    y: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_finite(self) -> TrainingPoint:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("polygon point values must be finite")
        return self


class DetectorAnnotation(BaseModel):
    model_config = ConfigDict(alias_generator=None, extra="forbid", populate_by_name=True)

    class_name: str = Field(alias="className", min_length=1, max_length=64)
    bbox: TrainingBoundingBox
    polygon: tuple[TrainingPoint, ...] = ()
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)

    @field_validator("class_name")
    @classmethod
    def normalize_class_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("annotation class name cannot be blank")
        return normalized

    @field_validator("polygon")
    @classmethod
    def validate_polygon(
        cls,
        value: tuple[TrainingPoint, ...],
    ) -> tuple[TrainingPoint, ...]:
        if value and not 3 <= len(value) <= 32:
            raise ValueError("annotation polygon must contain between 3 and 32 points")
        return value


class DetectorSample(BaseModel):
    """One independently reviewable image and all detector annotations on it.

    ``group_id`` must identify the correlated vehicle passage/identity.  The
    dataset builder assigns an entire group to one split, preventing adjacent
    frames or the same known vehicle from leaking into evaluation.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sample_id: str = Field(alias="sampleId", min_length=1, max_length=256)
    image_path: str = Field(alias="imagePath", min_length=1, max_length=1024)
    group_id: str = Field(alias="groupId", min_length=1, max_length=256)
    camera_id: str = Field(alias="cameraId", min_length=1, max_length=256)
    captured_at: datetime = Field(alias="capturedAt")
    split: DatasetSplit | None = None
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    annotations: tuple[DetectorAnnotation, ...] = ()

    @field_validator("sample_id", "group_id", "camera_id")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or "\x00" in normalized:
            raise ValueError("sample identifiers cannot be blank or contain NUL")
        return normalized

    @field_validator("image_path")
    @classmethod
    def validate_relative_image_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("imagePath must remain relative to the source directory")
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("detector images must use JPEG or PNG")
        return str(path)

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("capturedAt must contain a timezone")
        return value


class TrainingRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: DetectorRole
    output_directory: str
    log_path: str
    manifest_path: str
    exit_code: int
    command: tuple[str, ...]


class HubUploadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str
    repo_type: str
    revision: str | None = None
    url: str | None = None


class HubJobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    url: str | None = None
    status: str | None = None
