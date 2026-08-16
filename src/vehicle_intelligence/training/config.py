"""Strict configuration for offline vehicle and plate detector training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_intelligence.exceptions import ConfigurationError
from vehicle_intelligence.training.domain import DetectorRole

_OVERRIDE_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")
_CORPUS_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_CANONICAL_VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: float = Field(default=0.70, gt=0, lt=1)
    validation: float = Field(default=0.15, gt=0, lt=1)
    test: float = Field(default=0.15, gt=0, lt=1)
    seed: str = Field(default="detector-group-split-v1", min_length=1, max_length=128)
    require_non_empty: bool = True

    @model_validator(mode="after")
    def validate_total(self) -> SplitConfig:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("detector dataset split ratios must sum to one")
        return self


class DetectorDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: DetectorRole
    source_directory: Path
    annotations_file: Path = Path("annotations.jsonl")
    output_directory: Path = Path("datasets/detectors")
    classes: tuple[str, ...]
    split: SplitConfig = Field(default_factory=SplitConfig)
    maximum_samples: int = Field(default=100_000, ge=1, le=1_000_000)
    maximum_annotation_line_bytes: int = Field(default=1_000_000, ge=128, le=10_000_000)
    maximum_image_bytes: int = Field(default=20_000_000, ge=1024, le=100_000_000)
    maximum_image_pixels: int = Field(default=40_000_000, ge=1, le=200_000_000)

    @field_validator("classes")
    @classmethod
    def normalize_classes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip().lower() for item in value)
        if not normalized or any(not item for item in normalized):
            raise ValueError("detector dataset requires non-empty class names")
        if len(set(normalized)) != len(normalized):
            raise ValueError("detector dataset class names must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_role_classes(self) -> DetectorDatasetConfig:
        if self.role is DetectorRole.PLATE and self.classes != ("license_plate",):
            raise ValueError("plate detector v1 must use exactly one license_plate class")
        if self.role is DetectorRole.VEHICLE:
            unsupported = set(self.classes) - _CANONICAL_VEHICLE_CLASSES
            if unsupported:
                raise ValueError(f"unsupported canonical vehicle classes: {sorted(unsupported)}")
        return self


class DetectorReleaseGates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_map50: float | None = Field(default=None, ge=0, le=1)
    minimum_map50_95: float | None = Field(default=None, ge=0, le=1)
    minimum_precision: float | None = Field(default=None, ge=0, le=1)
    minimum_recall: float | None = Field(default=None, ge=0, le=1)
    minimum_group_recall: float | None = Field(default=None, ge=0, le=1)
    minimum_night_recall: float | None = Field(default=None, ge=0, le=1)
    minimum_full_bbox_coverage: float | None = Field(default=None, ge=0, le=1)
    minimum_per_class_recall: dict[str, float] = Field(default_factory=dict)

    @field_validator("minimum_per_class_recall")
    @classmethod
    def validate_per_class(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name, threshold in value.items():
            class_name = name.strip().lower()
            if not class_name or not 0 <= threshold <= 1:
                raise ValueError("per-class recall gates must use valid names and [0, 1] values")
            normalized[class_name] = threshold
        return normalized


ScalarOverride = str | int | float | bool


class PaddleDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_path: Path
    base_config: Path
    output_directory: Path
    python_executable: str = "python"
    paddle2onnx_executable: str = "paddle2onnx"
    device: Literal["cpu", "gpu"] = "gpu"
    gpus: tuple[str, ...] = ("0",)
    epochs: int = Field(default=80, ge=1, le=10_000)
    batch_size: int = Field(default=8, ge=1, le=1024)
    workers: int = Field(default=4, ge=0, le=128)
    snapshot_epoch: int = Field(default=5, ge=1, le=10_000)
    pretrain_weights: str | None = None
    maximum_runtime_seconds: int = Field(default=86_400, ge=60, le=2_592_000)
    export_maximum_runtime_seconds: int = Field(default=3_600, ge=30, le=86_400)
    onnx_opset: int = Field(default=11, ge=11, le=21)
    extra_overrides: dict[str, ScalarOverride] = Field(default_factory=dict)

    @field_validator("python_executable", "paddle2onnx_executable")
    @classmethod
    def validate_python_executable(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or "\x00" in stripped:
            raise ValueError("training Python executable is invalid")
        return stripped

    @field_validator("gpus")
    @classmethod
    def validate_gpus(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if not normalized or any(not item or not item.isdigit() for item in normalized):
            raise ValueError("PaddleDetection GPU identifiers must be numeric")
        return normalized

    @field_validator("pretrain_weights")
    @classmethod
    def validate_pretrain_weights(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or "\x00" in stripped:
            raise ValueError("pretrained checkpoint reference is invalid")
        parsed = urlsplit(stripped)
        if parsed.scheme and (
            parsed.scheme not in {"http", "https"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("pretrained checkpoint URL must not contain credentials or query data")
        return stripped

    @field_validator("extra_overrides")
    @classmethod
    def validate_overrides(cls, value: dict[str, ScalarOverride]) -> dict[str, ScalarOverride]:
        if len(value) > 64 or any(not _OVERRIDE_KEY.fullmatch(key) for key in value):
            raise ValueError("PaddleDetection override keys are invalid")
        return value

    @model_validator(mode="after")
    def validate_snapshot_epoch(self) -> PaddleDetectionConfig:
        if self.snapshot_epoch > self.epochs:
            raise ValueError("snapshot_epoch cannot exceed epochs")
        return self


class HubTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_repo: str
    model_repo: str

    @field_validator("dataset_repo", "model_repo")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        stripped = value.strip().strip("/")
        if stripped.count("/") != 1 or any(part in {"", ".", ".."} for part in stripped.split("/")):
            raise ValueError("Hugging Face repository id must be namespace/name")
        return stripped


class DetectorTargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DetectorDatasetConfig
    gates: DetectorReleaseGates = Field(default_factory=DetectorReleaseGates)
    paddledetection: PaddleDetectionConfig
    hub: HubTargetConfig


class HuggingFaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    private: Literal[True] = True
    jobs_enabled: bool = False
    job_image: str = ""
    job_flavor: str = "a10g-small"
    job_namespace: str | None = None
    job_output_bucket: str | None = None

    @field_validator("job_image", "job_flavor")
    @classmethod
    def strip_job_value(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_job_configuration(self) -> HuggingFaceConfig:
        if self.jobs_enabled and not self.enabled:
            raise ValueError("Hugging Face Jobs require the Hub integration to be enabled")
        if self.jobs_enabled and (not self.job_image or not self.job_output_bucket):
            raise ValueError(
                "enabled Hugging Face Jobs require job_image and a persistent output bucket"
            )
        return self


class DataCorpusConfig(BaseModel):
    """Founder-owned compilation identity; source ownership remains separate."""

    model_config = ConfigDict(extra="forbid")

    owner_namespace: str
    founder_id: str
    plate_corpus_id: str
    plate_output_directory: Path
    plate_external_sources_directory: Path = Path("datasets/source/plate/roboflow")
    plate_auxiliary_output_directory: Path = Path("datasets/corpora/plate-auxiliary")
    plate_additional_sources: tuple[Path, ...] = ()

    @field_validator("owner_namespace", "founder_id", "plate_corpus_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _CORPUS_ID.fullmatch(normalized):
            raise ValueError("corpus identity must use lowercase letters, digits, and hyphens")
        return normalized


class ModelTrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    vehicle: DetectorTargetConfig
    plate: DetectorTargetConfig
    corpus: DataCorpusConfig
    huggingface: HuggingFaceConfig = Field(default_factory=HuggingFaceConfig)

    @model_validator(mode="after")
    def validate_roles(self) -> ModelTrainingSettings:
        if self.vehicle.dataset.role is not DetectorRole.VEHICLE:
            raise ValueError("vehicle target must use role=vehicle")
        if self.plate.dataset.role is not DetectorRole.PLATE:
            raise ValueError("plate target must use role=plate")
        return self

    def target(self, role: DetectorRole) -> DetectorTargetConfig:
        return self.vehicle if role is DetectorRole.VEHICLE else self.plate


def load_training_settings(path: str | Path) -> ModelTrainingSettings:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw: Any = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict):
            raise ValueError("training configuration root must be a mapping")
        return ModelTrainingSettings.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid model training configuration: {config_path}") from exc
