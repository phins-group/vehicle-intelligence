"""Contracts for externally sourced, non-acceptance detector samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vehicle_intelligence.training.domain import DetectorRole, DetectorSample


class BootstrapSourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=128)
    dataset_url: str = Field(min_length=1, max_length=2048)
    revision: str = Field(min_length=1, max_length=128)
    annotation_license: str = Field(min_length=1, max_length=128)
    image_license: str = Field(min_length=1, max_length=128)
    license_review_status: str = "REVIEW_REQUIRED"
    acceptance_eligible: bool = False


@dataclass(frozen=True, slots=True)
class AcquiredDetectorSample:
    sample: DetectorSample
    image_bytes: bytes
    attribution: dict[str, str]


@dataclass(frozen=True, slots=True)
class BootstrapBuildResult:
    role: DetectorRole
    directory: Path
    manifest_sha256: str
    sample_count: int
    annotation_count: int
    reused: bool = False
