"""Provider-neutral contracts for durable detector model training runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from vehicle_intelligence.training.domain import DetectorRole


class ModelTrainingRunStatus(StrEnum):
    QUEUED = "QUEUED"
    SUBMITTING = "SUBMITTING"
    SCHEDULING = "SCHEDULING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

    @property
    def active(self) -> bool:
        return self in {
            ModelTrainingRunStatus.QUEUED,
            ModelTrainingRunStatus.SUBMITTING,
            ModelTrainingRunStatus.SCHEDULING,
            ModelTrainingRunStatus.RUNNING,
        }


@dataclass(frozen=True, slots=True)
class ModelTrainingParameters:
    epochs: int
    batch_size: int
    workers: int
    snapshot_epoch: int
    timeout_seconds: int
    hardware_flavor: str


@dataclass(frozen=True, slots=True)
class ModelTrainingRun:
    id: str
    role: DetectorRole
    status: ModelTrainingRunStatus
    source_id: str
    source_manifest_sha256: str
    export_id: str
    export_manifest_sha256: str
    dataset_repo_id: str
    dataset_revision: str
    dataset_commit_sha: str
    model_repo_id: str
    model_name: str
    model_version: str
    architecture: str
    parameters: ModelTrainingParameters
    requested_by: str
    dataset_rights_confirmed: bool
    compute_cost_confirmed: bool
    restricted_data_confirmed: bool
    created_at: datetime
    updated_at: datetime
    output_bucket: str
    output_path: str
    remote_job_id: str | None = None
    remote_job_url: str | None = None
    remote_message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ModelTrainingDefaults:
    role: DetectorRole
    architecture: str
    base_config: str
    model_repo_id: str
    dataset_repo_id: str
    epochs: int
    batch_size: int
    workers: int
    snapshot_epoch: int
    timeout_seconds: int
    hardware_flavor: str


@dataclass(frozen=True, slots=True)
class ModelTrainingCapabilities:
    enabled: bool
    jobs_enabled: bool
    credentials_configured: bool
    image_configured: bool
    output_bucket_configured: bool
    submissions_enabled: bool
    job_image: str | None
    output_bucket: str | None
    namespace: str | None
    defaults: ModelTrainingDefaults
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RemoteTrainingJob:
    id: str
    status: str
    url: str | None = None
    message: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RemoteTrainingSubmission:
    image: str
    command: tuple[str, ...]
    hardware_flavor: str
    dataset_repo_id: str
    dataset_revision: str
    output_bucket: str
    namespace: str | None
    timeout_seconds: int
    name: str
    labels: dict[str, str]


@dataclass(frozen=True, slots=True)
class ModelTrainingLog:
    run_id: str
    lines: tuple[str, ...]
    available: bool
