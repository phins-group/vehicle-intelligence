"""Domain contracts for immutable detector datasets and private Hub synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DatasetHubSyncStatus(StrEnum):
    QUEUED = "QUEUED"
    PREPARING_EXPORT = "PREPARING_EXPORT"
    UPLOADING = "UPLOADING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DetectorDatasetSampleKind(StrEnum):
    ALL = "ALL"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@dataclass(frozen=True, slots=True)
class DetectorDatasetSampleBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class DetectorDatasetSampleAnnotation:
    class_name: str
    bbox: DetectorDatasetSampleBox


@dataclass(frozen=True, slots=True)
class DetectorDatasetSamplePreview:
    source_id: str
    sample_id: str
    image_sha256: str
    camera_id: str
    group_id: str
    captured_at: datetime
    split: str | None
    lighting: str
    annotation_status: str | None
    negative: bool
    image_width: int
    image_height: int
    annotations: tuple[DetectorDatasetSampleAnnotation, ...]


@dataclass(frozen=True, slots=True)
class DetectorDatasetSamplePage:
    items: tuple[DetectorDatasetSamplePreview, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class DetectorDatasetSampleImage:
    path: str
    media_type: str
    sha256: str


@dataclass(frozen=True, slots=True)
class DetectorDatasetExport:
    export_id: str
    manifest_sha256: str
    created_at: datetime
    sample_count: int
    annotation_count: int
    negative_sample_count: int
    split_counts: dict[str, int]
    release_eligible: bool
    distribution_eligible: bool
    source_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetHubSyncJob:
    id: str
    source_id: str
    source_manifest_sha256: str
    export_id: str
    repo_id: str
    requested_revision: str
    status: DatasetHubSyncStatus
    requested_by: str
    restricted_transfer_confirmed: bool
    created_at: datetime
    updated_at: datetime
    export_manifest_sha256: str | None = None
    hub_commit_sha: str | None = None
    hub_url: str | None = None
    reused_export: bool = False
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class DetectorDatasetVersion:
    source_id: str
    source_manifest_sha256: str
    created_at: datetime
    sample_count: int
    annotation_count: int
    negative_sample_count: int
    review_queue_count: int
    release_eligible: bool
    distribution_eligible: bool
    privacy_classification: str
    parent_source_id: str | None
    export: DetectorDatasetExport | None = None
    latest_sync: DatasetHubSyncJob | None = None


@dataclass(frozen=True, slots=True)
class DatasetRegistryCapabilities:
    enabled: bool
    hub_enabled: bool
    repo_id: str | None
    credentials_configured: bool
    restricted_private_sync_enabled: bool
