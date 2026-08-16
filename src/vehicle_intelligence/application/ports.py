from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.domain import (
    ActionExecution,
    Alert,
    AlertStatus,
    AuditAction,
    AuditLog,
    AuditResourceType,
    Camera,
    CameraHealth,
    CameraTopologyEdge,
    DatasetSample,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    Detection,
    EmbeddingModel,
    EmbeddingVector,
    IdentityMergeReview,
    IdentityReviewResult,
    IdentitySplitReview,
    LifecycleReconcileResult,
    LiveFrameMetadata,
    LiveFramePacket,
    MediaKind,
    MediaRetentionClaim,
    ModelQualityReport,
    OCRResult,
    OnvifDiscoveredDevice,
    PlateDetection,
    PlateQuality,
    Principal,
    Rule,
    RuleAction,
    TrackedDetection,
    VectorNeighbor,
    VehicleEvent,
    VehicleFingerprint,
    VehicleIdentity,
    VideoFrame,
    WatchlistEntry,
    WatchlistType,
)


@dataclass(frozen=True, slots=True)
class ImageVariant:
    name: str
    image: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class EventQuery:
    limit: int = 50
    cursor: str | None = None
    camera_id: str | None = None
    plate: str | None = None
    event_type: str | None = None
    direction: str | None = None
    status: str | None = None
    from_time: datetime | None = None
    to_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class EventPage:
    items: tuple[VehicleEvent, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class AlertQuery:
    limit: int = 50
    cursor: str | None = None
    status: AlertStatus | None = None
    plate: str | None = None
    camera_id: str | None = None
    rule_id: str | None = None


@dataclass(frozen=True, slots=True)
class AlertPage:
    items: tuple[Alert, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class AuditQuery:
    limit: int = 50
    cursor: str | None = None
    actor_id: str | None = None
    action: AuditAction | None = None
    resource_type: AuditResourceType | None = None
    resource_id: str | None = None
    from_time: datetime | None = None
    to_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class AuditPage:
    items: tuple[AuditLog, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class DatasetSampleQuery:
    limit: int = 50
    cursor: str | None = None
    sample_type: DatasetSampleType | None = None
    status: DatasetSampleStatus | None = None
    reason: DatasetSampleReason | None = None
    source_event_id: str | None = None
    from_time: datetime | None = None
    to_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class DatasetSamplePage:
    items: tuple[DatasetSample, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class ActionContext:
    execution_id: str
    event: VehicleEvent
    rule: Rule
    action: RuleAction
    watchlist_types: tuple[WatchlistType, ...]


@dataclass(frozen=True, slots=True)
class EncryptedCameraCredential:
    camera_id: str
    token: str


@dataclass(frozen=True, slots=True)
class EventPolicyResult:
    matched_rules: int = 0
    actions_succeeded: int = 0
    actions_skipped: int = 0


@dataclass(frozen=True, slots=True)
class VectorSearchQuery:
    vector: tuple[float, ...]
    model: EmbeddingModel
    candidate_ids: tuple[str, ...]
    limit: int = 20
    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        if len(self.vector) != self.model.dimension:
            raise ValueError("vector search dimension does not match model")
        if not 1 <= self.limit <= 200:
            raise ValueError("vector search limit must be in [1, 200]")
        if not -1 <= self.minimum_score <= 1:
            raise ValueError("vector minimum score must be in [-1, 1]")
        if len(self.candidate_ids) > 5000:
            raise ValueError("vector candidate set is bounded to 5000")


@dataclass(frozen=True, slots=True)
class StreamHeartbeat:
    timestamp: datetime
    stream_epoch: int

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("stream heartbeat timestamp must be timezone-aware")
        if self.stream_epoch < 0:
            raise ValueError("stream heartbeat epoch cannot be negative")


@dataclass(frozen=True, slots=True)
class CameraConnectionTestResult:
    connected: bool
    latency_ms: float
    tested_at: datetime
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("camera connection latency cannot be negative")
        if self.tested_at.tzinfo is None:
            raise ValueError("camera connection test timestamp must be timezone-aware")


class CameraCreateOutcome(StrEnum):
    CREATED = "CREATED"
    CONFLICT = "CONFLICT"
    CAPACITY_REACHED = "CAPACITY_REACHED"


DetectorOutput = TypeVar("DetectorOutput")


@runtime_checkable
class Detector(Protocol[DetectorOutput]):
    def detect(self, image: NDArray[np.uint8]) -> list[DetectorOutput]: ...


@runtime_checkable
class VehicleDetector(Detector[Detection], Protocol):
    pass


@runtime_checkable
class BatchVehicleDetector(Protocol):
    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]: ...


@runtime_checkable
class PlateDetector(Detector[PlateDetection], Protocol):
    pass


@runtime_checkable
class BatchPlateDetector(Protocol):
    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[PlateDetection]]: ...


@runtime_checkable
class VehicleTracker(Protocol):
    def update(
        self, detections: Sequence[Detection], image: NDArray[np.uint8]
    ) -> list[TrackedDetection]: ...

    def reset(self) -> None: ...


@runtime_checkable
class OCRProvider(Protocol):
    def recognize(self, image: NDArray[np.uint8]) -> OCRResult: ...


@runtime_checkable
class PlatePreprocessor(Protocol):
    def variants(
        self,
        image: NDArray[np.uint8],
        quality: PlateQuality,
        detection: PlateDetection,
    ) -> list[ImageVariant]: ...


@runtime_checkable
class VideoSource(Protocol):
    @property
    def source_id(self) -> str: ...

    @property
    def source_fps(self) -> float: ...

    def frames(self) -> Iterator[VideoFrame | StreamHeartbeat]: ...

    def close(self) -> None: ...


@runtime_checkable
class ImageEncoder(Protocol):
    def encode_jpeg(self, image: NDArray[np.uint8]) -> bytes: ...


@dataclass(frozen=True, slots=True)
class DatasetImageTranscodeResult:
    jpeg: bytes | None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if (self.jpeg is None) == (self.error_code is None):
            raise ValueError("dataset image transcode result must contain JPEG data or an error")
        if self.jpeg == b"":
            raise ValueError("normalized dataset JPEG cannot be empty")
        if self.error_code is not None and not self.error_code.strip():
            raise ValueError("dataset image transcode error code cannot be blank")


@runtime_checkable
class DatasetImageTranscoder(Protocol):
    def normalize_jpeg(
        self,
        source: bytes,
        *,
        maximum_pixels: int,
        jpeg_quality: int,
    ) -> DatasetImageTranscodeResult: ...


@dataclass(frozen=True, slots=True)
class EncodedLivePreview:
    jpeg: bytes
    width: int
    height: int


@runtime_checkable
class LivePreviewEncoder(Protocol):
    def encode(
        self,
        image: NDArray[np.uint8],
        maximum_width: int,
        jpeg_quality: int,
    ) -> EncodedLivePreview: ...


@runtime_checkable
class LivePreviewSink(Protocol):
    async def report(
        self,
        image: NDArray[np.uint8],
        metadata: LiveFrameMetadata,
    ) -> bool: ...


@runtime_checkable
class MediaStorage(Protocol):
    async def put(self, key: str, data: bytes, content_type: str) -> str: ...


@runtime_checkable
class MediaStorageLifecycle(Protocol):
    async def close(self) -> None: ...


@runtime_checkable
class MediaUrlSigner(Protocol):
    async def presign_get(self, key: str, expires: timedelta) -> str | None: ...


@runtime_checkable
class MediaObjectInspector(Protocol):
    async def exists(self, key: str) -> bool: ...


@runtime_checkable
class MediaObjectReader(Protocol):
    async def get(self, key: str, maximum_bytes: int) -> bytes | None: ...


@runtime_checkable
class MediaObjectCleaner(Protocol):
    async def remove(self, key: str) -> None: ...


@runtime_checkable
class MediaLifecycleManager(Protocol):
    async def reconcile_lifecycle(self, debug_expiry_days: int) -> LifecycleReconcileResult: ...


@runtime_checkable
class RetentionRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def claim_media(
        self,
        kind: MediaKind,
        older_than: datetime,
        stale_before: datetime,
        lease_id: str,
        limit: int,
    ) -> list[MediaRetentionClaim]: ...

    async def mark_media_deleted(
        self,
        claim: MediaRetentionClaim,
        deleted_at: datetime,
    ) -> None: ...

    async def mark_media_failed(
        self,
        claim: MediaRetentionClaim,
        error_code: str,
        failed_at: datetime,
    ) -> None: ...

    async def delete_expired_events(self, older_than: datetime, limit: int) -> int: ...

    async def close(self) -> None: ...


@runtime_checkable
class CredentialCipher(Protocol):
    def encrypt(self, plaintext: str, context: str) -> str: ...

    def decrypt(self, token: str, context: str) -> str: ...


@runtime_checkable
class RotatingCredentialCipher(CredentialCipher, Protocol):
    @property
    def active_key_id(self) -> str: ...

    def needs_rotation(self, token: str) -> bool: ...


@runtime_checkable
class CameraRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def create(self, camera: Camera) -> bool: ...

    async def create_with_capacity(
        self,
        camera: Camera,
        maximum_cameras: int,
    ) -> CameraCreateOutcome: ...

    async def replace(self, camera: Camera, expected_revision: int) -> bool: ...

    async def get(self, camera_id: str) -> Camera | None: ...

    async def list(self, enabled_only: bool = False) -> list[Camera]: ...

    async def count(self) -> int: ...

    async def delete(self, camera_id: str) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class CameraHealthRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def save(self, health: CameraHealth) -> None: ...

    async def get(self, camera_id: str) -> CameraHealth | None: ...

    async def list(self) -> list[CameraHealth]: ...

    async def delete(self, camera_id: str) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class CameraConnectionTester(Protocol):
    async def test(self, camera: Camera) -> CameraConnectionTestResult: ...


@runtime_checkable
class OnvifDiscoveryProvider(Protocol):
    async def discover(self) -> list[OnvifDiscoveredDevice]: ...


@runtime_checkable
class CameraWorkerHandle(Protocol):
    @property
    def running(self) -> bool: ...

    @property
    def return_code(self) -> int | None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class CameraWorkerLauncher(Protocol):
    async def start(self, camera: Camera) -> CameraWorkerHandle: ...


@runtime_checkable
class InferenceServiceHandle(Protocol):
    @property
    def running(self) -> bool: ...

    @property
    def return_code(self) -> int | None: ...

    async def stop(self) -> None: ...


@runtime_checkable
class InferenceServiceLauncher(Protocol):
    async def start(self) -> InferenceServiceHandle: ...


@runtime_checkable
class WatchlistRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def create(self, entry: WatchlistEntry) -> bool: ...

    async def replace(self, entry: WatchlistEntry, expected_revision: int) -> bool: ...

    async def get(self, entry_id: str) -> WatchlistEntry | None: ...

    async def list(
        self,
        list_type: WatchlistType | None = None,
        enabled: bool | None = None,
        limit: int = 200,
    ) -> list[WatchlistEntry]: ...

    async def find_active_by_plate(
        self, plate: str, timestamp: datetime
    ) -> list[WatchlistEntry]: ...

    async def delete(self, entry_id: str) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class RuleRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def create(self, rule: Rule) -> bool: ...

    async def replace(self, rule: Rule, expected_revision: int) -> bool: ...

    async def get(self, rule_id: str) -> Rule | None: ...

    async def list(self, enabled_only: bool = False, limit: int = 200) -> list[Rule]: ...

    async def delete(self, rule_id: str) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class AlertRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def create(self, alert: Alert) -> bool: ...

    async def replace(self, alert: Alert, expected_revision: int) -> bool: ...

    async def get(self, alert_id: str) -> Alert | None: ...

    async def list(self, query: AlertQuery) -> AlertPage: ...

    async def close(self) -> None: ...


@runtime_checkable
class Authenticator(Protocol):
    async def authenticate(self, bearer_token: str) -> Principal | None: ...


@runtime_checkable
class CameraCredentialRotationRepository(Protocol):
    async def list_encrypted_credentials(
        self,
        after_camera_id: str | None,
        limit: int,
    ) -> tuple[EncryptedCameraCredential, ...]: ...

    async def replace_encrypted_credential(
        self,
        camera_id: str,
        expected_token: str,
        replacement_token: str,
        rotated_at: datetime,
    ) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class VehicleIdentityRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def register_observation(
        self,
        identity: VehicleIdentity,
        fingerprint: VehicleFingerprint,
    ) -> bool: ...

    async def get(self, vehicle_id: str) -> VehicleIdentity | None: ...

    async def get_fingerprint(self, fingerprint_id: str) -> VehicleFingerprint | None: ...

    async def get_fingerprints(
        self,
        fingerprint_ids: tuple[str, ...],
    ) -> tuple[VehicleFingerprint, ...]: ...

    async def list_fingerprints(
        self,
        vehicle_id: str,
        limit: int = 200,
    ) -> tuple[VehicleFingerprint, ...]: ...

    async def find_by_plate(
        self,
        plate: str,
        limit: int = 20,
    ) -> tuple[VehicleIdentity, ...]: ...

    async def find_fingerprints_by_camera_time(
        self,
        camera_id: str,
        from_time: datetime,
        to_time: datetime,
        limit: int,
    ) -> tuple[VehicleFingerprint, ...]: ...

    async def review_merge(self, review: IdentityMergeReview) -> IdentityReviewResult: ...

    async def review_split(self, review: IdentitySplitReview) -> IdentityReviewResult: ...

    async def get_review(self, review_id: str) -> IdentityReviewResult | None: ...

    async def close(self) -> None: ...


@runtime_checkable
class VehicleEventIdentityLinker(Protocol):
    async def assign_vehicle_id(self, event_id: str, vehicle_id: str) -> bool: ...

    async def reassign_vehicle_ids(
        self,
        event_ids: tuple[str, ...],
        source_vehicle_id: str,
        target_vehicle_id: str,
    ) -> int: ...


@runtime_checkable
class VectorRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def put(self, vector: EmbeddingVector) -> bool: ...

    async def get(self, vector_id: str) -> EmbeddingVector | None: ...

    async def search(self, query: VectorSearchQuery) -> tuple[VectorNeighbor, ...]: ...

    async def close(self) -> None: ...


@runtime_checkable
class VehicleEmbeddingProvider(Protocol):
    @property
    def model(self) -> EmbeddingModel: ...

    def embed(self, image: NDArray[np.uint8]) -> tuple[float, ...]: ...


@runtime_checkable
class CameraTopologyRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def create(self, edge: CameraTopologyEdge) -> bool: ...

    async def replace(
        self,
        edge: CameraTopologyEdge,
        expected_revision: int,
    ) -> bool: ...

    async def get(self, edge_id: str) -> CameraTopologyEdge | None: ...

    async def list(
        self,
        *,
        from_camera_id: str | None = None,
        to_camera_id: str | None = None,
        enabled_only: bool = False,
        limit: int = 200,
    ) -> tuple[CameraTopologyEdge, ...]: ...

    async def delete(self, edge_id: str) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class AuditLogRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def append(self, entry: AuditLog) -> None: ...

    async def get(self, entry_id: str) -> AuditLog | None: ...

    async def list(self, query: AuditQuery) -> AuditPage: ...

    async def close(self) -> None: ...


@runtime_checkable
class ActionExecutionRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def claim(
        self,
        execution: ActionExecution,
        stale_before: datetime,
        maximum_attempts: int,
    ) -> ActionExecution | None: ...

    async def get(self, execution_id: str) -> ActionExecution | None: ...

    async def mark_succeeded(self, execution_id: str, timestamp: datetime) -> None: ...

    async def mark_failed(
        self,
        execution_id: str,
        error_code: str,
        timestamp: datetime,
        *,
        terminal: bool,
        maximum_attempts: int,
        consume_attempt: bool = True,
    ) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class ActionHandler(Protocol):
    async def initialize(self) -> None: ...

    async def execute(self, context: ActionContext) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class VehicleEventPostProcessor(Protocol):
    async def initialize(self) -> None: ...

    async def process(self, event: VehicleEvent) -> EventPolicyResult: ...

    async def close(self) -> None: ...


@runtime_checkable
class VehicleEventRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def save(self, event: VehicleEvent) -> bool: ...

    async def get(self, event_id: str) -> VehicleEvent | None: ...

    async def list(self, query: EventQuery) -> EventPage: ...

    async def find_by_plate(self, plate: str, limit: int) -> list[VehicleEvent]: ...

    async def timeline(
        self,
        vehicle_id: str,
        *,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int = 1000,
        ascending: bool = True,
    ) -> tuple[VehicleEvent, ...]: ...

    async def update_plate_review(
        self,
        event: VehicleEvent,
        expected_revision: int,
    ) -> VehicleEvent | None: ...

    async def close(self) -> None: ...


@runtime_checkable
class DatasetSampleRepository(Protocol):
    async def ensure_indexes(self) -> None: ...

    async def create(self, sample: DatasetSample) -> bool: ...

    async def get(self, sample_id: str) -> DatasetSample | None: ...

    async def list(self, query: DatasetSampleQuery) -> DatasetSamplePage: ...

    async def claim_for_export(
        self,
        export_id: str,
        limit: int,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> tuple[DatasetSample, ...]: ...

    async def mark_exported(
        self,
        sample_ids: tuple[str, ...],
        export_id: str,
        manifest_sha256: str,
        exported_at: datetime,
    ) -> int: ...

    async def mark_export_failed(
        self,
        sample_ids: tuple[str, ...],
        export_id: str,
        error_code: str,
    ) -> int: ...

    async def close(self) -> None: ...


@runtime_checkable
class ModelQualityRepository(Protocol):
    async def summarize(
        self,
        from_time: datetime,
        to_time: datetime,
        generated_at: datetime,
        maximum_models: int,
    ) -> ModelQualityReport: ...

    async def close(self) -> None: ...


@runtime_checkable
class VehicleEventPublisher(Protocol):
    async def initialize(self) -> None: ...

    async def publish(self, event: VehicleEvent) -> bool: ...

    async def close(self) -> None: ...


@runtime_checkable
class VehicleEventCodec(Protocol):
    def encode(self, event: VehicleEvent) -> str: ...

    def decode(self, payload: str) -> VehicleEvent: ...


@dataclass(frozen=True, slots=True)
class BrokerMessage:
    message_id: str
    payload: str


@runtime_checkable
class EventMessageConsumer(Protocol):
    async def initialize(self) -> None: ...

    async def read_new(self) -> list[BrokerMessage]: ...

    async def reclaim_stale(self) -> list[BrokerMessage]: ...

    async def acknowledge(self, message_id: str) -> None: ...

    async def acknowledge_many(self, message_ids: tuple[str, ...]) -> None: ...

    async def dead_letter(self, message: BrokerMessage, reason: str) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class RealtimeEventPublisher(Protocol):
    async def initialize(self) -> None: ...

    async def publish(self, event: VehicleEvent) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class RealtimeEventSubscriber(Protocol):
    async def connect(self) -> None: ...

    async def receive(self, timeout_seconds: float) -> VehicleEvent | None: ...

    async def disconnect(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class LiveFramePublisher(Protocol):
    async def initialize(self) -> None: ...

    async def publish(self, packet: LiveFramePacket) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class LiveFrameSubscriber(Protocol):
    async def connect(self) -> None: ...

    async def receive(self, timeout_seconds: float) -> LiveFramePacket | None: ...

    async def disconnect(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class LiveFrameCodec(Protocol):
    def encode(self, packet: LiveFramePacket) -> str: ...

    def decode(self, payload: str) -> LiveFramePacket: ...
