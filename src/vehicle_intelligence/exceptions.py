"""Explicit platform exception types."""


class VehicleIntelligenceError(Exception):
    """Base exception for expected platform failures."""


class ConfigurationError(VehicleIntelligenceError):
    """Configuration is missing or internally inconsistent."""


class UnsupportedDetectorProvider(ConfigurationError):
    """A detector provider name is not registered by the composition factory."""


class DependencyUnavailableError(VehicleIntelligenceError):
    """An optional infrastructure dependency is not installed."""


class ModelLoadError(VehicleIntelligenceError):
    """A configured inference model cannot be loaded."""


class InferenceError(VehicleIntelligenceError):
    """An inference provider failed while processing an image."""


class InferenceProtocolError(InferenceError):
    """A local shared-inference message violates the bounded IPC contract."""


class VideoSourceError(VehicleIntelligenceError):
    """A video source cannot be opened or decoded."""


class CameraNotFoundError(VehicleIntelligenceError):
    """A requested camera does not exist."""


class CameraConflictError(VehicleIntelligenceError):
    """A camera create/update conflicts with current persisted state."""


class CameraCapacityError(CameraConflictError):
    """Configured camera capacity has been reached."""


class CameraDiscoveryError(VehicleIntelligenceError):
    """ONVIF discovery could not complete safely."""


class TopologyNotFoundError(VehicleIntelligenceError):
    """A requested topology edge or source fingerprint does not exist."""


class TopologyConflictError(VehicleIntelligenceError):
    """A topology create/update conflicts with current persisted state."""


class IdentityNotFoundError(VehicleIntelligenceError):
    """A requested identity, fingerprint, or review does not exist."""


class IdentityConflictError(VehicleIntelligenceError):
    """An identity review conflicts with current revision or ownership."""


class CredentialEncryptionError(VehicleIntelligenceError):
    """A protected camera credential cannot be encrypted or decrypted."""


class AuthenticationError(VehicleIntelligenceError):
    """A request did not present a valid configured identity."""


class AuthorizationError(VehicleIntelligenceError):
    """An authenticated identity lacks a required permission."""


class AuditWriteError(VehicleIntelligenceError):
    """A required actor-oriented audit record could not be persisted."""


class AuditNotFoundError(VehicleIntelligenceError):
    """A requested audit record does not exist."""


class CameraWorkerError(VehicleIntelligenceError):
    """A managed camera worker cannot be started or stopped cleanly."""


class PolicyNotFoundError(VehicleIntelligenceError):
    """A watchlist, rule, alert, or related policy resource does not exist."""


class PolicyConflictError(VehicleIntelligenceError):
    """A policy resource create/update conflicts with persisted state."""


class RuleValidationError(VehicleIntelligenceError):
    """A declarative rule contains an unsupported or unsafe expression."""


class ActionExecutionError(VehicleIntelligenceError):
    """A rule action could not be completed and may require retry."""


class ActionHandlerError(ActionExecutionError):
    """An action handler failed with a safe machine-readable code."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool = True,
        consume_attempt: bool = True,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.consume_attempt = consume_attempt


class PersistenceError(VehicleIntelligenceError):
    """A repository operation failed."""


class InvalidCursorError(PersistenceError):
    """A client-supplied pagination cursor is malformed or incompatible."""


class EventBusError(VehicleIntelligenceError):
    """An event bus operation failed."""


class EventContractError(VehicleIntelligenceError):
    """A cross-process event does not satisfy the versioned contract."""


class MediaStorageError(VehicleIntelligenceError):
    """A media object could not be stored."""


class FinalizationOutboxError(VehicleIntelligenceError):
    """A complete event/media finalization unit could not be staged durably."""


class FinalizationOutboxRetryableError(FinalizationOutboxError):
    """A transient local outbox failure can be retried without losing the track."""


class FinalizationOutboxFullError(FinalizationOutboxError):
    """The durable finalization outbox reached a configured hard capacity."""


class FinalizationOutboxCorruptionError(FinalizationOutboxError):
    """A finalization outbox entry failed integrity or path validation."""


class MediaAccessError(VehicleIntelligenceError):
    """A protected media reference could not be resolved safely."""


class VehicleEventNotFoundError(VehicleIntelligenceError):
    """A requested vehicle event does not exist."""


class PlateReviewConflictError(VehicleIntelligenceError):
    """A plate review was based on a stale review revision."""


class PlateReviewValidationError(VehicleIntelligenceError):
    """A plate review cannot be applied to the requested event or value."""


class DatasetExportError(VehicleIntelligenceError):
    """A dataset export could not be produced or verified safely."""


class DetectorDatasetError(VehicleIntelligenceError):
    """A vehicle or plate detector dataset is invalid or cannot be built."""


class DatasetReviewNotFoundError(VehicleIntelligenceError):
    """A detector review source, item, or promotion job does not exist."""


class DatasetReviewConflictError(VehicleIntelligenceError):
    """A detector review write conflicts with a newer immutable revision."""


class DatasetReviewValidationError(VehicleIntelligenceError):
    """A detector review decision or promotion request is invalid."""


class DatasetReviewStorageError(VehicleIntelligenceError):
    """Detector review evidence could not be read or persisted safely."""


class DatasetRegistryNotFoundError(VehicleIntelligenceError):
    """An immutable dataset source or Hub synchronization job does not exist."""


class DatasetRegistryConflictError(VehicleIntelligenceError):
    """A dataset export or synchronization conflicts with existing immutable state."""


class DatasetRegistryValidationError(VehicleIntelligenceError):
    """A dataset is not eligible for the requested registry operation."""


class DatasetRegistryStorageError(VehicleIntelligenceError):
    """Dataset catalog or synchronization evidence could not be persisted safely."""


class DetectorCorpusError(VehicleIntelligenceError):
    """A curated detector corpus cannot be imported or verified safely."""


class SampleDataAcquisitionError(VehicleIntelligenceError):
    """External bootstrap detector data could not be acquired or verified safely."""


class ModelTrainingError(VehicleIntelligenceError):
    """An offline detector training run could not be completed safely."""


class ModelTrainingNotFoundError(ModelTrainingError):
    """A persisted or remote detector training run does not exist."""


class ModelTrainingConflictError(ModelTrainingError):
    """A detector training operation conflicts with current run state."""


class ModelTrainingValidationError(ModelTrainingError):
    """A detector training request is not eligible or safely configured."""


class ModelTrainingStorageError(ModelTrainingError):
    """Detector training evidence could not be read or persisted safely."""


class ModelEvaluationError(VehicleIntelligenceError):
    """Detector predictions or release-gate evidence are invalid."""


class ModelArtifactError(VehicleIntelligenceError):
    """A promoted model artifact is incomplete, invalid, or tampered with."""


class ModelRegistryError(VehicleIntelligenceError):
    """A private remote dataset/model registry operation failed."""
