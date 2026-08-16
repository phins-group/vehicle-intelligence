"""Validated, YAML-backed and environment-overridable configuration."""

from __future__ import annotations

import re
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from vehicle_intelligence.exceptions import ConfigurationError


class AppConfig(BaseModel):
    environment: str = "development"
    log_level: str = "INFO"
    config_version: str = "default"

    @field_validator("environment")
    @classmethod
    def normalize_environment(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if not normalized or len(normalized) > 64:
            raise ValueError("application environment is invalid")
        return normalized

    @field_validator("config_version")
    @classmethod
    def validate_config_version(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 128:
            raise ValueError("application config version is invalid")
        return stripped


class CameraConfig(BaseModel):
    id: str = "sample-camera"
    name: str = "Sample Camera"
    zone: str | None = None
    fps_limit: float = Field(default=6.0, gt=0)
    direction: Literal["ENTRY", "EXIT", "BOTH"] = "BOTH"
    roi: list[tuple[float, float]] | None = None
    crossing_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    crossing_positive_to_negative: Literal["ENTER", "EXIT"] = "ENTER"
    finalize_on_crossing: bool = False

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or any(char in stripped for char in "/\\\0"):
            raise ValueError("camera id must be a non-empty path-safe value")
        return stripped

    @field_validator("roi")
    @classmethod
    def validate_roi(
        cls, value: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        if value is not None and len(value) < 3:
            raise ValueError("camera ROI requires at least three points")
        return value


class PicoDetOptions(BaseModel):
    """PicoDet ONNX preprocessing and head-decoding settings."""

    strides: tuple[int, ...] = (8, 16, 32, 64)
    nms_top_k: int = Field(default=1000, ge=1, le=100_000)
    keep_top_k: int = Field(default=100, ge=1, le=10_000)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    scale: float = Field(default=1.0 / 255.0, gt=0)

    @field_validator("strides")
    @classmethod
    def validate_strides(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(stride <= 0 for stride in value):
            raise ValueError("PicoDet strides must be positive")
        if len(set(value)) != len(value):
            raise ValueError("PicoDet strides must be unique")
        return value

    @field_validator("std")
    @classmethod
    def validate_std(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(component <= 0 for component in value):
            raise ValueError("PicoDet normalization std values must be positive")
        return value


class DetectorConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: str = "ultralytics"
    model_path: str | None = None
    model_name: str
    model_version: str
    model_hash: str | None = None
    confidence: float = Field(ge=0, le=1)
    iou: float = Field(ge=0, le=1)
    image_size: int = Field(default=640, gt=0)
    device: str | None = None
    execution_providers: list[str] = Field(default_factory=list, max_length=8)
    onnx_output_format: Literal["auto", "raw", "nms"] = "auto"
    model_classes: list[str] | None = None
    picodet: PicoDetOptions = Field(default_factory=PicoDetOptions)

    @field_validator("execution_providers")
    @classmethod
    def validate_execution_providers(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if len(normalized) != len(value):
            raise ValueError("execution provider names cannot be empty")
        return list(dict.fromkeys(normalized))

    @field_validator("model_classes")
    @classmethod
    def validate_model_classes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip().lower() for item in value if item.strip()]
        if not normalized or len(normalized) != len(value):
            raise ValueError("model class names cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("model class names must be unique")
        return normalized


class VehicleDetectorConfig(DetectorConfig):
    classes: list[str] = Field(default_factory=lambda: ["car", "motorcycle", "bus", "truck"])

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value if item.strip()]
        if not normalized:
            raise ValueError("at least one vehicle class is required")
        return list(dict.fromkeys(normalized))


class PlateCropConfig(BaseModel):
    horizontal_padding_ratio: float = Field(default=0.08, ge=0, le=1)
    vertical_padding_ratio: float = Field(default=0.08, ge=0, le=1)
    two_line_top_expansion_ratio: float = Field(default=1.0, ge=0, le=3)
    two_line_vehicle_classes: list[str] = Field(default_factory=lambda: ["motorcycle"])

    @field_validator("two_line_vehicle_classes")
    @classmethod
    def normalize_two_line_vehicle_classes(cls, value: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in value if item.strip()]
        return list(dict.fromkeys(normalized))


class QualityWeights(BaseModel):
    sharpness: float = Field(default=0.25, ge=0)
    brightness: float = Field(default=0.15, ge=0)
    contrast: float = Field(default=0.15, ge=0)
    resolution: float = Field(default=0.20, ge=0)
    angle: float = Field(default=0.10, ge=0)
    detector: float = Field(default=0.15, ge=0)

    @model_validator(mode="after")
    def validate_total(self) -> QualityWeights:
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("plate quality weights must have a positive sum")
        return self


class PlateQualityConfig(BaseModel):
    minimum: float = Field(default=0.55, ge=0, le=1)
    min_width: int = Field(default=48, gt=0)
    min_height: int = Field(default=16, gt=0)
    target_width: int = Field(default=160, gt=0)
    target_height: int = Field(default=50, gt=0)
    blur_reference: float = Field(default=180.0, gt=0)
    brightness_min: float = Field(default=45.0, ge=0, le=255)
    brightness_max: float = Field(default=220.0, ge=0, le=255)
    contrast_reference: float = Field(default=55.0, gt=0)
    aspect_ratio_min: float = Field(default=1.5, gt=0)
    aspect_ratio_ideal: float = Field(default=3.2, gt=0)
    aspect_ratio_max: float = Field(default=6.5, gt=0)
    weights: QualityWeights = Field(default_factory=QualityWeights)

    @model_validator(mode="after")
    def validate_brightness_range(self) -> PlateQualityConfig:
        if self.brightness_min >= self.brightness_max:
            raise ValueError("brightness_min must be less than brightness_max")
        if not self.aspect_ratio_min < self.aspect_ratio_ideal < self.aspect_ratio_max:
            raise ValueError("plate aspect ratios must satisfy min < ideal < max")
        return self


class PreprocessingConfig(BaseModel):
    enabled: bool = True
    resize_width: int = Field(default=320, gt=0)
    apply_clahe_below_contrast: float = Field(default=0.65, ge=0, le=1)
    denoise_below_sharpness: float = Field(default=0.45, ge=0, le=1)
    sharpen_below_sharpness: float = Field(default=0.70, ge=0, le=1)
    clahe_clip_limit: float = Field(default=2.0, gt=0)


class OCRConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: str = "paddleocr"
    minimum_confidence: float = Field(default=0.60, ge=0, le=1)
    detection_model_name: str = "PP-OCRv5_mobile_det"
    model_name: str = "PP-OCRv5_mobile_rec"
    model_version: str = "PP-OCRv5"
    model_hash: str | None = None
    detection_model_directory: str | None = None
    detection_model_hash: str | None = None
    recognition_model_directory: str | None = None
    recognition_model_hash: str | None = None
    device: str = "cpu"
    language: str = "en"
    allow_partial_plate: bool = False
    partial_min_characters: int = Field(default=4, ge=1, le=32)
    partial_max_characters: int = Field(default=12, ge=1, le=32)
    track_frame_interval: int = Field(default=2, ge=1, le=120)
    variant_early_stop_confidence: float | None = Field(default=0.95, ge=0, le=1)
    consensus_stop_min_observations: int | None = Field(default=3, ge=2, le=64)
    consensus_stop_min_confidence: float = Field(default=0.90, ge=0, le=1)

    @model_validator(mode="after")
    def validate_partial_length_range(self) -> OCRConfig:
        if self.partial_max_characters < self.partial_min_characters:
            raise ValueError("partial plate maximum cannot be below its minimum")
        if (
            self.variant_early_stop_confidence is not None
            and self.variant_early_stop_confidence < self.minimum_confidence
        ):
            raise ValueError("OCR variant early-stop confidence cannot be below the minimum")
        if self.consensus_stop_min_confidence < self.minimum_confidence:
            raise ValueError("OCR consensus-stop confidence cannot be below the minimum")
        return self


class SnapshotSelectionConfig(BaseModel):
    sharpness_reference: float = Field(default=220.0, gt=0)
    vehicle_area_weight: float = Field(default=0.35, ge=0)
    sharpness_weight: float = Field(default=0.35, ge=0)
    detector_confidence_weight: float = Field(default=0.30, ge=0)
    plate_quality_weight: float = Field(default=0.40, ge=0)
    plate_ocr_weight: float = Field(default=0.35, ge=0)
    plate_detector_weight: float = Field(default=0.25, ge=0)

    @model_validator(mode="after")
    def validate_weight_groups(self) -> SnapshotSelectionConfig:
        vehicle_total = (
            self.vehicle_area_weight + self.sharpness_weight + self.detector_confidence_weight
        )
        plate_total = self.plate_quality_weight + self.plate_ocr_weight + self.plate_detector_weight
        if vehicle_total <= 0 or plate_total <= 0:
            raise ValueError("snapshot selection weight groups must be positive")
        return self


class VisionConfig(BaseModel):
    plate_only: bool = False
    vehicle_detection: VehicleDetectorConfig
    plate_detection: DetectorConfig
    plate_crop: PlateCropConfig = Field(default_factory=PlateCropConfig)
    plate_quality: PlateQualityConfig = Field(default_factory=PlateQualityConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    snapshot_selection: SnapshotSelectionConfig = Field(default_factory=SnapshotSelectionConfig)


class TrackingConfig(BaseModel):
    provider: str = "bytetrack"
    activation_threshold: float = Field(default=0.25, ge=0, le=1)
    lost_track_buffer: int = Field(default=30, ge=1)
    minimum_matching_threshold: float = Field(default=0.80, ge=0, le=1)
    minimum_consecutive_frames: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=2.0, gt=0)
    max_trajectory_points: int = Field(default=256, ge=2)
    max_plate_observations: int = Field(default=64, ge=1)


class RTSPConfig(BaseModel):
    queue_size: int = Field(default=3, ge=1, le=32)
    reconnect_initial_seconds: float = Field(default=0.5, gt=0)
    reconnect_max_seconds: float = Field(default=30.0, gt=0)
    open_timeout_ms: int = Field(default=5000, gt=0)
    read_timeout_ms: int = Field(default=5000, gt=0)
    consumer_wait_seconds: float = Field(default=0.25, gt=0, le=5)
    shutdown_join_seconds: float = Field(default=6.0, gt=0, le=30)

    @model_validator(mode="after")
    def validate_reconnect_range(self) -> RTSPConfig:
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise ValueError("reconnect_max_seconds cannot be below the initial delay")
        return self


class CameraManagerConfig(BaseModel):
    reconcile_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    restart_backoff_seconds: float = Field(default=5.0, gt=0, le=300)
    restart_backoff_max_seconds: float = Field(default=120.0, gt=0, le=3600)
    restart_stability_seconds: float = Field(default=60.0, gt=0, le=3600)
    worker_shutdown_seconds: float = Field(default=15.0, gt=0, le=120)
    health_publish_interval_seconds: float = Field(default=5.0, gt=0, le=300)
    connection_test_concurrency: int = Field(default=4, ge=1, le=32)
    maximum_configured_cameras: int = Field(default=256, ge=1, le=10_000)
    maximum_active_workers: int = Field(default=32, ge=1, le=1024)
    maximum_starts_per_reconcile: int = Field(default=4, ge=1, le=128)
    batch_create_limit: int = Field(default=50, ge=1, le=100)
    worker_command: list[str] = Field(default_factory=lambda: ["vehicle-camera"])
    worker_config_path: Path = Path("configs/default.yaml")

    @field_validator("worker_command")
    @classmethod
    def validate_worker_command(cls, value: list[str]) -> list[str]:
        if not value or any(not part.strip() for part in value):
            raise ValueError("camera worker command requires non-empty arguments")
        return value

    @model_validator(mode="after")
    def validate_ingress_limits(self) -> CameraManagerConfig:
        if self.restart_backoff_max_seconds < self.restart_backoff_seconds:
            raise ValueError("camera restart maximum cannot be below initial backoff")
        if self.maximum_active_workers > self.maximum_configured_cameras:
            raise ValueError("active camera worker limit cannot exceed configured capacity")
        return self


class OnvifDiscoveryConfig(BaseModel):
    enabled: bool = True
    multicast_address: str = "239.255.255.250"
    port: int = Field(default=3702, ge=1, le=65535)
    interface_address: str | None = None
    timeout_seconds: float = Field(default=3.0, ge=0.2, le=30)
    probe_retries: int = Field(default=2, ge=1, le=5)
    multicast_ttl: int = Field(default=2, ge=1, le=32)
    maximum_results: int = Field(default=128, ge=1, le=1024)
    maximum_response_bytes: int = Field(default=65_535, ge=4096, le=1_048_576)

    @field_validator("multicast_address")
    @classmethod
    def validate_multicast_address(cls, value: str) -> str:
        address = IPv4Address(value)
        if not address.is_multicast:
            raise ValueError("ONVIF discovery address must be IPv4 multicast")
        return str(address)

    @field_validator("interface_address")
    @classmethod
    def validate_interface_address(cls, value: str | None) -> str | None:
        if value is None:
            return None
        address = IPv4Address(value)
        if address.is_multicast or address.is_unspecified:
            raise ValueError("ONVIF interface address must be a concrete IPv4 address")
        return str(address)


class CameraCredentialKeyConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    id: str
    key: SecretStr

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or "." in stripped or len(stripped) > 128:
            raise ValueError("camera credential key id is invalid")
        return stripped

    @field_validator("key", mode="before")
    @classmethod
    def validate_key(cls, value: object) -> object:
        if value == "":
            raise ValueError("camera credential key cannot be empty")
        return value


class SecurityConfig(BaseModel):
    """Camera secret keyring with backward-compatible single-key settings."""

    camera_credential_key: SecretStr | None = None
    camera_credential_key_id: str = "camera-key-v1"
    camera_credential_keys: list[CameraCredentialKeyConfig] = Field(default_factory=list)
    camera_credential_active_key_id: str | None = None

    @field_validator("camera_credential_key", mode="before")
    @classmethod
    def empty_key_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("camera_credential_key_id")
    @classmethod
    def validate_key_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or "." in stripped:
            raise ValueError("camera credential key id must be non-empty and cannot contain '.'")
        return stripped

    @model_validator(mode="after")
    def validate_keyring(self) -> SecurityConfig:
        key_ids = [item.id for item in self.camera_credential_keys]
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("camera credential key ids must be unique")
        active = self.camera_credential_active_key_id
        if active is not None:
            active = active.strip()
            if not active or "." in active:
                raise ValueError("active camera credential key id is invalid")
            self.camera_credential_active_key_id = active
        effective_ids = set(key_ids)
        if self.camera_credential_key is not None:
            effective_ids.add(self.camera_credential_key_id)
        if active is not None and active not in effective_ids:
            raise ValueError("active camera credential key id is not present in keyring")
        if self.camera_credential_keys and active is None:
            raise ValueError("a multi-key camera credential keyring requires an active key id")
        return self


class OIDCConfig(BaseModel):
    issuer: str = ""
    jwks_url: str = ""
    audiences: list[str] = Field(default_factory=list, max_length=16)
    algorithms: list[Literal["RS256", "ES256"]] = Field(
        default_factory=lambda: ["RS256"], max_length=2
    )
    roles_claim: str = "roles"
    name_claim: str = "name"
    role_mapping: dict[str, Literal["ADMIN", "OPERATOR", "VIEWER"]] = Field(
        default_factory=lambda: {
            "ADMIN": "ADMIN",
            "OPERATOR": "OPERATOR",
            "VIEWER": "VIEWER",
        },
        max_length=100,
    )
    leeway_seconds: int = Field(default=30, ge=0, le=300)
    jwks_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    jwks_cache_seconds: int = Field(default=300, ge=30, le=86_400)
    maximum_token_length: int = Field(default=16_384, ge=512, le=65_536)
    allow_insecure_http: bool = False

    @field_validator("issuer", "jwks_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        stripped = value.strip().rstrip("/")
        parsed = urlsplit(stripped)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or len(stripped) > 2048
        ):
            raise ValueError("OIDC issuer and JWKS URL must be safe HTTP(S) URLs")
        return stripped

    @field_validator("audiences")
    @classmethod
    def validate_audiences(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if any(len(item) > 256 for item in normalized):
            raise ValueError("OIDC audience is too long")
        return list(dict.fromkeys(normalized))

    @field_validator("roles_claim", "name_claim")
    @classmethod
    def validate_claim_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 128:
            raise ValueError("OIDC claim name is invalid")
        return stripped

    @model_validator(mode="after")
    def validate_security(self) -> OIDCConfig:
        if not self.allow_insecure_http and (
            urlsplit(self.issuer).scheme != "https" or urlsplit(self.jwks_url).scheme != "https"
        ):
            raise ValueError("OIDC requires HTTPS unless allow_insecure_http is explicit")
        if not self.audiences:
            raise ValueError("OIDC requires at least one audience")
        if not self.algorithms:
            raise ValueError("OIDC requires an explicit JWT algorithm allowlist")
        if not self.role_mapping:
            raise ValueError("OIDC requires at least one role mapping")
        return self


class AuthPrincipalConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    id: str
    display_name: str | None = None
    role: Literal["ADMIN", "OPERATOR", "VIEWER"]
    key_sha256: SecretStr
    enabled: bool = True

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 128 or any(char.isspace() for char in stripped):
            raise ValueError("auth principal id must be non-empty and contain no whitespace")
        return stripped

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or len(stripped) > 256:
            raise ValueError("auth principal display name is invalid")
        return stripped

    @field_validator("key_sha256", mode="before")
    @classmethod
    def validate_key_hash(cls, value: object) -> str:
        raw = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        if re.fullmatch(r"[0-9a-fA-F]{64}", raw) is None:
            raise ValueError("auth principal key_sha256 must be a SHA-256 hex digest")
        return raw.lower()


class AuthConfig(BaseModel):
    enabled: bool = False
    provider: Literal["api_key", "oidc"] = "api_key"
    realm: str = "vehicle-intelligence"
    minimum_token_length: int = Field(default=32, ge=16, le=256)
    principals: list[AuthPrincipalConfig] = Field(default_factory=list, max_length=1000)
    oidc: OIDCConfig | None = None

    @field_validator("realm")
    @classmethod
    def validate_realm(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 128:
            raise ValueError("authentication realm is invalid")
        return stripped

    @model_validator(mode="after")
    def validate_principals(self) -> AuthConfig:
        ids = [principal.id for principal in self.principals]
        hashes = [principal.key_sha256.get_secret_value() for principal in self.principals]
        if len(ids) != len(set(ids)):
            raise ValueError("authentication principal ids must be unique")
        if len(hashes) != len(set(hashes)):
            raise ValueError("authentication principal key hashes must be unique")
        if self.enabled and self.provider == "api_key":
            active = [principal for principal in self.principals if principal.enabled]
            if not active:
                raise ValueError("enabled authentication requires an active principal")
            if not any(principal.role == "ADMIN" for principal in active):
                raise ValueError("enabled authentication requires an active ADMIN")
        if self.enabled and self.provider == "oidc" and self.oidc is None:
            raise ValueError("OIDC authentication requires OIDC configuration")
        return self


class VotingConfig(BaseModel):
    minimum_observations: int = Field(default=2, ge=1)
    cluster_max_edit_distance: int = Field(default=2, ge=0, le=4)
    frequency_weight: float = Field(default=0.25, ge=0)
    ocr_confidence_weight: float = Field(default=0.30, ge=0)
    quality_weight: float = Field(default=0.20, ge=0)
    detection_confidence_weight: float = Field(default=0.15, ge=0)
    character_consensus_weight: float = Field(default=0.10, ge=0)

    @model_validator(mode="after")
    def validate_weights(self) -> VotingConfig:
        values = (
            self.frequency_weight,
            self.ocr_confidence_weight,
            self.quality_weight,
            self.detection_confidence_weight,
            self.character_consensus_weight,
        )
        if sum(values) <= 0:
            raise ValueError("voting weights must have a positive sum")
        return self


class EventConfig(BaseModel):
    minimum_plate_confidence: float = Field(default=0.75, ge=0, le=1)
    review_plate_confidence: float = Field(default=0.85, ge=0, le=1)
    duplicate_window_seconds: int = Field(default=10, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> EventConfig:
        if self.review_plate_confidence < self.minimum_plate_confidence:
            raise ValueError("review threshold cannot be below minimum plate confidence")
        return self


class VehicleEmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = False
    provider: Literal["torchscript"] = "torchscript"
    model_path: Path | None = None
    model_name: str = "vehicle-reid"
    model_version: str = "unset"
    model_hash: str | None = None
    dimension: int = Field(default=512, ge=1, le=65_536)
    image_width: int = Field(default=256, ge=32, le=2048)
    image_height: int = Field(default=256, ge=32, le=2048)
    device: str = "cpu"

    @model_validator(mode="after")
    def validate_model(self) -> VehicleEmbeddingConfig:
        if self.enabled and self.model_path is None:
            raise ValueError("enabled vehicle embedding requires a model path")
        if not self.model_name.strip() or not self.model_version.strip():
            raise ValueError("vehicle embedding model name/version are required")
        return self


class ReIDConfig(BaseModel):
    enabled: bool = True
    scoring_version: str = "reid-score-v2"
    plate_weight: float = Field(default=0.40, ge=0, le=1)
    embedding_weight: float = Field(default=0.25, ge=0, le=1)
    vehicle_type_weight: float = Field(default=0.10, ge=0, le=1)
    color_weight: float = Field(default=0.05, ge=0, le=1)
    travel_time_weight: float = Field(default=0.20, ge=0, le=1)
    match_threshold: float = Field(default=0.88, ge=0, le=1)
    review_threshold: float = Field(default=0.65, ge=0, le=1)
    minimum_match_evidence_coverage: float = Field(default=0.40, ge=0, le=1)
    minimum_match_identifying_coverage: float = Field(default=0.25, gt=0, le=1)
    maximum_scored_candidates: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def validate_reid(self) -> ReIDConfig:
        if not self.scoring_version.strip():
            raise ValueError("ReID scoring version is required")
        if self.match_threshold <= self.review_threshold:
            raise ValueError("ReID match threshold must exceed review threshold")
        weights = (
            self.plate_weight,
            self.embedding_weight,
            self.vehicle_type_weight,
            self.color_weight,
            self.travel_time_weight,
        )
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("at least one ReID score weight must be positive")
        maximum_identifying_coverage = (self.plate_weight + self.embedding_weight) / total_weight
        if self.minimum_match_identifying_coverage > maximum_identifying_coverage:
            raise ValueError(
                "minimum ReID identifying coverage exceeds configured plate/embedding weights"
            )
        return self


class IdentityConfig(BaseModel):
    enabled: bool = True
    fingerprint_schema_version: int = Field(default=1, ge=1)
    maximum_plate_aliases: int = Field(default=16, ge=1, le=16)
    vector_candidate_limit: int = Field(default=1000, ge=1, le=5000)
    topology_edge_limit: int = Field(default=64, ge=1, le=1000)
    candidates_per_edge: int = Field(default=100, ge=1, le=1000)
    cross_camera_candidate_limit: int = Field(default=200, ge=1, le=1000)
    journey_event_limit: int = Field(default=1000, ge=1, le=4999)
    embedding: VehicleEmbeddingConfig = Field(default_factory=VehicleEmbeddingConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)


class GPUSchedulerConfig(BaseModel):
    enabled: bool = False
    maximum_cameras: int = Field(default=32, ge=1, le=1024)
    maximum_clients: int = Field(default=128, ge=4, le=256)
    maximum_batch_size: int = Field(default=8, ge=1, le=256)
    per_camera_queue_size: int = Field(default=1, ge=1, le=8)
    maximum_frame_age_ms: float = Field(default=250.0, gt=0, le=60_000)
    batch_wait_ms: float = Field(default=5.0, ge=0, le=1000)
    socket_path: Path = Path("/tmp/vehicle-intelligence/shared-inference.sock")
    service_command: list[str] = Field(default_factory=lambda: ["vehicle-inference-service"])
    startup_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    shutdown_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    request_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    maximum_payload_bytes: int = Field(
        default=67_108_864,
        ge=1_048_576,
        le=268_435_456,
    )
    maximum_inflight_payload_bytes: int = Field(
        default=268_435_456,
        ge=1_048_576,
        le=1_073_741_824,
    )
    maximum_images_per_request: int = Field(default=64, ge=1, le=256)
    maximum_isolation_attempts: int = Field(default=15, ge=1, le=63)
    camera_failure_threshold: int = Field(default=3, ge=1, le=100)
    camera_quarantine_seconds: float = Field(default=30.0, gt=0, le=3600)
    provider_failure_threshold: int = Field(default=3, ge=1, le=100)
    provider_failure_minimum_cameras: int = Field(default=2, ge=1, le=32)

    @field_validator("socket_path")
    @classmethod
    def validate_socket_path(cls, value: Path) -> Path:
        if not value.is_absolute() or ".." in value.parts or len(str(value).encode("utf-8")) > 100:
            raise ValueError("GPU inference socket path must be absolute, normalized, and short")
        return value

    @field_validator("service_command")
    @classmethod
    def validate_service_command(cls, value: list[str]) -> list[str]:
        if not value or any(not part.strip() for part in value):
            raise ValueError("GPU inference service command requires non-empty arguments")
        return value

    @model_validator(mode="after")
    def validate_batch_capacity(self) -> GPUSchedulerConfig:
        if self.maximum_batch_size > self.maximum_images_per_request:
            raise ValueError("GPU batch size cannot exceed the IPC image bound")
        minimum_isolation_attempts = 2 * (self.maximum_batch_size - 1).bit_length() + 1
        if self.maximum_isolation_attempts < minimum_isolation_attempts:
            raise ValueError(
                "GPU isolation attempts cannot isolate one failing image within a batch"
            )
        if self.provider_failure_minimum_cameras > self.maximum_cameras:
            raise ValueError("GPU provider failure camera minimum cannot exceed camera capacity")
        if self.maximum_inflight_payload_bytes < self.maximum_payload_bytes:
            raise ValueError("GPU inflight payload budget cannot be smaller than one payload")
        if (
            self.maximum_cameras > 1
            and self.maximum_inflight_payload_bytes < self.maximum_payload_bytes * 2
        ):
            raise ValueError(
                "GPU inflight payload budget must reserve one payload for a peer camera"
            )
        effective_queue_deadline_ms = min(
            self.maximum_frame_age_ms,
            self.request_timeout_seconds * 1000,
        )
        if self.batch_wait_ms >= effective_queue_deadline_ms:
            raise ValueError("GPU batch wait must be shorter than request/frame deadlines")
        return self


class ModelQualityConfig(BaseModel):
    default_window_days: int = Field(default=30, ge=1, le=365)
    maximum_window_days: int = Field(default=365, ge=1, le=3650)
    maximum_models: int = Field(default=50, ge=1, le=200)
    in_memory_scan_limit: int = Field(default=100_000, ge=100, le=1_000_000)

    @model_validator(mode="after")
    def validate_windows(self) -> ModelQualityConfig:
        if self.default_window_days > self.maximum_window_days:
            raise ValueError("default quality window cannot exceed maximum window")
        return self


class DatasetExportConfig(BaseModel):
    output_directory: Path = Path("datasets/exports")
    batch_size: int = Field(default=500, ge=1, le=1000)
    maximum_image_bytes: int = Field(default=5_000_000, ge=1024, le=50_000_000)
    maximum_image_pixels: int = Field(default=20_000_000, ge=1, le=100_000_000)
    claim_stale_seconds: float = Field(default=600.0, gt=0, le=86_400)
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    validation_ratio: float = Field(default=0.1, gt=0, lt=1)
    test_ratio: float = Field(default=0.1, gt=0, lt=1)
    split_seed: str = "camera-split-v1"

    @field_validator("split_seed")
    @classmethod
    def validate_split_seed(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 128:
            raise ValueError("dataset split seed is invalid")
        return stripped

    @model_validator(mode="after")
    def validate_ratios(self) -> DatasetExportConfig:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("dataset export split ratios must sum to one")
        return self


class DatasetReviewConfig(BaseModel):
    enabled: bool = True
    sources_directory: Path = Path("datasets/source/plate-first-party")
    workspace_directory: Path = Path("datasets/reviews/detector")
    promoted_sources_directory: Path = Path("datasets/source/plate-first-party")
    maximum_sources: int = Field(default=50, ge=1, le=1000)
    maximum_queue_items_per_source: int = Field(default=100_000, ge=1, le=1_000_000)
    maximum_image_bytes: int = Field(default=20_000_000, ge=1024, le=100_000_000)
    maximum_image_pixels: int = Field(default=40_000_000, ge=1, le=200_000_000)


class DatasetRegistryConfig(BaseModel):
    """Local immutable dataset catalog and private Hub synchronization settings."""

    enabled: bool = True
    sources_directory: Path = Path("datasets/source/plate-first-party")
    exports_directory: Path = Path("datasets/detectors/plate")
    workspace_directory: Path = Path("datasets/registry")
    training_config: Path = Path("configs/model-training.yaml")
    maximum_sources: int = Field(default=100, ge=1, le=1000)
    maximum_exports: int = Field(default=1000, ge=1, le=10_000)
    maximum_jobs: int = Field(default=10_000, ge=1, le=100_000)
    restricted_private_sync_enabled: bool = False


class ModelTrainingRuntimeConfig(BaseModel):
    """Durable remote training-run orchestration exposed to authenticated operators."""

    enabled: bool = True
    workspace_directory: Path = Path("datasets/model-training")
    training_config: Path = Path("configs/model-training.yaml")
    maximum_runs: int = Field(default=10_000, ge=1, le=100_000)
    maximum_concurrent_runs: int = Field(default=1, ge=1, le=32)
    maximum_log_lines: int = Field(default=500, ge=10, le=5000)
    container_training_config: str = "/workspace/configs/model-training.hf.yaml"
    container_output_directory: str = "/output/model-training"

    @field_validator("container_training_config", "container_output_directory")
    @classmethod
    def validate_container_path(cls, value: str) -> str:
        stripped = value.strip()
        if (
            not stripped.startswith("/")
            or "\x00" in stripped
            or ".." in Path(stripped).parts
            or len(stripped) > 512
        ):
            raise ValueError("model training container path is invalid")
        return stripped.rstrip("/") or "/"


class EventBusConfig(BaseModel):
    backend: Literal["direct", "redis"] = "direct"


class ExternalActionTargetConfig(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    host: str
    require_https: bool = True
    authentication: Literal["none", "bearer", "hmac_sha256"] = "none"
    bearer_token: SecretStr | None = None
    hmac_secret: SecretStr | None = None
    hmac_key_id: str | None = None
    circuit_failure_threshold: int = Field(default=5, ge=1, le=100)
    circuit_recovery_seconds: float = Field(default=30.0, gt=0, le=3600)

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        normalized = value.strip().lower().rstrip(".")
        if (
            not normalized
            or "/" in normalized
            or ":" in normalized
            or "@" in normalized
            or len(normalized) > 253
        ):
            raise ValueError("external target host must be a hostname only")
        return normalized

    @field_validator("hmac_key_id")
    @classmethod
    def validate_hmac_key_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or len(stripped) > 128 or any(char.isspace() for char in stripped):
            raise ValueError("external target HMAC key id is invalid")
        return stripped

    @model_validator(mode="after")
    def validate_authentication(self) -> ExternalActionTargetConfig:
        if self.authentication == "bearer" and self.bearer_token is None:
            raise ValueError("Bearer external target requires bearer_token")
        if self.authentication == "hmac_sha256" and (
            self.hmac_secret is None or self.hmac_key_id is None
        ):
            raise ValueError("HMAC external target requires secret and key id")
        return self


class RuleEngineConfig(BaseModel):
    enabled: bool = True
    evaluation_max_rules: int = Field(default=1000, ge=1, le=10000)
    rule_cache_ttl_seconds: float = Field(default=2.0, ge=0.1, le=300)
    action_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
    action_max_attempts: int = Field(default=3, ge=1, le=20)
    action_claim_stale_seconds: float = Field(default=60.0, gt=0, le=3600)
    external_actions_enabled: bool = False
    external_allowed_hosts: list[str] = Field(default_factory=list)
    external_targets: list[ExternalActionTargetConfig] = Field(default_factory=list)
    external_maximum_url_length: int = Field(default=2048, ge=128, le=8192)

    @field_validator("external_allowed_hosts")
    @classmethod
    def validate_external_hosts(cls, value: list[str]) -> list[str]:
        normalized = [host.strip().lower() for host in value if host.strip()]
        if any("/" in host or ":" in host for host in normalized):
            raise ValueError("external action allowed hosts must contain hostnames only")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_external_actions(self) -> RuleEngineConfig:
        target_hosts = [target.host for target in self.external_targets]
        if len(target_hosts) != len(set(target_hosts)):
            raise ValueError("external action target hosts must be unique")
        if self.external_actions_enabled and not (self.external_allowed_hosts or target_hosts):
            raise ValueError("external actions require at least one allowed hostname")
        return self


class RedisConfig(BaseModel):
    url: SecretStr = SecretStr("redis://localhost:6379/0")
    stream: str = "vehicle.events"
    dead_letter_stream: str = "vehicle.events.dlq"
    consumer_group: str = "event-processors"
    max_length: int = Field(default=100_000, ge=100)
    dead_letter_max_length: int = Field(default=10_000, ge=10)
    batch_size: int = Field(default=25, ge=1, le=1000)
    worker_concurrency: int = Field(default=8, ge=1, le=128)
    block_ms: int = Field(default=1000, ge=1, le=60_000)
    claim_idle_ms: int = Field(default=30_000, ge=1000)
    reclaim_interval_ms: int = Field(default=5000, ge=100, le=60_000)
    connection_timeout_ms: int = Field(default=3000, ge=100)
    retry_delay_seconds: float = Field(default=1.0, gt=0, le=60)
    delete_after_ack: bool = True

    @field_validator("stream", "dead_letter_stream", "consumer_group")
    @classmethod
    def validate_redis_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Redis stream and consumer-group names cannot be empty")
        return stripped

    @model_validator(mode="after")
    def validate_streams(self) -> RedisConfig:
        if self.stream == self.dead_letter_stream:
            raise ValueError("Redis main stream and dead-letter stream must differ")
        return self


class RealtimeConfig(BaseModel):
    enabled: bool = False
    redis_channel: str = "vehicle.events.realtime"
    client_queue_size: int = Field(default=50, ge=1, le=1000)
    replay_size: int = Field(default=500, ge=1, le=10_000)
    heartbeat_seconds: float = Field(default=15.0, gt=0, le=60)
    websocket_auth_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    broker_poll_seconds: float = Field(default=1.0, gt=0, le=5)
    reconnect_initial_seconds: float = Field(default=0.5, gt=0, le=30)
    reconnect_max_seconds: float = Field(default=30.0, gt=0, le=300)

    @field_validator("redis_channel")
    @classmethod
    def validate_channel(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 256:
            raise ValueError("realtime Redis channel is invalid")
        return stripped

    @model_validator(mode="after")
    def validate_reconnect_range(self) -> RealtimeConfig:
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise ValueError("realtime reconnect maximum cannot be below initial delay")
        return self


class LiveMonitorConfig(BaseModel):
    enabled: bool = False
    redis_channel: str = "vehicle.live.frames"
    preview_fps: float = Field(default=2.0, gt=0, le=10)
    preview_max_width: int = Field(default=960, ge=160, le=1920)
    jpeg_quality: int = Field(default=72, ge=30, le=95)
    maximum_payload_bytes: int = Field(default=750_000, ge=32_768, le=2_000_000)
    publish_timeout_seconds: float = Field(default=1.0, gt=0, le=10)
    frame_buffer_size: int = Field(default=3, ge=1, le=10)
    maximum_cameras: int = Field(default=256, ge=1, le=10_000)
    stale_after_seconds: float = Field(default=5.0, gt=0, le=300)
    broker_poll_seconds: float = Field(default=0.25, gt=0, le=5)
    reconnect_initial_seconds: float = Field(default=0.5, gt=0, le=30)
    reconnect_max_seconds: float = Field(default=30.0, gt=0, le=300)

    @field_validator("redis_channel")
    @classmethod
    def validate_channel(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 256:
            raise ValueError("live monitor Redis channel is invalid")
        return stripped

    @model_validator(mode="after")
    def validate_reconnect_range(self) -> LiveMonitorConfig:
        if self.reconnect_max_seconds < self.reconnect_initial_seconds:
            raise ValueError("live monitor reconnect maximum cannot be below initial delay")
        return self


class StorageConfig(BaseModel):
    backend: Literal["local", "minio"] = "local"
    output_directory: Path = Path("output")
    snapshots: bool = True
    vehicle_crops: bool = True
    plate_crops: bool = True
    clips: bool = False


class FinalizationOutboxConfig(BaseModel):
    enabled: bool = True
    maximum_entries: int = Field(default=10_000, ge=1, le=1_000_000)
    maximum_bytes: int = Field(
        default=4 * 1024 * 1024 * 1024,
        ge=1024 * 1024,
        le=1024 * 1024 * 1024 * 1024,
    )
    maximum_entry_bytes: int = Field(
        default=32 * 1024 * 1024,
        ge=64 * 1024,
        le=32 * 1024 * 1024,
    )
    delivery_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    replay_interval_seconds: float = Field(default=5.0, gt=0, le=300)

    @model_validator(mode="after")
    def validate_capacity(self) -> FinalizationOutboxConfig:
        if self.maximum_entry_bytes > self.maximum_bytes:
            raise ValueError("finalization outbox entry limit cannot exceed its byte capacity")
        return self


class MongoConfig(BaseModel):
    enabled: bool = False
    uri: SecretStr = SecretStr("mongodb://localhost:27017")
    database: str = "vehicle_intelligence"
    server_selection_timeout_ms: int = Field(default=3000, gt=0)
    connect_timeout_ms: int = Field(default=3000, gt=0, le=120_000)
    socket_timeout_ms: int = Field(default=10_000, gt=0, le=300_000)
    transactions_enabled: bool = False
    transaction_max_commit_time_ms: int = Field(default=5000, gt=0, le=120_000)

    @field_validator("database")
    @classmethod
    def validate_database(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 64 or any(char in stripped for char in '/\\."$'):
            raise ValueError("MongoDB database name is invalid")
        return stripped


class MinioConfig(BaseModel):
    endpoint: str = "localhost:9000"
    public_endpoint: str | None = None
    region: str = "us-east-1"
    access_key: SecretStr = SecretStr("minioadmin")
    secret_key: SecretStr = SecretStr("minioadmin")
    bucket: str = "vehicle-media"
    secure: bool = False
    public_secure: bool | None = None
    presigned_url_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    connect_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        le=30,
        allow_inf_nan=False,
    )
    read_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        le=120,
        allow_inf_nan=False,
    )
    maximum_retries: int = Field(default=0, ge=0, le=5)
    retry_backoff_seconds: float = Field(
        default=0.2,
        ge=0,
        le=10,
        allow_inf_nan=False,
    )
    retry_backoff_max_seconds: float = Field(
        default=2.0,
        ge=0,
        le=30,
        allow_inf_nan=False,
    )

    @field_validator("endpoint", "public_endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if (
            not stripped
            or "://" in stripped
            or "/" in stripped
            or "@" in stripped
            or len(stripped) > 512
        ):
            raise ValueError("MinIO endpoint must be a host[:port] without scheme or path")
        return stripped

    @field_validator("region")
    @classmethod
    def validate_region(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 128:
            raise ValueError("MinIO region is invalid")
        return stripped

    @model_validator(mode="after")
    def validate_retry_backoff(self) -> MinioConfig:
        if self.retry_backoff_seconds > self.retry_backoff_max_seconds:
            raise ValueError("MinIO retry backoff cannot exceed its maximum")
        return self


_RESERVED_OBSERVABILITY_PATHS = frozenset(
    {
        "/docs",
        "/docs/oauth2-redirect",
        "/livez",
        "/openapi.json",
        "/readyz",
        "/redoc",
    }
)


class ObservabilityConfig(BaseModel):
    prometheus_enabled: bool = True
    prometheus_path: str = "/metrics"
    retention_metrics_port: int = Field(default=9101, ge=1024, le=65535)
    opentelemetry_enabled: bool = False
    otlp_traces_endpoint: str | None = None
    otlp_headers: SecretStr | None = None
    service_name: str = "vehicle-intelligence-api"
    service_version: str = "0.1.0"
    trace_sample_ratio: float = Field(default=0.10, ge=0, le=1)
    export_timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    @field_validator("prometheus_path")
    @classmethod
    def validate_prometheus_path(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped.startswith("/") or stripped.endswith("/") or "{" in stripped:
            raise ValueError("Prometheus path must be an absolute static path")
        if (
            stripped in _RESERVED_OBSERVABILITY_PATHS
            or stripped == "/api"
            or stripped.startswith("/api/")
        ):
            raise ValueError("Prometheus path collides with a reserved application path")
        return stripped

    @field_validator("service_name", "service_version")
    @classmethod
    def validate_resource_value(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or len(stripped) > 128:
            raise ValueError("OpenTelemetry resource values must be non-empty and bounded")
        return stripped

    @field_validator("otlp_traces_endpoint")
    @classmethod
    def validate_otlp_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        parsed = urlsplit(stripped)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or len(stripped) > 2048
        ):
            raise ValueError("OTLP endpoint must be a safe HTTP(S) URL")
        return stripped

    @model_validator(mode="after")
    def validate_telemetry(self) -> ObservabilityConfig:
        if self.opentelemetry_enabled and self.otlp_traces_endpoint is None:
            raise ValueError("enabled OpenTelemetry requires an OTLP traces endpoint")
        return self


class RetentionConfig(BaseModel):
    enabled: bool = False
    worker_interval_seconds: float = Field(default=3600.0, gt=0, le=86_400)
    batch_size: int = Field(default=100, ge=1, le=1000)
    claim_stale_seconds: float = Field(default=600.0, gt=0, le=86_400)
    vehicle_events_days: int = Field(default=365, ge=1, le=3650)
    snapshots_days: int = Field(default=30, ge=1, le=3650)
    vehicle_crops_days: int = Field(default=30, ge=1, le=3650)
    plate_crops_days: int = Field(default=30, ge=1, le=3650)
    event_clips_days: int = Field(default=14, ge=1, le=3650)
    debug_images_days: int = Field(default=7, ge=1, le=3650)
    minio_lifecycle_enabled: bool = True

    @model_validator(mode="after")
    def validate_retention_windows(self) -> RetentionConfig:
        media_maximum = max(
            self.snapshots_days,
            self.vehicle_crops_days,
            self.plate_crops_days,
            self.event_clips_days,
        )
        if self.vehicle_events_days < media_maximum:
            raise ValueError("vehicle-event retention cannot be shorter than media retention")
        return self


class DebugConfig(BaseModel):
    save_plate_candidates: bool = False
    save_vehicle_crops: bool = False
    draw_overlay: bool = False
    verbose_tracking: bool = False


class PrefixFilteredDotEnvSettingsSource(DotEnvSettingsSource):
    """Ignore non-application keys shared with Docker Compose in ``.env``."""

    def __call__(self) -> dict[str, Any]:
        values = super().__call__()
        prefix = self.env_prefix if self.case_sensitive else self.env_prefix.lower()
        unprefixed_names = {name for name in self.env_vars if not name.startswith(prefix)}
        return {name: value for name, value in values.items() if name not in unprefixed_names}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIP_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
        hide_input_in_errors=True,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vision: VisionConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    rtsp: RTSPConfig = Field(default_factory=RTSPConfig)
    camera_manager: CameraManagerConfig = Field(default_factory=CameraManagerConfig)
    onvif_discovery: OnvifDiscoveryConfig = Field(default_factory=OnvifDiscoveryConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)
    events: EventConfig = Field(default_factory=EventConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    gpu_scheduler: GPUSchedulerConfig = Field(default_factory=GPUSchedulerConfig)
    model_quality: ModelQualityConfig = Field(default_factory=ModelQualityConfig)
    dataset_export: DatasetExportConfig = Field(default_factory=DatasetExportConfig)
    dataset_review: DatasetReviewConfig = Field(default_factory=DatasetReviewConfig)
    dataset_registry: DatasetRegistryConfig = Field(default_factory=DatasetRegistryConfig)
    model_training: ModelTrainingRuntimeConfig = Field(default_factory=ModelTrainingRuntimeConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    rule_engine: RuleEngineConfig = Field(default_factory=RuleEngineConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    live_monitor: LiveMonitorConfig = Field(default_factory=LiveMonitorConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    finalization_outbox: FinalizationOutboxConfig = Field(default_factory=FinalizationOutboxConfig)
    mongodb: MongoConfig = Field(default_factory=MongoConfig)
    minio: MinioConfig = Field(default_factory=MinioConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    def validate_camera_finalization_budget(self) -> None:
        """Validate the delivery budget only for a camera pipeline process."""

        if self.storage.backend != "minio" or not self.finalization_outbox.enabled:
            return

        # A new camera namespace can issue HEAD bucket + PUT bucket + three media
        # PUTs. The explicit non-empty region passed to Minio prevents a separate
        # region-discovery request. Outbox replay owns retries, so the SDK default
        # is zero retries; custom retry/backoff settings are charged per request.
        request_budget_seconds = (
            self.minio.connect_timeout_seconds + self.minio.read_timeout_seconds
        ) * (self.minio.maximum_retries + 1) + (
            self.minio.retry_backoff_max_seconds * self.minio.maximum_retries
        )
        if self.event_bus.backend == "redis":
            redis_connect_seconds = self.redis.connection_timeout_ms / 1000
            redis_command_seconds = max(
                redis_connect_seconds,
                self.redis.block_ms / 1000 + 1,
            )
            publisher_budget_seconds = redis_connect_seconds + redis_command_seconds
        elif self.mongodb.enabled:
            publisher_budget_seconds = (
                self.mongodb.server_selection_timeout_ms
                + self.mongodb.connect_timeout_ms
                + self.mongodb.socket_timeout_ms
            ) / 1000
        else:
            # The single-writer JSONL development fallback has no network timer.
            publisher_budget_seconds = 3.0
        first_delivery_budget_seconds = 5 * request_budget_seconds + publisher_budget_seconds
        if first_delivery_budget_seconds >= self.finalization_outbox.delivery_timeout_seconds:
            raise ValueError(
                "MinIO first-delivery HTTP and publisher budget must be below "
                "finalization outbox delivery_timeout_seconds; reduce MinIO "
                "timeouts/retries or increase the outbox timeout"
            )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment and .env values intentionally override YAML/init values.
        filtered_dotenv = PrefixFilteredDotEnvSettingsSource(settings_cls)
        return env_settings, filtered_dotenv, init_settings, file_secret_settings


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration root must be a mapping: {config_path}")
    try:
        return Settings(**raw)
    except ValueError as exc:
        raise ConfigurationError(f"invalid configuration in {config_path}: {exc}") from exc
