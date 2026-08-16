from pathlib import Path

import pytest
from pydantic import ValidationError

from vehicle_intelligence.config import (
    AppConfig,
    CameraManagerConfig,
    DatasetExportConfig,
    DetectorConfig,
    FinalizationOutboxConfig,
    GPUSchedulerConfig,
    MinioConfig,
    ModelQualityConfig,
    ObservabilityConfig,
    OCRConfig,
    OnvifDiscoveryConfig,
    RetentionConfig,
    Settings,
    load_settings,
)


def test_application_environment_is_normalized_before_security_checks() -> None:
    assert AppConfig(environment=" Production ").environment == "production"
    with pytest.raises(ValidationError, match="environment is invalid"):
        AppConfig(environment="   ")


def test_environment_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("VIP_CAMERA__FPS_LIMIT", "9")
    monkeypatch.setenv("VIP_MONGODB__DATABASE", "test_vehicle_intelligence")
    monkeypatch.setenv("VIP_REDIS__URL", "redis://example.internal:6379/2")
    monkeypatch.setenv("VIP_MINIO__PUBLIC_ENDPOINT", "media.example.internal:9443")
    monkeypatch.setenv("VIP_MINIO__PRESIGNED_URL_TTL_SECONDS", "180")
    monkeypatch.setenv("VIP_MINIO__REGION", "eu-test-1")
    monkeypatch.setenv("VIP_MINIO__CONNECT_TIMEOUT_SECONDS", "4.5")
    monkeypatch.setenv("VIP_MINIO__READ_TIMEOUT_SECONDS", "9.5")
    monkeypatch.setenv("VIP_MINIO__MAXIMUM_RETRIES", "2")
    monkeypatch.setenv("VIP_OBSERVABILITY__RETENTION_METRICS_PORT", "9201")
    monkeypatch.setenv("VIP_FINALIZATION_OUTBOX__MAXIMUM_ENTRIES", "321")

    settings = load_settings()

    assert settings.camera.fps_limit == 9
    assert settings.mongodb.database == "test_vehicle_intelligence"
    assert "mongodb://" not in repr(settings.mongodb.uri)
    assert settings.redis.url.get_secret_value() == "redis://example.internal:6379/2"
    assert "example.internal" not in repr(settings.redis.url)
    assert settings.minio.public_endpoint == "media.example.internal:9443"
    assert settings.minio.presigned_url_ttl_seconds == 180
    assert settings.minio.region == "eu-test-1"
    assert settings.minio.connect_timeout_seconds == 4.5
    assert settings.minio.read_timeout_seconds == 9.5
    assert settings.minio.maximum_retries == 2
    assert settings.observability.retention_metrics_port == 9201
    assert settings.finalization_outbox.maximum_entries == 321


def test_finalization_outbox_capacity_is_bounded() -> None:
    with pytest.raises(ValidationError, match="entry limit cannot exceed"):
        FinalizationOutboxConfig(
            maximum_bytes=1024 * 1024,
            maximum_entry_bytes=2 * 1024 * 1024,
        )
    with pytest.raises(ValidationError):
        FinalizationOutboxConfig(maximum_entry_bytes=32 * 1024 * 1024 + 1)

    assert FinalizationOutboxConfig(maximum_entry_bytes=32 * 1024 * 1024).maximum_entry_bytes == (
        32 * 1024 * 1024
    )


@pytest.mark.parametrize(
    "path",
    (
        "/api",
        "/api/events",
        "/api/system/health",
        "/docs",
        "/docs/oauth2-redirect",
        "/livez",
        "/openapi.json",
        "/readyz",
        "/redoc",
    ),
)
def test_prometheus_path_cannot_shadow_application_routes(path: str) -> None:
    with pytest.raises(ValidationError, match="reserved application path"):
        ObservabilityConfig(prometheus_path=path)


def test_dotenv_ignores_unprefixed_compose_variables(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(
        "MONGODB_HOST_PORT=27018\nMINIO_API_STALE_UPLOADS_EXPIRY=24h\nVIP_CAMERA__FPS_LIMIT=7\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

    settings = load_settings(config_path)

    assert settings.camera.fps_limit == 7


@pytest.mark.parametrize(
    "endpoint",
    ("https://media.example", "user@media.example:9000", "media.example/path", ""),
)
def test_minio_endpoint_rejects_scheme_credentials_path_and_empty(endpoint: str) -> None:
    with pytest.raises(ValidationError, match=r"host\[:port\]"):
        MinioConfig(public_endpoint=endpoint)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("connect_timeout_seconds", 0),
        ("connect_timeout_seconds", float("inf")),
        ("read_timeout_seconds", 121),
        ("maximum_retries", 6),
        ("retry_backoff_seconds", float("nan")),
        ("retry_backoff_max_seconds", 31),
    ),
)
def test_minio_network_policy_rejects_unbounded_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MinioConfig(**{field: value})


def test_minio_network_policy_rejects_backoff_above_its_cap() -> None:
    with pytest.raises(ValidationError, match="backoff cannot exceed"):
        MinioConfig(retry_backoff_seconds=3, retry_backoff_max_seconds=2)


def _settings_with_minio(**updates: object) -> Settings:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    base = load_settings(config_path)
    values = base.model_dump()
    values["storage"] = base.storage.model_copy(update={"backend": "minio"})
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_default_minio_delivery_budget_fits_the_outbox_timeout_without_sdk_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _settings_with_minio()
    settings.validate_camera_finalization_budget()

    assert settings.minio.connect_timeout_seconds == 5
    assert settings.minio.read_timeout_seconds == 3
    assert settings.minio.maximum_retries == 0
    assert settings.finalization_outbox.delivery_timeout_seconds == 60


def test_minio_outbox_budget_accepts_and_rejects_custom_network_policies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    valid_minio = MinioConfig(
        connect_timeout_seconds=0.5,
        read_timeout_seconds=0.5,
        maximum_retries=1,
        retry_backoff_seconds=0.1,
        retry_backoff_max_seconds=0.2,
    )
    valid = _settings_with_minio(
        minio=valid_minio,
        finalization_outbox=FinalizationOutboxConfig(delivery_timeout_seconds=15),
    )
    valid.validate_camera_finalization_budget()

    assert valid.minio.maximum_retries == 1
    invalid = _settings_with_minio(
        minio=valid_minio,
        finalization_outbox=FinalizationOutboxConfig(delivery_timeout_seconds=14),
    )
    with pytest.raises(ValueError, match="first-delivery HTTP and publisher budget"):
        invalid.validate_camera_finalization_budget()


def test_minio_outbox_budget_includes_the_configured_redis_publisher_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    base = load_settings(config_path)

    settings = _settings_with_minio(
        event_bus=base.event_bus.model_copy(update={"backend": "redis"}),
        redis=base.redis.model_copy(update={"block_ms": 60_000}),
    )

    with pytest.raises(ValueError, match="first-delivery HTTP and publisher budget"):
        settings.validate_camera_finalization_budget()


def test_non_camera_minio_mongo_role_is_not_rejected_by_camera_outbox_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    base = load_settings(config_path)

    settings = _settings_with_minio(
        mongodb=base.mongodb.model_copy(update={"enabled": True}),
    )

    assert settings.storage.backend == "minio"
    assert settings.mongodb.enabled is True


def test_onvif_and_multi_camera_limits_are_validated() -> None:
    assert OnvifDiscoveryConfig().multicast_address == "239.255.255.250"
    with pytest.raises(ValidationError, match="IPv4 multicast"):
        OnvifDiscoveryConfig(multicast_address="127.0.0.1")
    with pytest.raises(ValidationError, match="active camera worker limit"):
        CameraManagerConfig(maximum_configured_cameras=2, maximum_active_workers=3)
    with pytest.raises(ValidationError, match="restart maximum"):
        CameraManagerConfig(restart_backoff_seconds=10, restart_backoff_max_seconds=5)


@pytest.mark.parametrize(
    "endpoint",
    (
        "grpc://collector.internal:4317",
        "http://user:secret@collector.internal/v1/traces",
        "http://collector.internal/v1/traces?token=secret",
    ),
)
def test_observability_rejects_unsafe_otlp_endpoints(endpoint: str) -> None:
    with pytest.raises(ValidationError, match=r"safe HTTP\(S\) URL"):
        ObservabilityConfig(otlp_traces_endpoint=endpoint)

    with pytest.raises(ValidationError, match="requires an OTLP traces endpoint"):
        ObservabilityConfig(opentelemetry_enabled=True)


def test_event_retention_cannot_expire_before_canonical_media() -> None:
    with pytest.raises(ValidationError, match="cannot be shorter"):
        RetentionConfig(vehicle_events_days=10, snapshots_days=11)

    config = RetentionConfig(vehicle_events_days=30, snapshots_days=30)
    assert config.vehicle_events_days == config.snapshots_days


def test_detector_optimization_configuration_is_bounded_and_normalized() -> None:
    config = DetectorConfig(
        provider="onnxruntime",
        model_name="detector",
        model_version="1",
        confidence=0.4,
        iou=0.5,
        execution_providers=["tensorrt", "CUDAExecutionProvider", "tensorrt"],
        model_classes=["Car", "Person"],
    )
    assert config.execution_providers == ["tensorrt", "CUDAExecutionProvider"]
    assert config.model_classes == ["car", "person"]
    assert config.picodet.strides == (8, 16, 32, 64)
    with pytest.raises(ValidationError, match="cannot be empty"):
        DetectorConfig(
            model_name="detector",
            model_version="1",
            confidence=0.4,
            iou=0.5,
            execution_providers=[""],
        )
    with pytest.raises(ValidationError, match="strides must be unique"):
        DetectorConfig(
            provider="picodet",
            model_name="detector",
            model_version="1",
            confidence=0.4,
            iou=0.5,
            picodet={"strides": [8, 8]},
        )


def test_gpu_scheduler_configuration_rejects_impossible_batch_capacity() -> None:
    with pytest.raises(ValidationError, match="IPC image bound"):
        GPUSchedulerConfig(
            maximum_batch_size=3,
            maximum_images_per_request=2,
        )
    with pytest.raises(ValidationError, match="isolation attempts"):
        GPUSchedulerConfig(
            maximum_batch_size=8,
            maximum_isolation_attempts=3,
        )
    with pytest.raises(ValidationError, match="camera minimum"):
        GPUSchedulerConfig(
            maximum_cameras=1,
            provider_failure_minimum_cameras=2,
        )


def test_quality_and_dataset_export_windows_are_bounded() -> None:
    with pytest.raises(ValidationError, match="default quality window"):
        ModelQualityConfig(default_window_days=31, maximum_window_days=30)
    with pytest.raises(ValidationError, match="ratios must sum to one"):
        DatasetExportConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.2)
    with pytest.raises(ValidationError):
        DatasetExportConfig(maximum_image_pixels=0)


def test_partial_plate_length_range_is_validated() -> None:
    with pytest.raises(ValidationError, match="maximum cannot be below"):
        OCRConfig(partial_min_characters=8, partial_max_characters=4)


def test_ocr_load_shaping_configuration_is_bounded_and_can_restore_legacy_behavior() -> None:
    with pytest.raises(ValidationError, match="early-stop confidence"):
        OCRConfig(variant_early_stop_confidence=0.5)
    with pytest.raises(ValidationError, match="consensus-stop confidence"):
        OCRConfig(consensus_stop_min_confidence=0.5)

    legacy = OCRConfig(
        track_frame_interval=1,
        variant_early_stop_confidence=None,
        consensus_stop_min_observations=None,
    )

    assert legacy.track_frame_interval == 1
    assert legacy.variant_early_stop_confidence is None
    assert legacy.consensus_stop_min_observations is None
