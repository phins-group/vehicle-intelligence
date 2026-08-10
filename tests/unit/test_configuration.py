from pathlib import Path

import pytest
from pydantic import ValidationError

from vehicle_intelligence.config import (
    CameraManagerConfig,
    DatasetExportConfig,
    DetectorConfig,
    GPUSchedulerConfig,
    MinioConfig,
    ModelQualityConfig,
    ObservabilityConfig,
    OCRConfig,
    OnvifDiscoveryConfig,
    RetentionConfig,
    load_settings,
)


def test_environment_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("VIP_CAMERA__FPS_LIMIT", "9")
    monkeypatch.setenv("VIP_MONGODB__DATABASE", "test_vehicle_intelligence")
    monkeypatch.setenv("VIP_REDIS__URL", "redis://example.internal:6379/2")
    monkeypatch.setenv("VIP_MINIO__PUBLIC_ENDPOINT", "media.example.internal:9443")
    monkeypatch.setenv("VIP_MINIO__PRESIGNED_URL_TTL_SECONDS", "180")
    monkeypatch.setenv("VIP_MINIO__REGION", "eu-test-1")
    monkeypatch.setenv("VIP_OBSERVABILITY__RETENTION_METRICS_PORT", "9201")

    settings = load_settings()

    assert settings.camera.fps_limit == 9
    assert settings.mongodb.database == "test_vehicle_intelligence"
    assert "mongodb://" not in repr(settings.mongodb.uri)
    assert settings.redis.url.get_secret_value() == "redis://example.internal:6379/2"
    assert "example.internal" not in repr(settings.redis.url)
    assert settings.minio.public_endpoint == "media.example.internal:9443"
    assert settings.minio.presigned_url_ttl_seconds == 180
    assert settings.minio.region == "eu-test-1"
    assert settings.observability.retention_metrics_port == 9201


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
    with pytest.raises(ValidationError, match="total scheduler capacity"):
        GPUSchedulerConfig(
            maximum_cameras=2,
            per_camera_queue_size=1,
            maximum_batch_size=3,
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
