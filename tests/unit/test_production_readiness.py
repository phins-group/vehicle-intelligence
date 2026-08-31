import argparse
import base64
import hashlib
import json
from pathlib import Path

from pydantic import SecretStr

from vehicle_intelligence.application.production_readiness import (
    ReadinessStatus,
    assess_production_readiness,
)
from vehicle_intelligence.config import (
    AppConfig,
    AuthConfig,
    AuthPrincipalConfig,
    EventBusConfig,
    MinioConfig,
    MongoConfig,
    ObservabilityConfig,
    OIDCConfig,
    OIDCConsoleConfig,
    RedisConfig,
    RetentionConfig,
    SecurityConfig,
    StorageConfig,
    load_settings,
)
from vehicle_intelligence.interfaces.readiness_cli import run
from vehicle_intelligence.model_artifact import sha256_directory


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _production_settings(tmp_path):
    settings = load_settings()
    vehicle_data = b"verified-vehicle-model"
    plate_data = b"verified-vietnam-plate-model"
    (tmp_path / "vehicle.pt").write_bytes(vehicle_data)
    (tmp_path / "plate.pt").write_bytes(plate_data)
    vehicle = settings.vision.vehicle_detection.model_copy(
        update={
            "model_path": "vehicle.pt",
            "model_version": "vehicle-2026.08",
            "model_hash": _sha256(vehicle_data),
        }
    )
    plate = settings.vision.plate_detection.model_copy(
        update={
            "model_path": "plate.pt",
            "model_version": "plate-2026.08",
            "model_hash": _sha256(plate_data),
        }
    )
    detection_directory = tmp_path / "ocr-detection"
    recognition_directory = tmp_path / "ocr-recognition"
    detection_directory.mkdir()
    recognition_directory.mkdir()
    (detection_directory / "inference.json").write_bytes(b"verified-ocr-detection")
    (recognition_directory / "inference.json").write_bytes(b"verified-ocr-recognition")
    ocr = settings.vision.ocr.model_copy(
        update={
            "detection_model_directory": detection_directory.name,
            "detection_model_hash": sha256_directory(detection_directory),
            "recognition_model_directory": recognition_directory.name,
            "recognition_model_hash": sha256_directory(recognition_directory),
        }
    )
    vision = settings.vision.model_copy(
        update={
            "vehicle_detection": vehicle,
            "plate_detection": plate,
            "ocr": ocr,
        }
    )
    reid = settings.identity.reid.model_copy(update={"embedding_weight": 0.0})
    identity = settings.identity.model_copy(update={"reid": reid})
    camera_key = base64.urlsafe_b64encode(bytes(range(32))).decode()
    return settings.model_copy(
        update={
            "app": AppConfig(environment="production", config_version="prod-2026.08"),
            "vision": vision,
            "identity": identity,
            "security": SecurityConfig(camera_credential_key=SecretStr(camera_key)),
            "auth": AuthConfig(
                enabled=True,
                principals=[
                    AuthPrincipalConfig(
                        id="admin",
                        role="ADMIN",
                        key_sha256=SecretStr("a" * 64),
                    )
                ],
            ),
            "mongodb": MongoConfig(
                enabled=True,
                transactions_enabled=True,
                uri=SecretStr("mongodb://vip-user:private-password@mongo:27017/?replicaSet=vip-rs"),
            ),
            "event_bus": EventBusConfig(backend="redis"),
            "redis": RedisConfig(url=SecretStr("rediss://:private-password@redis:6379/0")),
            "storage": StorageConfig(backend="minio"),
            "minio": MinioConfig(
                endpoint="minio.internal:9000",
                public_endpoint="media.example.com:443",
                access_key=SecretStr("vip-production-access"),
                secret_key=SecretStr("vip-production-secret-value"),
                secure=True,
            ),
            "observability": ObservabilityConfig(
                prometheus_enabled=True,
                opentelemetry_enabled=True,
                otlp_traces_endpoint="https://otel.example.com/v1/traces",
            ),
            "retention": RetentionConfig(enabled=True),
            "realtime": settings.realtime.model_copy(update={"enabled": True}),
        }
    )


def test_default_configuration_fails_closed_and_report_is_secret_safe(
    tmp_path,
    monkeypatch,
) -> None:
    config_path = Path("configs/default.yaml").resolve()
    monkeypatch.chdir(tmp_path)
    settings = load_settings(config_path)

    report = assess_production_readiness(settings, base_directory=tmp_path)
    document = report.to_document()
    rendered = json.dumps(document)
    failures = {check.id for check in report.checks if check.status is ReadinessStatus.FAIL}

    assert not report.ready
    assert document["schemaVersion"] == 1
    assert document["summary"]["FAIL"] == len(failures)
    assert {
        "app.environment",
        "auth.enabled",
        "camera.credentials",
        "mongodb.enabled",
        "event_bus.redis",
        "storage.minio",
        "model.plate_artifact",
    } <= failures
    assert "minioadmin" not in rendered
    assert "mongodb://" not in rendered
    assert "redis://" not in rendered


def test_versioned_production_configuration_passes_static_gate(tmp_path) -> None:
    report = assess_production_readiness(
        _production_settings(tmp_path),
        base_directory=tmp_path,
    )

    assert report.ready
    assert report.counts == {"PASS": len(report.checks), "WARN": 0, "FAIL": 0}
    statuses = {check.id: check.status for check in report.checks}
    assert statuses["model.ocr_artifacts"] is ReadinessStatus.PASS
    assert statuses["model.ocr_hash"] is ReadinessStatus.PASS


def test_production_gate_requires_durable_finalization_outbox(tmp_path) -> None:
    settings = _production_settings(tmp_path)
    settings = settings.model_copy(
        update={
            "finalization_outbox": settings.finalization_outbox.model_copy(
                update={"enabled": False}
            )
        }
    )

    report = assess_production_readiness(settings, base_directory=tmp_path)
    statuses = {check.id: check.status for check in report.checks}

    assert not report.ready
    assert statuses["durability.finalization_outbox"] is ReadinessStatus.FAIL


def test_production_oidc_requires_browser_pkce_metadata(tmp_path) -> None:
    settings = _production_settings(tmp_path)
    oidc = OIDCConfig(
        issuer="https://identity.example",
        jwks_url="https://identity.example/jwks",
        audiences=["vehicle-api"],
    )
    settings = settings.model_copy(
        update={"auth": AuthConfig(enabled=True, provider="oidc", oidc=oidc)}
    )

    missing = assess_production_readiness(settings, base_directory=tmp_path)
    missing_statuses = {check.id: check.status for check in missing.checks}

    assert missing_statuses["auth.oidc_console"] is ReadinessStatus.FAIL

    console = OIDCConsoleConfig(
        authorization_endpoint="https://identity.example/authorize",
        token_endpoint="https://identity.example/token",
        client_id="vehicle-console",
    )
    configured = settings.model_copy(
        update={
            "auth": AuthConfig(
                enabled=True,
                provider="oidc",
                oidc=oidc.model_copy(update={"console": console}),
            )
        }
    )
    report = assess_production_readiness(configured, base_directory=tmp_path)
    statuses = {check.id: check.status for check in report.checks}

    assert report.ready
    assert statuses["auth.oidc_console"] is ReadinessStatus.PASS


def test_model_tampering_fails_hash_gate(tmp_path) -> None:
    settings = _production_settings(tmp_path)
    (tmp_path / "vehicle.pt").write_bytes(b"tampered")

    report = assess_production_readiness(settings, base_directory=tmp_path)
    statuses = {check.id: check.status for check in report.checks}

    assert not report.ready
    assert statuses["model.vehicle_artifact"] is ReadinessStatus.PASS
    assert statuses["model.vehicle_hash"] is ReadinessStatus.FAIL


def test_ocr_directory_tampering_fails_hash_gate(tmp_path) -> None:
    settings = _production_settings(tmp_path)
    (tmp_path / "ocr-recognition" / "inference.json").write_bytes(b"tampered")

    report = assess_production_readiness(settings, base_directory=tmp_path)
    statuses = {check.id: check.status for check in report.checks}

    assert not report.ready
    assert statuses["model.ocr_artifacts"] is ReadinessStatus.PASS
    assert statuses["model.ocr_hash"] is ReadinessStatus.FAIL


def test_cli_writes_atomic_machine_readable_failure_report(tmp_path) -> None:
    output = tmp_path / "reports" / "readiness.json"

    exit_code = run(
        argparse.Namespace(
            config="configs/default.yaml",
            base_directory=tmp_path,
            strict_warnings=False,
            output=output,
        )
    )
    document = json.loads(output.read_text())

    assert exit_code == 4
    assert document["ready"] is False
    assert document["summary"]["FAIL"] > 0
    assert not output.with_name(f".{output.name}.tmp").exists()
