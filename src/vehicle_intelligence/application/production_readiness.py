"""Static, secret-safe production deployment preflight checks."""

from __future__ import annotations

import base64
import hashlib
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from vehicle_intelligence.config import DetectorConfig, Settings
from vehicle_intelligence.exceptions import ModelLoadError
from vehicle_intelligence.model_artifact import sha256_directory

_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_PLACEHOLDER_VERSIONS = {"", "default", "unset", "unknown"}


class ReadinessStatus(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


CheckAppender = Callable[[str, ReadinessStatus, str], None]


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    id: str
    status: ReadinessStatus
    message: str


@dataclass(frozen=True, slots=True)
class ProductionReadinessReport:
    environment: str
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status is not ReadinessStatus.FAIL for check in self.checks)

    @property
    def counts(self) -> dict[str, int]:
        counts = Counter(check.status.value for check in self.checks)
        return {status.value: counts[status.value] for status in ReadinessStatus}

    def to_document(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "environment": self.environment,
            "ready": self.ready,
            "summary": self.counts,
            "checks": [
                {
                    "id": check.id,
                    "status": check.status.value,
                    "message": check.message,
                }
                for check in self.checks
            ],
        }


def assess_production_readiness(
    settings: Settings,
    *,
    base_directory: Path | None = None,
) -> ProductionReadinessReport:
    """Assess static deployment inputs without contacting external services."""

    root = (base_directory or Path.cwd()).expanduser().resolve()
    checks: list[ReadinessCheck] = []

    def add(check_id: str, status: ReadinessStatus, message: str) -> None:
        checks.append(ReadinessCheck(check_id, status, message))

    _assess_application(settings, add)
    _assess_authentication(settings, add)
    _assess_camera_credentials(settings, add)
    _assess_mongodb(settings, add)
    _assess_event_bus(settings, add)
    _assess_storage(settings, add)
    _assess_runtime_services(settings, add)
    _assess_debug(settings, add)
    _assess_models(settings, root, add)
    _assess_external_actions(settings, add)

    return ProductionReadinessReport(
        environment=settings.app.environment,
        checks=tuple(checks),
    )


def _add_rule(
    add: CheckAppender,
    check_id: str,
    passed: bool,
    passed_message: str,
    failed_message: str,
    *,
    failure_status: ReadinessStatus = ReadinessStatus.FAIL,
) -> None:
    add(
        check_id,
        ReadinessStatus.PASS if passed else failure_status,
        passed_message if passed else failed_message,
    )


def _assess_application(settings: Settings, add: CheckAppender) -> None:
    production = settings.app.environment.strip().casefold() == "production"
    _add_rule(
        add,
        "app.environment",
        production,
        "Environment is production.",
        "Set app.environment=production for a production deployment.",
    )
    versioned = settings.app.config_version.strip().casefold() not in _PLACEHOLDER_VERSIONS
    _add_rule(
        add,
        "app.config_version",
        versioned,
        "Configuration has an explicit version.",
        "Replace the placeholder app.config_version with an immutable version.",
    )


def _assess_authentication(settings: Settings, add: CheckAppender) -> None:
    _add_rule(
        add,
        "auth.enabled",
        settings.auth.enabled,
        f"Authentication is enabled with {settings.auth.provider}.",
        "Authentication is disabled.",
    )
    if not (
        settings.auth.enabled
        and settings.auth.provider == "oidc"
        and settings.auth.oidc is not None
    ):
        return
    _add_rule(
        add,
        "auth.oidc_transport",
        not settings.auth.oidc.allow_insecure_http,
        "OIDC issuer and JWKS transport require HTTPS.",
        "OIDC insecure-HTTP override must be disabled in production.",
    )
    _add_rule(
        add,
        "auth.oidc_console",
        settings.auth.oidc.console is not None,
        "The operator console has public Authorization Code + PKCE metadata.",
        "Configure auth.oidc.console for operator-console OIDC login.",
    )


def _assess_camera_credentials(settings: Settings, add: CheckAppender) -> None:
    valid, key_count = _camera_keyring_state(settings)
    _add_rule(
        add,
        "camera.credentials",
        valid,
        f"Camera credentials use {key_count} valid AES-256-GCM key(s).",
        "Configure a valid URL-safe base64 AES-256-GCM camera keyring.",
    )


def _assess_mongodb(settings: Settings, add: CheckAppender) -> None:
    _add_rule(
        add,
        "mongodb.enabled",
        settings.mongodb.enabled,
        "MongoDB persistence is enabled.",
        "MongoDB persistence is disabled.",
    )
    transactions = settings.mongodb.enabled and settings.mongodb.transactions_enabled
    _add_rule(
        add,
        "mongodb.transactions",
        transactions,
        "MongoDB transaction boundaries are enabled.",
        "Enable MongoDB transactions for atomic resource/audit mutations.",
    )
    _assess_mongo_uri(settings, add)


def _assess_event_bus(settings: Settings, add: CheckAppender) -> None:
    _add_rule(
        add,
        "event_bus.redis",
        settings.event_bus.backend == "redis",
        "Redis Streams is the durable event bus.",
        "Production event delivery must use Redis Streams instead of direct mode.",
    )
    _assess_redis_uri(settings, add)


def _assess_storage(settings: Settings, add: CheckAppender) -> None:
    _add_rule(
        add,
        "durability.finalization_outbox",
        settings.finalization_outbox.enabled,
        "Durable event/media finalization staging is enabled.",
        "Enable the durable finalization outbox for production camera workers.",
    )
    minio_storage = settings.storage.backend == "minio"
    _add_rule(
        add,
        "storage.minio",
        minio_storage,
        "Canonical media uses MinIO.",
        "Production canonical media must not use local storage.",
    )
    default_credentials = (
        settings.minio.access_key.get_secret_value() == "minioadmin"
        or settings.minio.secret_key.get_secret_value() == "minioadmin"
        or len(settings.minio.secret_key.get_secret_value()) < 16
    )
    _add_rule(
        add,
        "storage.credentials",
        not default_credentials,
        "MinIO credentials are non-default and meet the static length gate.",
        "Replace default or weak MinIO credentials.",
    )
    _add_rule(
        add,
        "storage.public_endpoint",
        minio_storage and settings.minio.public_endpoint is not None,
        "A browser-reachable signed-media endpoint is configured.",
        "Configure MinIO public_endpoint for signed evidence delivery.",
    )
    _add_rule(
        add,
        "storage.transport",
        settings.minio.secure,
        "MinIO transport uses TLS.",
        "MinIO transport is plaintext; require a protected network or enable TLS.",
        failure_status=ReadinessStatus.WARN,
    )


def _assess_runtime_services(settings: Settings, add: CheckAppender) -> None:
    _add_rule(
        add,
        "retention.enabled",
        settings.retention.enabled,
        "Coordinated retention is enabled.",
        "Coordinated media/event retention is disabled.",
    )
    _add_rule(
        add,
        "observability.prometheus",
        settings.observability.prometheus_enabled,
        "Prometheus metrics are enabled.",
        "Prometheus metrics are disabled.",
    )
    _add_rule(
        add,
        "observability.tracing",
        settings.observability.opentelemetry_enabled,
        "OpenTelemetry export is enabled.",
        "OpenTelemetry export is disabled; production traces will be unavailable.",
        failure_status=ReadinessStatus.WARN,
    )
    _add_rule(
        add,
        "realtime.enabled",
        settings.realtime.enabled,
        "Realtime event delivery is enabled.",
        "Realtime delivery is disabled; clients must poll REST.",
        failure_status=ReadinessStatus.WARN,
    )


def _assess_debug(settings: Settings, add: CheckAppender) -> None:
    disabled = not any(bool(value) for value in settings.debug.model_dump().values())
    _add_rule(
        add,
        "debug.disabled",
        disabled,
        "Debug artifact generation is disabled.",
        "Disable debug artifacts and verbose tracking before production.",
    )


def _assess_models(settings: Settings, root: Path, add: CheckAppender) -> None:
    _assess_detector("vehicle", settings.vision.vehicle_detection, root, add)
    _assess_detector("plate", settings.vision.plate_detection, root, add)
    versioned_ocr = (
        settings.vision.ocr.model_version.strip().casefold() not in _PLACEHOLDER_VERSIONS
    )
    _add_rule(
        add,
        "model.ocr_version",
        versioned_ocr,
        "OCR model version is explicit.",
        "Configure an explicit OCR model version.",
    )
    _assess_ocr(settings, root, add)
    _assess_embedding(settings, root, add)


def _assess_external_actions(settings: Settings, add: CheckAppender) -> None:
    if not settings.rule_engine.external_actions_enabled:
        return
    all_https = bool(settings.rule_engine.external_targets) and all(
        target.require_https for target in settings.rule_engine.external_targets
    )
    _add_rule(
        add,
        "actions.https",
        all_https,
        "External action targets require HTTPS.",
        "Every enabled external action target must be explicit and require HTTPS.",
    )


def _assess_detector(
    role: str,
    config: DetectorConfig,
    root: Path,
    add: CheckAppender,
) -> None:
    versioned = config.model_version.strip().casefold() not in _PLACEHOLDER_VERSIONS
    add(
        f"model.{role}_version",
        ReadinessStatus.PASS if versioned else ReadinessStatus.FAIL,
        f"{role.title()} detector version is explicit."
        if versioned
        else f"Configure an explicit {role} detector version.",
    )
    if not config.model_path:
        add(
            f"model.{role}_artifact",
            ReadinessStatus.FAIL,
            f"Configure a local {role} detector artifact.",
        )
        add(
            f"model.{role}_hash",
            ReadinessStatus.FAIL,
            f"Pin the {role} detector SHA-256.",
        )
        return
    path = _resolve(root, Path(config.model_path))
    artifact_exists = path.is_file() and path.stat().st_size > 0
    add(
        f"model.{role}_artifact",
        ReadinessStatus.PASS if artifact_exists else ReadinessStatus.FAIL,
        f"{role.title()} detector artifact is present and non-empty."
        if artifact_exists
        else f"{role.title()} detector artifact is missing or empty.",
    )
    expected = config.model_hash
    hash_valid = artifact_exists and _valid_hash(expected)
    hash_matches = hash_valid and expected is not None and _sha256_file(path) == expected.casefold()
    add(
        f"model.{role}_hash",
        ReadinessStatus.PASS if hash_matches else ReadinessStatus.FAIL,
        f"{role.title()} detector SHA-256 matches configuration."
        if hash_matches
        else f"Pin and verify the {role} detector SHA-256.",
    )


def _assess_ocr(settings: Settings, root: Path, add: CheckAppender) -> None:
    ocr = settings.vision.ocr
    configured_directories = (
        ocr.detection_model_directory,
        ocr.recognition_model_directory,
    )
    paths = tuple(
        _resolve(root, Path(value)) if value else None for value in configured_directories
    )
    artifacts_present = all(path is not None and path.is_dir() for path in paths)
    add(
        "model.ocr_artifacts",
        ReadinessStatus.PASS if artifacts_present else ReadinessStatus.FAIL,
        (
            "OCR detection and recognition model directories are local."
            if artifacts_present
            else "Configure local OCR detection and recognition model directories."
        ),
    )

    expected_hashes = (ocr.detection_model_hash, ocr.recognition_model_hash)
    hashes_match = artifacts_present and all(_valid_hash(value) for value in expected_hashes)
    if hashes_match:
        try:
            actual_hashes = tuple(sha256_directory(path) for path in paths if path is not None)
            hashes_match = all(
                actual == expected.casefold()
                for actual, expected in zip(actual_hashes, expected_hashes, strict=True)
                if expected is not None
            )
        except (ModelLoadError, OSError):
            hashes_match = False
    add(
        "model.ocr_hash",
        ReadinessStatus.PASS if hashes_match else ReadinessStatus.FAIL,
        (
            "OCR detection and recognition directory hashes match configuration."
            if hashes_match
            else "Pin and verify both OCR model directory SHA-256 manifest hashes."
        ),
    )


def _assess_embedding(settings: Settings, root: Path, add: CheckAppender) -> None:
    embedding = settings.identity.embedding
    if not embedding.enabled:
        status = (
            ReadinessStatus.WARN
            if settings.identity.enabled
            and settings.identity.reid.enabled
            and settings.identity.reid.embedding_weight > 0
            else ReadinessStatus.PASS
        )
        add(
            "model.embedding",
            status,
            (
                "Visual embedding is disabled; ReID will renormalize available signals."
                if status is ReadinessStatus.WARN
                else "Visual embedding is intentionally disabled."
            ),
        )
        return
    path = _resolve(root, embedding.model_path or Path(""))
    exists = path.is_file() and path.stat().st_size > 0
    expected = embedding.model_hash
    valid = (
        exists
        and embedding.model_version.strip().casefold() not in _PLACEHOLDER_VERSIONS
        and _valid_hash(expected)
        and expected is not None
        and _sha256_file(path) == expected.casefold()
    )
    add(
        "model.embedding",
        ReadinessStatus.PASS if valid else ReadinessStatus.FAIL,
        (
            "Vehicle embedding artifact/version/hash are pinned."
            if valid
            else "Enabled vehicle embedding requires a present, versioned, SHA-256-pinned artifact."
        ),
    )


def _assess_mongo_uri(settings: Settings, add: CheckAppender) -> None:
    parsed = urlsplit(settings.mongodb.uri.get_secret_value())
    query = {key.casefold(): value for key, value in parse_qs(parsed.query).items()}
    replica_set = parsed.scheme.casefold() == "mongodb+srv" or "replicaset" in query
    add(
        "mongodb.replica_set",
        (
            ReadinessStatus.PASS
            if settings.mongodb.enabled and settings.mongodb.transactions_enabled and replica_set
            else ReadinessStatus.FAIL
        ),
        (
            "MongoDB URI identifies a transaction-capable topology."
            if settings.mongodb.enabled and settings.mongodb.transactions_enabled and replica_set
            else "Use mongodb+srv or an explicit replicaSet URI for transactions."
        ),
    )
    add(
        "mongodb.authentication",
        ReadinessStatus.PASS if parsed.username is not None else ReadinessStatus.WARN,
        (
            "MongoDB URI includes an authenticated identity."
            if parsed.username is not None
            else "MongoDB URI has no username; verify mTLS or another external auth boundary."
        ),
    )


def _assess_redis_uri(settings: Settings, add: CheckAppender) -> None:
    parsed = urlsplit(settings.redis.url.get_secret_value())
    add(
        "redis.authentication",
        ReadinessStatus.PASS if parsed.password is not None else ReadinessStatus.WARN,
        (
            "Redis URI includes authentication."
            if parsed.password is not None
            else "Redis URI has no password; verify ACL/mTLS or a protected external boundary."
        ),
    )
    add(
        "redis.transport",
        ReadinessStatus.PASS if parsed.scheme.casefold() == "rediss" else ReadinessStatus.WARN,
        (
            "Redis transport uses TLS."
            if parsed.scheme.casefold() == "rediss"
            else "Redis transport is plaintext; require a protected network or use rediss."
        ),
    )


def _camera_keyring_state(settings: Settings) -> tuple[bool, int]:
    encoded: list[str] = [
        item.key.get_secret_value() for item in settings.security.camera_credential_keys
    ]
    if settings.security.camera_credential_key is not None:
        encoded.append(settings.security.camera_credential_key.get_secret_value())
    if not encoded:
        return False, 0
    try:
        decoded = [
            base64.b64decode(
                value + "=" * (-len(value) % 4),
                altchars=b"-_",
                validate=True,
            )
            for value in encoded
        ]
    except (TypeError, ValueError):
        return False, len(encoded)
    return all(len(value) == 32 for value in decoded), len(decoded)


def _valid_hash(value: str | None) -> bool:
    return value is not None and _HASH_PATTERN.fullmatch(value) is not None


def _resolve(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
