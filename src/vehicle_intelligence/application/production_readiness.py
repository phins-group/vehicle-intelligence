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

    production = settings.app.environment.strip().casefold() == "production"
    add(
        "app.environment",
        ReadinessStatus.PASS if production else ReadinessStatus.FAIL,
        "Environment is production."
        if production
        else "Set app.environment=production for a production deployment.",
    )
    versioned_config = settings.app.config_version.strip().casefold() not in (
        _PLACEHOLDER_VERSIONS
    )
    add(
        "app.config_version",
        ReadinessStatus.PASS if versioned_config else ReadinessStatus.FAIL,
        "Configuration has an explicit version."
        if versioned_config
        else "Replace the placeholder app.config_version with an immutable version.",
    )

    add(
        "auth.enabled",
        ReadinessStatus.PASS if settings.auth.enabled else ReadinessStatus.FAIL,
        f"Authentication is enabled with {settings.auth.provider}."
        if settings.auth.enabled
        else "Authentication is disabled.",
    )
    if (
        settings.auth.enabled
        and settings.auth.provider == "oidc"
        and settings.auth.oidc is not None
    ):
        add(
            "auth.oidc_transport",
            (
                ReadinessStatus.FAIL
                if settings.auth.oidc.allow_insecure_http
                else ReadinessStatus.PASS
            ),
            (
                "OIDC issuer and JWKS transport require HTTPS."
                if not settings.auth.oidc.allow_insecure_http
                else "OIDC insecure-HTTP override must be disabled in production."
            ),
        )

    keyring_valid, keyring_count = _camera_keyring_state(settings)
    add(
        "camera.credentials",
        ReadinessStatus.PASS if keyring_valid else ReadinessStatus.FAIL,
        (
            f"Camera credentials use {keyring_count} valid AES-256-GCM key(s)."
            if keyring_valid
            else "Configure a valid URL-safe base64 AES-256-GCM camera keyring."
        ),
    )

    add(
        "mongodb.enabled",
        ReadinessStatus.PASS if settings.mongodb.enabled else ReadinessStatus.FAIL,
        "MongoDB persistence is enabled."
        if settings.mongodb.enabled
        else "MongoDB persistence is disabled.",
    )
    add(
        "mongodb.transactions",
        (
            ReadinessStatus.PASS
            if settings.mongodb.enabled and settings.mongodb.transactions_enabled
            else ReadinessStatus.FAIL
        ),
        (
            "MongoDB transaction boundaries are enabled."
            if settings.mongodb.enabled and settings.mongodb.transactions_enabled
            else "Enable MongoDB transactions for atomic resource/audit mutations."
        ),
    )
    _assess_mongo_uri(settings, add)

    add(
        "event_bus.redis",
        ReadinessStatus.PASS if settings.event_bus.backend == "redis" else ReadinessStatus.FAIL,
        "Redis Streams is the durable event bus."
        if settings.event_bus.backend == "redis"
        else "Production event delivery must use Redis Streams instead of direct mode.",
    )
    _assess_redis_uri(settings, add)

    minio_storage = settings.storage.backend == "minio"
    add(
        "storage.minio",
        ReadinessStatus.PASS if minio_storage else ReadinessStatus.FAIL,
        "Canonical media uses MinIO."
        if minio_storage
        else "Production canonical media must not use local storage.",
    )
    minio_defaults = (
        settings.minio.access_key.get_secret_value() == "minioadmin"
        or settings.minio.secret_key.get_secret_value() == "minioadmin"
        or len(settings.minio.secret_key.get_secret_value()) < 16
    )
    add(
        "storage.credentials",
        ReadinessStatus.FAIL if minio_defaults else ReadinessStatus.PASS,
        (
            "MinIO credentials are non-default and meet the static length gate."
            if not minio_defaults
            else "Replace default or weak MinIO credentials."
        ),
    )
    add(
        "storage.public_endpoint",
        (
            ReadinessStatus.PASS
            if minio_storage and settings.minio.public_endpoint is not None
            else ReadinessStatus.FAIL
        ),
        (
            "A browser-reachable signed-media endpoint is configured."
            if minio_storage and settings.minio.public_endpoint is not None
            else "Configure MinIO public_endpoint for signed evidence delivery."
        ),
    )
    add(
        "storage.transport",
        ReadinessStatus.PASS if settings.minio.secure else ReadinessStatus.WARN,
        (
            "MinIO transport uses TLS."
            if settings.minio.secure
            else "MinIO transport is plaintext; require a protected network or enable TLS."
        ),
    )

    add(
        "retention.enabled",
        ReadinessStatus.PASS if settings.retention.enabled else ReadinessStatus.FAIL,
        "Coordinated retention is enabled."
        if settings.retention.enabled
        else "Coordinated media/event retention is disabled.",
    )
    add(
        "observability.prometheus",
        (
            ReadinessStatus.PASS
            if settings.observability.prometheus_enabled
            else ReadinessStatus.FAIL
        ),
        (
            "Prometheus metrics are enabled."
            if settings.observability.prometheus_enabled
            else "Prometheus metrics are disabled."
        ),
    )
    add(
        "observability.tracing",
        (
            ReadinessStatus.PASS
            if settings.observability.opentelemetry_enabled
            else ReadinessStatus.WARN
        ),
        (
            "OpenTelemetry export is enabled."
            if settings.observability.opentelemetry_enabled
            else "OpenTelemetry export is disabled; production traces will be unavailable."
        ),
    )
    add(
        "realtime.enabled",
        ReadinessStatus.PASS if settings.realtime.enabled else ReadinessStatus.WARN,
        "Realtime event delivery is enabled."
        if settings.realtime.enabled
        else "Realtime delivery is disabled; clients must poll REST.",
    )

    debug_enabled = any(bool(value) for value in settings.debug.model_dump().values())
    add(
        "debug.disabled",
        ReadinessStatus.FAIL if debug_enabled else ReadinessStatus.PASS,
        (
            "Debug artifact generation is disabled."
            if not debug_enabled
            else "Disable debug artifacts and verbose tracking before production."
        ),
    )

    _assess_detector("vehicle", settings.vision.vehicle_detection, root, add)
    _assess_detector("plate", settings.vision.plate_detection, root, add)
    add(
        "model.ocr_version",
        (
            ReadinessStatus.PASS
            if settings.vision.ocr.model_version.strip().casefold()
            not in _PLACEHOLDER_VERSIONS
            else ReadinessStatus.FAIL
        ),
        (
            "OCR model version is explicit."
            if settings.vision.ocr.model_version.strip().casefold()
            not in _PLACEHOLDER_VERSIONS
            else "Configure an explicit OCR model version."
        ),
    )
    add(
        "model.ocr_hash",
        (
            ReadinessStatus.PASS
            if _valid_hash(settings.vision.ocr.model_hash)
            else ReadinessStatus.WARN
        ),
        (
            "OCR model hash is pinned."
            if _valid_hash(settings.vision.ocr.model_hash)
            else "OCR model hash is not pinned; preserve provider cache artifacts externally."
        ),
    )
    _assess_embedding(settings, root, add)

    if settings.rule_engine.external_actions_enabled:
        all_https = bool(settings.rule_engine.external_targets) and all(
            target.require_https for target in settings.rule_engine.external_targets
        )
        add(
            "actions.https",
            ReadinessStatus.PASS if all_https else ReadinessStatus.FAIL,
            (
                "External action targets require HTTPS."
                if all_https
                else "Every enabled external action target must be explicit and require HTTPS."
            ),
        )

    return ProductionReadinessReport(
        environment=settings.app.environment,
        checks=tuple(checks),
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
    hash_matches = hash_valid and _sha256_file(path) == expected.casefold()
    add(
        f"model.{role}_hash",
        ReadinessStatus.PASS if hash_matches else ReadinessStatus.FAIL,
        f"{role.title()} detector SHA-256 matches configuration."
        if hash_matches
        else f"Pin and verify the {role} detector SHA-256.",
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
            if settings.mongodb.enabled
            and settings.mongodb.transactions_enabled
            and replica_set
            else ReadinessStatus.FAIL
        ),
        (
            "MongoDB URI identifies a transaction-capable topology."
            if settings.mongodb.enabled
            and settings.mongodb.transactions_enabled
            and replica_set
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
