"""Strict edge model manifest loading, integrity checks, and env composition."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_intelligence.exceptions import ConfigurationError, DependencyUnavailableError
from vehicle_intelligence.infrastructure.vision.model_artifact import sha256_file
from vehicle_intelligence.infrastructure.vision.onnx_runtime import (
    select_execution_providers,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EdgeArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    role: Literal["vehicle", "plate"]
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=512)
    provider: Literal["onnxruntime", "tensorrt"]
    model_name: str = Field(alias="modelName", min_length=1, max_length=128)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=128)
    sha256: str
    size_bytes: int = Field(alias="sizeBytes", gt=0)
    execution_providers: list[str] = Field(
        default_factory=list,
        alias="executionProviders",
        max_length=8,
    )

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower().removeprefix("sha256:")
        if not _SHA256.fullmatch(normalized):
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        return normalized

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("artifact path must use portable forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.name:
            raise ValueError("artifact path must remain relative to model root")
        return path.as_posix()


class EdgeDeploymentManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_version: Literal[1] = Field(alias="schemaVersion")
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    config_version: str = Field(alias="configVersion", min_length=1, max_length=128)
    created_at: datetime = Field(alias="createdAt")
    artifacts: list[EdgeArtifact] = Field(min_length=2, max_length=8)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("edge manifest timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_roles(self) -> EdgeDeploymentManifest:
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("edge artifact roles must be unique")
        if set(roles) != {"vehicle", "plate"}:
            raise ValueError("edge manifest requires vehicle and plate artifacts")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedEdgeArtifact:
    artifact: EdgeArtifact
    path: Path
    execution_providers: tuple[str, ...]


def load_edge_manifest(path: Path, *, maximum_bytes: int = 128 * 1024) -> EdgeDeploymentManifest:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"edge manifest does not exist: {resolved}")
    if resolved.stat().st_size > maximum_bytes:
        raise ConfigurationError("edge manifest exceeds size limit")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
        return EdgeDeploymentManifest.model_validate(document)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid edge manifest: {resolved}") from exc


def resolve_edge_artifacts(
    manifest: EdgeDeploymentManifest,
    model_root: Path,
    *,
    check_runtime: bool = True,
    available_execution_providers: list[str] | None = None,
) -> dict[str, ResolvedEdgeArtifact]:
    root = model_root.expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"edge model root does not exist: {root}")
    available = available_execution_providers
    resolved: dict[str, ResolvedEdgeArtifact] = {}
    for artifact in manifest.artifacts:
        path = (root / artifact.relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ConfigurationError(
                f"edge artifact is missing or escapes model root: {artifact.relative_path}"
            )
        if path.stat().st_size != artifact.size_bytes:
            raise ConfigurationError(f"edge artifact size mismatch: {artifact.role}")
        if sha256_file(path) != artifact.sha256:
            raise ConfigurationError(f"edge artifact SHA-256 mismatch: {artifact.role}")
        providers: tuple[str, ...] = ()
        if path.suffix.lower() != ".onnx":
            raise ConfigurationError("edge detector artifacts must use .onnx")
        requested = artifact.execution_providers
        if artifact.provider == "tensorrt" and not requested:
            requested = ["tensorrt"]
        if check_runtime:
            if available is None:
                try:
                    import onnxruntime as ort
                except ImportError as exc:
                    raise DependencyUnavailableError(
                        "ONNX Runtime is required by the edge manifest"
                    ) from exc
                available = list(ort.get_available_providers())
            providers = select_execution_providers(requested, available)
        resolved[artifact.role] = ResolvedEdgeArtifact(artifact, path, providers)
    return resolved


def apply_edge_environment(
    manifest: EdgeDeploymentManifest,
    artifacts: dict[str, ResolvedEdgeArtifact],
) -> dict[str, str]:
    environment = {
        "VIP_APP__ENVIRONMENT": "edge",
        "VIP_APP__CONFIG_VERSION": manifest.config_version,
    }
    for role, prefix in (("vehicle", "VEHICLE_DETECTION"), ("plate", "PLATE_DETECTION")):
        resolved = artifacts[role]
        artifact = resolved.artifact
        stem = f"VIP_VISION__{prefix}"
        environment[f"{stem}__PROVIDER"] = artifact.provider
        environment[f"{stem}__MODEL_PATH"] = str(resolved.path)
        environment[f"{stem}__MODEL_NAME"] = artifact.model_name
        environment[f"{stem}__MODEL_VERSION"] = artifact.model_version
        environment[f"{stem}__MODEL_HASH"] = artifact.sha256
        if resolved.execution_providers:
            environment[f"{stem}__EXECUTION_PROVIDERS"] = json.dumps(
                list(resolved.execution_providers)
            )
    return environment
