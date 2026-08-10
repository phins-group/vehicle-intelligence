from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vehicle_intelligence.exceptions import ConfigurationError, DependencyUnavailableError
from vehicle_intelligence.infrastructure.deployment import (
    EdgeArtifact,
    EdgeDeploymentManifest,
    apply_edge_environment,
    load_edge_manifest,
    resolve_edge_artifacts,
)
from vehicle_intelligence.infrastructure.vision.model_artifact import sha256_file


def artifact(root, role: str, provider: str = "onnxruntime") -> EdgeArtifact:
    path = root / f"{role}.onnx"
    path.write_bytes(f"{role}-model".encode())
    return EdgeArtifact(
        role=role,
        relativePath=path.name,
        provider=provider,
        modelName=f"{role}-detector",
        modelVersion="1",
        sha256=sha256_file(path),
        sizeBytes=path.stat().st_size,
        executionProviders=[] if provider != "tensorrt" else ["tensorrt"],
    )


def manifest(root, plate_provider: str = "onnxruntime") -> EdgeDeploymentManifest:
    return EdgeDeploymentManifest(
        schemaVersion=1,
        nodeId="edge-01",
        configVersion="config-7",
        createdAt=datetime(2026, 8, 10, tzinfo=UTC),
        artifacts=[artifact(root, "vehicle"), artifact(root, "plate", plate_provider)],
    )


def test_manifest_loads_resolves_hashes_and_composes_provider_environment(tmp_path) -> None:
    value = manifest(tmp_path)
    manifest_path = tmp_path / "edge.json"
    manifest_path.write_text(
        json.dumps(value.model_dump(by_alias=True, mode="json")), encoding="utf-8"
    )

    loaded = load_edge_manifest(manifest_path)
    resolved = resolve_edge_artifacts(
        loaded,
        tmp_path,
        available_execution_providers=["CPUExecutionProvider"],
    )
    environment = apply_edge_environment(loaded, resolved)

    assert resolved["vehicle"].execution_providers == ("CPUExecutionProvider",)
    assert environment["VIP_APP__CONFIG_VERSION"] == "config-7"
    assert environment["VIP_VISION__PLATE_DETECTION__MODEL_HASH"] == value.artifacts[1].sha256
    assert json.loads(environment["VIP_VISION__VEHICLE_DETECTION__EXECUTION_PROVIDERS"]) == [
        "CPUExecutionProvider"
    ]


def test_manifest_rejects_traversal_duplicate_roles_and_changed_artifacts(tmp_path) -> None:
    with pytest.raises(ValidationError, match="relative"):
        EdgeArtifact(
            role="vehicle",
            relativePath="../vehicle.onnx",
            provider="onnxruntime",
            modelName="vehicle",
            modelVersion="1",
            sha256="0" * 64,
            sizeBytes=1,
        )
    vehicle = artifact(tmp_path, "vehicle")
    with pytest.raises(ValidationError, match="roles must be unique"):
        EdgeDeploymentManifest(
            schemaVersion=1,
            nodeId="edge-01",
            configVersion="1",
            createdAt=datetime.now(UTC),
            artifacts=[vehicle, vehicle],
        )
    value = manifest(tmp_path)
    (tmp_path / "vehicle.onnx").write_bytes(b"tampered-data")
    with pytest.raises(ConfigurationError, match="SHA-256"):
        resolve_edge_artifacts(
            value,
            tmp_path,
            available_execution_providers=["CPUExecutionProvider"],
        )


def test_manifest_requires_requested_tensorrt_runtime(tmp_path) -> None:
    value = manifest(tmp_path, plate_provider="tensorrt")
    with pytest.raises(DependencyUnavailableError, match="unavailable"):
        resolve_edge_artifacts(
            value,
            tmp_path,
            available_execution_providers=["CPUExecutionProvider"],
        )
