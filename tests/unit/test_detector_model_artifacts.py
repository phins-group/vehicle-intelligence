from __future__ import annotations

import json

import pytest

from vehicle_intelligence.exceptions import ModelArtifactError
from vehicle_intelligence.training.artifacts import (
    package_detector_candidate,
    verify_model_package,
)
from vehicle_intelligence.training.dataset import verify_detector_dataset
from vehicle_intelligence.training.domain import DetectorRole

from .training_fixtures import build_detector_dataset


def test_only_verified_gate_passed_candidate_is_packaged_atomically(tmp_path) -> None:
    dataset, config = build_detector_dataset(tmp_path)
    _, dataset_digest = verify_detector_dataset(dataset)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "type": "DETECTOR_EVALUATION",
                "role": "vehicle",
                "split": "test",
                "datasetManifestSha256": dataset_digest,
                "evidenceVerified": True,
                "metrics": {"recall": 1.0},
                "releaseGate": {"passed": True, "failures": []},
            }
        )
    )
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"test-onnx")

    result = package_detector_candidate(
        role=DetectorRole.VEHICLE,
        model_name="warehouse-vehicle",
        model_version="v1",
        classes=config.classes,
        onnx_path=model,
        dataset_directory=dataset,
        evaluation_path=evaluation,
        output_directory=tmp_path / "packages",
        validate_onnx=lambda _: None,
    )
    manifest, digest = verify_model_package(
        result.directory,
        validate_onnx=lambda _: None,
    )

    assert result.manifest_sha256 == digest
    assert manifest["licenseStatus"] == "REVIEW_REQUIRED"
    assert manifest["model"]["sha256"] == result.model_sha256
    assert "license: other" in (result.directory / "README.md").read_text()

    (result.directory / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ModelArtifactError, match="verification failed"):
        verify_model_package(result.directory, validate_onnx=lambda _: None)


def test_unverified_prediction_evidence_cannot_be_packaged(tmp_path) -> None:
    dataset, config = build_detector_dataset(tmp_path)
    _, dataset_digest = verify_detector_dataset(dataset)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "type": "DETECTOR_EVALUATION",
                "role": "vehicle",
                "split": "test",
                "datasetManifestSha256": dataset_digest,
                "evidenceVerified": False,
                "releaseGate": {"passed": True, "failures": []},
            }
        )
    )
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"test-onnx")

    with pytest.raises(ModelArtifactError, match="evidence"):
        package_detector_candidate(
            role=DetectorRole.VEHICLE,
            model_name="warehouse-vehicle",
            model_version="v1",
            classes=config.classes,
            onnx_path=model,
            dataset_directory=dataset,
            evaluation_path=evaluation,
            output_directory=tmp_path / "packages",
            validate_onnx=lambda _: None,
        )


def test_bootstrap_only_dataset_cannot_be_packaged(tmp_path) -> None:
    dataset, config = build_detector_dataset(tmp_path, acceptance_eligible=False)
    _, dataset_digest = verify_detector_dataset(dataset)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "type": "DETECTOR_EVALUATION",
                "role": "vehicle",
                "split": "test",
                "datasetManifestSha256": dataset_digest,
                "evidenceVerified": True,
                "releaseGate": {"passed": True, "failures": []},
            }
        )
    )
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"test-onnx")

    with pytest.raises(ModelArtifactError, match="bootstrap-only"):
        package_detector_candidate(
            role=DetectorRole.VEHICLE,
            model_name="warehouse-vehicle",
            model_version="v1",
            classes=config.classes,
            onnx_path=model,
            dataset_directory=dataset,
            evaluation_path=evaluation,
            output_directory=tmp_path / "packages",
            validate_onnx=lambda _: None,
        )
