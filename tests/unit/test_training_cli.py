from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vehicle_intelligence.training import cli
from vehicle_intelligence.training.config import load_training_settings
from vehicle_intelligence.training.domain import DetectorRole

_TRAINING_CONFIG = Path(__file__).resolve().parents[2] / "configs/model-training.yaml"


def test_verification_command_does_not_load_training_config(monkeypatch, capsys) -> None:
    manifest = {"type": "VERIFIED_SOURCE", "sampleCount": 3}
    monkeypatch.setattr(cli, "_verify_command", lambda _args: (manifest, "a" * 64))

    def fail_if_loaded(_path: Path) -> None:
        pytest.fail("verification commands must not load the training configuration")

    monkeypatch.setattr(cli, "load_training_settings", fail_if_loaded)

    assert cli.run(Namespace(command_name="verify-source")) == 0
    assert json.loads(capsys.readouterr().out) == {
        "manifest": manifest,
        "manifestSha256": "a" * 64,
    }


def test_package_command_skips_paddledetection_trainer(monkeypatch, tmp_path, capsys) -> None:
    settings = load_training_settings(_TRAINING_CONFIG)
    dataset = tmp_path / "dataset"
    onnx = tmp_path / "model.onnx"
    evaluation = tmp_path / "evaluation.json"
    output = tmp_path / "package"
    args = cli.build_parser().parse_args(
        [
            "package",
            "--role",
            "vehicle",
            str(dataset),
            "--onnx",
            str(onnx),
            "--evaluation",
            str(evaluation),
            "--model-name",
            "vehicle-detector",
            "--model-version",
            "v2",
            "--output",
            str(output),
        ]
    )
    captured: dict[str, object] = {}

    def package_candidate(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            directory=output,
            manifest_sha256="b" * 64,
            model_sha256="c" * 64,
            reused=False,
        )

    def unexpected_trainer(*_args, **_kwargs):
        pytest.fail("package must not construct a PaddleDetection trainer")

    monkeypatch.setattr(cli, "load_training_settings", lambda _path: settings)
    monkeypatch.setattr(cli, "package_detector_candidate", package_candidate)
    monkeypatch.setattr(cli, "PaddleDetectionTrainer", unexpected_trainer)

    assert cli.run(args) == 0
    assert captured == {
        "role": DetectorRole.VEHICLE,
        "model_name": "vehicle-detector",
        "model_version": "v2",
        "classes": ("car", "motorcycle", "bus", "truck"),
        "onnx_path": onnx,
        "dataset_directory": dataset,
        "evaluation_path": evaluation,
        "output_directory": output,
        "training_run_path": None,
    }
    assert json.loads(capsys.readouterr().out) == {
        "directory": str(output),
        "manifestSha256": "b" * 64,
        "modelSha256": "c" * 64,
        "reused": False,
    }


def test_export_command_constructs_trainer_lazily(monkeypatch, tmp_path, capsys) -> None:
    settings = load_training_settings(_TRAINING_CONFIG)
    dataset = tmp_path / "dataset"
    weights = tmp_path / "weights.pdparams"
    output = tmp_path / "plate.onnx"
    args = cli.build_parser().parse_args(
        [
            "export-onnx",
            "--role",
            "plate",
            str(dataset),
            "--weights",
            str(weights),
            "--output",
            str(output),
        ]
    )
    constructions: list[tuple[object, DetectorRole, tuple[str, ...]]] = []

    class FakeTrainer:
        def __init__(self, config, role, classes) -> None:
            constructions.append((config, role, classes))

        def export_onnx(self, dataset_path, weights_path, output_path):
            assert (dataset_path, weights_path, output_path) == (dataset, weights, output)
            output_path.write_bytes(b"optimized-onnx")
            return output_path

    monkeypatch.setattr(cli, "load_training_settings", lambda _path: settings)
    monkeypatch.setattr(cli, "PaddleDetectionTrainer", FakeTrainer)

    assert cli.run(args) == 0
    assert constructions == [
        (settings.plate.paddledetection, DetectorRole.PLATE, ("license_plate",))
    ]
    assert json.loads(capsys.readouterr().out) == {
        "onnx": str(output),
        "sha256": hashlib.sha256(b"optimized-onnx").hexdigest(),
    }
