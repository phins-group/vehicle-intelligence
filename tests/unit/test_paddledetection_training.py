from __future__ import annotations

import json
import subprocess
from pathlib import Path

from vehicle_intelligence.training.config import PaddleDetectionConfig
from vehicle_intelligence.training.domain import DetectorRole
from vehicle_intelligence.training.paddledetection import PaddleDetectionTrainer

from .training_fixtures import build_detector_dataset


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "PaddleDetection"
    (repository / "tools").mkdir(parents=True)
    (repository / "configs/picodet").mkdir(parents=True)
    (repository / "tools/train.py").write_text("# train\n")
    (repository / "tools/export_model.py").write_text("# export\n")
    (repository / "configs/picodet/base.yml").write_text("architecture: PicoDet\n")
    return repository


def _config(tmp_path: Path, repository: Path) -> PaddleDetectionConfig:
    return PaddleDetectionConfig(
        repository_path=repository,
        base_config=Path("configs/picodet/base.yml"),
        output_directory=tmp_path / "training",
        device="cpu",
        epochs=2,
        snapshot_epoch=1,
        batch_size=2,
        workers=0,
    )


def test_paddledetection_training_command_and_run_manifest_are_traceable(tmp_path) -> None:
    dataset, dataset_config = build_detector_dataset(tmp_path)
    repository = _repository(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write(b"training complete\n")
        return subprocess.CompletedProcess(command, 0)

    trainer = PaddleDetectionTrainer(
        _config(tmp_path, repository),
        DetectorRole.VEHICLE,
        dataset_config.classes,
        command_runner=runner,
    )

    result = trainer.train(dataset, "vehicle-run-v1")
    manifest = json.loads(Path(result.manifest_path).read_text())
    command = calls[0][0]

    assert result.exit_code == 0
    assert "tools/train.py" in command
    assert "use_gpu=False" in command
    assert "TrainDataset.anno_path=annotations/train.json" in command
    assert manifest["datasetExportId"] == "vehicle-v1"
    assert len(manifest["datasetManifestSha256"]) == 64
    assert manifest["logSha256"]


def test_paddledetection_export_runs_official_export_then_paddle2onnx(tmp_path) -> None:
    dataset, dataset_config = build_detector_dataset(tmp_path)
    repository = _repository(tmp_path)
    weights = tmp_path / "best_model.pdparams"
    weights.write_bytes(b"weights")
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(b"ok\n")
        if any(str(item).endswith("tools/export_model.py") for item in command):
            output_arg = next(item for item in command if item.startswith("--output_dir="))
            inference = Path(output_arg.split("=", 1)[1]) / "picodet"
            inference.mkdir(parents=True)
            (inference / "model.pdmodel").write_bytes(b"graph")
            (inference / "model.pdiparams").write_bytes(b"params")
        else:
            target = Path(command[command.index("--save_file") + 1])
            target.write_bytes(b"fake-onnx")
        return subprocess.CompletedProcess(command, 0)

    trainer = PaddleDetectionTrainer(
        _config(tmp_path, repository),
        DetectorRole.VEHICLE,
        dataset_config.classes,
        command_runner=runner,
    )
    target = tmp_path / "models/vehicle.onnx"

    result = trainer.export_onnx(dataset, weights, target)

    assert result.read_bytes() == b"fake-onnx"
    assert len(calls) == 2
    assert calls[1][0] == "paddle2onnx"
    assert "--enable_onnx_checker" in calls[1]
