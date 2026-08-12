"""Subprocess-isolated PaddleDetection/PicoDet training and ONNX export."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vehicle_intelligence.exceptions import ModelTrainingError
from vehicle_intelligence.training.config import PaddleDetectionConfig
from vehicle_intelligence.training.dataset import verify_detector_dataset
from vehicle_intelligence.training.domain import DetectorRole, TrainingRunResult

CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


class PaddleDetectionTrainer:
    """Launch official PaddleDetection tooling without importing it into runtime."""

    def __init__(
        self,
        config: PaddleDetectionConfig,
        role: DetectorRole,
        classes: tuple[str, ...],
        *,
        command_runner: CommandRunner = subprocess.run,
    ) -> None:
        self._config = config
        self._role = role
        self._classes = classes
        self._run_command = command_runner
        self._repository = config.repository_path.expanduser().resolve()

    def build_train_command(self, dataset_directory: Path, run_directory: Path) -> tuple[str, ...]:
        dataset = dataset_directory.expanduser().resolve()
        verify_detector_dataset(dataset)
        base_config = self._base_config()
        train_script = self._tool("tools/train.py")
        command: list[str]
        if self._config.device == "gpu":
            command = [
                self._config.python_executable,
                "-m",
                "paddle.distributed.launch",
                "--gpus",
                ",".join(self._config.gpus),
                str(train_script.relative_to(self._repository)),
            ]
        else:
            command = [
                self._config.python_executable,
                str(train_script.relative_to(self._repository)),
            ]
        command.extend(
            [
                "--eval",
                "-c",
                str(base_config.relative_to(self._repository)),
                "-o",
                *_dataset_overrides(dataset),
                "metric=COCO",
                f"num_classes={len(self._classes)}",
                f"epoch={self._config.epochs}",
                f"snapshot_epoch={self._config.snapshot_epoch}",
                f"worker_num={self._config.workers}",
                f"TrainReader.batch_size={self._config.batch_size}",
                f"save_dir={run_directory.expanduser().resolve()}",
                f"use_gpu={self._config.device == 'gpu'}",
            ]
        )
        if self._config.pretrain_weights:
            command.append(f"pretrain_weights={self._config.pretrain_weights}")
        command.extend(
            f"{key}={_render_override(value)}"
            for key, value in sorted(self._config.extra_overrides.items())
        )
        return tuple(command)

    def train(self, dataset_directory: Path, run_id: str) -> TrainingRunResult:
        if not _safe_run_id(run_id):
            raise ModelTrainingError("training run id is not path-safe")
        dataset = dataset_directory.expanduser().resolve()
        dataset_manifest, dataset_digest = verify_detector_dataset(dataset)
        if dataset_manifest["role"] != self._role.value:
            raise ModelTrainingError("training role does not match detector dataset")
        output_root = self._config.output_directory.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        run_directory = (output_root / run_id).resolve()
        if not run_directory.is_relative_to(output_root) or run_directory.exists():
            raise ModelTrainingError("training output directory already exists or is unsafe")
        run_directory.mkdir(parents=False, exist_ok=False)
        log_path = run_directory / "training.log"
        manifest_path = run_directory / "training-run.json"
        command = self.build_train_command(dataset, run_directory)
        started = datetime.now(UTC)
        exit_code = -1
        error_code: str | None = None
        try:
            with log_path.open("xb") as log:
                try:
                    completed = self._run_command(
                        list(command),
                        cwd=self._repository,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=self._config.maximum_runtime_seconds,
                    )
                    exit_code = int(completed.returncode)
                    if exit_code != 0:
                        error_code = "TRAINING_PROCESS_FAILED"
                except subprocess.TimeoutExpired:
                    error_code = "TRAINING_TIMEOUT"
        except OSError as exc:
            error_code = "TRAINING_PROCESS_START_FAILED"
            _write_log_failure(log_path, exc)
        finished = datetime.now(UTC)
        manifest = {
            "schemaVersion": 1,
            "type": "PADDLEDETECTION_TRAINING_RUN",
            "runId": run_id,
            "role": self._role.value,
            "classes": list(self._classes),
            "startedAt": _timestamp(started),
            "finishedAt": _timestamp(finished),
            "durationSeconds": (finished - started).total_seconds(),
            "exitCode": exit_code,
            "errorCode": error_code,
            "command": list(command),
            "datasetExportId": dataset_manifest["exportId"],
            "datasetManifestSha256": dataset_digest,
            "paddleDetectionRevision": _git_revision(self._repository),
            "baseConfig": str(self._base_config().relative_to(self._repository)),
            "baseConfigSha256": _sha256_file(self._base_config()),
            "logSha256": _sha256_file(log_path),
        }
        _write_new(manifest_path, _json_bytes(manifest))
        result = TrainingRunResult(
            role=self._role,
            output_directory=str(run_directory),
            log_path=str(log_path),
            manifest_path=str(manifest_path),
            exit_code=exit_code,
            command=command,
        )
        if error_code is not None:
            raise ModelTrainingError(f"{error_code}; inspect {log_path}")
        return result

    def export_onnx(
        self,
        dataset_directory: Path,
        weights_path: Path,
        target_onnx: Path,
    ) -> Path:
        dataset = dataset_directory.expanduser().resolve()
        manifest, _ = verify_detector_dataset(dataset)
        if manifest["role"] != self._role.value:
            raise ModelTrainingError("export role does not match detector dataset")
        weights = weights_path.expanduser().resolve()
        if not weights.is_file():
            raise ModelTrainingError("PaddleDetection weights file is missing")
        target = target_onnx.expanduser().resolve()
        if target.suffix.lower() != ".onnx" or target.exists():
            raise ModelTrainingError("ONNX export target must be a new .onnx file")
        target.parent.mkdir(parents=True, exist_ok=True)
        inference_root = target.parent / f".{target.stem}-paddle-inference"
        if inference_root.exists():
            raise ModelTrainingError("Paddle inference export directory already exists")
        export_log = target.with_suffix(".export.log")
        export_command = self.build_export_command(dataset, weights, inference_root)
        self._execute_stage(export_command, export_log, "PADDLE_EXPORT_FAILED")
        model_directory, model_filename, params_filename = _find_inference_model(inference_root)
        conversion_command = (
            self._config.paddle2onnx_executable,
            "--model_dir",
            str(model_directory),
            "--model_filename",
            model_filename,
            "--params_filename",
            params_filename,
            "--save_file",
            str(target),
            "--opset_version",
            str(self._config.onnx_opset),
            "--enable_onnx_checker",
            "True",
        )
        conversion_log = target.with_suffix(".paddle2onnx.log")
        self._execute_stage(conversion_command, conversion_log, "PADDLE2ONNX_FAILED")
        if not target.is_file() or target.stat().st_size == 0:
            raise ModelTrainingError("Paddle2ONNX completed without producing a model")
        return target

    def build_export_command(
        self,
        dataset_directory: Path,
        weights_path: Path,
        output_directory: Path,
    ) -> tuple[str, ...]:
        dataset = dataset_directory.expanduser().resolve()
        verify_detector_dataset(dataset)
        export_script = self._tool("tools/export_model.py")
        return (
            self._config.python_executable,
            str(export_script.relative_to(self._repository)),
            "-c",
            str(self._base_config().relative_to(self._repository)),
            f"--output_dir={output_directory.expanduser().resolve()}",
            "-o",
            *_dataset_overrides(dataset),
            f"num_classes={len(self._classes)}",
            f"weights={weights_path.expanduser().resolve()}",
        )

    def _execute_stage(
        self,
        command: tuple[str, ...],
        log_path: Path,
        error_code: str,
    ) -> None:
        try:
            with log_path.open("xb") as log:
                completed = self._run_command(
                    list(command),
                    cwd=self._repository,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=self._config.export_maximum_runtime_seconds,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ModelTrainingError(f"{error_code}; inspect {log_path}") from exc
        if completed.returncode != 0:
            raise ModelTrainingError(f"{error_code}; inspect {log_path}")

    def _base_config(self) -> Path:
        configured = self._config.base_config.expanduser()
        path = (
            configured.resolve()
            if configured.is_absolute()
            else (self._repository / configured).resolve()
        )
        if not path.is_relative_to(self._repository) or not path.is_file():
            raise ModelTrainingError("PaddleDetection base config is missing or outside repository")
        return path

    def _tool(self, relative: str) -> Path:
        path = (self._repository / relative).resolve()
        if not path.is_relative_to(self._repository) or not path.is_file():
            raise ModelTrainingError(f"PaddleDetection tool is missing: {relative}")
        return path


def _dataset_overrides(dataset: Path) -> tuple[str, ...]:
    return (
        f"TrainDataset.dataset_dir={dataset}",
        "TrainDataset.image_dir=",
        "TrainDataset.anno_path=annotations/train.json",
        f"EvalDataset.dataset_dir={dataset}",
        "EvalDataset.image_dir=",
        "EvalDataset.anno_path=annotations/validation.json",
        f"TestDataset.dataset_dir={dataset}",
        "TestDataset.image_dir=",
        "TestDataset.anno_path=annotations/test.json",
    )


def _find_inference_model(root: Path) -> tuple[Path, str, str]:
    model_names = ("model.pdmodel", "inference.pdmodel", "inference.json", "model.json")
    params_names = ("model.pdiparams", "inference.pdiparams")
    for directory, _, filenames in os.walk(root):
        names = set(filenames)
        model = next((name for name in model_names if name in names), None)
        params = next((name for name in params_names if name in names), None)
        if model is not None and params is not None:
            return Path(directory), model, params
    raise ModelTrainingError("PaddleDetection inference export is incomplete")


def _git_revision(repository: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = completed.stdout.strip().lower()
    return revision if completed.returncode == 0 and len(revision) == 40 else None


def _render_override(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _safe_run_id(value: str) -> bool:
    return bool(
        value and len(value) <= 128 and all(char.isalnum() or char in "_.-" for char in value)
    )


def _write_log_failure(path: Path, exc: OSError) -> None:
    if path.exists():
        return
    _write_new(path, f"process start failed: {type(exc).__name__}\n".encode())


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
