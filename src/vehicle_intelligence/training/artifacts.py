"""Checksum-verified detector candidate packaging after release gates pass."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from vehicle_intelligence.exceptions import ModelArtifactError
from vehicle_intelligence.training.dataset import verify_detector_dataset
from vehicle_intelligence.training.domain import DetectorRole

_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ModelPackageResult:
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    model_sha256: str
    reused: bool = False


def package_detector_candidate(
    *,
    role: DetectorRole,
    model_name: str,
    model_version: str,
    classes: tuple[str, ...],
    onnx_path: Path,
    dataset_directory: Path,
    evaluation_path: Path,
    output_directory: Path,
    training_run_path: Path | None = None,
    provider: str = "picodet",
    validate_onnx: Callable[[Path], None] | None = None,
) -> ModelPackageResult:
    artifact_id = f"{model_name}-{model_version}"
    if not _ARTIFACT_ID.fullmatch(artifact_id):
        raise ModelArtifactError("model name/version do not form a path-safe artifact id")
    source_model = onnx_path.expanduser().resolve()
    if source_model.suffix.lower() != ".onnx" or not source_model.is_file():
        raise ModelArtifactError("candidate ONNX model is missing")
    (validate_onnx or _validate_onnx)(source_model)
    dataset_root = dataset_directory.expanduser().resolve()
    dataset_manifest, dataset_digest = verify_detector_dataset(dataset_root)
    if dataset_manifest["role"] != role.value:
        raise ModelArtifactError("candidate role does not match dataset role")
    if dataset_manifest.get("acceptanceEligible") is not True:
        raise ModelArtifactError(
            "bootstrap-only dataset cannot be used as release acceptance evidence"
        )
    if tuple(dataset_manifest["classes"]) != classes:
        raise ModelArtifactError("candidate classes do not match dataset class order")

    evaluation = _read_json(evaluation_path, "detector evaluation")
    if (
        evaluation.get("schemaVersion") != 1
        or evaluation.get("type") != "DETECTOR_EVALUATION"
        or evaluation.get("role") != role.value
        or evaluation.get("split") != "test"
    ):
        raise ModelArtifactError("detector evaluation must be canonical test-split evidence")
    release_gate = evaluation.get("releaseGate")
    if not isinstance(release_gate, dict) or release_gate.get("passed") is not True:
        raise ModelArtifactError("detector evaluation release gate has not passed")
    if evaluation.get("evidenceVerified") is not True:
        raise ModelArtifactError("detector prediction evidence is not checksum verified")
    if evaluation.get("datasetManifestSha256") != dataset_digest:
        raise ModelArtifactError("detector evaluation was produced from another dataset")
    training_run: dict[str, Any] | None = None
    if training_run_path is not None:
        training_run = _read_json(training_run_path, "training run")
        if training_run.get("exitCode") != 0 or training_run.get("errorCode") is not None:
            raise ModelArtifactError("training run did not complete successfully")
        if training_run.get("datasetManifestSha256") != dataset_digest:
            raise ModelArtifactError("training run used another dataset")

    root = output_directory.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / artifact_id).resolve()
    if not target.is_relative_to(root):
        raise ModelArtifactError("model artifact target escapes configured root")
    if target.exists():
        manifest, digest = verify_model_package(target, validate_onnx=validate_onnx)
        return ModelPackageResult(
            target,
            target / "manifest.json",
            digest,
            str(manifest["model"]["sha256"]),
            reused=True,
        )

    temporary = root / f".{artifact_id}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(parents=False, exist_ok=False)
        model_target = temporary / "model.onnx"
        shutil.copyfile(source_model, model_target)
        evaluation_target = temporary / "evaluation.json"
        shutil.copyfile(evaluation_path.expanduser().resolve(), evaluation_target)
        dataset_target = temporary / "dataset-manifest.json"
        shutil.copyfile(dataset_root / "manifest.json", dataset_target)
        files = [
            _file_entry(path, temporary)
            for path in (model_target, evaluation_target, dataset_target)
        ]
        if training_run_path is not None:
            run_target = temporary / "training-run.json"
            shutil.copyfile(training_run_path.expanduser().resolve(), run_target)
            files.append(_file_entry(run_target, temporary))
        model_digest = _sha256_file(model_target)
        card_target = temporary / "README.md"
        _write_new(
            card_target,
            _model_card(
                role,
                model_name,
                model_version,
                provider,
                classes,
                dataset_manifest,
                dataset_digest,
                model_digest,
                evaluation,
            ).encode(),
        )
        files.append(_file_entry(card_target, temporary))
        manifest = {
            "schemaVersion": 1,
            "type": "DETECTOR_MODEL_CANDIDATE",
            "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "role": role.value,
            "provider": provider,
            "licenseStatus": "REVIEW_REQUIRED",
            "model": {
                "name": model_name,
                "version": model_version,
                "format": "ONNX",
                "path": "model.onnx",
                "sha256": model_digest,
            },
            "classes": list(classes),
            "dataset": {
                "exportId": dataset_manifest["exportId"],
                "manifestSha256": dataset_digest,
            },
            "evaluation": {
                "path": "evaluation.json",
                "sha256": _sha256_file(evaluation_target),
                "releaseGate": release_gate,
            },
            "trainingRun": (
                {
                    "path": "training-run.json",
                    "sha256": _sha256_file(temporary / "training-run.json"),
                    "runId": training_run.get("runId") if training_run else None,
                }
                if training_run_path is not None
                else None
            ),
            "files": sorted(files, key=lambda item: item["path"]),
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        _write_new(temporary / "manifest.json", manifest_bytes)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        temporary.replace(target)
        return ModelPackageResult(
            target,
            target / "manifest.json",
            manifest_digest,
            model_digest,
        )
    except ModelArtifactError:
        _remove_tree(temporary, root)
        raise
    except Exception as exc:
        _remove_tree(temporary, root)
        raise ModelArtifactError("cannot package detector model candidate") from exc


def verify_model_package(
    directory: Path,
    *,
    validate_onnx: Callable[[Path], None] | None = None,
) -> tuple[dict[str, Any], str]:
    root = directory.expanduser().resolve()
    if not _ARTIFACT_ID.fullmatch(root.name):
        raise ModelArtifactError("model artifact directory name is invalid")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, "model artifact manifest")
    if (
        manifest.get("schemaVersion") != 1
        or manifest.get("type") != "DETECTOR_MODEL_CANDIDATE"
        or manifest.get("role") not in {"vehicle", "plate"}
        or not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) > 32
    ):
        raise ModelArtifactError("model artifact manifest contract is invalid")
    paths: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ModelArtifactError("model artifact file entry is invalid")
        path = _safe_child(root, item["path"])
        if item["path"] in paths or not path.is_file():
            raise ModelArtifactError("model artifact file is missing or duplicated")
        paths.add(item["path"])
        if path.stat().st_size != int(item.get("size", -1)):
            raise ModelArtifactError("model artifact file size verification failed")
        if _sha256_file(path) != item.get("sha256"):
            raise ModelArtifactError("model artifact checksum verification failed")
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("path") != "model.onnx":
        raise ModelArtifactError("model artifact ONNX contract is invalid")
    model_path = root / "model.onnx"
    if _sha256_file(model_path) != model.get("sha256"):
        raise ModelArtifactError("model artifact ONNX hash does not match metadata")
    (validate_onnx or _validate_onnx)(model_path)
    return manifest, _sha256_file(manifest_path)


def _validate_onnx(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise ModelArtifactError("ONNX validation requires `pip install -e '.[training]'`") from exc
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ModelArtifactError("candidate ONNX model failed validation") from exc


def _model_card(
    role: DetectorRole,
    name: str,
    version: str,
    provider: str,
    classes: tuple[str, ...],
    dataset: dict[str, Any],
    dataset_digest: str,
    model_digest: str,
    evaluation: dict[str, Any],
) -> str:
    metrics = evaluation.get("metrics", evaluation)
    return f"""---
library_name: paddledetection
pipeline_tag: object-detection
tags:
  - vehicle-intelligence
  - {role.value}-detector
license: other
---

# {name} {version}

Private production candidate for the `{role.value}` detector role.

## Traceability

- Provider: `{provider}`
- Classes: `{", ".join(classes)}`
- Dataset export: `{dataset["exportId"]}`
- Dataset manifest SHA-256: `{dataset_digest}`
- ONNX SHA-256: `{model_digest}`
- License status: `REVIEW_REQUIRED` (framework, checkpoint, and dataset records
  must be approved together before commercial promotion)

## Evaluation

```json
{json.dumps(metrics, indent=2, sort_keys=True)}
```

This candidate must remain subject to site-specific shadow validation and the
license/source records for every dataset and pretrained checkpoint.
"""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"{description} cannot be read") from exc
    if not isinstance(value, dict):
        raise ModelArtifactError(f"{description} root must be an object")
    return value


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ModelArtifactError("model artifact path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise ModelArtifactError("model artifact path escapes package")
    return path


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


def _remove_tree(target: Path, root: Path) -> None:
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ModelArtifactError("refusing to remove unsafe model artifact path")
    if resolved.exists():
        shutil.rmtree(resolved)
