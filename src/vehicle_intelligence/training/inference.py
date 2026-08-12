"""Run any configured canonical detector against an immutable COCO split."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from vehicle_intelligence.application.ports import PlateDetector, VehicleDetector
from vehicle_intelligence.exceptions import ModelEvaluationError
from vehicle_intelligence.training.dataset import verify_detector_dataset
from vehicle_intelligence.training.domain import DatasetSplit, DetectorRole


@dataclass(frozen=True, slots=True)
class DatasetInferenceResult:
    predictions_path: Path
    manifest_path: Path
    prediction_count: int
    image_count: int
    predictions_sha256: str


def predict_dataset_split(
    dataset_directory: Path,
    split: DatasetSplit,
    role: DetectorRole,
    detector: VehicleDetector | PlateDetector,
    output_path: Path,
    *,
    model_info: dict[str, Any],
) -> DatasetInferenceResult:
    root = dataset_directory.expanduser().resolve()
    dataset_manifest, dataset_sha256 = verify_detector_dataset(root)
    if dataset_manifest["role"] != role.value:
        raise ModelEvaluationError("detector role does not match dataset role")
    annotation_path = root / "annotations" / f"{split.value}.json"
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelEvaluationError("cannot load detector evaluation split") from exc
    if not isinstance(document, dict) or not isinstance(document.get("images"), list):
        raise ModelEvaluationError("detector evaluation split contract is invalid")
    categories = {
        str(item["name"]).strip().lower(): int(item["id"])
        for item in document.get("categories", [])
        if isinstance(item, dict) and "name" in item and "id" in item
    }
    if not categories:
        raise ModelEvaluationError("detector evaluation categories are missing")

    predictions: list[dict[str, Any]] = []
    for image_record in sorted(document["images"], key=lambda item: int(item["id"])):
        image_id = int(image_record["id"])
        image_path = _safe_image(root, str(image_record["file_name"]))
        data = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ModelEvaluationError(f"cannot decode evaluation image id={image_id}")
        try:
            detections = detector.detect(image)
        except Exception as exc:
            raise ModelEvaluationError(
                f"detector inference failed for image id={image_id}"
            ) from exc
        if role is DetectorRole.VEHICLE:
            for detection in detections:
                class_name = detection.class_name.strip().lower()  # type: ignore[union-attr]
                category_id = categories.get(class_name)
                if category_id is None:
                    continue
                bbox = detection.bbox  # type: ignore[union-attr]
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [bbox.x1, bbox.y1, bbox.width, bbox.height],
                        "score": detection.confidence,  # type: ignore[union-attr]
                    }
                )
        else:
            category_id = categories.get("license_plate")
            if category_id is None:
                raise ModelEvaluationError("plate dataset lacks license_plate category")
            for detection in detections:
                bbox = detection.bbox  # type: ignore[union-attr]
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [bbox.x1, bbox.y1, bbox.width, bbox.height],
                        "score": detection.confidence,  # type: ignore[union-attr]
                    }
                )

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise ModelEvaluationError("detector prediction evidence already exists")
    predictions_bytes = (
        json.dumps(predictions, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    prediction_digest = hashlib.sha256(predictions_bytes).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "type": "DETECTOR_PREDICTIONS",
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "role": role.value,
        "split": split.value,
        "datasetExportId": dataset_manifest["exportId"],
        "datasetManifestSha256": dataset_sha256,
        "model": model_info,
        "imageCount": len(document["images"]),
        "predictionCount": len(predictions),
        "predictionsSha256": prediction_digest,
    }
    _write_new(output, predictions_bytes)
    try:
        _write_new(
            manifest_path,
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return DatasetInferenceResult(
        output,
        manifest_path,
        len(predictions),
        len(document["images"]),
        prediction_digest,
    )


def _safe_image(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ModelEvaluationError("COCO evaluation image path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ModelEvaluationError("COCO evaluation image is missing")
    return path


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
