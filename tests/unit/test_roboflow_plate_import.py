from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_intelligence.exceptions import DetectorCorpusError
from vehicle_intelligence.training.domain import DetectorSample
from vehicle_intelligence.training.roboflow import (
    RoboflowArchiveSpec,
    RoboflowPlateArchiveImporter,
    verify_roboflow_source,
)


def test_detection_archive_maps_classes_and_groups_augmentations(tmp_path: Path) -> None:
    archive = _detection_archive(tmp_path / "detection.zip")
    spec = _detection_spec(archive)
    result = _importer(spec, tmp_path).build(archive)
    manifest, digest = verify_roboflow_source(result.directory)
    samples = [
        DetectorSample.model_validate_json(line)
        for line in (result.directory / "annotations.jsonl").read_bytes().splitlines()
    ]

    assert result.manifest_sha256 == digest
    assert manifest["compilation"]["founderId"] == "duyhuynh"
    assert manifest["statistics"]["classMapping"] == {
        "0": "license_plate",
        "1": "license_plate",
    }
    assert len(samples) == 2
    assert len({sample.group_id for sample in samples}) == 1
    assert all(sample.split is None for sample in samples)
    assert {annotation.class_name for sample in samples for annotation in sample.annotations} == {
        "license_plate"
    }


def test_folder_archive_is_auxiliary_and_never_detector_negative_data(
    tmp_path: Path,
) -> None:
    archive = _classification_archive(tmp_path / "classification.zip")
    spec = _classification_spec(archive)
    result = _importer(spec, tmp_path).build(archive)
    manifest, _ = verify_roboflow_source(result.directory)
    samples = [
        json.loads(line)
        for line in (result.directory / "classification-samples.jsonl").read_bytes().splitlines()
    ]

    assert result.task == "classification"
    assert result.source_image_count == 3
    assert result.canonical_image_count == 2
    assert result.duplicate_images_merged == 1
    assert manifest["type"] == "AUXILIARY_CLASSIFICATION_SOURCE"
    assert manifest["statistics"]["detectorEligible"] is False
    assert not (result.directory / "annotations.jsonl").exists()
    assert {sample["label"] for sample in samples} == {
        "car_long_plate_context",
        "motorcycle_plate_context",
    }


def test_roboflow_import_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.jpg", _jpg(10))
    spec = _classification_spec(archive, expected_images=1)

    with pytest.raises(DetectorCorpusError, match="unsafe path"):
        _importer(spec, tmp_path).build(archive)


def test_roboflow_source_verifier_detects_tampering(tmp_path: Path) -> None:
    archive = _detection_archive(tmp_path / "detection.zip")
    result = _importer(_detection_spec(archive), tmp_path).build(archive)
    attribution = result.directory / "ATTRIBUTION.csv"
    attribution.write_text("tampered", encoding="utf-8")

    with pytest.raises(DetectorCorpusError, match="checksum verification failed"):
        verify_roboflow_source(result.directory)


def _importer(spec: RoboflowArchiveSpec, tmp_path: Path) -> RoboflowPlateArchiveImporter:
    return RoboflowPlateArchiveImporter(
        spec,
        owner_namespace="phins-group",
        founder_id="duyhuynh",
        detection_output_root=tmp_path / "detection-sources",
        auxiliary_output_root=tmp_path / "auxiliary-sources",
    )


def _detection_spec(path: Path) -> RoboflowArchiveSpec:
    return RoboflowArchiveSpec(
        source_id="roboflow-test-detection-v1",
        title="test detection",
        author="test author",
        dataset_url="https://universe.roboflow.com/test-workspace/test-project/dataset/1",
        workspace="test-workspace",
        project="test-project",
        version=1,
        archive_sha256=_sha256(path),
        exported_at=datetime(2026, 1, 1, tzinfo=UTC),
        expected_images=2,
        task="detection",
        class_names=("0", "plate"),
        class_mapping={0: "license_plate", 1: "license_plate"},
        augmented=True,
    )


def _classification_spec(
    path: Path,
    *,
    expected_images: int = 3,
) -> RoboflowArchiveSpec:
    return RoboflowArchiveSpec(
        source_id="roboflow-test-classification-v1",
        title="test classification",
        author="test author",
        dataset_url="https://universe.roboflow.com/test-workspace/test-project",
        workspace="test-workspace",
        project="test-project",
        version=1,
        archive_sha256=_sha256(path),
        exported_at=datetime(2026, 1, 1, tzinfo=UTC),
        expected_images=expected_images,
        task="classification",
        class_names=("Moto", "Car"),
        class_mapping={},
        classification_mapping={
            "Moto": "motorcycle_plate_context",
            "Car": "car_long_plate_context",
        },
        augmented=True,
    )


def _detection_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        _metadata(zipped, image_count=2, detection=True)
        for split, suffix, class_id, fill in (
            ("train", "a" * 32, 0, 40),
            ("valid", "b" * 32, 1, 80),
        ):
            stem = f"plate001_jpg.rf.{suffix}"
            zipped.writestr(f"{split}/images/{stem}.jpg", _jpg(fill))
            zipped.writestr(f"{split}/labels/{stem}.txt", f"{class_id} 0.5 0.5 0.4 0.2\n")
    return path


def _classification_archive(path: Path) -> Path:
    duplicate = _jpg(40)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        _metadata(zipped, image_count=3, detection=False)
        zipped.writestr(f"train/Moto/moto1_jpg.rf.{'a' * 32}.jpg", duplicate)
        zipped.writestr(f"train/Moto/moto1_jpg.rf.{'b' * 32}.jpg", duplicate)
        zipped.writestr(f"valid/Car/car1_jpg.rf.{'c' * 32}.jpg", _jpg(80))
    return path


def _metadata(
    zipped: zipfile.ZipFile,
    *,
    image_count: int,
    detection: bool,
) -> None:
    zipped.writestr(
        "README.dataset.txt",
        "https://universe.roboflow.com/test-workspace/test-project\nLicense: CC BY 4.0\n",
    )
    zipped.writestr(
        "README.roboflow.txt",
        f"The dataset includes {image_count} images.\n",
    )
    if detection:
        zipped.writestr(
            "data.yaml",
            "nc: 2\n"
            "names: ['0', 'plate']\n"
            "roboflow:\n"
            "  workspace: test-workspace\n"
            "  project: test-project\n"
            "  version: 1\n"
            "  license: CC BY 4.0\n",
        )


def _jpg(fill: int) -> bytes:
    image = np.full((64, 96, 3), fill, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
