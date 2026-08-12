from __future__ import annotations

import json

import pytest

from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.config import DetectorDatasetConfig
from vehicle_intelligence.training.dataset import (
    DetectorDatasetBuilder,
    verify_detector_dataset,
)
from vehicle_intelligence.training.domain import DetectorRole

from .training_fixtures import build_detector_dataset


def test_grouped_coco_dataset_is_immutable_and_has_no_split_leakage(tmp_path) -> None:
    directory, _ = build_detector_dataset(tmp_path)

    manifest, digest = verify_detector_dataset(directory)
    train = json.loads((directory / "annotations/train.json").read_text())
    validation = json.loads((directory / "annotations/validation.json").read_text())
    test = json.loads((directory / "annotations/test.json").read_text())

    assert len(digest) == 64
    assert manifest["role"] == "vehicle"
    assert manifest["sampleCount"] == 4
    assert manifest["splitCounts"] == {"test": 1, "train": 2, "validation": 1}
    assert {image["group_id"] for image in train["images"]} == {"group-train"}
    assert {image["group_id"] for image in validation["images"]} == {"group-validation"}
    assert {image["group_id"] for image in test["images"]} == {"group-test"}
    assert train["categories"][0]["name"] == "car"
    assert manifest["licenseStatus"] == "UNSPECIFIED"
    assert (directory / "ATTRIBUTION.csv").is_file()
    assert "license: other" in (directory / "README.md").read_text()

    image_path = directory / train["images"][0]["file_name"]
    image_path.write_bytes(b"tampered")
    with pytest.raises(DetectorDatasetError, match="verification failed"):
        verify_detector_dataset(directory)


def test_plate_dataset_requires_one_canonical_class(tmp_path) -> None:
    with pytest.raises(ValueError, match="license_plate"):
        DetectorDatasetConfig(
            role=DetectorRole.PLATE,
            source_directory=tmp_path,
            output_directory=tmp_path / "output",
            classes=("top_line", "bottom_line"),
        )


def test_one_group_cannot_declare_multiple_explicit_splits(tmp_path) -> None:
    directory, config = build_detector_dataset(tmp_path)
    assert directory.exists()
    source_annotations = config.source_directory / "annotations.jsonl"
    records = [json.loads(line) for line in source_annotations.read_text().splitlines()]
    records[1]["split"] = "test"
    source_annotations.write_text("".join(json.dumps(item) + "\n" for item in records))
    with pytest.raises(DetectorDatasetError, match="multiple explicit splits"):
        DetectorDatasetBuilder(config).build("vehicle-v2")
