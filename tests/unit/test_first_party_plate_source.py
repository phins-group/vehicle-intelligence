from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vehicle_intelligence.exceptions import DetectorDatasetError, ModelRegistryError
from vehicle_intelligence.training.config import DetectorDatasetConfig, SplitConfig
from vehicle_intelligence.training.dataset import DetectorDatasetBuilder, verify_detector_dataset
from vehicle_intelligence.training.domain import DetectorRole
from vehicle_intelligence.training.first_party import (
    FirstPartyPlateSourceBuilder,
    verify_first_party_detector_source,
)
from vehicle_intelligence.training.huggingface import HuggingFacePrivateRegistry


class _FakeHubApi:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None
        self.uploaded: dict[str, object] | None = None

    def create_repo(self, **kwargs):
        self.created = kwargs

    def repo_info(self, **_kwargs):
        return SimpleNamespace(private=True)

    def upload_folder(self, **kwargs):
        self.uploaded = kwargs
        return SimpleNamespace(oid="restricted-commit", commit_url="https://example.invalid")


def test_builds_release_eligible_source_from_exact_labels_and_isolates_review(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels"
    label_images = labels / "images"
    label_images.mkdir(parents=True)
    labeled_bytes = _jpg(80, plate=True)
    (label_images / "labeled.jpg").write_bytes(labeled_bytes)
    (labels / "annotations.jsonl").write_text(
        json.dumps(
            {
                "sampleId": "reference-sample",
                "imagePath": "images/labeled.jpg",
                "groupId": "reference-sequence-1",
                "cameraId": "reference-camera",
                "capturedAt": "2026-08-01T00:00:00Z",
                "attributes": {"acceptanceEligible": False},
                "annotations": [
                    {
                        "className": "license_plate",
                        "bbox": {"x": 20, "y": 25, "width": 70, "height": 25},
                    }
                ],
            }
        )
        + "\n"
    )

    auto = tmp_path / "auto"
    auto_images = auto / "images"
    auto_images.mkdir(parents=True)
    auto_bytes = _jpg(110, plate=True)
    (auto_images / "auto.jpg").write_bytes(auto_bytes)
    (auto / "annotations.auto.jsonl").write_text(
        json.dumps(
            {
                "sampleId": "auto-sample",
                "imagePath": "images/auto.jpg",
                "groupId": "auto-sequence",
                "cameraId": "auto-camera",
                "capturedAt": "2026-08-02T00:00:00Z",
                "attributes": {"reviewStatus": "PENDING_REVIEW"},
                "annotations": [
                    {
                        "className": "license_plate",
                        "bbox": {"x": 20, "y": 25, "width": 70, "height": 25},
                    }
                ],
            }
        )
        + "\n"
    )

    source = tmp_path / "collected"
    source.mkdir()
    (source / "a-labeled.jpg").write_bytes(labeled_bytes)
    (source / "b-labeled-duplicate.jpg").write_bytes(labeled_bytes)
    (source / "c-auto.jpg").write_bytes(auto_bytes)
    (source / "d-unmatched.jpg").write_bytes(_jpg(140, plate=False))
    (source / "notes.txt").write_text("operator note")

    output = tmp_path / "first-party-v1"
    result = FirstPartyPlateSourceBuilder(
        input_directory=source,
        output_directory=output,
        label_reference_directory=labels,
        auto_reference_directory=auto,
        source_id="phins-first-party-test-v1",
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()

    assert result.sample_count == 1
    assert result.annotation_count == 1
    assert result.review_queue_count == 2
    assert result.exact_duplicate_files_excluded == 1
    assert result.unsupported_file_count == 1
    manifest, digest = verify_first_party_detector_source(output)
    assert digest == result.manifest_sha256
    assert manifest["releaseEligible"] is True
    assert manifest["distributionEligible"] is False
    assert manifest["statistics"]["autoLabeledPendingReview"] == 1
    assert manifest["statistics"]["unlabeledPendingReview"] == 1

    record = json.loads((output / "annotations.jsonl").read_text())
    assert record["sampleId"].startswith("phins-first-party-plate-")
    assert record["groupId"].startswith("phins-group:first-party-plate:")
    assert record["attributes"]["sourceOwner"] == "duyhuynh"
    assert record["attributes"]["acceptanceEligible"] is True
    review = [json.loads(line) for line in (output / "REVIEW_QUEUE.jsonl").read_text().splitlines()]
    assert {item["reason"] for item in review} == {
        "MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW",
        "MISSING_VERIFIED_ANNOTATION",
    }

    dataset = DetectorDatasetBuilder(
        DetectorDatasetConfig(
            role=DetectorRole.PLATE,
            source_directory=output,
            output_directory=tmp_path / "datasets",
            classes=("license_plate",),
            split=SplitConfig(require_non_empty=False),
        )
    ).build("first-party-plate-v1")
    dataset_manifest, _ = verify_detector_dataset(dataset.directory)
    assert dataset_manifest["acceptanceEligible"] is True
    assert dataset_manifest["releaseEligible"] is True
    assert dataset_manifest["distributionEligible"] is False
    assert dataset_manifest["licenseStatus"] == "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED"

    api = _FakeHubApi()
    registry = HuggingFacePrivateRegistry(api=api)
    with pytest.raises(ModelRegistryError, match="not distribution-eligible"):
        registry.upload_dataset(dataset.directory, "phins-group/plate-private")
    uploaded = registry.upload_dataset(
        dataset.directory,
        "phins-group/plate-private",
        allow_restricted_private=True,
    )
    assert api.created is not None and api.created["private"] is True
    assert api.uploaded is not None and api.uploaded["repo_type"] == "dataset"
    assert uploaded.revision == "restricted-commit"


def test_verifier_detects_first_party_source_tampering(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    (labels / "images").mkdir(parents=True)
    data = _jpg(90, plate=True)
    (labels / "images/image.jpg").write_bytes(data)
    (labels / "annotations.jsonl").write_text(
        json.dumps(
            {
                "sampleId": "sample",
                "imagePath": "images/image.jpg",
                "groupId": "group",
                "cameraId": "camera",
                "capturedAt": "2026-08-01T00:00:00Z",
                "annotations": [],
            }
        )
        + "\n"
    )
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "image.jpg").write_bytes(data)
    output = tmp_path / "source"
    FirstPartyPlateSourceBuilder(
        input_directory=collected,
        output_directory=output,
        label_reference_directory=labels,
        source_id="source-v1",
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()

    image = next((output / "images").rglob("*.jpg"))
    image.write_bytes(b"tampered")
    with pytest.raises(DetectorDatasetError, match="size verification|checksum"):
        verify_first_party_detector_source(output)


def _jpg(fill: int, *, plate: bool) -> bytes:
    image = np.full((100, 140, 3), fill, dtype=np.uint8)
    if plate:
        cv2.rectangle(image, (20, 25), (90, 50), (245, 245, 245), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
