from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_intelligence.exceptions import (
    DetectorCorpusError,
    DetectorDatasetError,
    ModelRegistryError,
)
from vehicle_intelligence.training.config import DataCorpusConfig, DetectorDatasetConfig
from vehicle_intelligence.training.corpus import (
    VietnamPlateCorpusBuilder,
    verify_plate_corpus,
)
from vehicle_intelligence.training.dataset import (
    DetectorDatasetBuilder,
    verify_detector_dataset,
)
from vehicle_intelligence.training.domain import DetectorRole, DetectorSample
from vehicle_intelligence.training.huggingface import HuggingFacePrivateRegistry


class _NoUploadApi:
    def create_repo(self, **kwargs):
        raise AssertionError(f"distribution-ineligible data reached Hub API: {kwargs}")


def test_plate_corpus_owns_canonical_ids_but_preserves_source_provenance(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "archive.zip")
    config = _config(tmp_path)
    result = VietnamPlateCorpusBuilder(
        config,
        expected_archive_sha256=_sha256_file(archive),
    ).build(archive)
    manifest, digest = verify_plate_corpus(result.directory)

    assert result.manifest_sha256 == digest
    assert result.sample_count == 5
    assert result.annotation_count == 5
    assert result.duplicate_images_merged == 1
    assert manifest["compilation"] == {
        "founderId": "duyhuynh",
        "ownerNamespace": "phins-group",
        "sampleIdScheme": "phins-vnplate-sha256-v1",
        "sourceOwnershipClaimed": False,
    }
    assert manifest["distributionEligible"] is False
    assert manifest["sources"][0]["license"] == "UNKNOWN"

    samples = [
        DetectorSample.model_validate_json(line)
        for line in (result.directory / "annotations.jsonl").read_bytes().splitlines()
    ]
    assert all(sample.sample_id.startswith("phins-vnplate-") for sample in samples)
    assert all(sample.attributes["corpusOwner"] == "phins-group" for sample in samples)
    assert all(sample.annotations[0].polygon for sample in samples)
    assert {sample.split for sample in samples} == {
        "train",
        "validation",
        "test",
    }
    duplicate = next(sample for sample in samples if sample.attributes["sourceOriginalCount"] == 2)
    assert duplicate.annotations[0].attributes["sourceConsensusCount"] == 2
    provenance = (result.directory / "PROVENANCE.jsonl").read_text()
    assert "images/train/Tgmt_0001.png" in provenance
    assert "images/val/Tgmt_0002.png" in provenance


def test_corpus_polygon_reaches_coco_but_unknown_license_blocks_hub_upload(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "archive.zip")
    corpus = VietnamPlateCorpusBuilder(
        _config(tmp_path),
        expected_archive_sha256=_sha256_file(archive),
    ).build(archive)
    dataset = DetectorDatasetBuilder(
        DetectorDatasetConfig(
            role=DetectorRole.PLATE,
            source_directory=corpus.directory,
            output_directory=tmp_path / "datasets",
            classes=("license_plate",),
        )
    ).build("phins-vn-plate-dataset-v1")
    manifest, _ = verify_detector_dataset(dataset.directory)
    documents = [
        json.loads((dataset.directory / f"annotations/{split}.json").read_text())
        for split in ("train", "validation", "test")
    ]

    assert manifest["source"]["ownerNamespace"] == "phins-group"
    assert manifest["releaseEligible"] is False
    assert manifest["distributionEligible"] is False
    assert all(
        "segmentation" in annotation for item in documents for annotation in item["annotations"]
    )
    with pytest.raises(ModelRegistryError, match="not distribution-eligible"):
        HuggingFacePrivateRegistry(api=_NoUploadApi()).upload_dataset(
            dataset.directory,
            "phins-group/plate-dataset",
        )


def test_plate_corpus_verifier_detects_tampering(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "archive.zip")
    result = VietnamPlateCorpusBuilder(
        _config(tmp_path),
        expected_archive_sha256=_sha256_file(archive),
    ).build(archive)
    attribution = result.directory / "ATTRIBUTION.csv"
    attribution.write_text("tampered", encoding="utf-8")

    with pytest.raises(DetectorCorpusError, match="checksum verification failed"):
        verify_plate_corpus(result.directory)


def test_dataset_builder_refuses_tampered_plate_corpus(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "archive.zip")
    corpus = VietnamPlateCorpusBuilder(
        _config(tmp_path),
        expected_archive_sha256=_sha256_file(archive),
    ).build(archive)
    (corpus.directory / "ATTRIBUTION.csv").write_text("tampered", encoding="utf-8")

    with pytest.raises(DetectorDatasetError, match="integrity verification failed"):
        DetectorDatasetBuilder(
            DetectorDatasetConfig(
                role=DetectorRole.PLATE,
                source_directory=corpus.directory,
                output_directory=tmp_path / "datasets",
                classes=("license_plate",),
            )
        ).build("phins-vn-plate-dataset-v1")


def test_plate_corpus_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("dataset.yaml", "nc: 2\nnames: [BSD, BSV]\n")
        zipped.writestr("../escape.png", b"unsafe")

    with pytest.raises(DetectorCorpusError, match="unsafe path"):
        VietnamPlateCorpusBuilder(
            _config(tmp_path),
            expected_archive_sha256=_sha256_file(archive),
        ).build(archive)


def _config(tmp_path: Path) -> DataCorpusConfig:
    return DataCorpusConfig(
        owner_namespace="phins-group",
        founder_id="duyhuynh",
        plate_corpus_id="phins-vn-plate-corpus-v1",
        plate_output_directory=tmp_path / "corpora",
    )


def _archive(path: Path) -> Path:
    groups = {
        "Dieu": ("val", 60),
        "Hung": ("train", 80),
        "Tgmt": ("train", 100),
        "carlong": ("train", 120),
        "greenpack": ("train", 140),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(
            "dataset.yaml",
            "train: images/train\nval: images/val\nnc: 2\nnames: [BSD, BSV]\n",
        )
        for group, (split, fill) in groups.items():
            image = _png(fill)
            name = f"{group}_0001"
            zipped.writestr(f"images/{split}/{name}.png", image)
            zipped.writestr(
                f"labels/{split}/{name}.txt",
                "1 0.20 0.30 0.70 0.30 0.70 0.60 0.20 0.60\n",
            )
        duplicate = _png(groups["Tgmt"][1])
        zipped.writestr("images/val/Tgmt_0002.png", duplicate)
        zipped.writestr(
            "labels/val/Tgmt_0002.txt",
            "1 0.21 0.31 0.71 0.31 0.71 0.61 0.21 0.61\n",
        )
    return path


def _png(fill: int) -> bytes:
    image = np.full((64, 96, 3), fill, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
