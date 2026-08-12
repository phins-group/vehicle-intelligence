from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vehicle_intelligence.training.config import HuggingFaceConfig
from vehicle_intelligence.training.huggingface import (
    HuggingFaceJobRunner,
    HuggingFacePrivateRegistry,
)

from .training_fixtures import build_detector_dataset


class _FakeApi:
    def __init__(self) -> None:
        self.created = None
        self.uploaded = None

    def create_repo(self, **kwargs):
        self.created = kwargs

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=True)

    def upload_folder(self, **kwargs):
        self.uploaded = kwargs
        return SimpleNamespace(oid="commit123", commit_url="https://example.invalid/commit123")


def test_huggingface_dataset_upload_is_forced_private_and_verified(tmp_path) -> None:
    dataset, _ = build_detector_dataset(tmp_path)
    api = _FakeApi()

    result = HuggingFacePrivateRegistry(api=api).upload_dataset(
        dataset,
        "company/warehouse-vehicle-dataset",
    )

    assert api.created["private"] is True
    assert api.created["repo_type"] == "dataset"
    assert api.uploaded["folder_path"] == str(dataset.resolve())
    assert (dataset / "README.md").read_text().startswith("---\nlicense: other\n")
    assert result.revision == "commit123"


def test_huggingface_job_mounts_private_dataset_read_only(tmp_path) -> None:
    captured = {}

    def volume_factory(**kwargs):
        captured.setdefault("volumes", []).append(kwargs)
        return kwargs

    def run_job(**kwargs):
        captured["job"] = kwargs
        return SimpleNamespace(
            id="job123",
            url="https://example.invalid/job123",
            status=SimpleNamespace(stage="RUNNING"),
        )

    result = HuggingFaceJobRunner(
        run_job=run_job,
        volume_factory=volume_factory,
    ).submit(
        image="company/paddledetection:2.9",
        command=["python", "train.py"],
        flavor="a10g-small",
        dataset_repo="company/warehouse-vehicle-dataset",
        output_bucket="company/warehouse-training-output",
        secrets={"HF_TOKEN": "not-logged"},
    )

    assert captured["volumes"][0]["read_only"] is True
    assert captured["volumes"][0]["mount_path"] == "/data"
    assert captured["volumes"][1]["read_only"] is False
    assert captured["volumes"][1]["mount_path"] == "/output"
    assert captured["job"]["volumes"] == captured["volumes"]
    assert result.job_id == "job123"
    assert result.status == "RUNNING"


def test_huggingface_upload_can_be_enabled_without_jobs() -> None:
    config = HuggingFaceConfig(enabled=True)

    assert config.enabled is True
    assert config.jobs_enabled is False


def test_huggingface_jobs_require_image_and_persistent_output() -> None:
    with pytest.raises(ValidationError, match="job_image and a persistent output bucket"):
        HuggingFaceConfig(enabled=True, jobs_enabled=True)
