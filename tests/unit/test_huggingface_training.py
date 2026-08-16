from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vehicle_intelligence.infrastructure.training.huggingface_jobs import (
    HuggingFaceTrainingJobGateway,
)
from vehicle_intelligence.training.config import HuggingFaceConfig
from vehicle_intelligence.training.domain import DetectorRole
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


def test_huggingface_replaces_remote_with_attested_video_dataset(tmp_path) -> None:
    dataset, _ = build_detector_dataset(tmp_path, DetectorRole.PLATE)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "distributionEligible": False,
            "releaseEligible": True,
            "source": {
                "type": "FIRST_PARTY_SOURCE",
                "rightsAssertion": "USER_CONFIRMED_FIRST_PARTY_VIDEO_COLLECTION",
                "licenseReviewStatus": "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED",
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    api = _FakeApi()

    result = HuggingFacePrivateRegistry(api=api).upload_dataset(
        dataset,
        "phins-group/plate-dataset",
        allow_restricted_private=True,
        replace_remote=True,
    )

    assert api.uploaded["delete_patterns"] == "*"
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
        dataset_revision="a" * 40,
        output_bucket="company/warehouse-training-output",
        secrets={"HF_TOKEN": "not-logged"},
    )

    assert captured["volumes"][0]["read_only"] is True
    assert captured["volumes"][0]["mount_path"] == "/data"
    assert captured["volumes"][0]["revision"] == "a" * 40
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


def test_huggingface_training_logs_redact_credentials_and_signed_urls() -> None:
    configured_token = "hf_configuredSecret123456789"
    bearer = "AlphabeticBearerCredential"
    api_key = "api-secret-123456789"
    password = "correct horse battery staple"
    long_password = "x" * 9000
    signed_url = (
        "https://storage.example/train.bin?X-Amz-Credential=AKIAEXAMPLE%2Fscope"
        "&X-Amz-Signature=deadbeef1234567890"
    )
    gateway = HuggingFaceTrainingJobGateway(
        token=configured_token,
        inspect_job=lambda **_kwargs: None,
        fetch_job_logs=lambda **_kwargs: (
            f"configured={configured_token}",
            f"Authorization: Bearer {bearer}",
            f'api_key="{api_key}" password="{password}"',
            f"password={long_password}",
            f"artifact={signed_url}",
            "epoch=2 loss=0.125",
        ),
        cancel_job=lambda **_kwargs: None,
    )

    lines = gateway.logs("job-123", "phins", 10)
    rendered = "\n".join(lines)

    for secret in (configured_token, bearer, api_key, password, "deadbeef1234567890"):
        assert secret not in rendered
    assert "x" * 100 not in rendered
    assert "artifact=https://storage.example/train.bin?[REDACTED]" in rendered
    assert lines[-1] == "epoch=2 loss=0.125"
    assert rendered.count("[REDACTED]") >= 6
    assert all(len(line) <= 8000 for line in lines)


def test_huggingface_training_logs_redact_before_line_truncation() -> None:
    configured_token = "hf_boundarySecret123456789"
    gateway = HuggingFaceTrainingJobGateway(
        token=configured_token,
        inspect_job=lambda **_kwargs: None,
        fetch_job_logs=lambda **_kwargs: ("x" * 7985 + configured_token,),
        cancel_job=lambda **_kwargs: None,
    )

    (line,) = gateway.logs("job-123", None, 10)

    assert configured_token not in line
    assert line.endswith("[REDACTED]")


def test_huggingface_training_logs_do_not_over_redact_normal_diagnostics() -> None:
    expected = (
        "Tokenizer loaded; token count: 2048",
        "password policy check passed",
        "api_key_count=3",
        "metrics=https://example.com/report?epoch=2&loss=0.125",
    )
    gateway = HuggingFaceTrainingJobGateway(
        token="hf_unrelatedConfiguredSecret123",
        inspect_job=lambda **_kwargs: None,
        fetch_job_logs=lambda **_kwargs: expected,
        cancel_job=lambda **_kwargs: None,
    )

    assert gateway.logs("job-123", None, 10) == expected


def test_huggingface_training_status_message_is_redacted_before_persistence() -> None:
    configured_token = "hf_statusSecret123456789"
    job = SimpleNamespace(
        id="job-123",
        url="https://huggingface.co/jobs/job-123?token=url-secret-123",
        status=SimpleNamespace(
            stage="FAILED",
            message=f"remote error token={configured_token}",
        ),
        started_at=None,
        finished_at=None,
    )
    gateway = HuggingFaceTrainingJobGateway(
        token=configured_token,
        inspect_job=lambda **_kwargs: job,
        fetch_job_logs=lambda **_kwargs: (),
        cancel_job=lambda **_kwargs: None,
    )

    inspected = gateway.inspect("job-123", None)

    assert inspected.message == "remote error token=[REDACTED]"
    assert configured_token not in inspected.message
    assert inspected.url == "https://huggingface.co/jobs/job-123"
