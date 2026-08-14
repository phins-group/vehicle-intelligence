from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request

from vehicle_intelligence.application.audit import AuditService
from vehicle_intelligence.application.model_training import (
    ModelTrainingService,
    StartModelTrainingCommand,
)
from vehicle_intelligence.application.security import DevelopmentAuthenticator
from vehicle_intelligence.config import AuthConfig, ModelTrainingRuntimeConfig
from vehicle_intelligence.domain.dataset_registry import (
    DatasetHubSyncJob,
    DatasetHubSyncStatus,
    DetectorDatasetExport,
    DetectorDatasetVersion,
)
from vehicle_intelligence.domain.model_training import (
    ModelTrainingRunStatus,
    RemoteTrainingJob,
    RemoteTrainingSubmission,
)
from vehicle_intelligence.infrastructure.persistence.audit_memory import (
    InMemoryAuditLogRepository,
)
from vehicle_intelligence.infrastructure.training.model_training_files import (
    FileModelTrainingRunRepository,
)
from vehicle_intelligence.interfaces.model_training_api import build_model_training_router
from vehicle_intelligence.interfaces.request_context import resolve_request_id
from vehicle_intelligence.interfaces.security import APISecurity
from vehicle_intelligence.training.config import HuggingFaceConfig, load_training_settings

_SOURCE_SHA = "a" * 64
_EXPORT_SHA = "b" * 64
_COMMIT_SHA = "c" * 40


class _ReadyDatasets:
    def __init__(self, dataset: DetectorDatasetVersion) -> None:
        self.dataset = dataset

    async def list_datasets(self) -> tuple[DetectorDatasetVersion, ...]:
        return (self.dataset,)


class _FakeGateway:
    def __init__(self) -> None:
        self.submissions: list[RemoteTrainingSubmission] = []
        self.stage = "SCHEDULING"
        self.canceled: list[str] = []

    def submit(self, request: RemoteTrainingSubmission) -> RemoteTrainingJob:
        self.submissions.append(request)
        return RemoteTrainingJob(
            id="job-123",
            status=self.stage,
            url="https://huggingface.co/jobs/job-123",
        )

    def inspect(self, job_id: str, namespace: str | None) -> RemoteTrainingJob:
        del namespace
        return RemoteTrainingJob(
            id=job_id,
            status=self.stage,
            url=f"https://huggingface.co/jobs/{job_id}",
            message="remote status",
        )

    def logs(self, job_id: str, namespace: str | None, tail: int) -> tuple[str, ...]:
        del job_id, namespace
        return ("epoch 1", "loss=0.4", "validation mAP=0.8")[-tail:]

    def cancel(self, job_id: str, namespace: str | None) -> None:
        del namespace
        self.canceled.append(job_id)


@pytest.mark.asyncio
async def test_training_run_pins_dataset_commit_and_uses_fixed_safe_command(
    tmp_path: Path,
) -> None:
    service, gateway = _service(tmp_path)
    await service.initialize()

    run = await service.prepare_run(
        _command(),
        DevelopmentAuthenticator().principal,
    )
    assert run.status is ModelTrainingRunStatus.QUEUED
    assert run.source_manifest_sha256 == _SOURCE_SHA
    assert run.export_manifest_sha256 == _EXPORT_SHA
    assert run.dataset_commit_sha == _COMMIT_SHA

    service.dispatch_run(run)
    submitted = await _wait_for_remote_job(service, run.id)

    assert submitted.status is ModelTrainingRunStatus.SCHEDULING
    assert gateway.submissions[0].dataset_repo_id == "phins-group/plate-dataset"
    assert gateway.submissions[0].dataset_revision == _COMMIT_SHA
    assert gateway.submissions[0].output_bucket == "phins-group/training-output"
    assert gateway.submissions[0].command == (
        "vehicle-model-training",
        "--config",
        "/workspace/configs/model-training.hf.yaml",
        "train",
        "--role",
        "plate",
        "/data",
        "--run-id",
        run.id,
        "--epochs",
        "12",
        "--batch-size",
        "4",
        "--workers",
        "2",
        "--snapshot-epoch",
        "3",
        "--output-directory",
        "/output/model-training",
    )

    gateway.stage = "RUNNING"
    running = await service.get_run(run.id)
    assert running.status is ModelTrainingRunStatus.RUNNING
    assert (await service.logs(run.id, 20)).lines[-1] == "validation mAP=0.8"

    canceled = await service.cancel(run.id)
    assert canceled.status is ModelTrainingRunStatus.CANCELED
    assert gateway.canceled == ["job-123"]
    await service.close()

    restarted = FileModelTrainingRunRepository(_runtime(tmp_path))
    await restarted.initialize()
    persisted = await restarted.get(run.id)
    assert persisted.status is ModelTrainingRunStatus.CANCELED
    assert persisted.dataset_commit_sha == _COMMIT_SHA
    await restarted.close()


@pytest.mark.asyncio
async def test_training_http_workflow_exposes_preflight_run_logs_and_cancel(
    tmp_path: Path,
) -> None:
    service, gateway = _service(tmp_path)
    audits = AuditService(InMemoryAuditLogRepository())
    security = APISecurity(AuthConfig(enabled=False), DevelopmentAuthenticator())
    await service.initialize()
    await audits.initialize()
    app = FastAPI()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = resolve_request_id(request)
        return await call_next(request)

    app.include_router(build_model_training_router(service, security, audits))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        overview = await client.get("/api/model-training")
        assert overview.status_code == 200
        assert overview.json()["capabilities"]["submissionsEnabled"] is True
        assert overview.json()["capabilities"]["defaults"]["architecture"] == "PicoDet"

        accepted = await client.post(
            "/api/model-training/runs",
            json={
                "sourceId": "phins-vn-plate",
                "modelName": "phins-vn-plate-detector",
                "modelVersion": "v1",
                "epochs": 12,
                "batchSize": 4,
                "workers": 2,
                "snapshotEpoch": 3,
                "confirmDatasetRights": True,
                "confirmComputeCost": True,
                "confirmRestrictedData": True,
            },
        )
        assert accepted.status_code == 202
        run_id = accepted.json()["id"]
        await _wait_for_remote_job(service, run_id)

        gateway.stage = "RUNNING"
        status_response = await client.get(f"/api/model-training/runs/{run_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "RUNNING"
        assert status_response.json()["datasetCommitSha"] == _COMMIT_SHA

        logs = await client.get(f"/api/model-training/runs/{run_id}/logs?tail=2")
        assert logs.status_code == 200
        assert logs.json()["lines"] == ["loss=0.4", "validation mAP=0.8"]

        canceled = await client.post(f"/api/model-training/runs/{run_id}/cancel")
        assert canceled.status_code == 200
        assert canceled.json()["status"] == "CANCELED"

    await service.close()
    await audits.close()


def test_current_runtime_reports_honest_remote_training_blockers(tmp_path: Path) -> None:
    settings = load_training_settings("configs/model-training.yaml")
    service = ModelTrainingService(
        FileModelTrainingRunRepository(_runtime(tmp_path)),
        _ReadyDatasets(_ready_dataset()),
        _FakeGateway(),
        _runtime(tmp_path),
        settings,
        credentials_configured=lambda: True,
    )

    capabilities = service.capabilities()

    assert capabilities.submissions_enabled is False
    assert capabilities.output_bucket_configured is True
    assert "HUGGING_FACE_JOBS_DISABLED" in capabilities.blockers
    assert "TRAINING_IMAGE_MISSING" in capabilities.blockers
    assert "OUTPUT_BUCKET_MISSING" not in capabilities.blockers


def _service(tmp_path: Path) -> tuple[ModelTrainingService, _FakeGateway]:
    settings = load_training_settings("configs/model-training.yaml")
    settings = settings.model_copy(
        update={
            "huggingface": HuggingFaceConfig(
                enabled=True,
                jobs_enabled=True,
                job_image="ghcr.io/phins/picodet-trainer@sha256:" + "d" * 64,
                job_flavor="a10g-small",
                job_output_bucket="phins-group/training-output",
            )
        }
    )
    gateway = _FakeGateway()
    service = ModelTrainingService(
        FileModelTrainingRunRepository(_runtime(tmp_path)),
        _ReadyDatasets(_ready_dataset()),
        gateway,
        _runtime(tmp_path),
        settings,
        credentials_configured=lambda: True,
    )
    return service, gateway


def _runtime(tmp_path: Path) -> ModelTrainingRuntimeConfig:
    return ModelTrainingRuntimeConfig(workspace_directory=tmp_path / "model-training")


def _ready_dataset() -> DetectorDatasetVersion:
    created = datetime(2026, 8, 12, 1, 0, tzinfo=UTC)
    sync = DatasetHubSyncJob(
        id="dataset-sync-" + "1" * 32,
        source_id="phins-vn-plate",
        source_manifest_sha256=_SOURCE_SHA,
        export_id="phins-vn-plate",
        repo_id="phins-group/plate-dataset",
        requested_revision="main",
        status=DatasetHubSyncStatus.COMPLETED,
        requested_by="admin",
        restricted_transfer_confirmed=True,
        created_at=created,
        updated_at=created,
        export_manifest_sha256=_EXPORT_SHA,
        hub_commit_sha=_COMMIT_SHA,
    )
    export = DetectorDatasetExport(
        export_id="phins-vn-plate",
        manifest_sha256=_EXPORT_SHA,
        created_at=created,
        sample_count=38_122,
        annotation_count=54_502,
        negative_sample_count=100,
        split_counts={"train": 26_685, "validation": 5718, "test": 5719},
        release_eligible=True,
        distribution_eligible=False,
        source_manifest_sha256=_SOURCE_SHA,
    )
    return DetectorDatasetVersion(
        source_id="phins-vn-plate",
        source_manifest_sha256=_SOURCE_SHA,
        created_at=created,
        sample_count=38_122,
        annotation_count=54_502,
        negative_sample_count=100,
        review_queue_count=0,
        release_eligible=True,
        distribution_eligible=False,
        privacy_classification="RESTRICTED",
        parent_source_id=None,
        export=export,
        latest_sync=sync,
    )


def _command() -> StartModelTrainingCommand:
    return StartModelTrainingCommand(
        source_id="phins-vn-plate",
        model_name="phins-vn-plate-detector",
        model_version="v1",
        epochs=12,
        batch_size=4,
        workers=2,
        snapshot_epoch=3,
        dataset_rights_confirmed=True,
        compute_cost_confirmed=True,
        restricted_data_confirmed=True,
    )


async def _wait_for_remote_job(
    service: ModelTrainingService,
    run_id: str,
):
    for _ in range(100):
        run = await service.get_run(run_id, refresh=False)
        if run.remote_job_id or not run.status.active:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError("remote training submission did not finish")
