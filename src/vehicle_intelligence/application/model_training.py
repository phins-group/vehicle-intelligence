"""Auditable model-training orchestration independent of the remote Job provider."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Protocol

from pydantic import ValidationError

from vehicle_intelligence.application.dataset_registry import DatasetRegistryService
from vehicle_intelligence.config import ModelTrainingRuntimeConfig
from vehicle_intelligence.domain import Principal
from vehicle_intelligence.domain.dataset_registry import DatasetHubSyncStatus
from vehicle_intelligence.domain.model_training import (
    ModelTrainingCapabilities,
    ModelTrainingDefaults,
    ModelTrainingLog,
    ModelTrainingParameters,
    ModelTrainingRun,
    ModelTrainingRunStatus,
    RemoteTrainingJob,
    RemoteTrainingSubmission,
)
from vehicle_intelligence.exceptions import (
    ModelTrainingConflictError,
    ModelTrainingValidationError,
)
from vehicle_intelligence.training.config import (
    ModelTrainingSettings,
    PaddleDetectionConfig,
)
from vehicle_intelligence.training.domain import DetectorRole

logger = logging.getLogger(__name__)

_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ERROR_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class StartModelTrainingCommand:
    source_id: str
    model_name: str
    model_version: str
    epochs: int
    batch_size: int
    workers: int
    snapshot_epoch: int
    dataset_rights_confirmed: bool
    compute_cost_confirmed: bool
    restricted_data_confirmed: bool = False


class ModelTrainingRunRepository(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def create(self, run: ModelTrainingRun) -> None: ...

    async def save(self, run: ModelTrainingRun, expected_updated_at: datetime) -> None: ...

    async def get(self, run_id: str) -> ModelTrainingRun: ...

    async def list(self) -> tuple[ModelTrainingRun, ...]: ...


class RemoteModelTrainingGateway(Protocol):
    def submit(self, request: RemoteTrainingSubmission) -> RemoteTrainingJob: ...

    def inspect(self, job_id: str, namespace: str | None) -> RemoteTrainingJob: ...

    def logs(
        self,
        job_id: str,
        namespace: str | None,
        tail: int,
    ) -> tuple[str, ...]: ...

    def cancel(self, job_id: str, namespace: str | None) -> None: ...


class ModelTrainingService:
    """Bind one reviewed/synced dataset revision to one immutable remote run."""

    def __init__(
        self,
        repository: ModelTrainingRunRepository,
        datasets: DatasetRegistryService,
        gateway: RemoteModelTrainingGateway,
        runtime: ModelTrainingRuntimeConfig,
        settings: ModelTrainingSettings,
        *,
        credentials_configured: Callable[[], bool] = lambda: bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._datasets = datasets
        self._gateway = gateway
        self._runtime = runtime
        self._settings = settings
        self._credentials_configured = credentials_configured
        self._clock = clock
        self._tasks: set[asyncio.Task[None]] = set()

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._repository.close()

    def capabilities(self) -> ModelTrainingCapabilities:
        target = self._settings.target(DetectorRole.PLATE)
        hub = self._settings.huggingface
        blockers: list[str] = []
        if not self._runtime.enabled:
            blockers.append("MODEL_TRAINING_DISABLED")
        if not hub.enabled:
            blockers.append("HUGGING_FACE_DISABLED")
        if not hub.jobs_enabled:
            blockers.append("HUGGING_FACE_JOBS_DISABLED")
        if not self._credentials_configured():
            blockers.append("HUGGING_FACE_CREDENTIALS_MISSING")
        if not hub.job_image:
            blockers.append("TRAINING_IMAGE_MISSING")
        if not hub.job_output_bucket:
            blockers.append("OUTPUT_BUCKET_MISSING")
        defaults = ModelTrainingDefaults(
            role=DetectorRole.PLATE,
            architecture="PicoDet",
            base_config=str(target.paddledetection.base_config),
            model_repo_id=target.hub.model_repo,
            dataset_repo_id=target.hub.dataset_repo,
            epochs=target.paddledetection.epochs,
            batch_size=target.paddledetection.batch_size,
            workers=target.paddledetection.workers,
            snapshot_epoch=target.paddledetection.snapshot_epoch,
            timeout_seconds=target.paddledetection.maximum_runtime_seconds,
            hardware_flavor=hub.job_flavor,
        )
        return ModelTrainingCapabilities(
            enabled=self._runtime.enabled,
            jobs_enabled=hub.jobs_enabled,
            credentials_configured=self._credentials_configured(),
            image_configured=bool(hub.job_image),
            output_bucket_configured=bool(hub.job_output_bucket),
            submissions_enabled=not blockers,
            job_image=hub.job_image or None,
            output_bucket=hub.job_output_bucket,
            namespace=hub.job_namespace,
            defaults=defaults,
            blockers=tuple(blockers),
        )

    async def list_runs(self) -> tuple[ModelTrainingRun, ...]:
        return await self._repository.list()

    async def prepare_run(
        self,
        command: StartModelTrainingCommand,
        principal: Principal,
    ) -> ModelTrainingRun:
        capabilities = self.capabilities()
        if not capabilities.submissions_enabled:
            raise ModelTrainingValidationError(
                "remote model training is blocked by runtime preflight"
            )
        if not command.dataset_rights_confirmed:
            raise ModelTrainingValidationError(
                "dataset processing rights must be explicitly confirmed"
            )
        if not command.compute_cost_confirmed:
            raise ModelTrainingValidationError(
                "remote compute billing must be explicitly confirmed"
            )
        source_id = _model_identifier(command.source_id, "dataset source id")
        model_name = _model_identifier(command.model_name, "model name")
        model_version = _model_identifier(command.model_version, "model version")
        effective = _effective_training_config(command, self._settings)
        datasets = await self._datasets.list_datasets()
        dataset = next((item for item in datasets if item.source_id == source_id), None)
        if dataset is None:
            raise ModelTrainingValidationError("selected dataset source does not exist")
        if dataset.review_queue_count or not dataset.release_eligible:
            raise ModelTrainingValidationError(
                "selected dataset must be fully reviewed and release-eligible"
            )
        if not dataset.distribution_eligible and not command.restricted_data_confirmed:
            raise ModelTrainingValidationError(
                "restricted first-party data processing must be explicitly confirmed"
            )
        export = dataset.export
        sync = dataset.latest_sync
        if export is None:
            raise ModelTrainingValidationError("selected dataset has no verified COCO export")
        if sync is None or sync.status is not DatasetHubSyncStatus.COMPLETED:
            raise ModelTrainingValidationError(
                "selected dataset has not completed private Hugging Face synchronization"
            )
        target = self._settings.target(DetectorRole.PLATE)
        if (
            export.source_manifest_sha256 != dataset.source_manifest_sha256
            or sync.source_manifest_sha256 != dataset.source_manifest_sha256
            or sync.export_manifest_sha256 != export.manifest_sha256
            or sync.repo_id != target.hub.dataset_repo
            or not sync.hub_commit_sha
        ):
            raise ModelTrainingValidationError(
                "dataset export and private Hub commit do not form one verified revision"
            )
        now = _now(self._clock)
        run_id = f"model-training-{uuid.uuid4().hex}"
        output_root = PurePosixPath(self._runtime.container_output_directory)
        output_path = str(output_root / run_id)
        run = ModelTrainingRun(
            id=run_id,
            role=DetectorRole.PLATE,
            status=ModelTrainingRunStatus.QUEUED,
            source_id=dataset.source_id,
            source_manifest_sha256=dataset.source_manifest_sha256,
            export_id=export.export_id,
            export_manifest_sha256=export.manifest_sha256,
            dataset_repo_id=sync.repo_id,
            dataset_revision=sync.requested_revision,
            dataset_commit_sha=sync.hub_commit_sha,
            model_repo_id=target.hub.model_repo,
            model_name=model_name,
            model_version=model_version,
            architecture="PicoDet",
            parameters=ModelTrainingParameters(
                epochs=effective.epochs,
                batch_size=effective.batch_size,
                workers=effective.workers,
                snapshot_epoch=effective.snapshot_epoch,
                timeout_seconds=effective.maximum_runtime_seconds,
                hardware_flavor=self._settings.huggingface.job_flavor,
            ),
            requested_by=principal.id,
            dataset_rights_confirmed=True,
            compute_cost_confirmed=True,
            restricted_data_confirmed=command.restricted_data_confirmed,
            created_at=now,
            updated_at=now,
            output_bucket=str(self._settings.huggingface.job_output_bucket),
            output_path=output_path,
        )
        await self._repository.create(run)
        return run

    def dispatch_run(self, run: ModelTrainingRun) -> None:
        if run.status is not ModelTrainingRunStatus.QUEUED:
            return
        name = f"submit-{run.id}"
        if any(task.get_name() == name for task in self._tasks):
            return
        task = asyncio.create_task(self._submit(run.id), name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def fail_prepared_run(self, run_id: str, error_code: str) -> None:
        if not _ERROR_CODE.fullmatch(error_code):
            raise ModelTrainingValidationError("model training error code is invalid")
        current = await self._repository.get(run_id)
        if current.status is not ModelTrainingRunStatus.QUEUED:
            return
        failed = replace(
            current,
            status=ModelTrainingRunStatus.FAILED,
            updated_at=_now(self._clock),
            finished_at=_now(self._clock),
            error_code=error_code,
        )
        await self._repository.save(failed, current.updated_at)

    async def get_run(self, run_id: str, *, refresh: bool = True) -> ModelTrainingRun:
        run = await self._repository.get(run_id)
        if refresh and run.remote_job_id and run.status.active:
            return await self._refresh(run)
        return run

    async def logs(self, run_id: str, tail: int) -> ModelTrainingLog:
        run = await self._repository.get(run_id)
        maximum = self._runtime.maximum_log_lines
        if not 1 <= tail <= maximum:
            raise ModelTrainingValidationError(f"training log tail must be between 1 and {maximum}")
        if not run.remote_job_id:
            return ModelTrainingLog(run_id=run.id, lines=(), available=False)
        lines = await asyncio.to_thread(
            self._gateway.logs,
            run.remote_job_id,
            self._settings.huggingface.job_namespace,
            tail,
        )
        return ModelTrainingLog(run_id=run.id, lines=lines[-tail:], available=True)

    async def cancel(self, run_id: str) -> ModelTrainingRun:
        run = await self._repository.get(run_id)
        if not run.status.active:
            raise ModelTrainingConflictError("only an active training run can be canceled")
        if run.status is ModelTrainingRunStatus.QUEUED:
            return await self._save_status(run, ModelTrainingRunStatus.CANCELED)
        if not run.remote_job_id:
            raise ModelTrainingConflictError(
                "training submission is in progress; retry cancellation after a status refresh"
            )
        await asyncio.to_thread(
            self._gateway.cancel,
            run.remote_job_id,
            self._settings.huggingface.job_namespace,
        )
        return await self._save_status(run, ModelTrainingRunStatus.CANCELED)

    async def _submit(self, run_id: str) -> None:
        current = await self._repository.get(run_id)
        if current.status is not ModelTrainingRunStatus.QUEUED:
            return
        try:
            submitting = replace(
                current,
                status=ModelTrainingRunStatus.SUBMITTING,
                updated_at=_now(self._clock),
                error_code=None,
            )
            await self._repository.save(submitting, current.updated_at)
            request = self._submission(submitting)
            remote = await asyncio.to_thread(self._gateway.submit, request)
            mapped = _remote_status(remote.status)
            submitted = replace(
                submitting,
                status=mapped,
                remote_job_id=remote.id,
                remote_job_url=remote.url,
                remote_message=_message(remote.message),
                started_at=remote.started_at,
                finished_at=remote.finished_at,
                updated_at=_now(self._clock),
                error_code="REMOTE_JOB_ERROR" if mapped is ModelTrainingRunStatus.FAILED else None,
            )
            await self._repository.save(submitted, submitting.updated_at)
        except Exception:
            logger.exception(
                "remote model training submission failed",
                extra={"model_training_run_id": run_id},
            )
            latest = await self._repository.get(run_id)
            if latest.status.active:
                failed = replace(
                    latest,
                    status=ModelTrainingRunStatus.FAILED,
                    updated_at=_now(self._clock),
                    finished_at=_now(self._clock),
                    error_code="REMOTE_SUBMISSION_FAILED",
                )
                try:
                    await self._repository.save(failed, latest.updated_at)
                except Exception:
                    logger.exception(
                        "cannot persist failed model training submission",
                        extra={"model_training_run_id": run_id},
                    )

    def _submission(self, run: ModelTrainingRun) -> RemoteTrainingSubmission:
        parameters = run.parameters
        command = (
            "vehicle-model-training",
            "--config",
            self._runtime.container_training_config,
            "train",
            "--role",
            run.role.value,
            "/data",
            "--run-id",
            run.id,
            "--epochs",
            str(parameters.epochs),
            "--batch-size",
            str(parameters.batch_size),
            "--workers",
            str(parameters.workers),
            "--snapshot-epoch",
            str(parameters.snapshot_epoch),
            "--output-directory",
            self._runtime.container_output_directory,
        )
        return RemoteTrainingSubmission(
            image=str(self._settings.huggingface.job_image),
            command=command,
            hardware_flavor=parameters.hardware_flavor,
            dataset_repo_id=run.dataset_repo_id,
            dataset_revision=run.dataset_commit_sha,
            output_bucket=run.output_bucket,
            namespace=self._settings.huggingface.job_namespace,
            timeout_seconds=parameters.timeout_seconds,
            name=run.id,
            labels={
                "platform": "vehicle-intelligence",
                "role": run.role.value,
                "source": run.source_id[:64],
            },
        )

    async def _refresh(self, run: ModelTrainingRun) -> ModelTrainingRun:
        try:
            remote = await asyncio.to_thread(
                self._gateway.inspect,
                str(run.remote_job_id),
                self._settings.huggingface.job_namespace,
            )
        except Exception:
            logger.exception(
                "remote model training status refresh failed",
                extra={"model_training_run_id": run.id},
            )
            return run
        status = _remote_status(remote.status)
        if (
            status is run.status
            and _message(remote.message) == run.remote_message
            and remote.url == run.remote_job_url
        ):
            return run
        updated = replace(
            run,
            status=status,
            remote_job_url=remote.url or run.remote_job_url,
            remote_message=_message(remote.message),
            started_at=remote.started_at or run.started_at,
            finished_at=remote.finished_at or run.finished_at,
            updated_at=_now(self._clock),
            error_code="REMOTE_JOB_ERROR" if status is ModelTrainingRunStatus.FAILED else None,
        )
        try:
            await self._repository.save(updated, run.updated_at)
            return updated
        except ModelTrainingConflictError:
            return await self._repository.get(run.id)

    async def _save_status(
        self,
        run: ModelTrainingRun,
        status: ModelTrainingRunStatus,
    ) -> ModelTrainingRun:
        now = _now(self._clock)
        updated = replace(
            run,
            status=status,
            updated_at=now,
            finished_at=now if not status.active else run.finished_at,
        )
        await self._repository.save(updated, run.updated_at)
        return updated


def _effective_training_config(
    command: StartModelTrainingCommand,
    settings: ModelTrainingSettings,
) -> PaddleDetectionConfig:
    configured = settings.target(DetectorRole.PLATE).paddledetection
    values = configured.model_dump()
    values.update(
        {
            "epochs": command.epochs,
            "batch_size": command.batch_size,
            "workers": command.workers,
            "snapshot_epoch": command.snapshot_epoch,
        }
    )
    try:
        return PaddleDetectionConfig.model_validate(values)
    except ValidationError as exc:
        raise ModelTrainingValidationError("training parameters are invalid") from exc


def _remote_status(value: str) -> ModelTrainingRunStatus:
    normalized = value.strip().upper()
    return {
        "SCHEDULING": ModelTrainingRunStatus.SCHEDULING,
        "RUNNING": ModelTrainingRunStatus.RUNNING,
        "COMPLETED": ModelTrainingRunStatus.COMPLETED,
        "CANCELED": ModelTrainingRunStatus.CANCELED,
        "CANCELLED": ModelTrainingRunStatus.CANCELED,
        "DELETED": ModelTrainingRunStatus.CANCELED,
        "ERROR": ModelTrainingRunStatus.FAILED,
        "FAILED": ModelTrainingRunStatus.FAILED,
    }.get(normalized, ModelTrainingRunStatus.SCHEDULING)


def _model_identifier(value: str, label: str) -> str:
    stripped = value.strip()
    if not _MODEL_ID.fullmatch(stripped):
        raise ModelTrainingValidationError(f"{label} is invalid")
    return stripped


def _message(value: str | None) -> str | None:
    if value is None:
        return None
    safe = "".join(character for character in value if character >= " " or character == "\t")
    return safe.strip()[:1000] or None


def _now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if value.tzinfo is None:
        raise ModelTrainingValidationError("model training clock must be timezone-aware")
    return value.astimezone(UTC)
