"""Authenticated model-training workflow and remote Job observability API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.model_training import (
    ModelTrainingService,
    StartModelTrainingCommand,
)
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import AuditAction, AuditResourceType, Principal
from vehicle_intelligence.domain.model_training import (
    ModelTrainingCapabilities,
    ModelTrainingDefaults,
    ModelTrainingLog,
    ModelTrainingRun,
)
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    ModelRegistryError,
    ModelTrainingConflictError,
    ModelTrainingNotFoundError,
    ModelTrainingStorageError,
    ModelTrainingValidationError,
)
from vehicle_intelligence.interfaces.model_training_schemas import (
    StartModelTrainingRequest,
)
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.security import APISecurity


def build_model_training_router(
    service: ModelTrainingService,
    security: APISecurity,
    audits: AuditService,
) -> APIRouter:
    router = APIRouter(prefix="/api/model-training", tags=["model-training"])
    read_access = security.require(Permission.REVIEW_DATASETS)
    manage_access = security.require(Permission.MANAGE_DATASETS)

    @router.get("")
    async def overview(
        response: Response,
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> dict[str, object]:
        try:
            response.headers["Cache-Control"] = "no-store, private"
            runs = await service.list_runs()
            return {
                "capabilities": _capabilities_json(service.capabilities()),
                "items": [_run_json(item) for item in runs[:limit]],
            }
        except Exception as exc:
            _raise_model_training_http(exc)

    @router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
    async def start_run(
        payload: StartModelTrainingRequest,
        http_request: Request,
        response: Response,
        principal: Principal = Depends(manage_access),
    ) -> dict[str, object]:
        try:
            run = await service.prepare_run(
                StartModelTrainingCommand(
                    source_id=payload.source_id,
                    model_name=payload.model_name,
                    model_version=payload.model_version,
                    epochs=payload.epochs,
                    batch_size=payload.batch_size,
                    workers=payload.workers,
                    snapshot_epoch=payload.snapshot_epoch,
                    dataset_rights_confirmed=payload.confirm_dataset_rights,
                    compute_cost_confirmed=payload.confirm_compute_cost,
                    restricted_data_confirmed=payload.confirm_restricted_data,
                ),
                principal,
            )
            try:
                await audits.record(
                    AuditRecord(
                        principal=principal,
                        action=AuditAction.MODEL_TRAINING_STARTED,
                        resource_type=AuditResourceType.MODEL_TRAINING_RUN,
                        resource_id=run.id,
                        request_id=request_id(http_request),
                        after=_run_json(run),
                        metadata={
                            "sourceId": run.source_id,
                            "sourceManifestSha256": run.source_manifest_sha256,
                            "exportManifestSha256": run.export_manifest_sha256,
                            "datasetCommitSha": run.dataset_commit_sha,
                            "modelName": run.model_name,
                            "modelVersion": run.model_version,
                        },
                    )
                )
            except Exception:
                await service.fail_prepared_run(run.id, "AUDIT_WRITE_FAILED")
                raise
            service.dispatch_run(run)
            response.headers["Cache-Control"] = "no-store, private"
            return _run_json(run)
        except Exception as exc:
            _raise_model_training_http(exc)

    @router.get("/runs/{run_id}")
    async def get_run(
        run_id: str,
        response: Response,
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        try:
            response.headers["Cache-Control"] = "no-store, private"
            return _run_json(await service.get_run(run_id))
        except Exception as exc:
            _raise_model_training_http(exc)

    @router.get("/runs/{run_id}/logs")
    async def get_logs(
        run_id: str,
        response: Response,
        _principal: Principal = Depends(read_access),
        tail: Annotated[int, Query(ge=1, le=5000)] = 300,
    ) -> dict[str, object]:
        try:
            response.headers["Cache-Control"] = "no-store, private"
            return _log_json(await service.logs(run_id, tail))
        except Exception as exc:
            _raise_model_training_http(exc)

    @router.post("/runs/{run_id}/cancel")
    async def cancel_run(
        run_id: str,
        http_request: Request,
        response: Response,
        principal: Principal = Depends(manage_access),
    ) -> dict[str, object]:
        try:
            current = await service.get_run(run_id, refresh=False)
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.MODEL_TRAINING_CANCELED,
                    resource_type=AuditResourceType.MODEL_TRAINING_RUN,
                    resource_id=run_id,
                    request_id=request_id(http_request),
                    before=_run_json(current),
                    metadata={"intentRecordedBeforeRemoteCancellation": True},
                )
            )
            canceled = await service.cancel(run_id)
            response.headers["Cache-Control"] = "no-store, private"
            return _run_json(canceled)
        except Exception as exc:
            _raise_model_training_http(exc)

    return router


def _run_json(run: ModelTrainingRun) -> dict[str, object]:
    return {
        "id": run.id,
        "role": run.role.value,
        "status": run.status.value,
        "sourceId": run.source_id,
        "sourceManifestSha256": run.source_manifest_sha256,
        "exportId": run.export_id,
        "exportManifestSha256": run.export_manifest_sha256,
        "datasetRepoId": run.dataset_repo_id,
        "datasetRevision": run.dataset_revision,
        "datasetCommitSha": run.dataset_commit_sha,
        "modelRepoId": run.model_repo_id,
        "modelName": run.model_name,
        "modelVersion": run.model_version,
        "architecture": run.architecture,
        "parameters": {
            "epochs": run.parameters.epochs,
            "batchSize": run.parameters.batch_size,
            "workers": run.parameters.workers,
            "snapshotEpoch": run.parameters.snapshot_epoch,
            "timeoutSeconds": run.parameters.timeout_seconds,
            "hardwareFlavor": run.parameters.hardware_flavor,
        },
        "requestedBy": run.requested_by,
        "confirmations": {
            "datasetRights": run.dataset_rights_confirmed,
            "computeCost": run.compute_cost_confirmed,
            "restrictedData": run.restricted_data_confirmed,
        },
        "createdAt": run.created_at.isoformat(),
        "updatedAt": run.updated_at.isoformat(),
        "startedAt": run.started_at.isoformat() if run.started_at else None,
        "finishedAt": run.finished_at.isoformat() if run.finished_at else None,
        "outputBucket": run.output_bucket,
        "outputPath": run.output_path,
        "remoteJobId": run.remote_job_id,
        "remoteJobUrl": run.remote_job_url,
        "remoteMessage": run.remote_message,
        "errorCode": run.error_code,
    }


def _capabilities_json(item: ModelTrainingCapabilities) -> dict[str, object]:
    return {
        "enabled": item.enabled,
        "jobsEnabled": item.jobs_enabled,
        "credentialsConfigured": item.credentials_configured,
        "imageConfigured": item.image_configured,
        "outputBucketConfigured": item.output_bucket_configured,
        "submissionsEnabled": item.submissions_enabled,
        "jobImage": item.job_image,
        "outputBucket": item.output_bucket,
        "namespace": item.namespace,
        "defaults": _defaults_json(item.defaults),
        "blockers": list(item.blockers),
    }


def _defaults_json(item: ModelTrainingDefaults) -> dict[str, object]:
    return {
        "role": item.role.value,
        "architecture": item.architecture,
        "baseConfig": item.base_config,
        "modelRepoId": item.model_repo_id,
        "datasetRepoId": item.dataset_repo_id,
        "epochs": item.epochs,
        "batchSize": item.batch_size,
        "workers": item.workers,
        "snapshotEpoch": item.snapshot_epoch,
        "timeoutSeconds": item.timeout_seconds,
        "hardwareFlavor": item.hardware_flavor,
    }


def _log_json(item: ModelTrainingLog) -> dict[str, object]:
    return {"runId": item.run_id, "lines": list(item.lines), "available": item.available}


def _raise_model_training_http(exc: Exception) -> None:
    if isinstance(exc, ModelTrainingNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, ModelTrainingConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if isinstance(exc, (ModelTrainingValidationError, ValueError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (ModelTrainingStorageError, ModelRegistryError, AuditWriteError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model training service or remote provider is unavailable",
        ) from exc
    raise exc
