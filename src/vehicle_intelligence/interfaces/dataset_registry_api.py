"""Dataset catalog and private Hugging Face synchronization API."""

from __future__ import annotations

from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.dataset_registry import (
    DatasetHubSyncCommand,
    DatasetRegistryService,
    DetectorDatasetSampleQuery,
)
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import AuditAction, AuditResourceType, Principal
from vehicle_intelligence.domain.dataset_registry import (
    DatasetHubSyncJob,
    DatasetRegistryCapabilities,
    DetectorDatasetExport,
    DetectorDatasetSampleAnnotation,
    DetectorDatasetSampleKind,
    DetectorDatasetSamplePreview,
    DetectorDatasetVersion,
)
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    DatasetRegistryConflictError,
    DatasetRegistryNotFoundError,
    DatasetRegistryStorageError,
    DatasetRegistryValidationError,
    InvalidCursorError,
)
from vehicle_intelligence.interfaces.dataset_registry_schemas import DatasetHubSyncRequest
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.security import APISecurity


def build_dataset_registry_router(
    service: DatasetRegistryService,
    security: APISecurity,
    audits: AuditService,
) -> APIRouter:
    router = APIRouter(prefix="/api/datasets", tags=["detector-dataset-registry"])
    read_access = security.require(Permission.REVIEW_DATASETS)
    manage_access = security.require(Permission.MANAGE_DATASETS)

    @router.get("")
    async def list_datasets(
        response: Response,
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        try:
            capabilities = await service.capabilities()
            datasets = await service.list_datasets()
            response.headers["Cache-Control"] = "no-store, private"
            return {
                "items": [_dataset_json(item) for item in datasets],
                "hub": _capabilities_json(capabilities),
            }
        except Exception as exc:
            _raise_dataset_registry_http(exc)

    @router.get("/{source_id}/samples")
    async def list_samples(
        source_id: str,
        response: Response,
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=50)] = 12,
        cursor: str | None = None,
        kind: DetectorDatasetSampleKind = DetectorDatasetSampleKind.ALL,
        lighting: Annotated[str | None, Query(min_length=1, max_length=16)] = None,
    ) -> dict[str, object]:
        try:
            page = await service.list_samples(
                DetectorDatasetSampleQuery(
                    source_id=source_id,
                    limit=limit,
                    cursor=cursor,
                    kind=kind,
                    lighting=lighting,
                )
            )
            response.headers["Cache-Control"] = "no-store, private"
            return {
                "items": [_sample_json(item) for item in page.items],
                "nextCursor": page.next_cursor,
            }
        except Exception as exc:
            _raise_dataset_registry_http(exc)

    @router.get("/{source_id}/samples/{image_sha256}/image")
    async def sample_image(
        source_id: str,
        image_sha256: str,
        _principal: Principal = Depends(read_access),
    ) -> FileResponse:
        try:
            image = await service.get_sample_image(source_id, image_sha256)
            return FileResponse(
                image.path,
                media_type=image.media_type,
                headers={
                    "Cache-Control": "private, max-age=300",
                    "ETag": f'"{image.sha256}"',
                    "X-Content-Type-Options": "nosniff",
                },
            )
        except Exception as exc:
            _raise_dataset_registry_http(exc)

    @router.post(
        "/{source_id}/syncs",
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def start_sync(
        source_id: str,
        payload: DatasetHubSyncRequest,
        http_request: Request,
        response: Response,
        principal: Principal = Depends(manage_access),
    ) -> dict[str, object]:
        try:
            job = await service.prepare_sync(
                source_id,
                DatasetHubSyncCommand(
                    export_id=payload.export_id,
                    revision=payload.revision,
                    restricted_transfer_confirmed=(
                        payload.confirm_restricted_private_transfer
                    ),
                ),
                principal,
            )
            try:
                await audits.record(
                    AuditRecord(
                        principal=principal,
                        action=AuditAction.DETECTOR_DATASET_HF_SYNC_STARTED,
                        resource_type=AuditResourceType.DETECTOR_DATASET,
                        resource_id=source_id,
                        request_id=request_id(http_request),
                        after=_sync_json(job),
                        metadata={
                            "datasetSyncJobId": job.id,
                            "exportId": job.export_id,
                            "repoId": job.repo_id,
                            "revision": job.requested_revision,
                            "restrictedTransferConfirmed": (
                                job.restricted_transfer_confirmed
                            ),
                        },
                    )
                )
            except Exception:
                await service.fail_prepared_sync(job.id, "AUDIT_WRITE_FAILED")
                raise
            service.dispatch_sync(job)
            response.headers["Cache-Control"] = "no-store, private"
            return _sync_json(job)
        except Exception as exc:
            _raise_dataset_registry_http(exc)

    @router.get("/syncs/{job_id}")
    async def sync_status(
        job_id: str,
        response: Response,
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        try:
            response.headers["Cache-Control"] = "no-store, private"
            return _sync_json(await service.get_sync(job_id))
        except Exception as exc:
            _raise_dataset_registry_http(exc)

    return router


def _dataset_json(item: DetectorDatasetVersion) -> dict[str, object]:
    return {
        "sourceId": item.source_id,
        "sourceManifestSha256": item.source_manifest_sha256,
        "createdAt": item.created_at.isoformat(),
        "sampleCount": item.sample_count,
        "annotationCount": item.annotation_count,
        "negativeSampleCount": item.negative_sample_count,
        "reviewQueueCount": item.review_queue_count,
        "releaseEligible": item.release_eligible,
        "distributionEligible": item.distribution_eligible,
        "privacyClassification": item.privacy_classification,
        "parentSourceId": item.parent_source_id,
        "export": _export_json(item.export) if item.export is not None else None,
        "latestSync": _sync_json(item.latest_sync) if item.latest_sync is not None else None,
    }


def _export_json(item: DetectorDatasetExport) -> dict[str, object]:
    return {
        "exportId": item.export_id,
        "manifestSha256": item.manifest_sha256,
        "createdAt": item.created_at.isoformat(),
        "sampleCount": item.sample_count,
        "annotationCount": item.annotation_count,
        "negativeSampleCount": item.negative_sample_count,
        "splitCounts": item.split_counts,
        "releaseEligible": item.release_eligible,
        "distributionEligible": item.distribution_eligible,
        "sourceManifestSha256": item.source_manifest_sha256,
    }


def _sample_json(item: DetectorDatasetSamplePreview) -> dict[str, object]:
    return {
        "sourceId": item.source_id,
        "sampleId": item.sample_id,
        "imageSha256": item.image_sha256,
        "cameraId": item.camera_id,
        "groupId": item.group_id,
        "capturedAt": item.captured_at.isoformat(),
        "split": item.split,
        "lighting": item.lighting,
        "annotationStatus": item.annotation_status,
        "negative": item.negative,
        "image": {"width": item.image_width, "height": item.image_height},
        "annotations": [_sample_annotation_json(value) for value in item.annotations],
        "imageUrl": (
            f"/api/datasets/{quote(item.source_id, safe='')}/samples/"
            f"{quote(item.image_sha256, safe='')}/image"
        ),
    }


def _sample_annotation_json(item: DetectorDatasetSampleAnnotation) -> dict[str, object]:
    return {
        "className": item.class_name,
        "bbox": {
            "x": item.bbox.x,
            "y": item.bbox.y,
            "width": item.bbox.width,
            "height": item.bbox.height,
        },
    }


def _sync_json(item: DatasetHubSyncJob) -> dict[str, object]:
    return {
        "id": item.id,
        "sourceId": item.source_id,
        "sourceManifestSha256": item.source_manifest_sha256,
        "exportId": item.export_id,
        "repoId": item.repo_id,
        "requestedRevision": item.requested_revision,
        "status": item.status.value,
        "requestedBy": item.requested_by,
        "restrictedTransferConfirmed": item.restricted_transfer_confirmed,
        "createdAt": item.created_at.isoformat(),
        "updatedAt": item.updated_at.isoformat(),
        "exportManifestSha256": item.export_manifest_sha256,
        "hubCommitSha": item.hub_commit_sha,
        "hubUrl": item.hub_url,
        "reusedExport": item.reused_export,
        "errorCode": item.error_code,
    }


def _capabilities_json(item: DatasetRegistryCapabilities) -> dict[str, object]:
    return {
        "enabled": item.enabled,
        "hubEnabled": item.hub_enabled,
        "repoId": item.repo_id,
        "credentialsConfigured": item.credentials_configured,
        "restrictedPrivateSyncEnabled": item.restricted_private_sync_enabled,
    }


def _raise_dataset_registry_http(exc: Exception) -> None:
    if isinstance(exc, DatasetRegistryNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, DatasetRegistryConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if isinstance(exc, (DatasetRegistryValidationError, InvalidCursorError, ValueError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (DatasetRegistryStorageError, AuditWriteError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dataset registry persistence is unavailable",
        ) from exc
    raise exc
