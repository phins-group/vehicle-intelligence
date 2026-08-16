"""Authenticated detector-dataset labeling and promotion API."""

from __future__ import annotations

from typing import Annotated, Never
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.dataset_review import (
    DetectorDatasetReviewService,
    DetectorReviewCommand,
    DetectorReviewQuery,
)
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import AuditAction, AuditResourceType, Principal
from vehicle_intelligence.domain.dataset_review import (
    DetectorPromotionJob,
    DetectorReviewAnnotation,
    DetectorReviewBox,
    DetectorReviewDecision,
    DetectorReviewItem,
    DetectorReviewStatus,
)
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    DatasetReviewConflictError,
    DatasetReviewNotFoundError,
    DatasetReviewStorageError,
    DatasetReviewValidationError,
    InvalidCursorError,
)
from vehicle_intelligence.interfaces.dataset_review_schemas import (
    DetectorPromotionRequest,
    DetectorReviewDecisionRequest,
)
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.security import APISecurity


def build_dataset_review_router(
    service: DetectorDatasetReviewService,
    security: APISecurity,
    audits: AuditService,
) -> APIRouter:
    router = APIRouter(prefix="/api/detector-review", tags=["detector-dataset-review"])
    review_access = security.require(Permission.REVIEW_DATASETS)
    promotion_access = security.require(Permission.MANAGE_DATASETS)

    @router.get("/sources")
    async def list_sources(
        response: Response,
        _principal: Principal = Depends(review_access),
    ) -> dict[str, object]:
        try:
            response.headers["Cache-Control"] = "no-store, private"
            return {
                "items": [
                    {
                        "sourceId": item.source_id,
                        "sourceManifestSha256": item.source_manifest_sha256,
                        "sourceType": item.source_type,
                        "collectionMethod": item.collection_method,
                        "rightsStatus": item.rights_status,
                        "promotionEligible": item.promotion_eligible,
                        "releaseEligible": item.release_eligible,
                        "distributionEligible": item.distribution_eligible,
                        "queueCount": item.queue_count,
                        "statusCounts": item.status_counts,
                        "reasonCounts": item.reason_counts,
                        "reviewedCount": item.reviewed_count,
                        "pendingCount": item.pending_count,
                    }
                    for item in await service.list_sources()
                ]
            }
        except Exception as exc:
            _raise_dataset_review_http(exc)

    @router.get("/items")
    async def list_items(
        response: Response,
        source_id: Annotated[str, Query(alias="sourceId", min_length=1, max_length=128)],
        _principal: Principal = Depends(review_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: str | None = None,
        review_status: Annotated[DetectorReviewStatus | None, Query(alias="status")] = None,
        reason: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    ) -> dict[str, object]:
        try:
            page = await service.list_items(
                DetectorReviewQuery(
                    source_id=source_id,
                    limit=limit,
                    cursor=cursor,
                    status=review_status,
                    reason=reason,
                )
            )
            response.headers["Cache-Control"] = "no-store, private"
            return {
                "items": [_item_json(item, include_dimensions=False) for item in page.items],
                "nextCursor": page.next_cursor,
            }
        except Exception as exc:
            _raise_dataset_review_http(exc)

    @router.get("/sources/{source_id}/items/{review_id}")
    async def get_item(
        source_id: str,
        review_id: str,
        response: Response,
        _principal: Principal = Depends(review_access),
    ) -> dict[str, object]:
        try:
            item = await service.get_item(source_id, review_id)
            response.headers["Cache-Control"] = "no-store, private"
            return _item_json(item, include_dimensions=True)
        except Exception as exc:
            _raise_dataset_review_http(exc)

    @router.get("/sources/{source_id}/items/{review_id}/image")
    async def get_image(
        source_id: str,
        review_id: str,
        _principal: Principal = Depends(review_access),
    ) -> FileResponse:
        try:
            image = await service.get_image(source_id, review_id)
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
            _raise_dataset_review_http(exc)

    @router.put("/sources/{source_id}/items/{review_id}")
    async def review_item(
        source_id: str,
        review_id: str,
        payload: DetectorReviewDecisionRequest,
        http_request: Request,
        response: Response,
        principal: Principal = Depends(review_access),
    ) -> dict[str, object]:
        try:
            annotations = tuple(
                DetectorReviewAnnotation(
                    bbox=DetectorReviewBox(
                        x=item.x,
                        y=item.y,
                        width=item.width,
                        height=item.height,
                    )
                )
                for item in payload.annotations
            )
            prepared = await service.prepare_review(
                source_id,
                review_id,
                DetectorReviewCommand(
                    action=payload.action,
                    expected_revision=payload.expected_revision,
                    reviewer=principal,
                    annotations=annotations,
                    note=payload.note,
                ),
            )
            audit_entry = audits.prepare(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.DETECTOR_SAMPLE_REVIEWED,
                    resource_type=AuditResourceType.DETECTOR_DATASET_SAMPLE,
                    resource_id=f"{source_id}:{review_id}",
                    request_id=request_id(http_request),
                    before=_item_json(prepared.before, include_dimensions=False),
                    after=_item_json(prepared.after, include_dimensions=False),
                    metadata={
                        "sourceId": source_id,
                        "reviewAction": payload.action.value,
                        "reviewRevision": prepared.after.revision,
                    },
                )
            )
            reviewed = await service.commit_review(prepared, audit_entry)
            delivered = await service.deliver_audit(audit_entry, audits)
            response.headers["X-Audit-Delivery"] = "delivered" if delivered else "pending"
            response.headers["Cache-Control"] = "no-store, private"
            return _item_json(reviewed, include_dimensions=True)
        except Exception as exc:
            _raise_dataset_review_http(exc)

    @router.get("/sources/{source_id}/items/{review_id}/history")
    async def review_history(
        source_id: str,
        review_id: str,
        response: Response,
        _principal: Principal = Depends(review_access),
    ) -> dict[str, object]:
        try:
            history = await service.history(source_id, review_id)
            response.headers["Cache-Control"] = "no-store, private"
            return {"items": [_decision_json(item) for item in history]}
        except Exception as exc:
            _raise_dataset_review_http(exc)

    @router.post("/sources/{source_id}/promotions", status_code=status.HTTP_202_ACCEPTED)
    async def promote_source(
        source_id: str,
        payload: DetectorPromotionRequest,
        http_request: Request,
        response: Response,
        principal: Principal = Depends(promotion_access),
    ) -> dict[str, object]:
        try:
            job = await service.prepare_promotion(source_id, payload.target_source_id, principal)
            try:
                await audits.record(
                    AuditRecord(
                        principal=principal,
                        action=AuditAction.DETECTOR_DATASET_PROMOTION_STARTED,
                        resource_type=AuditResourceType.DETECTOR_DATASET,
                        resource_id=payload.target_source_id,
                        request_id=request_id(http_request),
                        after=_job_json(job),
                        metadata={"sourceId": source_id, "promotionJobId": job.id},
                    )
                )
            except Exception:
                await service.fail_prepared_promotion(job.id, "AUDIT_WRITE_FAILED")
                raise
            service.dispatch_promotion(job)
            response.headers["Cache-Control"] = "no-store, private"
            return _job_json(job)
        except Exception as exc:
            _raise_dataset_review_http(exc)

    @router.get("/promotions/{job_id}")
    async def promotion_status(
        job_id: str,
        response: Response,
        _principal: Principal = Depends(promotion_access),
    ) -> dict[str, object]:
        try:
            response.headers["Cache-Control"] = "no-store, private"
            return _job_json(await service.get_promotion(job_id))
        except Exception as exc:
            _raise_dataset_review_http(exc)

    return router


def _item_json(item: DetectorReviewItem, *, include_dimensions: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "sourceId": item.source_id,
        "reviewId": item.review_id,
        "sourceImageSha256": item.source_image_sha256,
        "sourceFilenameSha256": item.source_filename_sha256,
        "reason": item.reason,
        "status": item.status.value,
        "revision": item.revision,
        "suggestions": [_annotation_json(value) for value in item.suggestions],
        "decision": _decision_json(item.decision) if item.decision is not None else None,
        "imageUrl": (
            "/api/detector-review/sources/"
            f"{quote(item.source_id, safe='')}/items/{quote(item.review_id, safe='')}/image"
        ),
    }
    if include_dimensions:
        result["image"] = {"width": item.image_width, "height": item.image_height}
    return result


def _annotation_json(annotation: DetectorReviewAnnotation) -> dict[str, object]:
    return {
        "className": annotation.class_name,
        "bbox": {
            "x": annotation.bbox.x,
            "y": annotation.bbox.y,
            "width": annotation.bbox.width,
            "height": annotation.bbox.height,
        },
        "attributes": annotation.attributes,
    }


def _decision_json(decision: DetectorReviewDecision) -> dict[str, object]:
    return {
        "action": decision.action.value,
        "status": decision.status.value,
        "annotations": [_annotation_json(value) for value in decision.annotations],
        "revision": decision.revision,
        "reviewedBy": {
            "id": decision.reviewed_by,
            "displayName": decision.reviewer_display_name,
        },
        "reviewedAt": decision.reviewed_at.isoformat(),
        "note": decision.note,
    }


def _job_json(job: DetectorPromotionJob) -> dict[str, object]:
    return {
        "id": job.id,
        "sourceId": job.source_id,
        "targetSourceId": job.target_source_id,
        "status": job.status.value,
        "createdAt": job.created_at.isoformat(),
        "updatedAt": job.updated_at.isoformat(),
        "requestedBy": job.requested_by,
        "reviewedSampleCount": job.reviewed_sample_count,
        "pendingSampleCount": job.pending_sample_count,
        "decisionSnapshotSha256": job.decision_snapshot_sha256,
        "outputDirectory": job.output_directory,
        "manifestSha256": job.manifest_sha256,
        "errorCode": job.error_code,
    }


def _raise_dataset_review_http(exc: Exception) -> Never:
    if isinstance(exc, DatasetReviewNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, DatasetReviewConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if isinstance(exc, (DatasetReviewValidationError, ValueError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, InvalidCursorError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if isinstance(exc, (DatasetReviewStorageError, AuditWriteError)):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="detector dataset review persistence is unavailable",
        ) from exc
    raise exc
