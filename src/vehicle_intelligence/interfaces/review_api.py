"""Authenticated human OCR review and dataset-feedback API."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.ports import DatasetSampleQuery
from vehicle_intelligence.application.review import (
    HumanPlateReviewService,
    PlateReviewCommand,
)
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import (
    AuditAction,
    AuditResourceType,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    Principal,
)
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    InvalidCursorError,
    MediaStorageError,
    PersistenceError,
    PlateReviewConflictError,
    PlateReviewValidationError,
    VehicleEventNotFoundError,
)
from vehicle_intelligence.infrastructure.review_serialization import (
    dataset_sample_to_jsonable,
)
from vehicle_intelligence.infrastructure.serialization import event_to_jsonable
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.review_schemas import (
    PlateReviewRequest,
    PlateReviewResponse,
)
from vehicle_intelligence.interfaces.security import APISecurity


def build_review_router(
    service: HumanPlateReviewService,
    security: APISecurity,
    audits: AuditService,
    mutation_transaction: Callable[[], AsyncIterator[None]] | None = None,
) -> APIRouter:
    router = APIRouter(tags=["human-review"])
    review_access = security.require(Permission.REVIEW_PLATES)
    mutation_dependencies = (
        [Depends(mutation_transaction)] if mutation_transaction is not None else []
    )

    @router.put(
        "/api/events/{event_id}/plate-review",
        response_model=PlateReviewResponse,
        dependencies=mutation_dependencies,
    )
    async def review_plate(
        event_id: str,
        http_request: Request,
        request: PlateReviewRequest,
        response: Response,
        principal: Principal = Depends(review_access),
    ) -> PlateReviewResponse:
        try:
            result = await service.review(
                event_id,
                PlateReviewCommand(
                    text=request.text,
                    expected_revision=request.expected_revision,
                    reviewer=principal,
                    note=request.note,
                ),
            )
            if result.changed:
                action = (
                    AuditAction.PLATE_CONFIRMED
                    if result.reason is DatasetSampleReason.HUMAN_CONFIRMATION
                    else AuditAction.PLATE_CORRECTED
                )
                await audits.record(
                    AuditRecord(
                        principal=principal,
                        action=action,
                        resource_type=AuditResourceType.VEHICLE_EVENT,
                        resource_id=event_id,
                        request_id=request_id(http_request),
                        before=event_to_jsonable(result.before),
                        after=event_to_jsonable(result.event),
                        metadata={
                            "reviewRevision": result.event.plate.review_revision,
                            "feedbackReason": result.reason.value,
                            "datasetSampleId": result.dataset_sample_id,
                        },
                    )
                )
            response.headers["Cache-Control"] = "no-store, private"
            return PlateReviewResponse(
                event=event_to_jsonable(result.event),
                changed=result.changed,
                feedbackReason=result.reason.value,
                datasetSampleId=result.dataset_sample_id,
            )
        except Exception as exc:
            _raise_review_http(exc)

    @router.get("/api/dataset-samples")
    async def list_dataset_samples(
        response: Response,
        _principal: Principal = Depends(review_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: str | None = None,
        sample_type: Annotated[DatasetSampleType | None, Query(alias="type")] = None,
        sample_status: Annotated[DatasetSampleStatus | None, Query(alias="status")] = None,
        reason: DatasetSampleReason | None = None,
        source_event_id: Annotated[str | None, Query(alias="sourceEventId")] = None,
    ) -> dict[str, object]:
        try:
            page = await service.list_samples(
                DatasetSampleQuery(
                    limit=limit,
                    cursor=cursor,
                    sample_type=sample_type,
                    status=sample_status,
                    reason=reason,
                    source_event_id=source_event_id,
                )
            )
            response.headers["Cache-Control"] = "no-store, private"
            return {
                "items": [dataset_sample_to_jsonable(item) for item in page.items],
                "nextCursor": page.next_cursor,
            }
        except Exception as exc:
            _raise_review_http(exc)

    return router


def _raise_review_http(exc: Exception) -> None:
    if isinstance(exc, AuditWriteError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="audit persistence is unavailable",
        ) from exc
    if isinstance(exc, VehicleEventNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if isinstance(exc, PlateReviewConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if isinstance(exc, PlateReviewValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    if isinstance(exc, InvalidCursorError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if isinstance(exc, PersistenceError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="human-review persistence is unavailable",
        ) from exc
    if isinstance(exc, MediaStorageError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="media storage is unavailable",
        ) from exc
    raise exc
