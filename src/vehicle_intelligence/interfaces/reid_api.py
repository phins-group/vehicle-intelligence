"""ReID scoring and audited human merge/split routes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.reid import IdentityReviewService, ReIDScoringService
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import AuditAction, AuditResourceType, Principal
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    IdentityConflictError,
    IdentityNotFoundError,
    PersistenceError,
    TopologyNotFoundError,
)
from vehicle_intelligence.interfaces.reid_schemas import (
    IdentityReviewResultPublic,
    MergeIdentitiesRequest,
    ReIDScoreListPublic,
    ReIDScorePublic,
    SplitIdentityRequest,
)
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.security import APISecurity


def build_reid_router(
    scoring: ReIDScoringService,
    reviews: IdentityReviewService,
    security: APISecurity,
    audits: AuditService,
    mutation_transaction: Callable[[], AsyncIterator[None]] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["vehicle-reid"])
    read_access = security.require(Permission.READ_PLATFORM)
    review_access = security.require(Permission.REVIEW_IDENTITIES)
    mutation_dependencies = (
        [Depends(mutation_transaction)] if mutation_transaction is not None else []
    )

    @router.get(
        "/vehicle-fingerprints/{fingerprint_id}/reid-candidates",
        response_model=ReIDScoreListPublic,
    )
    async def score_candidates(
        fingerprint_id: str,
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> ReIDScoreListPublic:
        try:
            values = await scoring.score_candidates(fingerprint_id, limit)
            return ReIDScoreListPublic(
                sourceFingerprintId=fingerprint_id,
                items=[ReIDScorePublic.from_domain(value) for value in values],
            )
        except Exception as exc:
            _raise_reid_http(exc)

    @router.post(
        "/vehicle-identities/merge",
        response_model=IdentityReviewResultPublic,
        dependencies=mutation_dependencies,
    )
    async def merge_identities(
        http_request: Request,
        request: MergeIdentitiesRequest,
        principal: Principal = Depends(review_access),
    ) -> IdentityReviewResultPublic:
        try:
            result = IdentityReviewResultPublic.from_domain(
                await reviews.merge(request.to_command(), principal)
            )
            if not result.idempotent:
                await audits.record(
                    AuditRecord(
                        principal=principal,
                        action=AuditAction.VEHICLE_IDENTITIES_MERGED,
                        resource_type=AuditResourceType.VEHICLE_IDENTITY,
                        resource_id=result.result_vehicle_id,
                        request_id=request_id(http_request),
                        after=_snapshot(result),
                        metadata={"reviewId": result.review_id},
                    )
                )
            return result
        except Exception as exc:
            _raise_reid_http(exc)

    @router.post(
        "/vehicle-identities/split",
        response_model=IdentityReviewResultPublic,
        dependencies=mutation_dependencies,
    )
    async def split_identity(
        http_request: Request,
        request: SplitIdentityRequest,
        principal: Principal = Depends(review_access),
    ) -> IdentityReviewResultPublic:
        try:
            result = IdentityReviewResultPublic.from_domain(
                await reviews.split(request.to_command(), principal)
            )
            if not result.idempotent:
                await audits.record(
                    AuditRecord(
                        principal=principal,
                        action=AuditAction.VEHICLE_IDENTITY_SPLIT,
                        resource_type=AuditResourceType.VEHICLE_IDENTITY,
                        resource_id=result.result_vehicle_id,
                        request_id=request_id(http_request),
                        after=_snapshot(result),
                        metadata={"reviewId": result.review_id},
                    )
                )
            return result
        except Exception as exc:
            _raise_reid_http(exc)

    @router.get(
        "/vehicle-identity-reviews/{review_id}",
        response_model=IdentityReviewResultPublic,
    )
    async def get_review(
        review_id: str,
        _principal: Principal = Depends(read_access),
    ) -> IdentityReviewResultPublic:
        try:
            return IdentityReviewResultPublic.from_domain(
                await reviews.get_review(review_id)
            )
        except Exception as exc:
            _raise_reid_http(exc)

    return router


def _snapshot(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json", by_alias=True)


def _raise_reid_http(exc: Exception) -> NoReturn:
    if isinstance(exc, AuditWriteError):
        raise HTTPException(status_code=503, detail="audit persistence is unavailable") from exc
    if isinstance(exc, (IdentityNotFoundError, TopologyNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, IdentityConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, PersistenceError):
        raise HTTPException(status_code=503, detail="identity persistence unavailable") from exc
    raise exc
