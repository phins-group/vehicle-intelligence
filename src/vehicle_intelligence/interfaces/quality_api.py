"""Authenticated model-quality reporting API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from vehicle_intelligence.application.model_quality import ModelQualityService
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import Principal
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.quality_serialization import quality_report_to_jsonable
from vehicle_intelligence.interfaces.security import APISecurity


def build_quality_router(service: ModelQualityService, security: APISecurity) -> APIRouter:
    router = APIRouter(tags=["model-quality"])
    read_access = security.require(Permission.READ_PLATFORM)

    @router.get("/api/model-quality")
    async def model_quality(
        response: Response,
        _principal: Principal = Depends(read_access),
        from_time: Annotated[datetime | None, Query(alias="from")] = None,
        to_time: Annotated[datetime | None, Query(alias="to")] = None,
    ) -> dict[str, object]:
        try:
            report = await service.report(from_time, to_time)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except PersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="model-quality persistence is unavailable",
            ) from exc
        response.headers["Cache-Control"] = "no-store, private"
        return quality_report_to_jsonable(report)

    return router
