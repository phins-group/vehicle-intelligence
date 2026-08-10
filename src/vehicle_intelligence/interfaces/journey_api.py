"""Read-only logical vehicle timeline and journey routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query

from vehicle_intelligence.application.journeys import VehicleJourneyService
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import Principal
from vehicle_intelligence.exceptions import IdentityNotFoundError, PersistenceError
from vehicle_intelligence.interfaces.journey_schemas import (
    JourneyObservationPublic,
    VehicleJourneyPublic,
    VehicleTimelinePublic,
)
from vehicle_intelligence.interfaces.security import APISecurity


def build_journey_router(
    service: VehicleJourneyService,
    security: APISecurity,
) -> APIRouter:
    router = APIRouter(prefix="/api/vehicles", tags=["vehicle-journey"])
    read_access = security.require(Permission.READ_PLATFORM)

    @router.get("/{vehicle_id}/timeline", response_model=VehicleTimelinePublic)
    async def timeline(
        vehicle_id: str,
        _principal: Principal = Depends(read_access),
        from_time: Annotated[datetime | None, Query(alias="from")] = None,
        to_time: Annotated[datetime | None, Query(alias="to")] = None,
        limit: Annotated[int, Query(ge=1, le=4999)] = 1000,
    ) -> VehicleTimelinePublic:
        try:
            events = await service.timeline(
                vehicle_id,
                from_time=from_time,
                to_time=to_time,
                limit=limit,
            )
            items = [
                JourneyObservationPublic(
                    eventId=event.id,
                    cameraId=event.camera.id,
                    cameraName=event.camera.name,
                    zone=event.camera.zone,
                    occurredAt=event.occurred_at,
                    eventType=event.event_type.value,
                    direction=event.direction.value,
                    status=event.status.value,
                    plate=(
                        event.plate.final_normalized
                        if event.plate is not None
                        else None
                    ),
                    vehicleType=event.vehicle.type,
                )
                for event in events
            ]
            return VehicleTimelinePublic(vehicleId=vehicle_id, items=items)
        except Exception as exc:
            _raise_journey_http(exc)

    @router.get("/{vehicle_id}/journey", response_model=VehicleJourneyPublic)
    async def journey(
        vehicle_id: str,
        _principal: Principal = Depends(read_access),
        from_time: Annotated[datetime | None, Query(alias="from")] = None,
        to_time: Annotated[datetime | None, Query(alias="to")] = None,
        limit: Annotated[int, Query(ge=1, le=4999)] = 1000,
    ) -> VehicleJourneyPublic:
        try:
            value = await service.journey(
                vehicle_id,
                from_time=from_time,
                to_time=to_time,
                limit=limit,
            )
            return VehicleJourneyPublic.from_domain(value)
        except Exception as exc:
            _raise_journey_http(exc)

    return router


def _raise_journey_http(exc: Exception) -> NoReturn:
    if isinstance(exc, IdentityNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, PersistenceError):
        raise HTTPException(status_code=503, detail="journey persistence unavailable") from exc
    raise exc
