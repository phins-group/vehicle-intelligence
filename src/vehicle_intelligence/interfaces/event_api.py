"""Read-only event search and vehicle identity routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from vehicle_intelligence.application.journeys import VehicleJourneyService
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.ports import (
    EventQuery,
    VehicleEventRepository,
    VehicleIdentityRepository,
)
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import Principal
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.identity_serialization import (
    fingerprint_to_jsonable,
    identity_to_jsonable,
)
from vehicle_intelligence.infrastructure.serialization import event_to_jsonable
from vehicle_intelligence.interfaces.security import APISecurity


def build_event_router(
    events: VehicleEventRepository,
    identities: VehicleIdentityRepository,
    journeys: VehicleJourneyService,
    normalizer: VietnamPlateNormalizer,
    security: APISecurity,
) -> APIRouter:
    """Build event and vehicle read routes without coupling them to app composition."""

    router = APIRouter()
    read_access = security.require(Permission.READ_PLATFORM)

    @router.get("/api/events")
    async def list_events(
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: str | None = None,
        camera_id: Annotated[str | None, Query(alias="cameraId")] = None,
        plate: str | None = None,
        event_type: Annotated[str | None, Query(alias="eventType")] = None,
        direction: str | None = None,
        status: str | None = None,
        from_time: Annotated[datetime | None, Query(alias="from")] = None,
        to_time: Annotated[datetime | None, Query(alias="to")] = None,
    ) -> dict[str, object]:
        canonical = _canonical_plate(normalizer, plate)
        _validate_aware(from_time, "from")
        _validate_aware(to_time, "to")
        try:
            page = await events.list(
                EventQuery(
                    limit=limit,
                    cursor=cursor,
                    camera_id=camera_id,
                    plate=canonical,
                    event_type=event_type,
                    direction=direction,
                    status=status,
                    from_time=from_time,
                    to_time=to_time,
                )
            )
        except PersistenceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "items": [event_to_jsonable(event) for event in page.items],
            "nextCursor": page.next_cursor,
        }

    @router.get("/api/events/{event_id}")
    async def get_event(
        event_id: str,
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        event = await events.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="vehicle event not found")
        return event_to_jsonable(event)

    @router.get("/api/vehicles/search")
    async def search_vehicles(
        plate: Annotated[str, Query(min_length=4)],
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        cursor: str | None = None,
    ) -> dict[str, object]:
        canonical = _canonical_plate(normalizer, plate)
        if canonical is None:
            raise HTTPException(status_code=422, detail="plate is required")
        try:
            page = await events.list(EventQuery(limit=limit, cursor=cursor, plate=canonical))
        except PersistenceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "query": canonical,
            "items": [event_to_jsonable(event) for event in page.items],
            "nextCursor": page.next_cursor,
        }

    @router.get("/api/vehicles/{vehicle_id}")
    async def get_vehicle_identity(
        vehicle_id: str,
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        identity = await identities.get(vehicle_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="vehicle identity not found")
        result = identity_to_jsonable(identity)
        latest = await journeys.latest(vehicle_id)
        result["latestEvent"] = event_to_jsonable(latest) if latest is not None else None
        return result

    @router.get("/api/vehicles/{vehicle_id}/fingerprints")
    async def list_vehicle_fingerprints(
        vehicle_id: str,
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, object]:
        identity = await identities.get(vehicle_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="vehicle identity not found")
        fingerprints = await identities.list_fingerprints(vehicle_id, limit)
        return {
            "vehicleId": vehicle_id,
            "items": [fingerprint_to_jsonable(item) for item in fingerprints],
        }

    return router


def _canonical_plate(normalizer: VietnamPlateNormalizer, plate: str | None) -> str | None:
    if plate is None:
        return None
    normalized = normalizer.normalize(plate)
    if not normalized.valid or normalized.normalized is None:
        raise HTTPException(status_code=422, detail="invalid Vietnamese plate format")
    return normalized.normalized


def _validate_aware(value: datetime | None, field: str) -> None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(status_code=422, detail=f"{field} timestamp must include a timezone")
