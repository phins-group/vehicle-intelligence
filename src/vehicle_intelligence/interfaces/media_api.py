"""RBAC-protected, event-scoped media access API."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field

from vehicle_intelligence.application.media_access import (
    EventMediaAccess,
    SignedMediaAsset,
    VehicleEventMediaService,
)
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import Principal
from vehicle_intelligence.exceptions import (
    MediaAccessError,
    VehicleEventNotFoundError,
)
from vehicle_intelligence.interfaces.security import APISecurity


class MediaAssetPublic(BaseModel):
    key: str
    url: str | None
    content_type: str = Field(serialization_alias="contentType")
    status: Literal["AVAILABLE", "MISSING"]

    @classmethod
    def from_domain(cls, asset: SignedMediaAsset) -> MediaAssetPublic:
        return cls(
            key=asset.key,
            url=asset.url,
            content_type=asset.content_type,
            status=asset.status,
        )


class MediaSlotsPublic(BaseModel):
    snapshot: MediaAssetPublic | None
    vehicle_crop: MediaAssetPublic | None = Field(serialization_alias="vehicleCrop")
    plate_crop: MediaAssetPublic | None = Field(serialization_alias="plateCrop")
    clip: MediaAssetPublic | None


class EventMediaPublic(BaseModel):
    event_id: str = Field(serialization_alias="eventId")
    expires_at: str = Field(serialization_alias="expiresAt")
    media: MediaSlotsPublic

    @classmethod
    def from_domain(cls, access: EventMediaAccess) -> EventMediaPublic:
        def public(asset: SignedMediaAsset | None) -> MediaAssetPublic | None:
            return MediaAssetPublic.from_domain(asset) if asset is not None else None

        return cls(
            event_id=access.event_id,
            expires_at=access.expires_at.isoformat().replace("+00:00", "Z"),
            media=MediaSlotsPublic(
                snapshot=public(access.snapshot),
                vehicle_crop=public(access.vehicle_crop),
                plate_crop=public(access.plate_crop),
                clip=public(access.clip),
            ),
        )


def build_media_router(
    service: VehicleEventMediaService | None,
    security: APISecurity,
) -> APIRouter:
    router = APIRouter(prefix="/api/events", tags=["media"])
    read_access = security.require(Permission.READ_PLATFORM)

    @router.get("/{event_id}/media", response_model=EventMediaPublic)
    async def event_media(
        event_id: str,
        response: Response,
        _principal: Principal = Depends(read_access),
    ) -> EventMediaPublic:
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Pragma"] = "no-cache"
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="media access requires configured object storage",
            )
        try:
            return EventMediaPublic.from_domain(await service.resolve(event_id))
        except VehicleEventNotFoundError as exc:
            raise HTTPException(status_code=404, detail="vehicle event not found") from exc
        except MediaAccessError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="media evidence is temporarily unavailable",
            ) from exc

    return router
