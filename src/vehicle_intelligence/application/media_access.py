"""Event-scoped, short-lived access to persisted media evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

from vehicle_intelligence.application.ports import MediaUrlSigner, VehicleEventRepository
from vehicle_intelligence.exceptions import (
    MediaAccessError,
    VehicleEventNotFoundError,
)


@dataclass(frozen=True, slots=True)
class SignedMediaAsset:
    key: str
    url: str | None
    content_type: str

    @property
    def status(self) -> str:
        return "AVAILABLE" if self.url is not None else "MISSING"


@dataclass(frozen=True, slots=True)
class EventMediaAccess:
    event_id: str
    expires_at: datetime
    snapshot: SignedMediaAsset | None
    vehicle_crop: SignedMediaAsset | None
    plate_crop: SignedMediaAsset | None
    clip: SignedMediaAsset | None


class VehicleEventMediaService:
    def __init__(
        self,
        repository: VehicleEventRepository,
        signer: MediaUrlSigner,
        ttl_seconds: int,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("media URL TTL must be positive")
        self._repository = repository
        self._signer = signer
        self._ttl = timedelta(seconds=ttl_seconds)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def resolve(self, event_id: str) -> EventMediaAccess:
        event = await self._repository.get(event_id)
        if event is None:
            raise VehicleEventNotFoundError(f"vehicle event not found: {event_id}")

        now = self._clock()
        if now.tzinfo is None:
            raise MediaAccessError("media access clock must be timezone-aware")

        requests = (
            (event.media.snapshot_key, "image/jpeg"),
            (event.media.vehicle_crop_key, "image/jpeg"),
            (event.media.plate_crop_key, "image/jpeg"),
            (event.media.clip_key, "video/mp4"),
        )
        for key, _content_type in requests:
            if key is not None:
                self._validate_key(key)
        assets = await asyncio.gather(
            *(self._resolve_asset(key, content_type) for key, content_type in requests)
        )
        return EventMediaAccess(
            event_id=event.id,
            expires_at=now.astimezone(UTC) + self._ttl,
            snapshot=assets[0],
            vehicle_crop=assets[1],
            plate_crop=assets[2],
            clip=assets[3],
        )

    async def _resolve_asset(
        self,
        key: str | None,
        content_type: str,
    ) -> SignedMediaAsset | None:
        if key is None:
            return None
        try:
            url = await self._signer.presign_get(key, self._ttl)
        except MediaAccessError:
            raise
        except Exception as exc:
            raise MediaAccessError(f"cannot authorize media object: {key}") from exc
        return SignedMediaAsset(key=key, url=url, content_type=content_type)

    @staticmethod
    def _validate_key(key: str) -> None:
        path = PurePosixPath(key)
        if (
            not key
            or not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "//" in key
            or "\\" in key
            or "\x00" in key
            or any(not part or part == "." for part in path.parts)
        ):
            raise MediaAccessError("vehicle event contains an unsafe media key")
