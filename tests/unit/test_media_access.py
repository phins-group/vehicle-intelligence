from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.media_access import VehicleEventMediaService
from vehicle_intelligence.domain import MediaReferences
from vehicle_intelligence.exceptions import MediaAccessError, VehicleEventNotFoundError
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)


class FakeMediaUrlSigner:
    def __init__(self, urls: dict[str, str | None]) -> None:
        self.urls = urls
        self.calls: list[tuple[str, timedelta]] = []

    async def presign_get(self, key: str, expires: timedelta) -> str | None:
        self.calls.append((key, expires))
        return self.urls.get(key)


async def test_media_access_resolves_available_and_missing_event_assets(sample_event) -> None:
    repository = InMemoryVehicleEventRepository()
    event = replace(
        sample_event,
        media=MediaReferences(
            snapshot_key="vehicles/test/snapshot.jpg",
            vehicle_crop_key="vehicles/test/vehicle.jpg",
            plate_crop_key="vehicles/test/plate.jpg",
            clip_key="vehicles/test/event.mp4",
        ),
    )
    await repository.save(event)
    signer = FakeMediaUrlSigner(
        {
            "vehicles/test/snapshot.jpg": "https://media.example/snapshot?signature=one",
            "vehicles/test/vehicle.jpg": None,
            "vehicles/test/plate.jpg": "https://media.example/plate?signature=two",
            "vehicles/test/event.mp4": "https://media.example/clip?signature=three",
        }
    )
    now = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
    service = VehicleEventMediaService(repository, signer, 300, clock=lambda: now)

    access = await service.resolve(event.id)

    assert access.event_id == event.id
    assert access.expires_at == now + timedelta(seconds=300)
    assert access.snapshot is not None and access.snapshot.status == "AVAILABLE"
    assert access.vehicle_crop is not None and access.vehicle_crop.status == "MISSING"
    assert access.plate_crop is not None and access.plate_crop.content_type == "image/jpeg"
    assert access.clip is not None and access.clip.content_type == "video/mp4"
    assert {key for key, _ in signer.calls} == {
        "vehicles/test/snapshot.jpg",
        "vehicles/test/vehicle.jpg",
        "vehicles/test/plate.jpg",
        "vehicles/test/event.mp4",
    }
    assert all(expires == timedelta(seconds=300) for _, expires in signer.calls)


async def test_media_access_rejects_unknown_event_and_unsafe_key(sample_event) -> None:
    repository = InMemoryVehicleEventRepository()
    signer = FakeMediaUrlSigner({})
    service = VehicleEventMediaService(repository, signer, 60)

    with pytest.raises(VehicleEventNotFoundError):
        await service.resolve("evt-missing")

    unsafe = replace(
        sample_event,
        id="evt-unsafe-media",
        track_id="gate-01:unsafe-media",
        media=MediaReferences(
            snapshot_key="vehicles/test/snapshot.jpg",
            plate_crop_key="../private/secret.jpg",
        ),
    )
    await repository.save(unsafe)
    with pytest.raises(MediaAccessError, match="unsafe media key"):
        await service.resolve(unsafe.id)
    assert signer.calls == []


def test_media_access_requires_positive_ttl() -> None:
    repository = InMemoryVehicleEventRepository()
    signer = FakeMediaUrlSigner({})

    with pytest.raises(ValueError, match="TTL must be positive"):
        VehicleEventMediaService(repository, signer, 0)
