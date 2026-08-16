import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vehicle_intelligence.application.cameras import (
    CameraBatchStatus,
    CameraCreate,
    CameraService,
    CameraUpdate,
)
from vehicle_intelligence.application.ports import CameraConnectionTestResult
from vehicle_intelligence.domain import Camera, CameraDirection, CameraHealth, CameraStatus, Point
from vehicle_intelligence.exceptions import (
    CameraCapacityError,
    CameraConflictError,
    CameraNotFoundError,
)
from vehicle_intelligence.infrastructure.persistence.camera_memory import (
    InMemoryCameraHealthRepository,
    InMemoryCameraRepository,
)
from vehicle_intelligence.interfaces.camera_schemas import CameraGeometryInput


class FakeConnectionTester:
    def __init__(self) -> None:
        self.urls: list[str] = []

    async def test(self, camera):
        self.urls.append(camera.rtsp_url.reveal())
        return CameraConnectionTestResult(True, 12.5, camera.updated_at)


class AtomicCreateCameraRepository(InMemoryCameraRepository):
    async def get(self, camera_id: str) -> Camera | None:
        del camera_id
        raise AssertionError("camera creation must not preflight a separate get")

    async def list(self, enabled_only: bool = False) -> list[Camera]:
        del enabled_only
        raise AssertionError("camera creation must not list cameras")

    async def count(self) -> int:
        raise AssertionError("camera creation must not preflight a separate count")


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_camera_geometry_rejects_non_finite_coordinates(coordinate: float) -> None:
    with pytest.raises(ValidationError):
        CameraGeometryInput.model_validate({"vehicleRoi": [[coordinate, 0], [100, 0], [100, 100]]})
    with pytest.raises(ValueError, match="coordinates must be finite"):
        Point(coordinate, 0)


def create_command(camera_id: str = "gate-01") -> CameraCreate:
    return CameraCreate(
        id=camera_id,
        name="Main Gate",
        rtsp_url="rtsp://admin:secret@camera.example/live",
        fps_limit=6,
        direction=CameraDirection.BOTH,
        enabled=True,
        vehicle_confidence=0.4,
        plate_confidence=0.45,
    )


async def test_camera_service_crud_revision_health_and_connection() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    tester = FakeConnectionTester()
    now = datetime(2026, 8, 9, 9, 0, tzinfo=UTC)
    service = CameraService(cameras, health, tester, clock=lambda: now)
    await service.initialize()

    created = await service.create(create_command())
    assert created.revision == 1
    assert [item.id for item in await service.list()] == ["gate-01"]
    with pytest.raises(CameraConflictError):
        await service.create(create_command())

    updated = await service.update(
        created.id,
        CameraUpdate(
            revision=created.revision,
            name="Renamed Gate",
            rtsp_url=None,
            fps_limit=8,
            direction=CameraDirection.ENTRY,
            enabled=True,
            vehicle_confidence=0.5,
            plate_confidence=0.55,
        ),
    )
    assert updated.revision == 2
    assert updated.rtsp_url.reveal() == created.rtsp_url.reveal()
    with pytest.raises(ValueError, match="RTSP URL"):
        await service.update(
            created.id,
            CameraUpdate(
                revision=updated.revision,
                name="Empty credential",
                rtsp_url="",
            ),
        )
    with pytest.raises(CameraConflictError, match="revision conflict"):
        await service.update(
            created.id,
            CameraUpdate(
                revision=1,
                name="Stale",
                rtsp_url=None,
            ),
        )

    disabled = await service.set_enabled(created.id, False)
    assert not disabled.enabled
    assert await service.list(enabled_only=True) == []
    result = await service.test_connection(created.id)
    assert result.connected
    assert tester.urls == ["rtsp://admin:secret@camera.example/live"]

    await health.save(
        CameraHealth(
            camera_id=created.id,
            status=CameraStatus.OFFLINE,
            source_fps=0,
            decode_fps=0,
            queue_size=0,
            dropped_frames=0,
            reconnect_count=0,
            connection_failures=1,
            stream_epoch=0,
            last_frame_at=None,
            updated_at=now,
        )
    )
    assert (await service.get_health(created.id)).status is CameraStatus.OFFLINE

    await service.delete(created.id)
    assert await health.get(created.id) is None
    with pytest.raises(CameraNotFoundError):
        await service.get(created.id)
    await service.close()


async def test_camera_batch_is_bounded_partial_and_capacity_aware() -> None:
    service = CameraService(
        InMemoryCameraRepository(),
        InMemoryCameraHealthRepository(),
        FakeConnectionTester(),
        maximum_cameras=2,
        batch_create_limit=3,
    )
    await service.create(create_command("existing"))

    result = await service.create_many(
        (
            create_command("existing"),
            create_command("new-camera"),
            create_command("over-capacity"),
        )
    )

    assert [item.status for item in result.items] == [
        CameraBatchStatus.CONFLICT,
        CameraBatchStatus.CREATED,
        CameraBatchStatus.CAPACITY_REACHED,
    ]
    assert [camera.id for camera in result.created] == ["new-camera"]

    with pytest.raises(ValueError, match="IDs must be unique"):
        await service.create_many((create_command("same"), create_command("same")))
    with pytest.raises(ValueError, match="exceeds configured limit"):
        await service.create_many(tuple(create_command(f"gate-{i}") for i in range(4)))


async def test_camera_creation_uses_one_atomic_repository_operation() -> None:
    cameras = AtomicCreateCameraRepository()
    service = CameraService(
        cameras,
        InMemoryCameraHealthRepository(),
        FakeConnectionTester(),
        maximum_cameras=1,
    )

    created = await service.create(create_command())
    assert created.id == "gate-01"
    assert await InMemoryCameraRepository.count(cameras) == 1

    with pytest.raises(CameraCapacityError, match="capacity reached"):
        await service.create(create_command("gate-02"))


async def test_concurrent_camera_creation_cannot_exceed_capacity() -> None:
    cameras = InMemoryCameraRepository()
    service = CameraService(
        cameras,
        InMemoryCameraHealthRepository(),
        FakeConnectionTester(),
        maximum_cameras=1,
    )

    results = await asyncio.gather(
        service.create(create_command("concurrent-a")),
        service.create(create_command("concurrent-b")),
        return_exceptions=True,
    )

    assert sum(isinstance(result, Camera) for result in results) == 1
    assert sum(isinstance(result, CameraCapacityError) for result in results) == 1
    assert await cameras.count() == 1
