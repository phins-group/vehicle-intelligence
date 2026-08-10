"""Camera configuration use cases independent of FastAPI and MongoDB."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum

from vehicle_intelligence.application.ports import (
    CameraConnectionTester,
    CameraConnectionTestResult,
    CameraHealthRepository,
    CameraRepository,
)
from vehicle_intelligence.domain import (
    Camera,
    CameraDirection,
    CameraHealth,
    Direction,
    Point,
    SecretUri,
)
from vehicle_intelligence.exceptions import (
    CameraCapacityError,
    CameraConflictError,
    CameraNotFoundError,
)


@dataclass(frozen=True, slots=True)
class CameraCreate:
    id: str
    name: str
    rtsp_url: str = field(repr=False)
    fps_limit: float
    direction: CameraDirection
    enabled: bool
    vehicle_confidence: float
    plate_confidence: float
    location: str | None = None
    zone: str | None = None
    roi: tuple[Point, ...] | None = None
    crossing_line: tuple[Point, Point] | None = None
    crossing_positive_to_negative: Direction = Direction.ENTER
    finalize_on_crossing: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CameraUpdate:
    revision: int
    name: str
    rtsp_url: str | None = field(repr=False)
    fps_limit: float = 6.0
    direction: CameraDirection = CameraDirection.BOTH
    enabled: bool = True
    vehicle_confidence: float = 0.4
    plate_confidence: float = 0.45
    location: str | None = None
    zone: str | None = None
    roi: tuple[Point, ...] | None = None
    crossing_line: tuple[Point, Point] | None = None
    crossing_positive_to_negative: Direction = Direction.ENTER
    finalize_on_crossing: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class CameraBatchStatus(StrEnum):
    CREATED = "CREATED"
    CONFLICT = "CONFLICT"
    CAPACITY_REACHED = "CAPACITY_REACHED"


@dataclass(frozen=True, slots=True)
class CameraBatchItem:
    camera_id: str
    status: CameraBatchStatus
    camera: Camera | None = None


@dataclass(frozen=True, slots=True)
class CameraBatchResult:
    items: tuple[CameraBatchItem, ...]

    @property
    def created(self) -> tuple[Camera, ...]:
        return tuple(item.camera for item in self.items if item.camera is not None)


class CameraService:
    def __init__(
        self,
        cameras: CameraRepository,
        health: CameraHealthRepository,
        connection_tester: CameraConnectionTester,
        maximum_cameras: int = 256,
        batch_create_limit: int = 50,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if maximum_cameras < 1 or batch_create_limit < 1:
            raise ValueError("camera capacity and batch limit must be positive")
        self._cameras = cameras
        self._health = health
        self._connection_tester = connection_tester
        self._maximum_cameras = maximum_cameras
        self._batch_create_limit = batch_create_limit
        self._clock = clock

    async def initialize(self) -> None:
        await self._cameras.ensure_indexes()
        await self._health.ensure_indexes()

    async def close(self) -> None:
        try:
            await self._cameras.close()
        finally:
            await self._health.close()

    async def create(self, command: CameraCreate) -> Camera:
        camera_id = command.id.strip()
        if await self._cameras.get(camera_id) is not None:
            raise CameraConflictError(f"camera already exists: {camera_id}")
        if len(await self._cameras.list()) >= self._maximum_cameras:
            raise CameraCapacityError(
                f"camera capacity reached: {self._maximum_cameras}"
            )
        now = self._now()
        camera = Camera(
            id=camera_id,
            name=command.name.strip(),
            rtsp_url=SecretUri(command.rtsp_url),
            fps_limit=command.fps_limit,
            direction=command.direction,
            enabled=command.enabled,
            vehicle_confidence=command.vehicle_confidence,
            plate_confidence=command.plate_confidence,
            location=command.location,
            zone=command.zone,
            roi=command.roi,
            crossing_line=command.crossing_line,
            crossing_positive_to_negative=command.crossing_positive_to_negative,
            finalize_on_crossing=command.finalize_on_crossing,
            metadata=dict(command.metadata),
            created_at=now,
            updated_at=now,
        )
        if not await self._cameras.create(camera):
            raise CameraConflictError(f"camera already exists: {camera.id}")
        return camera

    async def create_many(self, commands: tuple[CameraCreate, ...]) -> CameraBatchResult:
        if not commands:
            raise ValueError("camera batch cannot be empty")
        if len(commands) > self._batch_create_limit:
            raise ValueError(
                f"camera batch exceeds configured limit: {self._batch_create_limit}"
            )
        ids = [command.id.strip() for command in commands]
        if len(ids) != len(set(ids)):
            raise ValueError("camera batch IDs must be unique")

        items: list[CameraBatchItem] = []
        for command in commands:
            try:
                camera = await self.create(command)
            except CameraCapacityError:
                items.append(
                    CameraBatchItem(command.id.strip(), CameraBatchStatus.CAPACITY_REACHED)
                )
            except CameraConflictError:
                items.append(CameraBatchItem(command.id.strip(), CameraBatchStatus.CONFLICT))
            else:
                items.append(
                    CameraBatchItem(camera.id, CameraBatchStatus.CREATED, camera=camera)
                )
        return CameraBatchResult(tuple(items))

    async def update(self, camera_id: str, command: CameraUpdate) -> Camera:
        current = await self._required(camera_id)
        if current.revision != command.revision:
            raise CameraConflictError(
                f"camera revision conflict: expected {command.revision}, current {current.revision}"
            )
        updated = Camera(
            id=current.id,
            name=command.name.strip(),
            rtsp_url=(
                SecretUri(command.rtsp_url)
                if command.rtsp_url is not None
                else current.rtsp_url
            ),
            fps_limit=command.fps_limit,
            direction=command.direction,
            enabled=command.enabled,
            vehicle_confidence=command.vehicle_confidence,
            plate_confidence=command.plate_confidence,
            location=command.location,
            zone=command.zone,
            roi=command.roi,
            crossing_line=command.crossing_line,
            crossing_positive_to_negative=command.crossing_positive_to_negative,
            finalize_on_crossing=command.finalize_on_crossing,
            metadata=dict(command.metadata),
            schema_version=current.schema_version,
            revision=current.revision + 1,
            created_at=current.created_at,
            updated_at=self._now(),
        )
        if not await self._cameras.replace(updated, current.revision):
            raise CameraConflictError(f"camera was concurrently updated: {camera_id}")
        return updated

    async def get(self, camera_id: str) -> Camera:
        return await self._required(camera_id)

    async def list(self, enabled_only: bool = False) -> list[Camera]:
        return await self._cameras.list(enabled_only)

    async def delete(self, camera_id: str) -> None:
        if not await self._cameras.delete(camera_id):
            raise CameraNotFoundError(f"camera not found: {camera_id}")
        await self._health.delete(camera_id)

    async def set_enabled(self, camera_id: str, enabled: bool) -> Camera:
        current = await self._required(camera_id)
        if current.enabled == enabled:
            return current
        updated = replace(
            current,
            enabled=enabled,
            revision=current.revision + 1,
            updated_at=self._now(),
        )
        if not await self._cameras.replace(updated, current.revision):
            raise CameraConflictError(f"camera was concurrently updated: {camera_id}")
        return updated

    async def test_connection(self, camera_id: str) -> CameraConnectionTestResult:
        return await self._connection_tester.test(await self._required(camera_id))

    async def get_health(self, camera_id: str) -> CameraHealth | None:
        await self._required(camera_id)
        return await self._health.get(camera_id)

    async def list_health(self) -> list[CameraHealth]:
        return await self._health.list()

    async def _required(self, camera_id: str) -> Camera:
        camera = await self._cameras.get(camera_id)
        if camera is None:
            raise CameraNotFoundError(f"camera not found: {camera_id}")
        return camera

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("camera service clock must be timezone-aware")
        return value.astimezone(UTC)
