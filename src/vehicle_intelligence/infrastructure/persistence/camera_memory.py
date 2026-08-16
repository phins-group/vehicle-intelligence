"""In-memory camera and latest-health repositories for tests/local development."""

from __future__ import annotations

import asyncio

from vehicle_intelligence.application.ports import CameraCreateOutcome
from vehicle_intelligence.domain import Camera, CameraHealth


class InMemoryCameraRepository:
    def __init__(self) -> None:
        self._cameras: dict[str, Camera] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def create(self, camera: Camera) -> bool:
        async with self._lock:
            if camera.id in self._cameras:
                return False
            self._cameras[camera.id] = camera
            return True

    async def create_with_capacity(
        self,
        camera: Camera,
        maximum_cameras: int,
    ) -> CameraCreateOutcome:
        if maximum_cameras < 1:
            raise ValueError("camera capacity must be positive")
        async with self._lock:
            if camera.id in self._cameras:
                return CameraCreateOutcome.CONFLICT
            if len(self._cameras) >= maximum_cameras:
                return CameraCreateOutcome.CAPACITY_REACHED
            self._cameras[camera.id] = camera
            return CameraCreateOutcome.CREATED

    async def replace(self, camera: Camera, expected_revision: int) -> bool:
        if camera.revision != expected_revision + 1:
            raise ValueError("replacement camera revision must increment by one")
        async with self._lock:
            current = self._cameras.get(camera.id)
            if current is None or current.revision != expected_revision:
                return False
            self._cameras[camera.id] = camera
            return True

    async def get(self, camera_id: str) -> Camera | None:
        return self._cameras.get(camera_id)

    async def list(self, enabled_only: bool = False) -> list[Camera]:
        cameras = [
            camera for camera in self._cameras.values() if not enabled_only or camera.enabled
        ]
        return sorted(cameras, key=lambda camera: (camera.name.casefold(), camera.id))

    async def count(self) -> int:
        return len(self._cameras)

    async def delete(self, camera_id: str) -> bool:
        async with self._lock:
            return self._cameras.pop(camera_id, None) is not None

    async def close(self) -> None:
        return None


class InMemoryCameraHealthRepository:
    def __init__(self) -> None:
        self._health: dict[str, CameraHealth] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def save(self, health: CameraHealth) -> None:
        async with self._lock:
            self._health[health.camera_id] = health

    async def get(self, camera_id: str) -> CameraHealth | None:
        return self._health.get(camera_id)

    async def list(self) -> list[CameraHealth]:
        return sorted(self._health.values(), key=lambda health: health.camera_id)

    async def delete(self, camera_id: str) -> None:
        async with self._lock:
            self._health.pop(camera_id, None)

    async def close(self) -> None:
        return None
