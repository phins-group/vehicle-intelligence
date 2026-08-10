"""In-memory directed camera-topology repository."""

from __future__ import annotations

import asyncio

from vehicle_intelligence.domain import CameraTopologyEdge


class InMemoryCameraTopologyRepository:
    def __init__(self) -> None:
        self._edges: dict[str, CameraTopologyEdge] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def create(self, edge: CameraTopologyEdge) -> bool:
        async with self._lock:
            if edge.id in self._edges or any(
                existing.from_camera_id == edge.from_camera_id
                and existing.to_camera_id == edge.to_camera_id
                for existing in self._edges.values()
            ):
                return False
            self._edges[edge.id] = edge
            return True

    async def replace(
        self,
        edge: CameraTopologyEdge,
        expected_revision: int,
    ) -> bool:
        if edge.revision != expected_revision + 1:
            raise ValueError("replacement topology revision must increment by one")
        async with self._lock:
            current = self._edges.get(edge.id)
            if current is None or current.revision != expected_revision:
                return False
            if any(
                existing.id != edge.id
                and existing.from_camera_id == edge.from_camera_id
                and existing.to_camera_id == edge.to_camera_id
                for existing in self._edges.values()
            ):
                return False
            self._edges[edge.id] = edge
            return True

    async def get(self, edge_id: str) -> CameraTopologyEdge | None:
        return self._edges.get(edge_id)

    async def list(
        self,
        *,
        from_camera_id: str | None = None,
        to_camera_id: str | None = None,
        enabled_only: bool = False,
        limit: int = 200,
    ) -> tuple[CameraTopologyEdge, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("topology list limit must be in [1, 1000]")
        values = [
            edge
            for edge in self._edges.values()
            if (from_camera_id is None or edge.from_camera_id == from_camera_id)
            and (to_camera_id is None or edge.to_camera_id == to_camera_id)
            and (not enabled_only or edge.enabled)
        ]
        values.sort(key=lambda edge: (edge.from_camera_id, edge.to_camera_id, edge.id))
        return tuple(values[:limit])

    async def delete(self, edge_id: str) -> bool:
        async with self._lock:
            return self._edges.pop(edge_id, None) is not None

    async def close(self) -> None:
        return None
