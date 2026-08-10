"""Directed camera topology and bounded travel-time candidate generation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from vehicle_intelligence.application.ports import (
    CameraTopologyRepository,
    VehicleIdentityRepository,
)
from vehicle_intelligence.config import IdentityConfig
from vehicle_intelligence.domain import (
    CameraTopologyEdge,
    CrossCameraCandidate,
)
from vehicle_intelligence.exceptions import TopologyConflictError, TopologyNotFoundError


@dataclass(frozen=True, slots=True)
class TopologyCreate:
    id: str
    from_camera_id: str
    to_camera_id: str
    minimum_travel_seconds: float
    maximum_travel_seconds: float
    typical_travel_seconds: float
    enabled: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TopologyUpdate:
    revision: int
    from_camera_id: str
    to_camera_id: str
    minimum_travel_seconds: float
    maximum_travel_seconds: float
    typical_travel_seconds: float
    enabled: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


class CameraTopologyService:
    def __init__(
        self,
        repository: CameraTopologyRepository,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._clock = clock

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def close(self) -> None:
        await self._repository.close()

    async def create(self, command: TopologyCreate) -> CameraTopologyEdge:
        now = self._now()
        edge = CameraTopologyEdge(
            id=command.id.strip(),
            from_camera_id=command.from_camera_id.strip(),
            to_camera_id=command.to_camera_id.strip(),
            minimum_travel_seconds=command.minimum_travel_seconds,
            maximum_travel_seconds=command.maximum_travel_seconds,
            typical_travel_seconds=command.typical_travel_seconds,
            enabled=command.enabled,
            metadata=dict(command.metadata),
            created_at=now,
            updated_at=now,
        )
        if not await self._repository.create(edge):
            raise TopologyConflictError(f"topology edge already exists: {edge.id}")
        return edge

    async def update(self, edge_id: str, command: TopologyUpdate) -> CameraTopologyEdge:
        current = await self.get(edge_id)
        if current.revision != command.revision:
            raise TopologyConflictError(
                f"topology revision conflict: expected {command.revision}, "
                f"current {current.revision}"
            )
        updated = CameraTopologyEdge(
            id=current.id,
            from_camera_id=command.from_camera_id.strip(),
            to_camera_id=command.to_camera_id.strip(),
            minimum_travel_seconds=command.minimum_travel_seconds,
            maximum_travel_seconds=command.maximum_travel_seconds,
            typical_travel_seconds=command.typical_travel_seconds,
            enabled=command.enabled,
            metadata=dict(command.metadata),
            created_at=current.created_at,
            updated_at=self._now(),
            revision=current.revision + 1,
            schema_version=current.schema_version,
        )
        if not await self._repository.replace(updated, current.revision):
            raise TopologyConflictError(f"topology was concurrently updated: {edge_id}")
        return updated

    async def get(self, edge_id: str) -> CameraTopologyEdge:
        edge = await self._repository.get(edge_id)
        if edge is None:
            raise TopologyNotFoundError(f"topology edge not found: {edge_id}")
        return edge

    async def list(
        self,
        *,
        from_camera_id: str | None = None,
        to_camera_id: str | None = None,
        enabled_only: bool = False,
        limit: int = 200,
    ) -> tuple[CameraTopologyEdge, ...]:
        return await self._repository.list(
            from_camera_id=from_camera_id,
            to_camera_id=to_camera_id,
            enabled_only=enabled_only,
            limit=limit,
        )

    async def delete(self, edge_id: str) -> None:
        if not await self._repository.delete(edge_id):
            raise TopologyNotFoundError(f"topology edge not found: {edge_id}")

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("topology service clock must be timezone-aware")
        return value.astimezone(UTC)


class CrossCameraCandidateGenerator:
    """Find only feasible prior observations on explicitly connected cameras."""

    def __init__(
        self,
        identities: VehicleIdentityRepository,
        topology: CameraTopologyRepository,
        config: IdentityConfig,
    ) -> None:
        self._identities = identities
        self._topology = topology
        self._edge_limit = config.topology_edge_limit
        self._per_edge_limit = config.candidates_per_edge
        self._candidate_limit = config.cross_camera_candidate_limit

    async def generate(
        self,
        fingerprint_id: str,
        limit: int | None = None,
    ) -> tuple[CrossCameraCandidate, ...]:
        requested = self._candidate_limit if limit is None else limit
        if not 1 <= requested <= self._candidate_limit:
            raise ValueError(
                f"candidate limit must be in [1, {self._candidate_limit}]"
            )
        source = await self._identities.get_fingerprint(fingerprint_id)
        if source is None:
            raise TopologyNotFoundError(f"vehicle fingerprint not found: {fingerprint_id}")
        edges = await self._topology.list(
            to_camera_id=source.camera_id,
            enabled_only=True,
            limit=self._edge_limit,
        )
        candidates: list[CrossCameraCandidate] = []
        for edge in edges:
            earliest = source.observed_at - timedelta(
                seconds=edge.maximum_travel_seconds
            )
            latest = source.observed_at - timedelta(
                seconds=edge.minimum_travel_seconds
            )
            observations = await self._identities.find_fingerprints_by_camera_time(
                edge.from_camera_id,
                earliest,
                latest,
                self._per_edge_limit,
            )
            for observation in observations:
                if (
                    observation.id == source.id
                    or observation.vehicle_id == source.vehicle_id
                ):
                    continue
                travel_seconds = (
                    source.observed_at - observation.observed_at
                ).total_seconds()
                candidates.append(
                    CrossCameraCandidate(
                        fingerprint_id=observation.id,
                        vehicle_id=observation.vehicle_id,
                        camera_id=observation.camera_id,
                        observed_at=observation.observed_at,
                        topology_edge_id=edge.id,
                        travel_seconds=travel_seconds,
                        time_score=_time_score(edge, travel_seconds),
                    )
                )
        candidates.sort(
            key=lambda item: (item.time_score, item.observed_at, item.fingerprint_id),
            reverse=True,
        )
        return tuple(candidates[:requested])


def _time_score(edge: CameraTopologyEdge, travel_seconds: float) -> float:
    if not edge.minimum_travel_seconds <= travel_seconds <= edge.maximum_travel_seconds:
        return 0.0
    if travel_seconds == edge.typical_travel_seconds:
        return 1.0
    if travel_seconds < edge.typical_travel_seconds:
        span = edge.typical_travel_seconds - edge.minimum_travel_seconds
    else:
        span = edge.maximum_travel_seconds - edge.typical_travel_seconds
    if span <= 0:
        return 1.0
    score = 1.0 - abs(travel_seconds - edge.typical_travel_seconds) / span
    return min(1.0, max(0.0, score))
