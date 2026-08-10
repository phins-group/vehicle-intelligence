"""In-process event publisher for local development and Phase 1."""

from __future__ import annotations

from vehicle_intelligence.application.ports import VehicleEventRepository
from vehicle_intelligence.domain import VehicleEvent


class RepositoryEventPublisher:
    """Publish directly to a repository while preserving the publisher boundary."""

    def __init__(self, repository: VehicleEventRepository) -> None:
        self._repository = repository

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def publish(self, event: VehicleEvent) -> bool:
        return await self._repository.save(event)

    async def close(self) -> None:
        await self._repository.close()
