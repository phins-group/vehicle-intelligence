"""Fan-out repository for local JSONL plus optional MongoDB persistence."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from vehicle_intelligence.application.ports import (
    EventPage,
    EventQuery,
    VehicleEventRepository,
)
from vehicle_intelligence.domain import VehicleEvent


class CompositeVehicleEventRepository:
    def __init__(self, repositories: Sequence[VehicleEventRepository]) -> None:
        if not repositories:
            raise ValueError("composite repository requires at least one target")
        self._repositories = tuple(repositories)

    async def ensure_indexes(self) -> None:
        await asyncio.gather(*(item.ensure_indexes() for item in self._repositories))

    async def save(self, event: VehicleEvent) -> bool:
        results = await asyncio.gather(*(item.save(event) for item in self._repositories))
        return any(results)

    async def get(self, event_id: str) -> VehicleEvent | None:
        return await self._repositories[0].get(event_id)

    async def list(self, query: EventQuery) -> EventPage:
        return await self._repositories[0].list(query)

    async def find_by_plate(self, plate: str, limit: int) -> list[VehicleEvent]:
        return await self._repositories[0].find_by_plate(plate, limit)

    async def update_plate_review(
        self,
        event: VehicleEvent,
        expected_revision: int,
    ) -> VehicleEvent | None:
        results = await asyncio.gather(
            *(item.update_plate_review(event, expected_revision) for item in self._repositories)
        )
        return event if results and all(result is not None for result in results) else None

    async def close(self) -> None:
        await asyncio.gather(*(item.close() for item in self._repositories))
