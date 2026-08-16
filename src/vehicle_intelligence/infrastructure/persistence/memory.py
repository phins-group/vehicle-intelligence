"""Deterministic repository for tests and embedded API use."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from vehicle_intelligence.application.ports import EventPage, EventQuery
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor


class InMemoryVehicleEventRepository:
    def __init__(self) -> None:
        self._events: dict[str, VehicleEvent] = {}
        self._semantic_keys: set[tuple[str, str, str]] = set()
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def save(self, event: VehicleEvent) -> bool:
        key = self._semantic_key(event)
        async with self._lock:
            if event.id in self._events or key in self._semantic_keys:
                return False
            self._events[event.id] = event
            self._semantic_keys.add(key)
            return True

    async def conflicts_with(self, event: VehicleEvent) -> bool:
        """Return whether the ID or idempotency key is already indexed."""
        key = self._semantic_key(event)
        async with self._lock:
            return event.id in self._events or key in self._semantic_keys

    async def get(self, event_id: str) -> VehicleEvent | None:
        return self._events.get(event_id)

    async def list(self, query: EventQuery) -> EventPage:
        items = [event for event in self._events.values() if self._matches(event, query)]
        items.sort(key=lambda event: (event.occurred_at, event.id), reverse=True)
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            items = [
                event for event in items if (event.occurred_at, event.id) < (cursor_time, cursor_id)
            ]
        page = items[: query.limit + 1]
        has_more = len(page) > query.limit
        page = page[: query.limit]
        next_cursor = (
            encode_cursor(page[-1].occurred_at, page[-1].id) if has_more and page else None
        )
        return EventPage(tuple(page), next_cursor)

    async def find_by_plate(self, plate: str, limit: int) -> list[VehicleEvent]:
        items = [
            event
            for event in self._events.values()
            if event.plate is not None and event.plate.final_normalized == plate
        ]
        items.sort(key=lambda event: (event.occurred_at, event.id), reverse=True)
        return items[:limit]

    async def timeline(
        self,
        vehicle_id: str,
        *,
        from_time=None,
        to_time=None,
        limit: int = 1000,
        ascending: bool = True,
    ) -> tuple[VehicleEvent, ...]:
        if not 1 <= limit <= 5000:
            raise ValueError("vehicle timeline limit must be in [1, 5000]")
        items = [
            event
            for event in self._events.values()
            if event.vehicle_id == vehicle_id
            and (from_time is None or event.occurred_at >= from_time)
            and (to_time is None or event.occurred_at <= to_time)
        ]
        items.sort(
            key=lambda event: (event.occurred_at, event.id),
            reverse=not ascending,
        )
        return tuple(items[:limit])

    async def update_plate_review(
        self,
        event: VehicleEvent,
        expected_revision: int,
    ) -> VehicleEvent | None:
        if event.plate is None or event.plate.review_revision != expected_revision + 1:
            raise ValueError("plate review revision must increment by one")
        async with self._lock:
            current = self._events.get(event.id)
            if (
                current is None
                or current.plate is None
                or current.plate.review_revision != expected_revision
            ):
                return None
            self._events[event.id] = event
            return event

    async def assign_vehicle_id(self, event_id: str, vehicle_id: str) -> bool:
        async with self._lock:
            current = self._events.get(event_id)
            if current is None or current.vehicle_id not in {None, vehicle_id}:
                return False
            self._events[event_id] = replace(current, vehicle_id=vehicle_id)
            return True

    async def reassign_vehicle_ids(
        self,
        event_ids: tuple[str, ...],
        source_vehicle_id: str,
        target_vehicle_id: str,
    ) -> int:
        async with self._lock:
            moved = 0
            for event_id in dict.fromkeys(event_ids):
                current = self._events.get(event_id)
                if current is None or current.vehicle_id != source_vehicle_id:
                    continue
                self._events[event_id] = replace(current, vehicle_id=target_vehicle_id)
                moved += 1
            return moved

    async def close(self) -> None:
        return None

    async def snapshot(self) -> tuple[VehicleEvent, ...]:
        async with self._lock:
            return tuple(self._events.values())

    @staticmethod
    def _semantic_key(event: VehicleEvent) -> tuple[str, str, str]:
        return (event.camera.id, event.track_id, event.event_type.value)

    @staticmethod
    def _matches(event: VehicleEvent, query: EventQuery) -> bool:
        return not any(
            (
                query.camera_id and event.camera.id != query.camera_id,
                query.plate
                and (event.plate is None or event.plate.final_normalized != query.plate),
                query.event_type and event.event_type.value != query.event_type,
                query.direction and event.direction.value != query.direction,
                query.status and event.status.value != query.status,
                query.from_time and event.occurred_at < query.from_time,
                query.to_time and event.occurred_at > query.to_time,
            )
        )
