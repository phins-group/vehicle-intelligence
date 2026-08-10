"""Local durable JSONL event repository used by the Phase 1 CLI."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path

from vehicle_intelligence.application.ports import EventPage, EventQuery
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.serialization import (
    document_to_event,
    event_to_jsonable,
)


class JsonlVehicleEventRepository:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._memory = InMemoryVehicleEventRepository()
        self._loaded = False
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        await self._load()

    async def save(self, event: VehicleEvent) -> bool:
        await self._load()
        async with self._lock:
            created = await self._memory.save(event)
            if not created:
                return False
            line = json.dumps(event_to_jsonable(event), ensure_ascii=False, separators=(",", ":"))
            try:
                await asyncio.to_thread(self._append, line)
            except OSError as exc:
                raise PersistenceError(f"cannot append event JSONL: {self._path}") from exc
            return True

    async def get(self, event_id: str) -> VehicleEvent | None:
        await self._load()
        return await self._memory.get(event_id)

    async def list(self, query: EventQuery) -> EventPage:
        await self._load()
        return await self._memory.list(query)

    async def find_by_plate(self, plate: str, limit: int) -> list[VehicleEvent]:
        await self._load()
        return await self._memory.find_by_plate(plate, limit)

    async def timeline(
        self,
        vehicle_id: str,
        *,
        from_time=None,
        to_time=None,
        limit: int = 1000,
        ascending: bool = True,
    ) -> tuple[VehicleEvent, ...]:
        await self._load()
        return await self._memory.timeline(
            vehicle_id,
            from_time=from_time,
            to_time=to_time,
            limit=limit,
            ascending=ascending,
        )

    async def assign_vehicle_id(self, event_id: str, vehicle_id: str) -> bool:
        await self._load()
        async with self._lock:
            current = await self._memory.get(event_id)
            if current is None or current.vehicle_id not in {None, vehicle_id}:
                return False
            events = list(await self._memory.snapshot())
            rewritten = [
                replace(item, vehicle_id=vehicle_id) if item.id == event_id else item
                for item in events
            ]
            try:
                await asyncio.to_thread(self._rewrite, rewritten)
            except OSError as exc:
                raise PersistenceError(f"cannot rewrite event JSONL: {self._path}") from exc
            return await self._memory.assign_vehicle_id(event_id, vehicle_id)

    async def reassign_vehicle_ids(
        self,
        event_ids: tuple[str, ...],
        source_vehicle_id: str,
        target_vehicle_id: str,
    ) -> int:
        await self._load()
        ids = set(event_ids)
        async with self._lock:
            events = list(await self._memory.snapshot())
            moved_ids = {
                item.id
                for item in events
                if item.id in ids and item.vehicle_id == source_vehicle_id
            }
            if not moved_ids:
                return 0
            rewritten = [
                replace(item, vehicle_id=target_vehicle_id)
                if item.id in moved_ids
                else item
                for item in events
            ]
            try:
                await asyncio.to_thread(self._rewrite, rewritten)
            except OSError as exc:
                raise PersistenceError(f"cannot rewrite event JSONL: {self._path}") from exc
            return await self._memory.reassign_vehicle_ids(
                tuple(moved_ids),
                source_vehicle_id,
                target_vehicle_id,
            )

    async def update_plate_review(
        self,
        event: VehicleEvent,
        expected_revision: int,
    ) -> VehicleEvent | None:
        await self._load()
        if event.plate is None or event.plate.review_revision != expected_revision + 1:
            raise ValueError("plate review revision must increment by one")
        async with self._lock:
            current = await self._memory.get(event.id)
            if (
                current is None
                or current.plate is None
                or current.plate.review_revision != expected_revision
            ):
                return None
            events = list(await self._memory.snapshot())
            rewritten = [event if item.id == event.id else item for item in events]
            try:
                await asyncio.to_thread(self._rewrite, rewritten)
            except OSError as exc:
                raise PersistenceError(f"cannot rewrite event JSONL: {self._path}") from exc
            return await self._memory.update_plate_review(event, expected_revision)

    async def close(self) -> None:
        return None

    async def _load(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            try:
                documents = await asyncio.to_thread(self._read_documents)
            except (OSError, ValueError, KeyError) as exc:
                raise PersistenceError(f"cannot load event JSONL: {self._path}") from exc
            for document in documents:
                await self._memory.save(document_to_event(document))
            self._loaded = True

    def _read_documents(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        documents: list[dict[str, object]] = []
        with self._path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at line {line_number}") from exc
                documents.append(document)
        return documents

    def _append(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.write("\n")
            stream.flush()

    def _rewrite(self, events: list[VehicleEvent]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                for event in events:
                    stream.write(
                        json.dumps(
                            event_to_jsonable(event),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)
