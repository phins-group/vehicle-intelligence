"""In-memory append-only audit repository for local development and tests."""

from __future__ import annotations

import asyncio

from vehicle_intelligence.application.ports import AuditPage, AuditQuery
from vehicle_intelligence.domain import AuditLog
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor


class InMemoryAuditLogRepository:
    def __init__(self) -> None:
        self._entries: dict[str, AuditLog] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def append(self, entry: AuditLog) -> None:
        async with self._lock:
            if entry.id in self._entries:
                raise PersistenceError(f"audit record already exists: {entry.id}")
            self._entries[entry.id] = entry

    async def get(self, entry_id: str) -> AuditLog | None:
        return self._entries.get(entry_id)

    async def list(self, query: AuditQuery) -> AuditPage:
        entries = [entry for entry in self._entries.values() if _matches(entry, query)]
        entries.sort(key=lambda item: (item.occurred_at, item.id), reverse=True)
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            entries = [
                entry
                for entry in entries
                if (entry.occurred_at, entry.id) < (cursor_time, cursor_id)
            ]
        page = entries[: query.limit + 1]
        has_more = len(page) > query.limit
        page = page[: query.limit]
        next_cursor = (
            encode_cursor(page[-1].occurred_at, page[-1].id) if has_more and page else None
        )
        return AuditPage(tuple(page), next_cursor)

    async def close(self) -> None:
        return None


def _matches(entry: AuditLog, query: AuditQuery) -> bool:
    return not any(
        (
            query.actor_id is not None and entry.actor.id != query.actor_id,
            query.action is not None and entry.action is not query.action,
            query.resource_type is not None
            and entry.resource_type is not query.resource_type,
            query.resource_id is not None and entry.resource_id != query.resource_id,
            query.from_time is not None and entry.occurred_at < query.from_time,
            query.to_time is not None and entry.occurred_at > query.to_time,
        )
    )

