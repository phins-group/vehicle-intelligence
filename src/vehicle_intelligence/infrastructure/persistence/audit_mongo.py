"""MongoDB append-only audit-log repository."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.application.ports import AuditPage, AuditQuery
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    AuditAction,
    AuditActor,
    AuditLog,
    AuditResourceType,
    AuthenticationMethod,
    UserRole,
)
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import AUDIT_LOGS
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class MongoAuditLogRepository:
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        binding = bind_mongo(config)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[AUDIT_LOGS]

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [("occurredAt", DESCENDING), ("_id", DESCENDING)],
                        name="ix_audit_cursor",
                    ),
                    IndexModel(
                        [("actor.id", ASCENDING), ("occurredAt", DESCENDING)],
                        name="ix_audit_actor_time",
                    ),
                    IndexModel(
                        [
                            ("resource.type", ASCENDING),
                            ("resource.id", ASCENDING),
                            ("occurredAt", DESCENDING),
                        ],
                        name="ix_audit_resource_time",
                    ),
                    IndexModel(
                        [("action", ASCENDING), ("occurredAt", DESCENDING)],
                        name="ix_audit_action_time",
                    ),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB audit indexes") from exc

    async def append(self, entry: AuditLog) -> None:
        try:
            await self._collection.insert_one(_to_document(entry))
        except DuplicateKeyError as exc:
            raise PersistenceError(f"audit record already exists: {entry.id}") from exc
        except PyMongoError as exc:
            raise PersistenceError(f"cannot append audit record: {entry.id}") from exc

    async def get(self, entry_id: str) -> AuditLog | None:
        try:
            document = await self._collection.find_one({"_id": entry_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read audit record: {entry_id}") from exc
        return _from_document(document) if document is not None else None

    async def list(self, query: AuditQuery) -> AuditPage:
        filters: dict[str, Any] = {}
        if query.actor_id is not None:
            filters["actor.id"] = query.actor_id
        if query.action is not None:
            filters["action"] = query.action.value
        if query.resource_type is not None:
            filters["resource.type"] = query.resource_type.value
        if query.resource_id is not None:
            filters["resource.id"] = query.resource_id
        time_filter: dict[str, datetime] = {}
        if query.from_time is not None:
            time_filter["$gte"] = query.from_time
        if query.to_time is not None:
            time_filter["$lte"] = query.to_time
        if time_filter:
            filters["occurredAt"] = time_filter
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            filters["$or"] = [
                {"occurredAt": {"$lt": cursor_time}},
                {"occurredAt": cursor_time, "_id": {"$lt": cursor_id}},
            ]
        try:
            cursor = (
                self._collection.find(filters)
                .sort([("occurredAt", DESCENDING), ("_id", DESCENDING)])
                .limit(query.limit + 1)
            )
            documents = [document async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot list audit records") from exc
        has_more = len(documents) > query.limit
        entries = tuple(_from_document(document) for document in documents[: query.limit])
        next_cursor = (
            encode_cursor(entries[-1].occurred_at, entries[-1].id)
            if has_more and entries
            else None
        )
        return AuditPage(entries, next_cursor)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def _to_document(entry: AuditLog) -> dict[str, Any]:
    return {
        "_id": entry.id,
        "schemaVersion": entry.schema_version,
        "actor": {
            "id": entry.actor.id,
            "displayName": entry.actor.display_name,
            "role": entry.actor.role.value,
            "authenticationMethod": entry.actor.authentication_method.value,
        },
        "action": entry.action.value,
        "resource": {
            "type": entry.resource_type.value,
            "id": entry.resource_id,
        },
        "requestId": entry.request_id,
        "before": entry.before,
        "after": entry.after,
        "metadata": entry.metadata,
        "occurredAt": entry.occurred_at.astimezone(UTC),
    }


def _from_document(document: dict[str, Any]) -> AuditLog:
    actor = document["actor"]
    resource = document["resource"]
    return AuditLog(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        actor=AuditActor(
            id=str(actor["id"]),
            display_name=str(actor["displayName"]),
            role=UserRole(actor["role"]),
            authentication_method=AuthenticationMethod(actor["authenticationMethod"]),
        ),
        action=AuditAction(document["action"]),
        resource_type=AuditResourceType(resource["type"]),
        resource_id=str(resource["id"]),
        request_id=str(document["requestId"]),
        before=document.get("before"),
        after=document.get("after"),
        metadata=document.get("metadata") or {},
        occurred_at=_aware(document["occurredAt"]),
    )
