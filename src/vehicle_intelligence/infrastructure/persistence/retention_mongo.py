"""Leased MongoDB retention coordination for canonical events and media keys."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, AsyncMongoClient, ReturnDocument
from pymongo.errors import PyMongoError

from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import MediaKind, MediaRetentionClaim
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import (
    DATASET_SAMPLES,
    VEHICLE_EVENTS,
)

_MEDIA_FIELDS = {
    MediaKind.SNAPSHOT: "media.snapshotKey",
    MediaKind.VEHICLE_CROP: "media.vehicleCropKey",
    MediaKind.PLATE_CROP: "media.plateCropKey",
    MediaKind.EVENT_CLIP: "media.clipKey",
}


class MongoRetentionRepository:
    def __init__(self, config: MongoConfig) -> None:
        self._client = AsyncMongoClient(
            config.uri.get_secret_value(),
            tz_aware=True,
            serverSelectionTimeoutMS=config.server_selection_timeout_ms,
            connectTimeoutMS=config.connect_timeout_ms,
            socketTimeoutMS=config.socket_timeout_ms,
        )
        database = self._client[config.database]
        self._events = database[VEHICLE_EVENTS]
        self._samples = database[DATASET_SAMPLES]

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._events.create_index(
                [("occurredAt", ASCENDING), ("_id", ASCENDING)],
                name="ix_retention_oldest",
            )
            await self._samples.create_index(
                [("imageKey", ASCENDING), ("status", ASCENDING)],
                name="ix_dataset_image_status",
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize retention indexes") from exc

    async def claim_media(
        self,
        kind: MediaKind,
        older_than: datetime,
        stale_before: datetime,
        lease_id: str,
        limit: int,
    ) -> list[MediaRetentionClaim]:
        field = _MEDIA_FIELDS[kind]
        state_path = self._state_path(kind)
        try:
            candidates = [
                document
                async for document in self._events.find(
                    {
                        "occurredAt": {"$lt": older_than},
                        field: {"$type": "string"},
                        "$or": [
                            {state_path: {"$exists": False}},
                            {state_path: "FAILED"},
                        ],
                    },
                    {"_id": 1, "occurredAt": 1, field: 1},
                )
                .sort([("occurredAt", ASCENDING), ("_id", ASCENDING)])
                .limit(limit * 4)
            ]
            pinned = await self._pinned_keys(kind, candidates)
            claims: list[MediaRetentionClaim] = []
            for document in candidates:
                key = _nested(document, field)
                if not isinstance(key, str) or key in pinned:
                    continue
                claimed = await self._events.find_one_and_update(
                    {
                        "_id": document["_id"],
                        field: key,
                        "$or": [
                            {state_path: {"$exists": False}},
                            {state_path: "FAILED"},
                        ],
                    },
                    {
                        "$set": {
                            self._retention_path(kind): {
                                "state": "DELETING",
                                "key": key,
                                "leaseId": lease_id,
                                "updatedAt": datetime.now(UTC),
                            }
                        },
                        "$unset": {field: ""},
                    },
                    return_document=ReturnDocument.AFTER,
                )
                if claimed is not None:
                    claims.append(self._claim(document, kind, key, lease_id))
                if len(claims) >= limit:
                    return claims

            remaining = limit - len(claims)
            if remaining:
                stale_cursor = (
                    self._events.find(
                        {
                            "occurredAt": {"$lt": older_than},
                            state_path: "DELETING",
                            f"{self._retention_path(kind)}.updatedAt": {"$lt": stale_before},
                            f"{self._retention_path(kind)}.key": {"$type": "string"},
                        },
                        {
                            "_id": 1,
                            "occurredAt": 1,
                            self._retention_path(kind): 1,
                        },
                    )
                    .sort([("occurredAt", ASCENDING), ("_id", ASCENDING)])
                    .limit(remaining)
                )
                async for document in stale_cursor:
                    retention = _nested(document, self._retention_path(kind)) or {}
                    key = retention.get("key")
                    previous_lease = retention.get("leaseId")
                    if not isinstance(key, str):
                        continue
                    reclaimed = await self._events.find_one_and_update(
                        {
                            "_id": document["_id"],
                            state_path: "DELETING",
                            f"{self._retention_path(kind)}.leaseId": previous_lease,
                            f"{self._retention_path(kind)}.updatedAt": {"$lt": stale_before},
                        },
                        {
                            "$set": {
                                f"{self._retention_path(kind)}.leaseId": lease_id,
                                f"{self._retention_path(kind)}.updatedAt": datetime.now(UTC),
                            }
                        },
                        return_document=ReturnDocument.AFTER,
                    )
                    if reclaimed is not None:
                        claims.append(self._claim(document, kind, key, lease_id))
            return claims
        except PyMongoError as exc:
            raise PersistenceError(f"cannot claim {kind.value} retention work") from exc

    async def mark_media_deleted(
        self,
        claim: MediaRetentionClaim,
        deleted_at: datetime,
    ) -> None:
        try:
            result = await self._events.update_one(
                {
                    "_id": claim.event_id,
                    f"{self._retention_path(claim.kind)}.leaseId": claim.lease_id,
                    self._state_path(claim.kind): "DELETING",
                },
                {
                    "$set": {
                        self._retention_path(claim.kind): {
                            "state": "DELETED",
                            "key": claim.key,
                            "deletedAt": deleted_at.astimezone(UTC),
                            "updatedAt": deleted_at.astimezone(UTC),
                        }
                    }
                },
            )
        except PyMongoError as exc:
            raise PersistenceError(f"cannot complete media retention: {claim.event_id}") from exc
        if result.matched_count != 1:
            raise PersistenceError(f"media retention lease was lost: {claim.event_id}")

    async def mark_media_failed(
        self,
        claim: MediaRetentionClaim,
        error_code: str,
        failed_at: datetime,
    ) -> None:
        field = _MEDIA_FIELDS[claim.kind]
        try:
            result = await self._events.update_one(
                {
                    "_id": claim.event_id,
                    f"{self._retention_path(claim.kind)}.leaseId": claim.lease_id,
                    self._state_path(claim.kind): "DELETING",
                },
                {
                    "$set": {
                        field: claim.key,
                        self._retention_path(claim.kind): {
                            "state": "FAILED",
                            "key": claim.key,
                            "errorCode": error_code[:128],
                            "updatedAt": failed_at.astimezone(UTC),
                        },
                    }
                },
            )
        except PyMongoError as exc:
            raise PersistenceError(f"cannot release media retention: {claim.event_id}") from exc
        if result.matched_count != 1:
            raise PersistenceError(f"media retention lease was lost: {claim.event_id}")

    async def delete_expired_events(self, older_than: datetime, limit: int) -> int:
        media_ready: dict[str, Any] = {field: None for field in _MEDIA_FIELDS.values()}
        for kind in MediaKind:
            media_ready[self._state_path(kind)] = {"$ne": "DELETING"}
        try:
            documents = [
                document
                async for document in self._events.find(
                    {"occurredAt": {"$lt": older_than}, **media_ready},
                    {"_id": 1},
                )
                .sort([("occurredAt", ASCENDING), ("_id", ASCENDING)])
                .limit(limit * 2)
            ]
            ids = [str(document["_id"]) for document in documents]
            if not ids:
                return 0
            pinned = set(
                await self._samples.distinct(
                    "sourceEventId",
                    {
                        "sourceEventId": {"$in": ids},
                        "status": {"$in": ["READY", "EXPORTING", "EXPORT_FAILED"]},
                    },
                )
            )
            deletable = [event_id for event_id in ids if event_id not in pinned][:limit]
            if not deletable:
                return 0
            result = await self._events.delete_many({"_id": {"$in": deletable}})
            return result.deleted_count
        except PyMongoError as exc:
            raise PersistenceError("cannot delete expired vehicle events") from exc

    async def close(self) -> None:
        await self._client.close()

    async def _pinned_keys(
        self,
        kind: MediaKind,
        candidates: list[dict[str, Any]],
    ) -> set[str]:
        field = _MEDIA_FIELDS[kind]
        keys = [key for item in candidates if isinstance((key := _nested(item, field)), str)]
        if not keys:
            return set()
        return set(
            await self._samples.distinct(
                "imageKey",
                {
                    "imageKey": {"$in": keys},
                    "status": {"$in": ["READY", "EXPORTING", "EXPORT_FAILED"]},
                },
            )
        )

    @staticmethod
    def _retention_path(kind: MediaKind) -> str:
        return f"retention.media.{kind.value}"

    @classmethod
    def _state_path(cls, kind: MediaKind) -> str:
        return f"{cls._retention_path(kind)}.state"

    @staticmethod
    def _claim(
        document: dict[str, Any],
        kind: MediaKind,
        key: str,
        lease_id: str,
    ) -> MediaRetentionClaim:
        occurred_at = document["occurredAt"]
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=UTC)
        return MediaRetentionClaim(
            event_id=str(document["_id"]),
            kind=kind,
            key=key,
            lease_id=lease_id,
            occurred_at=occurred_at.astimezone(UTC),
        )


def _nested(document: dict[str, Any], path: str) -> Any:
    value: Any = document
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
