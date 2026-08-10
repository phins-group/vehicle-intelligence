"""Async PyMongo repository for canonical vehicle events."""

from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.application.ports import EventPage, EventQuery
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import VEHICLE_EVENTS
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo
from vehicle_intelligence.infrastructure.serialization import document_to_event, event_to_document


class MongoVehicleEventRepository:
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        binding = bind_mongo(config)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[VEHICLE_EVENTS]

    async def ensure_indexes(self) -> None:
        indexes = [
            IndexModel(
                [("camera.id", ASCENDING), ("trackId", ASCENDING), ("eventType", ASCENDING)],
                unique=True,
                name="uq_event_track_type",
            ),
            IndexModel(
                [("plate.normalized", ASCENDING), ("occurredAt", DESCENDING)],
                name="ix_plate_time",
                partialFilterExpression={"plate.normalized": {"$type": "string"}},
            ),
            IndexModel(
                [("plate.final", ASCENDING), ("occurredAt", DESCENDING)],
                name="ix_plate_final_time",
                partialFilterExpression={"plate.final": {"$type": "string"}},
            ),
            IndexModel(
                [("camera.id", ASCENDING), ("occurredAt", DESCENDING)],
                name="ix_camera_time",
            ),
            IndexModel(
                [("vehicleId", ASCENDING), ("occurredAt", DESCENDING)],
                name="ix_vehicle_time",
                partialFilterExpression={"vehicleId": {"$type": "string"}},
            ),
            IndexModel(
                [("eventType", ASCENDING), ("occurredAt", DESCENDING)],
                name="ix_type_time",
            ),
            IndexModel(
                [("occurredAt", DESCENDING), ("_id", DESCENDING)],
                name="ix_time_cursor",
            ),
            IndexModel(
                [("status", ASCENDING), ("occurredAt", DESCENDING), ("_id", DESCENDING)],
                name="ix_status_time_cursor",
            ),
        ]
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(indexes)
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB vehicle-event indexes") from exc

    async def save(self, event: VehicleEvent) -> bool:
        try:
            await self._collection.insert_one(event_to_document(event))
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot persist vehicle event: {event.id}") from exc

    async def get(self, event_id: str) -> VehicleEvent | None:
        try:
            document = await self._collection.find_one({"_id": event_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read vehicle event: {event_id}") from exc
        return document_to_event(document) if document is not None else None

    async def list(self, query: EventQuery) -> EventPage:
        filters = self._filters(query)
        clauses: list[dict[str, Any]] = []
        if query.plate:
            clauses.append(self._plate_filter(query.plate))
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            clauses.append(
                {
                    "$or": [
                        {"occurredAt": {"$lt": cursor_time}},
                        {"occurredAt": cursor_time, "_id": {"$lt": cursor_id}},
                    ]
                }
            )
        if clauses:
            filters["$and"] = clauses
        try:
            cursor = (
                self._collection.find(filters)
                .sort([("occurredAt", DESCENDING), ("_id", DESCENDING)])
                .limit(query.limit + 1)
            )
            documents = [document async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot list vehicle events") from exc
        has_more = len(documents) > query.limit
        documents = documents[: query.limit]
        events = tuple(document_to_event(document) for document in documents)
        next_cursor = (
            encode_cursor(events[-1].occurred_at, events[-1].id) if has_more and events else None
        )
        return EventPage(events, next_cursor)

    async def find_by_plate(self, plate: str, limit: int) -> list[VehicleEvent]:
        try:
            cursor = (
                self._collection.find(self._plate_filter(plate))
                .sort([("occurredAt", DESCENDING), ("_id", DESCENDING)])
                .limit(limit)
            )
            return [document_to_event(document) async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError(f"cannot search events for plate: {plate}") from exc

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
        filters: dict[str, Any] = {"vehicleId": vehicle_id}
        if from_time is not None or to_time is not None:
            occurred: dict[str, Any] = {}
            if from_time is not None:
                occurred["$gte"] = from_time
            if to_time is not None:
                occurred["$lte"] = to_time
            filters["occurredAt"] = occurred
        direction = ASCENDING if ascending else DESCENDING
        try:
            cursor = (
                self._collection.find(filters)
                .sort([("occurredAt", direction), ("_id", direction)])
                .limit(limit)
            )
            return tuple([document_to_event(document) async for document in cursor])
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read vehicle timeline: {vehicle_id}") from exc

    async def update_plate_review(
        self,
        event: VehicleEvent,
        expected_revision: int,
    ) -> VehicleEvent | None:
        if event.plate is None or event.plate.review_revision != expected_revision + 1:
            raise ValueError("plate review revision must increment by one")
        filters: dict[str, Any] = {"_id": event.id}
        if expected_revision == 0:
            filters["$or"] = [
                {"plate.review": {"$exists": False}},
                {"plate.review": None},
            ]
        else:
            filters["plate.review.revision"] = expected_revision
        plate_document = event_to_document(event)["plate"]
        try:
            document = await self._collection.find_one_and_update(
                filters,
                {
                    "$set": {
                        "schemaVersion": event.schema_version,
                        "status": event.status.value,
                        "plate": plate_document,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            raise PersistenceError(f"cannot review vehicle event plate: {event.id}") from exc
        return document_to_event(document) if document is not None else None

    async def assign_vehicle_id(self, event_id: str, vehicle_id: str) -> bool:
        try:
            result = await self._collection.update_one(
                {
                    "_id": event_id,
                    "$or": [
                        {"vehicleId": None},
                        {"vehicleId": {"$exists": False}},
                        {"vehicleId": vehicle_id},
                    ],
                },
                {"$set": {"vehicleId": vehicle_id}},
            )
            return result.matched_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot assign vehicle identity: {event_id}") from exc

    async def reassign_vehicle_ids(
        self,
        event_ids: tuple[str, ...],
        source_vehicle_id: str,
        target_vehicle_id: str,
    ) -> int:
        if len(event_ids) > 1000:
            raise ValueError("identity event reassignment is bounded to 1000 events")
        if not event_ids:
            return 0
        try:
            result = await self._collection.update_many(
                {
                    "_id": {"$in": list(dict.fromkeys(event_ids))},
                    "vehicleId": source_vehicle_id,
                },
                {"$set": {"vehicleId": target_vehicle_id}},
            )
            return result.modified_count
        except PyMongoError as exc:
            raise PersistenceError("cannot reassign vehicle event identities") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    @staticmethod
    def _filters(query: EventQuery) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if query.camera_id:
            filters["camera.id"] = query.camera_id
        if query.event_type:
            filters["eventType"] = query.event_type
        if query.direction:
            filters["direction"] = query.direction
        if query.status:
            filters["status"] = query.status
        if query.from_time or query.to_time:
            occurred: dict[str, Any] = {}
            if query.from_time:
                occurred["$gte"] = query.from_time
            if query.to_time:
                occurred["$lte"] = query.to_time
            filters["occurredAt"] = occurred
        return filters

    @staticmethod
    def _plate_filter(plate: str) -> dict[str, Any]:
        return {
            "$or": [
                {"plate.final": plate},
                {
                    "plate.final": {"$exists": False},
                    "plate.normalized": plate,
                },
            ]
        }
