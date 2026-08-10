"""MongoDB camera-topology repository with directional travel-time indexes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, IndexModel
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import CameraTopologyEdge
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import CAMERA_TOPOLOGY
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo


class MongoCameraTopologyRepository:
    def __init__(self, source: MongoConfig | MongoRuntime) -> None:
        binding = bind_mongo(source)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[CAMERA_TOPOLOGY]

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [("fromCameraId", ASCENDING), ("toCameraId", ASCENDING)],
                        unique=True,
                        name="uq_topology_direction",
                    ),
                    IndexModel(
                        [
                            ("toCameraId", ASCENDING),
                            ("enabled", ASCENDING),
                            ("fromCameraId", ASCENDING),
                        ],
                        name="ix_topology_inbound",
                    ),
                    IndexModel(
                        [
                            ("fromCameraId", ASCENDING),
                            ("enabled", ASCENDING),
                            ("toCameraId", ASCENDING),
                        ],
                        name="ix_topology_outbound",
                    ),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize camera-topology indexes") from exc

    async def create(self, edge: CameraTopologyEdge) -> bool:
        try:
            await self._collection.insert_one(_to_document(edge))
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot create topology edge: {edge.id}") from exc

    async def replace(
        self,
        edge: CameraTopologyEdge,
        expected_revision: int,
    ) -> bool:
        if edge.revision != expected_revision + 1:
            raise ValueError("replacement topology revision must increment by one")
        try:
            result = await self._collection.replace_one(
                {"_id": edge.id, "revision": expected_revision},
                _to_document(edge),
            )
            return result.matched_count == 1
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot update topology edge: {edge.id}") from exc

    async def get(self, edge_id: str) -> CameraTopologyEdge | None:
        try:
            document = await self._collection.find_one({"_id": edge_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read topology edge: {edge_id}") from exc
        return _from_document(document) if document is not None else None

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
        query: dict[str, object] = {}
        if from_camera_id is not None:
            query["fromCameraId"] = from_camera_id
        if to_camera_id is not None:
            query["toCameraId"] = to_camera_id
        if enabled_only:
            query["enabled"] = True
        try:
            cursor = (
                self._collection.find(query)
                .sort(
                    [
                        ("fromCameraId", ASCENDING),
                        ("toCameraId", ASCENDING),
                        ("_id", ASCENDING),
                    ]
                )
                .limit(limit)
            )
            return tuple([_from_document(item) async for item in cursor])
        except PyMongoError as exc:
            raise PersistenceError("cannot list topology edges") from exc

    async def delete(self, edge_id: str) -> bool:
        try:
            result = await self._collection.delete_one({"_id": edge_id})
            return result.deleted_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot delete topology edge: {edge_id}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def _to_document(edge: CameraTopologyEdge) -> dict[str, Any]:
    return {
        "_id": edge.id,
        "schemaVersion": edge.schema_version,
        "revision": edge.revision,
        "fromCameraId": edge.from_camera_id,
        "toCameraId": edge.to_camera_id,
        "travelTime": {
            "minimumSeconds": edge.minimum_travel_seconds,
            "maximumSeconds": edge.maximum_travel_seconds,
            "typicalSeconds": edge.typical_travel_seconds,
        },
        "enabled": edge.enabled,
        "metadata": edge.metadata,
        "createdAt": edge.created_at.astimezone(UTC),
        "updatedAt": edge.updated_at.astimezone(UTC),
    }


def _from_document(document: dict[str, Any]) -> CameraTopologyEdge:
    travel = document["travelTime"]
    created_at: datetime = document["createdAt"]
    updated_at: datetime = document["updatedAt"]
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return CameraTopologyEdge(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        revision=int(document["revision"]),
        from_camera_id=str(document["fromCameraId"]),
        to_camera_id=str(document["toCameraId"]),
        minimum_travel_seconds=float(travel["minimumSeconds"]),
        maximum_travel_seconds=float(travel["maximumSeconds"]),
        typical_travel_seconds=float(travel["typicalSeconds"]),
        enabled=bool(document["enabled"]),
        metadata=document.get("metadata") or {},
        created_at=created_at.astimezone(UTC),
        updated_at=updated_at.astimezone(UTC),
    )
