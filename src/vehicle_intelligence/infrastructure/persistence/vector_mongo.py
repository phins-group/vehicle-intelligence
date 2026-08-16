"""Mongo vector storage with explicitly bounded candidate-only cosine search."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.application.ports import VectorSearchQuery
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import EmbeddingModel, EmbeddingVector, VectorNeighbor
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import VEHICLE_EMBEDDINGS
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo


class MongoVectorRepository:
    def __init__(
        self,
        source: MongoConfig | MongoRuntime,
        maximum_candidates: int = 1000,
    ) -> None:
        if not 1 <= maximum_candidates <= 5000:
            raise ValueError("Mongo vector candidate limit must be in [1, 5000]")
        binding = bind_mongo(source)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[VEHICLE_EMBEDDINGS]
        self._maximum_candidates = maximum_candidates

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [
                            ("model.name", ASCENDING),
                            ("model.version", ASCENDING),
                            ("createdAt", DESCENDING),
                        ],
                        name="ix_embedding_model_time",
                    ),
                    IndexModel(
                        [("metadata.vehicleId", ASCENDING), ("createdAt", DESCENDING)],
                        name="ix_embedding_vehicle_time",
                        partialFilterExpression={"metadata.vehicleId": {"$type": "string"}},
                    ),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize vehicle embedding indexes") from exc

    async def put(self, vector: EmbeddingVector) -> bool:
        try:
            await self._collection.insert_one(_to_document(vector))
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot persist embedding vector: {vector.id}") from exc

    async def get(self, vector_id: str) -> EmbeddingVector | None:
        try:
            document = await self._collection.find_one({"_id": vector_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read embedding vector: {vector_id}") from exc
        return _from_document(document) if document is not None else None

    async def search(self, query: VectorSearchQuery) -> tuple[VectorNeighbor, ...]:
        candidates = tuple(dict.fromkeys(query.candidate_ids))
        if not candidates:
            return ()
        if len(candidates) > self._maximum_candidates:
            raise ValueError("vector query exceeds configured candidate limit")
        query_norm = math.sqrt(sum(value * value for value in query.vector))
        if query_norm <= 0:
            raise ValueError("vector search query cannot have zero norm")
        normalized = tuple(value / query_norm for value in query.vector)
        try:
            cursor = self._collection.find(
                {
                    "_id": {"$in": list(candidates)},
                    "model.name": query.model.name,
                    "model.version": query.model.version,
                    "model.hash": query.model.model_hash,
                    "model.dimension": query.model.dimension,
                }
            ).limit(self._maximum_candidates)
            vectors = [_from_document(document) async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot load bounded vector candidates") from exc
        neighbors: list[VectorNeighbor] = []
        for vector in vectors:
            score = sum(
                left * right
                for left, right in zip(
                    normalized,
                    vector.normalized_values,
                    strict=True,
                )
            )
            if score >= query.minimum_score:
                neighbors.append(VectorNeighbor(vector.id, min(1.0, max(-1.0, score))))
        neighbors.sort(key=lambda item: (item.score, item.vector_id), reverse=True)
        return tuple(neighbors[: query.limit])

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def _to_document(vector: EmbeddingVector) -> dict[str, Any]:
    return {
        "_id": vector.id,
        "schemaVersion": 1,
        "model": {
            "name": vector.model.name,
            "version": vector.model.version,
            "hash": vector.model.model_hash,
            "dimension": vector.model.dimension,
        },
        "values": list(vector.normalized_values),
        "createdAt": vector.created_at.astimezone(UTC),
        "metadata": vector.metadata,
    }


def _from_document(document: dict[str, Any]) -> EmbeddingVector:
    model = document["model"]
    timestamp: datetime = document["createdAt"]
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return EmbeddingVector(
        id=str(document["_id"]),
        model=EmbeddingModel(
            name=str(model["name"]),
            version=str(model["version"]),
            model_hash=model.get("hash"),
            dimension=int(model["dimension"]),
        ),
        values=tuple(float(value) for value in document["values"]),
        created_at=timestamp.astimezone(UTC),
        metadata=document.get("metadata") or {},
    )
