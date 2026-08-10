"""MongoDB persistence for bounded, labeled retraining samples."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.application.ports import DatasetSamplePage, DatasetSampleQuery
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import DatasetSample, DatasetSampleStatus
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import DATASET_SAMPLES
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo
from vehicle_intelligence.infrastructure.review_serialization import (
    dataset_sample_to_document,
    document_to_dataset_sample,
)


class MongoDatasetSampleRepository:
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        binding = bind_mongo(config)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[DATASET_SAMPLES]

    async def ensure_indexes(self) -> None:
        indexes = [
            IndexModel(
                [("sourceEventId", ASCENDING), ("review.revision", ASCENDING)],
                unique=True,
                name="uq_dataset_event_review",
            ),
            IndexModel(
                [
                    ("type", ASCENDING),
                    ("status", ASCENDING),
                    ("createdAt", DESCENDING),
                    ("_id", DESCENDING),
                ],
                name="ix_dataset_type_status_cursor",
            ),
            IndexModel(
                [("reason", ASCENDING), ("createdAt", DESCENDING)],
                name="ix_dataset_reason_time",
            ),
            IndexModel(
                [("createdAt", DESCENDING), ("_id", DESCENDING)],
                name="ix_dataset_cursor",
            ),
            IndexModel(
                [("status", ASCENDING), ("export.claimedAt", ASCENDING), ("createdAt", ASCENDING)],
                name="ix_dataset_export_claim",
            ),
            IndexModel(
                [("export.id", ASCENDING), ("status", ASCENDING), ("createdAt", ASCENDING)],
                name="ix_dataset_export_resume",
                partialFilterExpression={"export.id": {"$type": "string"}},
            ),
        ]
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(indexes)
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB dataset-sample indexes") from exc

    async def create(self, sample: DatasetSample) -> bool:
        try:
            await self._collection.insert_one(dataset_sample_to_document(sample))
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot create dataset sample: {sample.id}") from exc

    async def get(self, sample_id: str) -> DatasetSample | None:
        try:
            document = await self._collection.find_one({"_id": sample_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read dataset sample: {sample_id}") from exc
        return document_to_dataset_sample(document) if document is not None else None

    async def list(self, query: DatasetSampleQuery) -> DatasetSamplePage:
        filters: dict[str, Any] = {}
        if query.sample_type is not None:
            filters["type"] = query.sample_type.value
        if query.status is not None:
            filters["status"] = query.status.value
        if query.reason is not None:
            filters["reason"] = query.reason.value
        if query.source_event_id is not None:
            filters["sourceEventId"] = query.source_event_id
        if query.from_time is not None or query.to_time is not None:
            reviewed_at: dict[str, datetime] = {}
            if query.from_time is not None:
                reviewed_at["$gte"] = query.from_time
            if query.to_time is not None:
                reviewed_at["$lt"] = query.to_time
            filters["review.reviewedAt"] = reviewed_at
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            filters["$or"] = [
                {"createdAt": {"$lt": cursor_time}},
                {"createdAt": cursor_time, "_id": {"$lt": cursor_id}},
            ]
        try:
            cursor = (
                self._collection.find(filters)
                .sort([("createdAt", DESCENDING), ("_id", DESCENDING)])
                .limit(query.limit + 1)
            )
            documents = [document async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot list dataset samples") from exc
        has_more = len(documents) > query.limit
        samples = tuple(
            document_to_dataset_sample(document) for document in documents[: query.limit]
        )
        next_cursor = (
            encode_cursor(samples[-1].created_at, samples[-1].id)
            if has_more and samples
            else None
        )
        return DatasetSamplePage(samples, next_cursor)

    async def claim_for_export(
        self,
        export_id: str,
        limit: int,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> tuple[DatasetSample, ...]:
        if not export_id.strip() or not 1 <= limit <= 1000:
            raise ValueError("dataset export claim is invalid")
        try:
            resumed_cursor = (
                self._collection.find(
                    {"status": DatasetSampleStatus.EXPORTING.value, "export.id": export_id}
                )
                .sort([("createdAt", ASCENDING), ("_id", ASCENDING)])
                .limit(limit)
            )
            documents = [document async for document in resumed_cursor]
            selected_ids = [str(document["_id"]) for document in documents]
            while len(documents) < limit:
                eligible: dict[str, Any] = {
                    "$or": [
                        {
                            "status": {
                                "$in": [
                                    DatasetSampleStatus.READY.value,
                                    DatasetSampleStatus.EXPORT_FAILED.value,
                                ]
                            }
                        },
                        {
                            "status": DatasetSampleStatus.EXPORTING.value,
                            "export.claimedAt": {"$lt": stale_before},
                        },
                    ]
                }
                if selected_ids:
                    eligible["_id"] = {"$nin": selected_ids}
                document = await self._collection.find_one_and_update(
                    eligible,
                    [
                        {
                            "$set": {
                                "schemaVersion": 2,
                                "status": DatasetSampleStatus.EXPORTING.value,
                                "export": {
                                    "id": export_id,
                                    "attempts": {
                                        "$add": [
                                            {"$ifNull": ["$export.attempts", 0]},
                                            1,
                                        ]
                                    },
                                    "claimedAt": claimed_at,
                                    "exportedAt": None,
                                    "manifestSha256": None,
                                    "errorCode": None,
                                },
                            }
                        }
                    ],
                    sort=[("createdAt", ASCENDING), ("_id", ASCENDING)],
                    return_document=ReturnDocument.AFTER,
                )
                if document is None:
                    break
                documents.append(document)
                selected_ids.append(str(document["_id"]))
            return tuple(document_to_dataset_sample(document) for document in documents)
        except PyMongoError as exc:
            raise PersistenceError(f"cannot claim dataset export: {export_id}") from exc

    async def mark_exported(
        self,
        sample_ids: tuple[str, ...],
        export_id: str,
        manifest_sha256: str,
        exported_at: datetime,
    ) -> int:
        if not sample_ids:
            return 0
        try:
            result = await self._collection.update_many(
                {
                    "_id": {"$in": list(dict.fromkeys(sample_ids))},
                    "export.id": export_id,
                    "status": {
                        "$in": [
                            DatasetSampleStatus.EXPORTING.value,
                            DatasetSampleStatus.EXPORTED.value,
                        ]
                    },
                    "$or": [
                        {"export.manifestSha256": None},
                        {"export.manifestSha256": manifest_sha256},
                    ],
                },
                {
                    "$set": {
                        "status": DatasetSampleStatus.EXPORTED.value,
                        "export.exportedAt": exported_at,
                        "export.manifestSha256": manifest_sha256,
                        "export.errorCode": None,
                    }
                },
            )
            return result.matched_count
        except PyMongoError as exc:
            raise PersistenceError(f"cannot complete dataset export: {export_id}") from exc

    async def mark_export_failed(
        self,
        sample_ids: tuple[str, ...],
        export_id: str,
        error_code: str,
    ) -> int:
        if not sample_ids:
            return 0
        try:
            result = await self._collection.update_many(
                {
                    "_id": {"$in": list(dict.fromkeys(sample_ids))},
                    "export.id": export_id,
                    "status": DatasetSampleStatus.EXPORTING.value,
                },
                {
                    "$set": {
                        "status": DatasetSampleStatus.EXPORT_FAILED.value,
                        "export.errorCode": error_code,
                    }
                },
            )
            return result.modified_count
        except PyMongoError as exc:
            raise PersistenceError(f"cannot fail dataset export: {export_id}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()
