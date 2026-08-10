"""Server-side MongoDB aggregation for model-quality reporting."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo.errors import PyMongoError

from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    DailyQualityPoint,
    DatasetFeedbackMetrics,
    DatasetSampleReason,
    DatasetSampleStatus,
    ModelMetadata,
    ModelQualityReport,
    ModelQualitySlice,
)
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import (
    DATASET_SAMPLES,
    VEHICLE_EVENTS,
)
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo
from vehicle_intelligence.infrastructure.persistence.quality_common import QualityCounts


class MongoModelQualityRepository:
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        binding = bind_mongo(config)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._events = binding.database[VEHICLE_EVENTS]
        self._samples = binding.database[DATASET_SAMPLES]

    async def summarize(
        self,
        from_time: datetime,
        to_time: datetime,
        generated_at: datetime,
        maximum_models: int,
    ) -> ModelQualityReport:
        try:
            event_cursor = await self._events.aggregate(
                _event_pipeline(from_time, to_time, maximum_models)
            )
            event_documents = [document async for document in event_cursor]
            feedback_cursor = await self._samples.aggregate(
                [
                    {
                        "$match": {
                            "review.reviewedAt": {"$gte": from_time, "$lt": to_time}
                        }
                    },
                    {
                        "$group": {
                            "_id": {"status": "$status", "reason": "$reason"},
                            "count": {"$sum": 1},
                        }
                    },
                ]
            )
            feedback_documents = [document async for document in feedback_cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot aggregate model quality") from exc

        facets = event_documents[0] if event_documents else {}
        totals = facets.get("totals") or []
        models = facets.get("models") or []
        daily = facets.get("daily") or []
        return ModelQualityReport(
            from_time=from_time,
            to_time=to_time,
            generated_at=generated_at,
            totals=_metrics(totals[0] if totals else {}),
            models=tuple(
                ModelQualitySlice(
                    model=_model(document.get("_id")),
                    metrics=_metrics(document),
                )
                for document in models
            ),
            daily=tuple(
                DailyQualityPoint(day=str(document["_id"]), metrics=_metrics(document))
                for document in daily
            ),
            feedback=_feedback(feedback_documents),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


def _event_pipeline(
    from_time: datetime,
    to_time: datetime,
    maximum_models: int,
) -> list[dict[str, Any]]:
    group = _group_fields()
    return [
        {"$match": {"occurredAt": {"$gte": from_time, "$lt": to_time}}},
        {
            "$set": {
                "_qualityModel": {
                    "$cond": [
                        {"$eq": [{"$ifNull": ["$ai.ocr", None]}, None]},
                        None,
                        {
                            "name": {"$ifNull": ["$ai.ocr.model", "unknown"]},
                            "version": {"$ifNull": ["$ai.ocr.version", "unknown"]},
                            "hash": {"$ifNull": ["$ai.ocr.hash", None]},
                        },
                    ]
                },
                "_qualityReadable": {"$cond": [{"$ne": ["$plate", None]}, 1, 0]},
                "_qualityReviewed": {
                    "$cond": [
                        {"$ne": [{"$ifNull": ["$plate.review", None]}, None]},
                        1,
                        0,
                    ]
                },
                "_qualityCorrected": {
                    "$cond": [
                        {
                            "$and": [
                                {
                                    "$ne": [
                                        {"$ifNull": ["$plate.review", None]},
                                        None,
                                    ]
                                },
                                {
                                    "$ne": [
                                        "$plate.review.normalized",
                                        {
                                            "$ifNull": [
                                                "$plate.prediction.normalized",
                                                "$plate.normalized",
                                            ]
                                        },
                                    ]
                                },
                            ]
                        },
                        1,
                        0,
                    ]
                },
                "_qualityConfidence": {
                    "$cond": [
                        {"$ne": ["$plate", None]},
                        {
                            "$ifNull": [
                                "$plate.prediction.confidence",
                                "$plate.confidence",
                            ]
                        },
                        None,
                    ]
                },
                "_qualityDay": {
                    "$dateToString": {
                        "date": "$occurredAt",
                        "format": "%Y-%m-%d",
                        "timezone": "UTC",
                    }
                },
            }
        },
        {
            "$facet": {
                "totals": [{"$group": {"_id": None, **group}}],
                "models": [
                    {"$group": {"_id": "$_qualityModel", **group}},
                    {"$sort": {"eventCount": -1, "_id.name": 1, "_id.version": 1}},
                    {"$limit": maximum_models},
                ],
                "daily": [
                    {"$group": {"_id": "$_qualityDay", **group}},
                    {"$sort": {"_id": 1}},
                ],
            }
        },
    ]


def _group_fields() -> dict[str, Any]:
    def status(name: str) -> dict[str, Any]:
        return {"$sum": {"$cond": [{"$eq": ["$status", name]}, 1, 0]}}

    return {
        "eventCount": {"$sum": 1},
        "readablePlateCount": {"$sum": "$_qualityReadable"},
        "confirmedCount": status("CONFIRMED"),
        "needsReviewCount": status("NEEDS_REVIEW"),
        "noPlateCount": status("NO_PLATE"),
        "unreadableCount": status("UNREADABLE"),
        "reviewedCount": {"$sum": "$_qualityReviewed"},
        "correctedCount": {"$sum": "$_qualityCorrected"},
        "averagePlateConfidence": {"$avg": "$_qualityConfidence"},
    }


def _metrics(document: dict[str, Any]):
    counts = QualityCounts(
        event_count=int(document.get("eventCount", 0)),
        readable_plate_count=int(document.get("readablePlateCount", 0)),
        confirmed_count=int(document.get("confirmedCount", 0)),
        needs_review_count=int(document.get("needsReviewCount", 0)),
        no_plate_count=int(document.get("noPlateCount", 0)),
        unreadable_count=int(document.get("unreadableCount", 0)),
        reviewed_count=int(document.get("reviewedCount", 0)),
        corrected_count=int(document.get("correctedCount", 0)),
    )
    average = document.get("averagePlateConfidence")
    if average is not None and counts.readable_plate_count:
        counts.confidence_count = counts.readable_plate_count
        counts.confidence_sum = float(average) * counts.readable_plate_count
    return counts.metrics()


def _model(value: dict[str, Any] | None) -> ModelMetadata | None:
    if not value:
        return None
    return ModelMetadata(
        name=str(value["name"]),
        version=str(value["version"]),
        hash=value.get("hash"),
    )


def _feedback(documents: list[dict[str, Any]]) -> DatasetFeedbackMetrics:
    statuses: dict[str, int] = {}
    reasons: dict[str, int] = {}
    total = 0
    for document in documents:
        count = int(document["count"])
        key = document.get("_id") or {}
        statuses[str(key.get("status"))] = statuses.get(str(key.get("status")), 0) + count
        reasons[str(key.get("reason"))] = reasons.get(str(key.get("reason")), 0) + count
        total += count
    return DatasetFeedbackMetrics(
        total=total,
        ready=statuses.get(DatasetSampleStatus.READY.value, 0),
        exporting=statuses.get(DatasetSampleStatus.EXPORTING.value, 0),
        exported=statuses.get(DatasetSampleStatus.EXPORTED.value, 0),
        export_failed=statuses.get(DatasetSampleStatus.EXPORT_FAILED.value, 0),
        corrections=reasons.get(DatasetSampleReason.HUMAN_CORRECTION.value, 0),
        confirmations=reasons.get(DatasetSampleReason.HUMAN_CONFIRMATION.value, 0),
    )
