"""Bounded MongoDB identity/fingerprint documents; event history stays separate."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    EmbeddingModel,
    EmbeddingReference,
    IdentityMergeReview,
    IdentityReviewAction,
    IdentityReviewResult,
    IdentitySplitReview,
    PlateIdentitySignal,
    VehicleFingerprint,
    VehicleIdentity,
    VehicleIdentityStatus,
)
from vehicle_intelligence.exceptions import (
    IdentityConflictError,
    IdentityNotFoundError,
    PersistenceError,
)
from vehicle_intelligence.infrastructure.persistence.constants import (
    IDENTITY_REVIEWS,
    VEHICLE_EVENTS,
    VEHICLE_FINGERPRINTS,
    VEHICLES,
)
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    identity_from_fingerprints,
    merge_identity,
)
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class MongoVehicleIdentityRepository:
    def __init__(self, source: MongoConfig | MongoRuntime) -> None:
        binding = bind_mongo(source)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._database = binding.database
        self._identities = self._database[VEHICLES]
        self._fingerprints = self._database[VEHICLE_FINGERPRINTS]
        self._reviews = self._database[IDENTITY_REVIEWS]
        self._events = self._database[VEHICLE_EVENTS]
        self._runtime = source if isinstance(source, MongoRuntime) else None

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._identities.create_indexes(
                [
                    IndexModel(
                        [("primaryPlate", ASCENDING), ("lastSeenAt", DESCENDING)],
                        name="ix_vehicle_primary_plate_time",
                        partialFilterExpression={"primaryPlate": {"$type": "string"}},
                    ),
                    IndexModel(
                        [("plates.text", ASCENDING), ("lastSeenAt", DESCENDING)],
                        name="ix_vehicle_plate_alias_time",
                    ),
                    IndexModel(
                        [("status", ASCENDING), ("lastSeenAt", DESCENDING)],
                        name="ix_vehicle_status_time",
                    ),
                ]
            )
            await self._fingerprints.create_indexes(
                [
                    IndexModel(
                        [("sourceEventId", ASCENDING)],
                        unique=True,
                        name="uq_fingerprint_source_event",
                    ),
                    IndexModel(
                        [("vehicleId", ASCENDING), ("observedAt", DESCENDING)],
                        name="ix_fingerprint_vehicle_time",
                    ),
                    IndexModel(
                        [("cameraId", ASCENDING), ("observedAt", DESCENDING)],
                        name="ix_fingerprint_camera_time",
                    ),
                    IndexModel(
                        [("embedding.id", ASCENDING)],
                        name="ix_fingerprint_embedding",
                        partialFilterExpression={"embedding.id": {"$type": "string"}},
                    ),
                ]
            )
            await self._reviews.create_indexes(
                [
                    IndexModel(
                        [("sourceVehicleId", ASCENDING), ("reviewedAt", DESCENDING)],
                        name="ix_identity_review_source_time",
                    ),
                    IndexModel(
                        [("reviewer.id", ASCENDING), ("reviewedAt", DESCENDING)],
                        name="ix_identity_review_actor_time",
                    ),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize vehicle identity indexes") from exc

    async def register_observation(
        self,
        identity: VehicleIdentity,
        fingerprint: VehicleFingerprint,
    ) -> bool:
        if identity.id != fingerprint.vehicle_id:
            raise ValueError("fingerprint vehicle_id must match identity")
        try:
            async with self._transaction():
                inserted = await self._fingerprints.update_one(
                    {"_id": fingerprint.id},
                    {"$setOnInsert": _fingerprint_to_document(fingerprint)},
                    upsert=True,
                )
                if inserted.upserted_id is None:
                    return False
                current_document = await self._identities.find_one({"_id": identity.id})
                if current_document is None:
                    await self._identities.insert_one(_identity_to_document(identity))
                else:
                    current = _document_to_identity(current_document)
                    merged = merge_identity(current, identity)
                    result = await self._identities.replace_one(
                        {"_id": identity.id, "revision": current.revision},
                        _identity_to_document(merged),
                    )
                    if result.matched_count != 1:
                        raise PersistenceError(
                            f"concurrent vehicle identity update: {identity.id}"
                        )
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(
                f"cannot register vehicle fingerprint: {fingerprint.id}"
            ) from exc

    async def get(self, vehicle_id: str) -> VehicleIdentity | None:
        try:
            document = await self._identities.find_one({"_id": vehicle_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read vehicle identity: {vehicle_id}") from exc
        return _document_to_identity(document) if document is not None else None

    async def get_fingerprint(self, fingerprint_id: str) -> VehicleFingerprint | None:
        try:
            document = await self._fingerprints.find_one({"_id": fingerprint_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read vehicle fingerprint: {fingerprint_id}") from exc
        return _document_to_fingerprint(document) if document is not None else None

    async def list_fingerprints(
        self,
        vehicle_id: str,
        limit: int = 200,
    ) -> tuple[VehicleFingerprint, ...]:
        if not 1 <= limit <= 1000:
            raise ValueError("fingerprint list limit must be in [1, 1000]")
        try:
            cursor = (
                self._fingerprints.find({"vehicleId": vehicle_id})
                .sort([("observedAt", DESCENDING), ("_id", DESCENDING)])
                .limit(limit)
            )
            return tuple([_document_to_fingerprint(item) async for item in cursor])
        except PyMongoError as exc:
            raise PersistenceError(f"cannot list fingerprints: {vehicle_id}") from exc

    async def find_by_plate(
        self,
        plate: str,
        limit: int = 20,
    ) -> tuple[VehicleIdentity, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("vehicle identity search limit must be in [1, 200]")
        try:
            cursor = (
                self._identities.find({"plates.text": plate})
                .sort([("lastSeenAt", DESCENDING), ("_id", DESCENDING)])
                .limit(limit)
            )
            return tuple([_document_to_identity(item) async for item in cursor])
        except PyMongoError as exc:
            raise PersistenceError(f"cannot find vehicle identities by plate: {plate}") from exc

    async def find_fingerprints_by_camera_time(
        self,
        camera_id: str,
        from_time: datetime,
        to_time: datetime,
        limit: int,
    ) -> tuple[VehicleFingerprint, ...]:
        if from_time.tzinfo is None or to_time.tzinfo is None or from_time > to_time:
            raise ValueError("fingerprint time window is invalid")
        if not 1 <= limit <= 1000:
            raise ValueError("fingerprint window limit must be in [1, 1000]")
        try:
            cursor = (
                self._fingerprints.find(
                    {
                        "cameraId": camera_id,
                        "observedAt": {
                            "$gte": from_time.astimezone(UTC),
                            "$lte": to_time.astimezone(UTC),
                        },
                    }
                )
                .sort([("observedAt", DESCENDING), ("_id", DESCENDING)])
                .limit(limit)
            )
            return tuple([_document_to_fingerprint(item) async for item in cursor])
        except PyMongoError as exc:
            raise PersistenceError(
                f"cannot find fingerprints in camera window: {camera_id}"
            ) from exc

    async def review_merge(self, review: IdentityMergeReview) -> IdentityReviewResult:
        try:
            async with self._transaction():
                prior = await self._reviews.find_one({"_id": review.id})
                if prior is not None:
                    if not _review_matches(prior, review):
                        raise IdentityConflictError("identity review ID was reused")
                    return replace(_review_result(prior), idempotent=True)
                source_document = await self._identities.find_one(
                    {"_id": review.source_vehicle_id}
                )
                target_document = await self._identities.find_one(
                    {"_id": review.target_vehicle_id}
                )
                if source_document is None or target_document is None:
                    raise IdentityNotFoundError("merge identity not found")
                source = _document_to_identity(source_document)
                target = _document_to_identity(target_document)
                if (
                    source.status is not VehicleIdentityStatus.ACTIVE
                    or target.status is not VehicleIdentityStatus.ACTIVE
                ):
                    raise IdentityConflictError("only active identities can be merged")
                if (
                    source.revision != review.expected_source_revision
                    or target.revision != review.expected_target_revision
                ):
                    raise IdentityConflictError("identity merge revision conflict")
                source_fingerprints = await self._bounded_fingerprints(source.id)
                target_fingerprints = await self._bounded_fingerprints(target.id)
                if not source_fingerprints or not target_fingerprints:
                    raise IdentityConflictError(
                        "identity merge requires fingerprint evidence"
                    )
                moved = tuple(
                    replace(item, vehicle_id=target.id)
                    for item in source_fingerprints
                )
                merged = identity_from_fingerprints(
                    target.id,
                    tuple(target_fingerprints) + moved,
                    revision=target.revision + 1,
                    metadata=dict(target.metadata),
                )
                merged_source = replace(
                    source,
                    status=VehicleIdentityStatus.MERGED,
                    revision=source.revision + 1,
                    metadata={
                        **source.metadata,
                        "mergedInto": target.id,
                        "reviewId": review.id,
                    },
                )
                target_update = await self._identities.replace_one(
                    {"_id": target.id, "revision": target.revision},
                    _identity_to_document(merged),
                )
                source_update = await self._identities.replace_one(
                    {"_id": source.id, "revision": source.revision},
                    _identity_to_document(merged_source),
                )
                if target_update.matched_count != 1 or source_update.matched_count != 1:
                    raise IdentityConflictError("identity merge revision conflict")
                fingerprint_ids = [item.id for item in source_fingerprints]
                fingerprint_update = await self._fingerprints.update_many(
                    {"_id": {"$in": fingerprint_ids}, "vehicleId": source.id},
                    {"$set": {"vehicleId": target.id}},
                )
                if fingerprint_update.modified_count != len(fingerprint_ids):
                    raise IdentityConflictError("identity fingerprint ownership changed")
                event_ids = [item.source_event_id for item in source_fingerprints]
                event_update = await self._events.update_many(
                    {"_id": {"$in": event_ids}, "vehicleId": source.id},
                    {"$set": {"vehicleId": target.id}},
                )
                result = IdentityReviewResult(
                    review_id=review.id,
                    action=IdentityReviewAction.MERGE,
                    source_vehicle_id=source.id,
                    result_vehicle_id=target.id,
                    moved_fingerprints=len(fingerprint_ids),
                    moved_events=event_update.modified_count,
                    reviewed_at=review.reviewed_at,
                )
                await self._reviews.insert_one(_merge_review_document(review, result))
                return result
        except (IdentityConflictError, IdentityNotFoundError):
            raise
        except PyMongoError as exc:
            raise PersistenceError("cannot apply identity merge review") from exc

    async def review_split(self, review: IdentitySplitReview) -> IdentityReviewResult:
        try:
            async with self._transaction():
                prior = await self._reviews.find_one({"_id": review.id})
                if prior is not None:
                    if not _review_matches(prior, review):
                        raise IdentityConflictError("identity review ID was reused")
                    return replace(_review_result(prior), idempotent=True)
                source_document = await self._identities.find_one(
                    {"_id": review.source_vehicle_id}
                )
                if source_document is None:
                    raise IdentityNotFoundError("split identity not found")
                source = _document_to_identity(source_document)
                if source.status is not VehicleIdentityStatus.ACTIVE:
                    raise IdentityConflictError("only an active identity can be split")
                if source.revision != review.expected_source_revision:
                    raise IdentityConflictError("identity split revision conflict")
                if await self._identities.find_one({"_id": review.new_vehicle_id}):
                    raise IdentityConflictError("split destination identity already exists")
                all_fingerprints = await self._bounded_fingerprints(source.id)
                selected_ids = set(review.fingerprint_ids)
                selected = tuple(
                    item for item in all_fingerprints if item.id in selected_ids
                )
                remaining = tuple(
                    item for item in all_fingerprints if item.id not in selected_ids
                )
                if len(selected) != len(selected_ids):
                    raise IdentityConflictError("split fingerprint ownership changed")
                if not remaining:
                    raise IdentityConflictError(
                        "split must leave evidence on the source identity"
                    )
                moved = tuple(
                    replace(item, vehicle_id=review.new_vehicle_id)
                    for item in selected
                )
                remaining_identity = identity_from_fingerprints(
                    source.id,
                    remaining,
                    revision=source.revision + 1,
                    metadata={**source.metadata, "lastSplitReviewId": review.id},
                )
                new_identity = identity_from_fingerprints(
                    review.new_vehicle_id,
                    moved,
                    revision=1,
                    metadata={"splitFrom": source.id, "reviewId": review.id},
                )
                source_update = await self._identities.replace_one(
                    {"_id": source.id, "revision": source.revision},
                    _identity_to_document(remaining_identity),
                )
                if source_update.matched_count != 1:
                    raise IdentityConflictError("identity split revision conflict")
                await self._identities.insert_one(_identity_to_document(new_identity))
                fingerprint_ids = [item.id for item in selected]
                fingerprint_update = await self._fingerprints.update_many(
                    {"_id": {"$in": fingerprint_ids}, "vehicleId": source.id},
                    {"$set": {"vehicleId": review.new_vehicle_id}},
                )
                if fingerprint_update.modified_count != len(fingerprint_ids):
                    raise IdentityConflictError("identity fingerprint ownership changed")
                event_ids = [item.source_event_id for item in selected]
                event_update = await self._events.update_many(
                    {"_id": {"$in": event_ids}, "vehicleId": source.id},
                    {"$set": {"vehicleId": review.new_vehicle_id}},
                )
                result = IdentityReviewResult(
                    review_id=review.id,
                    action=IdentityReviewAction.SPLIT,
                    source_vehicle_id=source.id,
                    result_vehicle_id=review.new_vehicle_id,
                    moved_fingerprints=len(fingerprint_ids),
                    moved_events=event_update.modified_count,
                    reviewed_at=review.reviewed_at,
                )
                await self._reviews.insert_one(_split_review_document(review, result))
                return result
        except (IdentityConflictError, IdentityNotFoundError):
            raise
        except DuplicateKeyError as exc:
            raise IdentityConflictError("identity split destination or review exists") from exc
        except PyMongoError as exc:
            raise PersistenceError("cannot apply identity split review") from exc

    async def get_review(self, review_id: str) -> IdentityReviewResult | None:
        try:
            document = await self._reviews.find_one({"_id": review_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read identity review: {review_id}") from exc
        return _review_result(document) if document is not None else None

    async def _bounded_fingerprints(
        self,
        vehicle_id: str,
    ) -> tuple[VehicleFingerprint, ...]:
        documents = [
            item
            async for item in self._fingerprints.find({"vehicleId": vehicle_id})
            .sort([("observedAt", DESCENDING), ("_id", DESCENDING)])
            .limit(1001)
        ]
        if len(documents) > 1000:
            raise IdentityConflictError(
                "identity review exceeds the bounded 1000-fingerprint operation"
            )
        return tuple(_document_to_fingerprint(item) for item in documents)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        if self._runtime is None:
            yield
            return
        async with self._runtime.transaction():
            yield


def _identity_to_document(identity: VehicleIdentity) -> dict[str, Any]:
    return {
        "_id": identity.id,
        "schemaVersion": identity.schema_version,
        "revision": identity.revision,
        "status": identity.status.value,
        "primaryPlate": identity.primary_plate,
        "plates": [
            {
                "text": plate.text,
                "confidence": plate.confidence,
                "firstSeenAt": plate.first_seen_at.astimezone(UTC),
                "lastSeenAt": plate.last_seen_at.astimezone(UTC),
            }
            for plate in identity.plates
        ],
        "attributes": {"type": identity.vehicle_type, "color": identity.color},
        "firstSeenAt": identity.first_seen_at.astimezone(UTC),
        "lastSeenAt": identity.last_seen_at.astimezone(UTC),
        "observationCount": identity.observation_count,
        "metadata": identity.metadata,
    }


def _document_to_identity(document: dict[str, Any]) -> VehicleIdentity:
    attributes = document.get("attributes") or {}
    return VehicleIdentity(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        revision=int(document.get("revision", 1)),
        status=VehicleIdentityStatus(document.get("status", "ACTIVE")),
        primary_plate=document.get("primaryPlate"),
        plates=tuple(
            PlateIdentitySignal(
                text=str(plate["text"]),
                confidence=float(plate["confidence"]),
                first_seen_at=_aware(plate["firstSeenAt"]),
                last_seen_at=_aware(plate["lastSeenAt"]),
            )
            for plate in document.get("plates") or []
        ),
        vehicle_type=attributes.get("type"),
        color=attributes.get("color"),
        first_seen_at=_aware(document["firstSeenAt"]),
        last_seen_at=_aware(document["lastSeenAt"]),
        observation_count=int(document["observationCount"]),
        metadata=document.get("metadata") or {},
    )


def _fingerprint_to_document(fingerprint: VehicleFingerprint) -> dict[str, Any]:
    embedding = fingerprint.embedding
    return {
        "_id": fingerprint.id,
        "schemaVersion": fingerprint.schema_version,
        "vehicleId": fingerprint.vehicle_id,
        "sourceEventId": fingerprint.source_event_id,
        "cameraId": fingerprint.camera_id,
        "observedAt": fingerprint.observed_at.astimezone(UTC),
        "plate": (
            {"text": fingerprint.plate, "confidence": fingerprint.plate_confidence}
            if fingerprint.plate is not None
            else None
        ),
        "vehicle": {
            "type": fingerprint.vehicle_type,
            "confidence": fingerprint.vehicle_confidence,
            "color": fingerprint.color,
        },
        "embedding": (
            {
                "id": embedding.id,
                "model": {
                    "name": embedding.model.name,
                    "version": embedding.model.version,
                    "hash": embedding.model.model_hash,
                    "dimension": embedding.model.dimension,
                },
            }
            if embedding is not None
            else None
        ),
    }


def _document_to_fingerprint(document: dict[str, Any]) -> VehicleFingerprint:
    plate = document.get("plate")
    vehicle = document["vehicle"]
    raw_embedding = document.get("embedding")
    embedding = None
    if raw_embedding is not None:
        raw_model = raw_embedding["model"]
        embedding = EmbeddingReference(
            id=str(raw_embedding["id"]),
            model=EmbeddingModel(
                name=str(raw_model["name"]),
                version=str(raw_model["version"]),
                model_hash=raw_model.get("hash"),
                dimension=int(raw_model["dimension"]),
            ),
        )
    return VehicleFingerprint(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        vehicle_id=str(document["vehicleId"]),
        source_event_id=str(document["sourceEventId"]),
        camera_id=str(document["cameraId"]),
        observed_at=_aware(document["observedAt"]),
        plate=str(plate["text"]) if plate is not None else None,
        plate_confidence=float(plate["confidence"]) if plate is not None else None,
        vehicle_type=str(vehicle["type"]),
        vehicle_confidence=float(vehicle["confidence"]),
        color=vehicle.get("color"),
        embedding=embedding,
    )


def _merge_review_document(
    review: IdentityMergeReview,
    result: IdentityReviewResult,
) -> dict[str, Any]:
    return {
        "_id": review.id,
        "schemaVersion": 1,
        "action": IdentityReviewAction.MERGE.value,
        "sourceVehicleId": review.source_vehicle_id,
        "resultVehicleId": review.target_vehicle_id,
        "expectedSourceRevision": review.expected_source_revision,
        "expectedTargetRevision": review.expected_target_revision,
        "fingerprints": {
            "source": review.source_fingerprint_id,
            "target": review.target_fingerprint_id,
        },
        "score": review.score,
        "reviewer": {
            "id": review.reviewer.id,
            "displayName": review.reviewer.display_name,
        },
        "reason": review.reason,
        "reviewedAt": review.reviewed_at.astimezone(UTC),
        "result": {
            "movedFingerprints": result.moved_fingerprints,
            "movedEvents": result.moved_events,
        },
    }


def _split_review_document(
    review: IdentitySplitReview,
    result: IdentityReviewResult,
) -> dict[str, Any]:
    return {
        "_id": review.id,
        "schemaVersion": 1,
        "action": IdentityReviewAction.SPLIT.value,
        "sourceVehicleId": review.source_vehicle_id,
        "resultVehicleId": review.new_vehicle_id,
        "expectedSourceRevision": review.expected_source_revision,
        "fingerprintIds": sorted(review.fingerprint_ids),
        "reviewer": {
            "id": review.reviewer.id,
            "displayName": review.reviewer.display_name,
        },
        "reason": review.reason,
        "reviewedAt": review.reviewed_at.astimezone(UTC),
        "result": {
            "movedFingerprints": result.moved_fingerprints,
            "movedEvents": result.moved_events,
        },
    }


def _review_matches(
    document: dict[str, Any],
    review: IdentityMergeReview | IdentitySplitReview,
) -> bool:
    common = (
        document.get("sourceVehicleId") == review.source_vehicle_id
        and document.get("expectedSourceRevision") == review.expected_source_revision
        and (document.get("reviewer") or {}).get("id") == review.reviewer.id
        and document.get("reason") == review.reason
    )
    if isinstance(review, IdentityMergeReview):
        fingerprints = document.get("fingerprints") or {}
        return bool(
            common
            and document.get("action") == IdentityReviewAction.MERGE.value
            and document.get("resultVehicleId") == review.target_vehicle_id
            and document.get("expectedTargetRevision")
            == review.expected_target_revision
            and fingerprints.get("source") == review.source_fingerprint_id
            and fingerprints.get("target") == review.target_fingerprint_id
        )
    return bool(
        common
        and document.get("action") == IdentityReviewAction.SPLIT.value
        and document.get("resultVehicleId") == review.new_vehicle_id
        and sorted(document.get("fingerprintIds") or [])
        == sorted(review.fingerprint_ids)
    )


def _review_result(document: dict[str, Any]) -> IdentityReviewResult:
    result = document["result"]
    reviewed_at: datetime = document["reviewedAt"]
    if reviewed_at.tzinfo is None:
        reviewed_at = reviewed_at.replace(tzinfo=UTC)
    return IdentityReviewResult(
        review_id=str(document["_id"]),
        action=IdentityReviewAction(document["action"]),
        source_vehicle_id=str(document["sourceVehicleId"]),
        result_vehicle_id=str(document["resultVehicleId"]),
        moved_fingerprints=int(result["movedFingerprints"]),
        moved_events=int(result["movedEvents"]),
        reviewed_at=reviewed_at.astimezone(UTC),
    )
