"""In-memory identity and bounded vector adapters for tests/embedded mode."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from datetime import datetime

from vehicle_intelligence.application.ports import (
    VectorSearchQuery,
    VehicleEventIdentityLinker,
)
from vehicle_intelligence.domain import (
    EmbeddingVector,
    IdentityMergeReview,
    IdentityReviewAction,
    IdentityReviewResult,
    IdentitySplitReview,
    PlateIdentitySignal,
    VectorNeighbor,
    VehicleFingerprint,
    VehicleIdentity,
    VehicleIdentityStatus,
)
from vehicle_intelligence.exceptions import IdentityConflictError, IdentityNotFoundError


def merge_identity(
    current: VehicleIdentity,
    observation: VehicleIdentity,
) -> VehicleIdentity:
    plates = list(current.plates)
    for signal in observation.plates:
        matched = next(
            (index for index, item in enumerate(plates) if item.text == signal.text),
            None,
        )
        if matched is None:
            if len(plates) < 16:
                plates.append(signal)
            continue
        existing = plates[matched]
        plates[matched] = PlateIdentitySignal(
            text=existing.text,
            confidence=max(existing.confidence, signal.confidence),
            first_seen_at=min(existing.first_seen_at, signal.first_seen_at),
            last_seen_at=max(existing.last_seen_at, signal.last_seen_at),
        )
    return replace(
        current,
        primary_plate=current.primary_plate or observation.primary_plate,
        plates=tuple(plates),
        vehicle_type=current.vehicle_type or observation.vehicle_type,
        color=current.color or observation.color,
        first_seen_at=min(current.first_seen_at, observation.first_seen_at),
        last_seen_at=max(current.last_seen_at, observation.last_seen_at),
        observation_count=current.observation_count + 1,
        revision=current.revision + 1,
    )


class InMemoryVehicleIdentityRepository:
    def __init__(self, events: VehicleEventIdentityLinker | None = None) -> None:
        self._identities: dict[str, VehicleIdentity] = {}
        self._fingerprints: dict[str, VehicleFingerprint] = {}
        self._source_events: set[str] = set()
        self._reviews: dict[
            str,
            tuple[IdentityMergeReview | IdentitySplitReview, IdentityReviewResult],
        ] = {}
        self._events = events
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def register_observation(
        self,
        identity: VehicleIdentity,
        fingerprint: VehicleFingerprint,
    ) -> bool:
        if identity.id != fingerprint.vehicle_id:
            raise ValueError("fingerprint vehicle_id must match identity")
        async with self._lock:
            if (
                fingerprint.id in self._fingerprints
                or fingerprint.source_event_id in self._source_events
            ):
                return False
            current = self._identities.get(identity.id)
            self._identities[identity.id] = (
                identity if current is None else merge_identity(current, identity)
            )
            self._fingerprints[fingerprint.id] = fingerprint
            self._source_events.add(fingerprint.source_event_id)
            return True

    async def get(self, vehicle_id: str) -> VehicleIdentity | None:
        return self._identities.get(vehicle_id)

    async def get_fingerprint(self, fingerprint_id: str) -> VehicleFingerprint | None:
        return self._fingerprints.get(fingerprint_id)

    async def get_fingerprints(
        self,
        fingerprint_ids: tuple[str, ...],
    ) -> tuple[VehicleFingerprint, ...]:
        unique_ids = tuple(dict.fromkeys(fingerprint_ids))
        if len(unique_ids) > 1000:
            raise ValueError("fingerprint batch is bounded to 1000 IDs")
        return tuple(
            fingerprint
            for fingerprint_id in unique_ids
            if (fingerprint := self._fingerprints.get(fingerprint_id)) is not None
        )

    async def list_fingerprints(
        self,
        vehicle_id: str,
        limit: int = 200,
    ) -> tuple[VehicleFingerprint, ...]:
        values = [item for item in self._fingerprints.values() if item.vehicle_id == vehicle_id]
        values.sort(key=lambda item: (item.observed_at, item.id), reverse=True)
        return tuple(values[:limit])

    async def find_by_plate(
        self,
        plate: str,
        limit: int = 20,
    ) -> tuple[VehicleIdentity, ...]:
        values = [
            item for item in self._identities.values() if any(p.text == plate for p in item.plates)
        ]
        values.sort(key=lambda item: (item.last_seen_at, item.id), reverse=True)
        return tuple(values[:limit])

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
        values = [
            item
            for item in self._fingerprints.values()
            if item.camera_id == camera_id and from_time <= item.observed_at <= to_time
        ]
        values.sort(key=lambda item: (item.observed_at, item.id), reverse=True)
        return tuple(values[:limit])

    async def review_merge(self, review: IdentityMergeReview) -> IdentityReviewResult:
        async with self._lock:
            prior = self._reviews.get(review.id)
            if prior is not None:
                if _review_request_key(prior[0]) != _review_request_key(review):
                    raise IdentityConflictError("identity review ID was reused")
                return replace(prior[1], idempotent=True)
            source = self._identities.get(review.source_vehicle_id)
            target = self._identities.get(review.target_vehicle_id)
            if source is None or target is None:
                raise IdentityNotFoundError("merge identity not found")
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
            source_fingerprints = [
                item for item in self._fingerprints.values() if item.vehicle_id == source.id
            ]
            target_fingerprints = [
                item for item in self._fingerprints.values() if item.vehicle_id == target.id
            ]
            if not source_fingerprints or not target_fingerprints:
                raise IdentityConflictError("identity merge requires fingerprint evidence")
            moved = [replace(item, vehicle_id=target.id) for item in source_fingerprints]
            merged = identity_from_fingerprints(
                target.id,
                tuple(target_fingerprints + moved),
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
            for fingerprint in moved:
                self._fingerprints[fingerprint.id] = fingerprint
            self._identities[target.id] = merged
            self._identities[source.id] = merged_source
            event_ids = tuple(item.source_event_id for item in source_fingerprints)
            moved_events = (
                await self._events.reassign_vehicle_ids(event_ids, source.id, target.id)
                if self._events is not None
                else 0
            )
            result = IdentityReviewResult(
                review_id=review.id,
                action=IdentityReviewAction.MERGE,
                source_vehicle_id=source.id,
                result_vehicle_id=target.id,
                moved_fingerprints=len(moved),
                moved_events=moved_events,
                reviewed_at=review.reviewed_at,
            )
            self._reviews[review.id] = (review, result)
            return result

    async def review_split(self, review: IdentitySplitReview) -> IdentityReviewResult:
        async with self._lock:
            prior = self._reviews.get(review.id)
            if prior is not None:
                if _review_request_key(prior[0]) != _review_request_key(review):
                    raise IdentityConflictError("identity review ID was reused")
                return replace(prior[1], idempotent=True)
            source = self._identities.get(review.source_vehicle_id)
            if source is None:
                raise IdentityNotFoundError("split identity not found")
            if source.status is not VehicleIdentityStatus.ACTIVE:
                raise IdentityConflictError("only an active identity can be split")
            if source.revision != review.expected_source_revision:
                raise IdentityConflictError("identity split revision conflict")
            if review.new_vehicle_id in self._identities:
                raise IdentityConflictError("split destination identity already exists")
            all_fingerprints = [
                item for item in self._fingerprints.values() if item.vehicle_id == source.id
            ]
            selected_ids = set(review.fingerprint_ids)
            selected = [item for item in all_fingerprints if item.id in selected_ids]
            remaining = [item for item in all_fingerprints if item.id not in selected_ids]
            if len(selected) != len(selected_ids):
                raise IdentityConflictError("split fingerprint ownership changed")
            if not remaining:
                raise IdentityConflictError("split must leave evidence on the source identity")
            moved = [replace(item, vehicle_id=review.new_vehicle_id) for item in selected]
            self._identities[source.id] = identity_from_fingerprints(
                source.id,
                tuple(remaining),
                revision=source.revision + 1,
                metadata={**source.metadata, "lastSplitReviewId": review.id},
            )
            self._identities[review.new_vehicle_id] = identity_from_fingerprints(
                review.new_vehicle_id,
                tuple(moved),
                revision=1,
                metadata={"splitFrom": source.id, "reviewId": review.id},
            )
            for fingerprint in moved:
                self._fingerprints[fingerprint.id] = fingerprint
            event_ids = tuple(item.source_event_id for item in selected)
            moved_events = (
                await self._events.reassign_vehicle_ids(
                    event_ids,
                    source.id,
                    review.new_vehicle_id,
                )
                if self._events is not None
                else 0
            )
            result = IdentityReviewResult(
                review_id=review.id,
                action=IdentityReviewAction.SPLIT,
                source_vehicle_id=source.id,
                result_vehicle_id=review.new_vehicle_id,
                moved_fingerprints=len(moved),
                moved_events=moved_events,
                reviewed_at=review.reviewed_at,
            )
            self._reviews[review.id] = (review, result)
            return result

    async def get_review(self, review_id: str) -> IdentityReviewResult | None:
        value = self._reviews.get(review_id)
        return value[1] if value is not None else None

    async def close(self) -> None:
        return None


class InMemoryVectorRepository:
    def __init__(self) -> None:
        self._vectors: dict[str, EmbeddingVector] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def put(self, vector: EmbeddingVector) -> bool:
        async with self._lock:
            if vector.id in self._vectors:
                return False
            self._vectors[vector.id] = vector
            return True

    async def get(self, vector_id: str) -> EmbeddingVector | None:
        return self._vectors.get(vector_id)

    async def search(self, query: VectorSearchQuery) -> tuple[VectorNeighbor, ...]:
        norm = math.sqrt(sum(value * value for value in query.vector))
        if norm <= 0:
            raise ValueError("vector search query cannot have zero norm")
        query_values = tuple(value / norm for value in query.vector)
        neighbors: list[VectorNeighbor] = []
        for vector_id in dict.fromkeys(query.candidate_ids):
            candidate = self._vectors.get(vector_id)
            if candidate is None or candidate.model.key != query.model.key:
                continue
            score = sum(
                left * right
                for left, right in zip(
                    query_values,
                    candidate.normalized_values,
                    strict=True,
                )
            )
            if score >= query.minimum_score:
                neighbors.append(VectorNeighbor(vector_id, min(1.0, max(-1.0, score))))
        neighbors.sort(key=lambda item: (item.score, item.vector_id), reverse=True)
        return tuple(neighbors[: query.limit])

    async def close(self) -> None:
        return None


def identity_from_fingerprints(
    vehicle_id: str,
    fingerprints: tuple[VehicleFingerprint, ...],
    *,
    revision: int,
    metadata: dict[str, object],
) -> VehicleIdentity:
    if not fingerprints:
        raise ValueError("identity aggregate requires at least one fingerprint")
    plates: dict[str, PlateIdentitySignal] = {}
    for item in fingerprints:
        if item.plate is None or item.plate_confidence is None:
            continue
        current = plates.get(item.plate)
        signal = PlateIdentitySignal(
            text=item.plate,
            confidence=item.plate_confidence,
            first_seen_at=item.observed_at,
            last_seen_at=item.observed_at,
        )
        if current is None:
            plates[item.plate] = signal
        else:
            plates[item.plate] = PlateIdentitySignal(
                text=current.text,
                confidence=max(current.confidence, signal.confidence),
                first_seen_at=min(current.first_seen_at, signal.first_seen_at),
                last_seen_at=max(current.last_seen_at, signal.last_seen_at),
            )
    plate_signals = sorted(
        plates.values(),
        key=lambda item: (item.confidence, item.last_seen_at, item.text),
        reverse=True,
    )[:16]
    type_scores: dict[str, float] = {}
    color_counts: dict[str, int] = {}
    for item in fingerprints:
        type_scores[item.vehicle_type] = (
            type_scores.get(item.vehicle_type, 0.0) + item.vehicle_confidence
        )
        if item.color is not None:
            color_counts[item.color] = color_counts.get(item.color, 0) + 1
    vehicle_type = max(type_scores, key=lambda value: (type_scores[value], value))
    color = (
        max(color_counts, key=lambda value: (color_counts[value], value)) if color_counts else None
    )
    return VehicleIdentity(
        id=vehicle_id,
        primary_plate=plate_signals[0].text if plate_signals else None,
        plates=tuple(plate_signals),
        vehicle_type=vehicle_type,
        color=color,
        first_seen_at=min(item.observed_at for item in fingerprints),
        last_seen_at=max(item.observed_at for item in fingerprints),
        observation_count=len(fingerprints),
        revision=revision,
        metadata=metadata,
    )


def _review_request_key(
    review: IdentityMergeReview | IdentitySplitReview,
) -> tuple[object, ...]:
    if isinstance(review, IdentityMergeReview):
        return (
            IdentityReviewAction.MERGE,
            review.source_vehicle_id,
            review.target_vehicle_id,
            review.expected_source_revision,
            review.expected_target_revision,
            review.reviewer.id,
            review.reason,
            review.source_fingerprint_id,
            review.target_fingerprint_id,
        )
    return (
        IdentityReviewAction.SPLIT,
        review.source_vehicle_id,
        review.new_vehicle_id,
        review.expected_source_revision,
        review.reviewer.id,
        review.reason,
        tuple(sorted(review.fingerprint_ids)),
    )
