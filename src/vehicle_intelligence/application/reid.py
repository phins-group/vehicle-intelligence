"""Versioned multi-signal ReID scoring and explicit human identity reviews."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from vehicle_intelligence.application.ports import (
    VectorRepository,
    VectorSearchQuery,
    VehicleIdentityRepository,
)
from vehicle_intelligence.application.topology import CrossCameraCandidateGenerator
from vehicle_intelligence.config import ReIDConfig
from vehicle_intelligence.domain import (
    IdentityMergeReview,
    IdentityReviewer,
    IdentityReviewResult,
    IdentitySplitReview,
    Principal,
    ReIDScore,
    ReIDSignals,
    ReIDVerdict,
    VehicleFingerprint,
)
from vehicle_intelligence.exceptions import IdentityConflictError, IdentityNotFoundError


@dataclass(frozen=True, slots=True)
class MergeIdentities:
    review_id: str
    source_vehicle_id: str
    target_vehicle_id: str
    expected_source_revision: int
    expected_target_revision: int
    reason: str
    source_fingerprint_id: str | None = None
    target_fingerprint_id: str | None = None


@dataclass(frozen=True, slots=True)
class SplitIdentity:
    review_id: str
    source_vehicle_id: str
    expected_source_revision: int
    fingerprint_ids: tuple[str, ...]
    reason: str


class ReIDScoringService:
    def __init__(
        self,
        identities: VehicleIdentityRepository,
        candidates: CrossCameraCandidateGenerator,
        vectors: VectorRepository,
        config: ReIDConfig,
    ) -> None:
        self._identities = identities
        self._candidates = candidates
        self._vectors = vectors
        self._config = config

    async def initialize(self) -> None:
        await self._vectors.ensure_indexes()

    async def close(self) -> None:
        await self._vectors.close()

    async def score_candidates(
        self,
        source_fingerprint_id: str,
        limit: int | None = None,
    ) -> tuple[ReIDScore, ...]:
        requested = self._config.maximum_scored_candidates if limit is None else limit
        if not 1 <= requested <= self._config.maximum_scored_candidates:
            raise ValueError(
                "scored candidate limit must be in "
                f"[1, {self._config.maximum_scored_candidates}]"
            )
        source = await self._identities.get_fingerprint(source_fingerprint_id)
        if source is None:
            raise IdentityNotFoundError(
                f"vehicle fingerprint not found: {source_fingerprint_id}"
            )
        feasible = await self._candidates.generate(source_fingerprint_id, requested)
        results: list[ReIDScore] = []
        for candidate in feasible:
            fingerprint = await self._identities.get_fingerprint(candidate.fingerprint_id)
            if fingerprint is None:
                continue
            results.append(
                await self._score_pair(
                    source,
                    fingerprint,
                    candidate.time_score,
                    candidate.topology_edge_id,
                )
            )
        results.sort(
            key=lambda item: (item.score, item.candidate_fingerprint_id),
            reverse=True,
        )
        return tuple(results[:requested])

    async def _score_pair(
        self,
        source: VehicleFingerprint,
        candidate: VehicleFingerprint,
        time_score: float,
        topology_edge_id: str,
    ) -> ReIDScore:
        signals = ReIDSignals(
            plate=(
                _text_similarity(source.plate, candidate.plate)
                if source.plate is not None and candidate.plate is not None
                else None
            ),
            embedding=await self._embedding_similarity(source, candidate),
            vehicle_type=(1.0 if source.vehicle_type == candidate.vehicle_type else 0.0),
            color=_attribute_similarity(source.color, candidate.color),
            travel_time=time_score,
        )
        score = _weighted_score(signals, self._config)
        if score >= self._config.match_threshold:
            verdict = ReIDVerdict.MATCH
        elif score >= self._config.review_threshold:
            verdict = ReIDVerdict.REVIEW
        else:
            verdict = ReIDVerdict.REJECT
        return ReIDScore(
            source_fingerprint_id=source.id,
            candidate_fingerprint_id=candidate.id,
            source_vehicle_id=source.vehicle_id,
            candidate_vehicle_id=candidate.vehicle_id,
            score=score,
            verdict=verdict,
            signals=signals,
            scoring_version=self._config.scoring_version,
            topology_edge_id=topology_edge_id,
        )

    async def _embedding_similarity(
        self,
        source: VehicleFingerprint,
        candidate: VehicleFingerprint,
    ) -> float | None:
        if source.embedding is None or candidate.embedding is None:
            return None
        if source.embedding.model != candidate.embedding.model:
            return None
        source_vector = await self._vectors.get(source.embedding.id)
        if source_vector is None:
            return None
        neighbors = await self._vectors.search(
            VectorSearchQuery(
                vector=source_vector.normalized_values,
                model=source.embedding.model,
                candidate_ids=(candidate.embedding.id,),
                limit=1,
                minimum_score=-1,
            )
        )
        if not neighbors:
            return None
        return min(1.0, max(0.0, neighbors[0].score))


class IdentityReviewService:
    def __init__(
        self,
        identities: VehicleIdentityRepository,
        scoring: ReIDScoringService,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._identities = identities
        self._scoring = scoring
        self._clock = clock

    async def merge(
        self,
        command: MergeIdentities,
        principal: Principal,
    ) -> IdentityReviewResult:
        score: float | None = None
        if (command.source_fingerprint_id is None) != (
            command.target_fingerprint_id is None
        ):
            raise ValueError("both merge fingerprints must be supplied together")
        if await self._identities.get_review(command.review_id) is not None:
            retry = IdentityMergeReview(
                id=command.review_id,
                source_vehicle_id=command.source_vehicle_id,
                target_vehicle_id=command.target_vehicle_id,
                expected_source_revision=command.expected_source_revision,
                expected_target_revision=command.expected_target_revision,
                reviewer=IdentityReviewer(principal.id, principal.display_name),
                reviewed_at=self._now(),
                reason=command.reason,
                source_fingerprint_id=command.source_fingerprint_id,
                target_fingerprint_id=command.target_fingerprint_id,
            )
            return await self._identities.review_merge(retry)
        if command.source_fingerprint_id is not None:
            values = await self._scoring.score_candidates(command.source_fingerprint_id)
            scored = next(
                (
                    value
                    for value in values
                    if value.candidate_fingerprint_id == command.target_fingerprint_id
                ),
                None,
            )
            if scored is None:
                raise IdentityConflictError(
                    "merge fingerprints are not a feasible topology/time candidate"
                )
            if (
                scored.source_vehicle_id != command.source_vehicle_id
                or scored.candidate_vehicle_id != command.target_vehicle_id
            ):
                raise IdentityConflictError("merge fingerprint ownership changed")
            score = scored.score
        review = IdentityMergeReview(
            id=command.review_id,
            source_vehicle_id=command.source_vehicle_id,
            target_vehicle_id=command.target_vehicle_id,
            expected_source_revision=command.expected_source_revision,
            expected_target_revision=command.expected_target_revision,
            reviewer=IdentityReviewer(principal.id, principal.display_name),
            reviewed_at=self._now(),
            reason=command.reason,
            source_fingerprint_id=command.source_fingerprint_id,
            target_fingerprint_id=command.target_fingerprint_id,
            score=score,
        )
        return await self._identities.review_merge(review)

    async def split(
        self,
        command: SplitIdentity,
        principal: Principal,
    ) -> IdentityReviewResult:
        new_vehicle_id = _split_vehicle_id(command.review_id)
        review = IdentitySplitReview(
            id=command.review_id,
            source_vehicle_id=command.source_vehicle_id,
            new_vehicle_id=new_vehicle_id,
            fingerprint_ids=command.fingerprint_ids,
            expected_source_revision=command.expected_source_revision,
            reviewer=IdentityReviewer(principal.id, principal.display_name),
            reviewed_at=self._now(),
            reason=command.reason,
        )
        return await self._identities.review_split(review)

    async def get_review(self, review_id: str) -> IdentityReviewResult:
        result = await self._identities.get_review(review_id)
        if result is None:
            raise IdentityNotFoundError(f"identity review not found: {review_id}")
        return result

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("identity review clock must be timezone-aware")
        return value.astimezone(UTC)


def _split_vehicle_id(review_id: str) -> str:
    digest = hashlib.sha256(f"identity-split|{review_id}".encode()).hexdigest()[:32]
    return f"veh_{digest}"


def _text_similarity(left: str, right: str) -> float:
    left_compact = "".join(char for char in left.upper() if char.isalnum())
    right_compact = "".join(char for char in right.upper() if char.isalnum())
    size = max(len(left_compact), len(right_compact))
    if size == 0:
        return 0.0
    previous = list(range(len(right_compact) + 1))
    for row, left_char in enumerate(left_compact, start=1):
        current = [row]
        for column, right_char in enumerate(right_compact, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return 1.0 - previous[-1] / size


def _weighted_score(signals: ReIDSignals, config: ReIDConfig) -> float:
    weighted = (
        (signals.plate, config.plate_weight),
        (signals.embedding, config.embedding_weight),
        (signals.vehicle_type, config.vehicle_type_weight),
        (signals.color, config.color_weight),
        (signals.travel_time, config.travel_time_weight),
    )
    available = [(value, weight) for value, weight in weighted if value is not None and weight > 0]
    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        return 0.0
    return min(
        1.0,
        max(0.0, sum(value * weight for value, weight in available) / total_weight),
    )


def _attribute_similarity(left: str | None, right: str | None) -> float | None:
    if left is None or right is None:
        return None
    return 1.0 if left.casefold() == right.casefold() else 0.0
