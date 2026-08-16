from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from vehicle_intelligence.application.reid import (
    IdentityReviewService,
    MergeIdentities,
    ReIDScoringService,
    SplitIdentity,
)
from vehicle_intelligence.application.topology import (
    CameraTopologyService,
    CrossCameraCandidateGenerator,
    TopologyCreate,
)
from vehicle_intelligence.config import IdentityConfig, ReIDConfig
from vehicle_intelligence.domain import (
    AuthenticationMethod,
    EmbeddingModel,
    EmbeddingReference,
    EmbeddingVector,
    Principal,
    ReIDVerdict,
    UserRole,
    VehicleFingerprint,
    VehicleIdentity,
    VehicleIdentityStatus,
)
from vehicle_intelligence.exceptions import IdentityConflictError
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    InMemoryVectorRepository,
    InMemoryVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.topology_memory import (
    InMemoryCameraTopologyRepository,
)


def test_reid_configuration_requires_identifying_signal_weight() -> None:
    with pytest.raises(ValueError, match="identifying coverage exceeds"):
        ReIDConfig(plate_weight=0, embedding_weight=0)


class _CountingIdentityRepository(InMemoryVehicleIdentityRepository):
    def __init__(self, events) -> None:
        super().__init__(events)
        self.batch_reads = 0

    async def get_fingerprints(self, fingerprint_ids):
        self.batch_reads += 1
        return await super().get_fingerprints(fingerprint_ids)


class _CountingVectorRepository(InMemoryVectorRepository):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0
        self.search_calls = 0
        self.searched_candidate_ids: tuple[str, ...] = ()

    async def get(self, vector_id):
        self.get_calls += 1
        return await super().get(vector_id)

    async def search(self, query):
        self.search_calls += 1
        self.searched_candidate_ids = query.candidate_ids
        return await super().search(query)


def _observation(
    fingerprint_id: str,
    vehicle_id: str,
    event_id: str,
    camera_id: str,
    observed_at: datetime,
    *,
    embedding: EmbeddingReference | None = None,
    plate: str | None = "51H-123.45",
    color: str | None = "white",
    vehicle_type: str = "car",
) -> tuple[VehicleIdentity, VehicleFingerprint]:
    return (
        VehicleIdentity(
            id=vehicle_id,
            primary_plate=None,
            plates=(),
            vehicle_type=vehicle_type,
            color=color,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
            observation_count=1,
        ),
        VehicleFingerprint(
            id=fingerprint_id,
            vehicle_id=vehicle_id,
            source_event_id=event_id,
            camera_id=camera_id,
            observed_at=observed_at,
            vehicle_type=vehicle_type,
            vehicle_confidence=0.95,
            plate=plate,
            plate_confidence=0.95 if plate is not None else None,
            color=color,
            embedding=embedding,
        ),
    )


async def _services(sample_event):
    now = sample_event.occurred_at
    events = InMemoryVehicleEventRepository()
    identities = _CountingIdentityRepository(events)
    topology_repository = InMemoryCameraTopologyRepository()
    topology = CameraTopologyService(topology_repository, clock=lambda: now)
    await topology.create(TopologyCreate("a-b", "camera-a", "camera-b", 60, 600, 300))
    vectors = _CountingVectorRepository()
    model = EmbeddingModel("vehicle-reid", "v1", 3, "sha256-test")
    source_reference = EmbeddingReference("emb-source", model)
    target_reference = EmbeddingReference("emb-target", model)
    observations = (
        _observation(
            "fp-source",
            "veh-source",
            "evt-source",
            "camera-b",
            now,
            embedding=source_reference,
        ),
        _observation(
            "fp-target",
            "veh-target",
            "evt-target",
            "camera-a",
            now - timedelta(seconds=300),
            embedding=target_reference,
        ),
    )
    for identity, fingerprint in observations:
        assert await identities.register_observation(identity, fingerprint)
    assert await vectors.put(EmbeddingVector("emb-source", model, (1, 0, 0), now))
    assert await vectors.put(EmbeddingVector("emb-target", model, (0.99, 0.01, 0), now))
    config = IdentityConfig(reid=ReIDConfig(match_threshold=0.85, review_threshold=0.60))
    candidates = CrossCameraCandidateGenerator(
        identities,
        topology_repository,
        config,
    )
    scoring = ReIDScoringService(identities, candidates, vectors, config.reid)
    reviews = IdentityReviewService(identities, scoring, clock=lambda: now)
    return events, identities, scoring, reviews, vectors


async def test_reid_scoring_combines_versioned_signals_without_mutating_identity(
    sample_event,
) -> None:
    _events, identities, scoring, _reviews, _vectors = await _services(sample_event)

    scores = await scoring.score_candidates("fp-source")

    assert len(scores) == 1
    assert scores[0].verdict is ReIDVerdict.MATCH
    assert scores[0].score > 0.99
    assert scores[0].scoring_version == "reid-score-v2"
    assert scores[0].signals.embedding is not None
    assert scores[0].signals.plate == 1
    assert (await identities.get("veh-source")).status is VehicleIdentityStatus.ACTIVE
    assert (await identities.get("veh-target")).observation_count == 1


async def test_merge_is_reviewed_atomic_idempotent_and_split_is_reversible(
    sample_event,
) -> None:
    events, identities, _scoring, reviews, _vectors = await _services(sample_event)
    source_event = replace(
        sample_event,
        id="evt-source",
        track_id="camera-b:source",
        vehicle_id="veh-source",
    )
    target_event = replace(
        sample_event,
        id="evt-target",
        track_id="camera-a:target",
        occurred_at=sample_event.occurred_at - timedelta(seconds=300),
        created_at=sample_event.created_at - timedelta(seconds=300),
        vehicle_id="veh-target",
    )
    assert await events.save(source_event)
    assert await events.save(target_event)
    principal = Principal(
        "operator-1",
        "Identity Operator",
        UserRole.OPERATOR,
        AuthenticationMethod.API_KEY,
    )
    merge_command = MergeIdentities(
        review_id="review-merge-001",
        source_vehicle_id="veh-source",
        target_vehicle_id="veh-target",
        expected_source_revision=1,
        expected_target_revision=1,
        reason="Same vehicle confirmed from both snapshots",
        source_fingerprint_id="fp-source",
        target_fingerprint_id="fp-target",
    )

    merged = await reviews.merge(merge_command, principal)
    retry = await reviews.merge(merge_command, principal)

    assert merged.moved_fingerprints == merged.moved_events == 1
    assert retry.idempotent is True
    assert (await identities.get("veh-source")).status is VehicleIdentityStatus.MERGED
    target = await identities.get("veh-target")
    assert target is not None and target.observation_count == 2 and target.revision == 2
    assert (await events.get("evt-source")).vehicle_id == "veh-target"
    assert (await identities.get_fingerprint("fp-source")).vehicle_id == "veh-target"

    split = await reviews.split(
        SplitIdentity(
            review_id="review-split-001",
            source_vehicle_id="veh-target",
            expected_source_revision=2,
            fingerprint_ids=("fp-source",),
            reason="Operator found distinguishing damage pattern",
        ),
        principal,
    )
    split_retry = await reviews.split(
        SplitIdentity(
            review_id="review-split-001",
            source_vehicle_id="veh-target",
            expected_source_revision=2,
            fingerprint_ids=("fp-source",),
            reason="Operator found distinguishing damage pattern",
        ),
        principal,
    )

    assert split.moved_fingerprints == split.moved_events == 1
    assert split_retry.idempotent is True
    assert (await identities.get("veh-target")).observation_count == 1
    assert (await identities.get(split.result_vehicle_id)).observation_count == 1
    assert (await events.get("evt-source")).vehicle_id == split.result_vehicle_id


async def test_merge_rejects_stale_revision(sample_event) -> None:
    _events, _identities, _scoring, reviews, _vectors = await _services(sample_event)
    principal = Principal(
        "operator-1",
        "Identity Operator",
        UserRole.OPERATOR,
        AuthenticationMethod.API_KEY,
    )
    with pytest.raises(IdentityConflictError, match="revision"):
        await reviews.merge(
            MergeIdentities(
                review_id="review-stale-001",
                source_vehicle_id="veh-source",
                target_vehicle_id="veh-target",
                expected_source_revision=9,
                expected_target_revision=1,
                reason="stale review must fail",
            ),
            principal,
        )


async def test_reid_batches_candidate_and_vector_reads(sample_event) -> None:
    _events, identities, scoring, _reviews, vectors = await _services(sample_event)
    now = sample_event.occurred_at
    source = await identities.get_fingerprint("fp-source")
    assert source is not None and source.embedding is not None
    second_reference = EmbeddingReference("emb-target-2", source.embedding.model)
    identity, fingerprint = _observation(
        "fp-target-2",
        "veh-target-2",
        "evt-target-2",
        "camera-a",
        now - timedelta(seconds=360),
        embedding=second_reference,
    )
    assert await identities.register_observation(identity, fingerprint)
    assert await vectors.put(
        EmbeddingVector("emb-target-2", source.embedding.model, (0.98, 0.02, 0), now)
    )

    scores = await scoring.score_candidates("fp-source")

    assert len(scores) == 2
    assert identities.batch_reads == 1
    assert vectors.get_calls == 1
    assert vectors.search_calls == 1
    assert set(vectors.searched_candidate_ids) == {"emb-target", "emb-target-2"}


@pytest.mark.parametrize(
    ("vehicle_type", "expected_type_signal"),
    [("car", 1.0), ("unknown", None)],
)
async def test_reid_sparse_type_and_travel_evidence_cannot_match(
    sample_event,
    vehicle_type: str,
    expected_type_signal: float | None,
) -> None:
    now = sample_event.occurred_at
    events = InMemoryVehicleEventRepository()
    identities = InMemoryVehicleIdentityRepository(events)
    topology_repository = InMemoryCameraTopologyRepository()
    topology = CameraTopologyService(topology_repository, clock=lambda: now)
    await topology.create(TopologyCreate("a-b", "camera-a", "camera-b", 60, 600, 300))
    for observation in (
        _observation(
            "fp-sparse-source",
            "veh-sparse-source",
            "evt-sparse-source",
            "camera-b",
            now,
            plate=None,
            color=None,
            vehicle_type=vehicle_type,
        ),
        _observation(
            "fp-sparse-target",
            "veh-sparse-target",
            "evt-sparse-target",
            "camera-a",
            now - timedelta(seconds=300),
            plate=None,
            color=None,
            vehicle_type=vehicle_type,
        ),
    ):
        assert await identities.register_observation(*observation)
    config = IdentityConfig()
    scoring = ReIDScoringService(
        identities,
        CrossCameraCandidateGenerator(identities, topology_repository, config),
        InMemoryVectorRepository(),
        config.reid,
    )

    scores = await scoring.score_candidates("fp-sparse-source")

    assert len(scores) == 1
    assert scores[0].score == 1.0
    assert scores[0].signals.vehicle_type == expected_type_signal
    assert scores[0].verdict is ReIDVerdict.REVIEW


async def test_reid_strong_embedding_evidence_still_matches_without_plate(sample_event) -> None:
    now = sample_event.occurred_at
    events = InMemoryVehicleEventRepository()
    identities = InMemoryVehicleIdentityRepository(events)
    topology_repository = InMemoryCameraTopologyRepository()
    topology = CameraTopologyService(topology_repository, clock=lambda: now)
    await topology.create(TopologyCreate("a-b", "camera-a", "camera-b", 60, 600, 300))
    vectors = InMemoryVectorRepository()
    model = EmbeddingModel("vehicle-reid", "v1", 3, "sha256-test")
    source_reference = EmbeddingReference("emb-sparse-source", model)
    target_reference = EmbeddingReference("emb-sparse-target", model)
    for observation in (
        _observation(
            "fp-embedding-source",
            "veh-embedding-source",
            "evt-embedding-source",
            "camera-b",
            now,
            embedding=source_reference,
            plate=None,
            color=None,
            vehicle_type="unknown",
        ),
        _observation(
            "fp-embedding-target",
            "veh-embedding-target",
            "evt-embedding-target",
            "camera-a",
            now - timedelta(seconds=300),
            embedding=target_reference,
            plate=None,
            color=None,
            vehicle_type="unknown",
        ),
    ):
        assert await identities.register_observation(*observation)
    assert await vectors.put(EmbeddingVector(source_reference.id, model, (1, 0, 0), now))
    assert await vectors.put(EmbeddingVector(target_reference.id, model, (0.99, 0.01, 0), now))
    config = IdentityConfig()
    scoring = ReIDScoringService(
        identities,
        CrossCameraCandidateGenerator(identities, topology_repository, config),
        vectors,
        config.reid,
    )

    scores = await scoring.score_candidates("fp-embedding-source")

    assert len(scores) == 1
    assert scores[0].signals.vehicle_type is None
    assert scores[0].signals.embedding is not None
    assert scores[0].verdict is ReIDVerdict.MATCH
