from __future__ import annotations

import os
import uuid
from dataclasses import replace
from datetime import timedelta

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
from vehicle_intelligence.config import IdentityConfig, MongoConfig
from vehicle_intelligence.domain import (
    AuthenticationMethod,
    Principal,
    UserRole,
    VehicleFingerprint,
    VehicleIdentity,
    VehicleIdentityStatus,
)
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    InMemoryVectorRepository,
)
from vehicle_intelligence.infrastructure.persistence.identity_mongo import (
    MongoVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.persistence.topology_mongo import (
    MongoCameraTopologyRepository,
)


def _observation(
    fingerprint_id: str,
    vehicle_id: str,
    event_id: str,
    camera_id: str,
    observed_at,
) -> tuple[VehicleIdentity, VehicleFingerprint]:
    return (
        VehicleIdentity(
            id=vehicle_id,
            primary_plate=None,
            plates=(),
            vehicle_type="car",
            color="white",
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
            vehicle_type="car",
            vehicle_confidence=0.95,
            plate="51H-123.45",
            plate_confidence=0.95,
            color="white",
        ),
    )


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_identity_merge_split_is_transactional_and_idempotent(sample_event) -> None:
    suffix = uuid.uuid4().hex
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
        transactions_enabled=True,
    )
    runtime = MongoRuntime(config)
    await runtime.initialize()
    events = MongoVehicleEventRepository(runtime)
    identities = MongoVehicleIdentityRepository(runtime)
    topology_repository = MongoCameraTopologyRepository(runtime)
    topology = CameraTopologyService(topology_repository, clock=lambda: sample_event.occurred_at)
    identity_config = IdentityConfig()
    candidates = CrossCameraCandidateGenerator(
        identities,
        topology_repository,
        identity_config,
    )
    vectors = InMemoryVectorRepository()
    scoring = ReIDScoringService(identities, candidates, vectors, identity_config.reid)
    reviews = IdentityReviewService(
        identities,
        scoring,
        clock=lambda: sample_event.occurred_at,
    )
    source_vehicle_id = f"veh-source-{suffix}"
    target_vehicle_id = f"veh-target-{suffix}"
    source_fingerprint_id = f"fp-source-{suffix}"
    target_fingerprint_id = f"fp-target-{suffix}"
    source_event_id = f"evt-source-{suffix}"
    target_event_id = f"evt-target-{suffix}"
    edge_id = f"edge-{suffix}"
    review_ids = (f"merge-{suffix}", f"split-{suffix}")
    principal = Principal(
        "operator-mongo",
        "Mongo Identity Reviewer",
        UserRole.OPERATOR,
        AuthenticationMethod.API_KEY,
    )
    try:
        await events.ensure_indexes()
        await identities.ensure_indexes()
        await topology.initialize()
        await scoring.initialize()
        await topology.create(
            TopologyCreate(edge_id, f"camera-a-{suffix}", f"camera-b-{suffix}", 60, 600, 300)
        )
        source_event = replace(
            sample_event,
            id=source_event_id,
            track_id=f"camera-b:{suffix}:source",
            vehicle_id=source_vehicle_id,
        )
        target_event = replace(
            sample_event,
            id=target_event_id,
            track_id=f"camera-a:{suffix}:target",
            occurred_at=sample_event.occurred_at - timedelta(seconds=300),
            created_at=sample_event.created_at - timedelta(seconds=300),
            vehicle_id=target_vehicle_id,
        )
        assert await events.save(source_event)
        assert await events.save(target_event)
        for identity, fingerprint in (
            _observation(
                source_fingerprint_id,
                source_vehicle_id,
                source_event_id,
                f"camera-b-{suffix}",
                sample_event.occurred_at,
            ),
            _observation(
                target_fingerprint_id,
                target_vehicle_id,
                target_event_id,
                f"camera-a-{suffix}",
                sample_event.occurred_at - timedelta(seconds=300),
            ),
        ):
            assert await identities.register_observation(identity, fingerprint)

        command = MergeIdentities(
            review_id=review_ids[0],
            source_vehicle_id=source_vehicle_id,
            target_vehicle_id=target_vehicle_id,
            expected_source_revision=1,
            expected_target_revision=1,
            reason="Mongo transaction acceptance merge",
            source_fingerprint_id=source_fingerprint_id,
            target_fingerprint_id=target_fingerprint_id,
        )
        merged = await reviews.merge(command, principal)
        retry = await reviews.merge(command, principal)
        assert merged.moved_fingerprints == merged.moved_events == 1
        assert retry.idempotent
        assert (await identities.get(source_vehicle_id)).status is VehicleIdentityStatus.MERGED
        assert (await events.get(source_event_id)).vehicle_id == target_vehicle_id

        split = await reviews.split(
            SplitIdentity(
                review_id=review_ids[1],
                source_vehicle_id=target_vehicle_id,
                expected_source_revision=2,
                fingerprint_ids=(source_fingerprint_id,),
                reason="Mongo transaction acceptance split",
            ),
            principal,
        )
        split_retry = await reviews.split(
            SplitIdentity(
                review_id=review_ids[1],
                source_vehicle_id=target_vehicle_id,
                expected_source_revision=2,
                fingerprint_ids=(source_fingerprint_id,),
                reason="Mongo transaction acceptance split",
            ),
            principal,
        )
        assert split.moved_fingerprints == split.moved_events == 1
        assert split_retry.idempotent
        assert (await events.get(source_event_id)).vehicle_id == split.result_vehicle_id
        assert (await identities.get(split.result_vehicle_id)).observation_count == 1
    finally:
        await identities._reviews.delete_many({"_id": {"$in": list(review_ids)}})
        await topology_repository._collection.delete_many({"_id": edge_id})
        await identities._fingerprints.delete_many(
            {"_id": {"$in": [source_fingerprint_id, target_fingerprint_id]}}
        )
        await identities._identities.delete_many(
            {
                "$or": [
                    {"_id": {"$in": [source_vehicle_id, target_vehicle_id]}},
                    {"metadata.reviewId": review_ids[1]},
                ]
            }
        )
        await events._collection.delete_many({"_id": {"$in": [source_event_id, target_event_id]}})
        await scoring.close()
        await topology.close()
        await identities.close()
        await events.close()
        await runtime.close()
