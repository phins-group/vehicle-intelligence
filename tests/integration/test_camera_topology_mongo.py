from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.topology import (
    CameraTopologyService,
    CrossCameraCandidateGenerator,
    TopologyCreate,
    TopologyUpdate,
)
from vehicle_intelligence.config import IdentityConfig, MongoConfig
from vehicle_intelligence.domain import VehicleFingerprint, VehicleIdentity
from vehicle_intelligence.infrastructure.persistence.identity_mongo import (
    MongoVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.persistence.topology_mongo import (
    MongoCameraTopologyRepository,
)


def _observation(
    fingerprint_id: str,
    vehicle_id: str,
    camera_id: str,
    observed_at: datetime,
) -> tuple[VehicleIdentity, VehicleFingerprint]:
    return (
        VehicleIdentity(
            id=vehicle_id,
            primary_plate=None,
            plates=(),
            vehicle_type="car",
            color=None,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
            observation_count=1,
        ),
        VehicleFingerprint(
            id=fingerprint_id,
            vehicle_id=vehicle_id,
            source_event_id=f"evt-{fingerprint_id}",
            camera_id=camera_id,
            observed_at=observed_at,
            vehicle_type="car",
            vehicle_confidence=0.9,
        ),
    )


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_topology_and_travel_candidate_query_are_bounded() -> None:
    suffix = uuid.uuid4().hex
    runtime = MongoRuntime(
        MongoConfig(
            enabled=True,
            uri=os.environ["TEST_MONGODB_URI"],
            database="vehicle_intelligence_test",
            transactions_enabled=True,
        )
    )
    await runtime.initialize()
    identities = MongoVehicleIdentityRepository(runtime)
    topology_repository = MongoCameraTopologyRepository(runtime)
    now = datetime(2026, 8, 10, 10, tzinfo=UTC)
    topology = CameraTopologyService(topology_repository, clock=lambda: now)
    generator = CrossCameraCandidateGenerator(
        identities,
        topology_repository,
        IdentityConfig(cross_camera_candidate_limit=10),
    )
    edge_id = f"edge-{suffix}"
    fingerprint_ids = [f"fp-{name}-{suffix}" for name in ("current", "match", "old")]
    vehicle_ids = [f"veh-{name}-{suffix}" for name in ("current", "match", "old")]
    try:
        await identities.ensure_indexes()
        await topology.initialize()
        created = await topology.create(
            TopologyCreate(edge_id, f"camera-a-{suffix}", f"camera-b-{suffix}", 60, 600, 300)
        )
        updated = await topology.update(
            edge_id,
            TopologyUpdate(1, created.from_camera_id, created.to_camera_id, 60, 700, 300),
        )
        assert updated.revision == 2

        observations = (
            _observation(fingerprint_ids[0], vehicle_ids[0], created.to_camera_id, now),
            _observation(
                fingerprint_ids[1],
                vehicle_ids[1],
                created.from_camera_id,
                now - timedelta(seconds=300),
            ),
            _observation(
                fingerprint_ids[2],
                vehicle_ids[2],
                created.from_camera_id,
                now - timedelta(seconds=701),
            ),
        )
        for identity, fingerprint in observations:
            assert await identities.register_observation(identity, fingerprint)

        candidates = await generator.generate(fingerprint_ids[0], 10)
        assert [candidate.fingerprint_id for candidate in candidates] == [fingerprint_ids[1]]
        assert candidates[0].time_score == 1
        inbound = await topology.list(to_camera_id=created.to_camera_id, enabled_only=True)
        assert [edge.id for edge in inbound] == [edge_id]
    finally:
        await topology_repository._collection.delete_many({"_id": edge_id})
        await identities._fingerprints.delete_many({"_id": {"$in": fingerprint_ids}})
        await identities._identities.delete_many({"_id": {"$in": vehicle_ids}})
        await topology.close()
        await identities.close()
        await runtime.close()
