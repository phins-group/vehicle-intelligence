from __future__ import annotations

import os
import uuid
from dataclasses import replace
from datetime import timedelta

import pytest

from vehicle_intelligence.application.journeys import VehicleJourneyService
from vehicle_intelligence.application.topology import CameraTopologyService, TopologyCreate
from vehicle_intelligence.config import IdentityConfig, MongoConfig
from vehicle_intelligence.domain import CameraSnapshot, VehicleFingerprint, VehicleIdentity
from vehicle_intelligence.infrastructure.persistence.identity_mongo import (
    MongoVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.persistence.topology_mongo import (
    MongoCameraTopologyRepository,
)


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_vehicle_journey_uses_identity_time_index(sample_event) -> None:
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
    events = MongoVehicleEventRepository(runtime)
    identities = MongoVehicleIdentityRepository(runtime)
    topology_repository = MongoCameraTopologyRepository(runtime)
    topology = CameraTopologyService(topology_repository, clock=lambda: sample_event.occurred_at)
    vehicle_id = f"veh-journey-{suffix}"
    fingerprint_id = f"fp-journey-{suffix}"
    event_ids = (f"evt-a-{suffix}", f"evt-b-{suffix}")
    edge_id = f"edge-journey-{suffix}"
    try:
        await events.ensure_indexes()
        await identities.ensure_indexes()
        await topology.initialize()
        identity = VehicleIdentity(
            vehicle_id,
            None,
            (),
            "car",
            None,
            sample_event.occurred_at,
            sample_event.occurred_at,
            1,
        )
        fingerprint = VehicleFingerprint(
            fingerprint_id,
            vehicle_id,
            event_ids[0],
            f"camera-a-{suffix}",
            sample_event.occurred_at,
            "car",
            0.9,
        )
        assert await identities.register_observation(identity, fingerprint)
        await topology.create(
            TopologyCreate(
                edge_id,
                f"camera-a-{suffix}",
                f"camera-b-{suffix}",
                30,
                120,
                60,
            )
        )
        first = replace(
            sample_event,
            id=event_ids[0],
            track_id=f"camera-a:{suffix}:1",
            vehicle_id=vehicle_id,
            camera=CameraSnapshot(f"camera-a-{suffix}", "Gate A", None),
        )
        second = replace(
            sample_event,
            id=event_ids[1],
            track_id=f"camera-b:{suffix}:2",
            vehicle_id=vehicle_id,
            camera=CameraSnapshot(f"camera-b-{suffix}", "Warehouse", None),
            occurred_at=sample_event.occurred_at + timedelta(seconds=60),
            created_at=sample_event.created_at + timedelta(seconds=60),
        )
        assert await events.save(second)
        assert await events.save(first)
        service = VehicleJourneyService(
            identities,
            events,
            topology_repository,
            IdentityConfig(journey_event_limit=10),
        )

        journey = await service.journey(vehicle_id, limit=10)

        assert [item.event_id for item in journey.observations] == list(event_ids)
        assert journey.segments[0].topology_edge_id == edge_id
        assert journey.segments[0].feasible is True
    finally:
        await events._collection.delete_many({"_id": {"$in": list(event_ids)}})
        await identities._fingerprints.delete_many({"_id": fingerprint_id})
        await identities._identities.delete_many({"_id": vehicle_id})
        await topology_repository._collection.delete_many({"_id": edge_id})
        await topology.close()
        await identities.close()
        await events.close()
        await runtime.close()
