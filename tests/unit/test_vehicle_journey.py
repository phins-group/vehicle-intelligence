from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from vehicle_intelligence.application.journeys import VehicleJourneyService
from vehicle_intelligence.application.topology import CameraTopologyService, TopologyCreate
from vehicle_intelligence.config import IdentityConfig
from vehicle_intelligence.domain import CameraSnapshot, VehicleFingerprint, VehicleIdentity
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    InMemoryVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.topology_memory import (
    InMemoryCameraTopologyRepository,
)


async def test_journey_is_chronological_topology_aware_and_bounded(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    identities = InMemoryVehicleIdentityRepository(events)
    topology_repository = InMemoryCameraTopologyRepository()
    now = sample_event.occurred_at
    identity = VehicleIdentity(
        "veh-journey",
        None,
        (),
        "car",
        "white",
        now,
        now,
        1,
    )
    fingerprint = VehicleFingerprint(
        "fp-journey",
        identity.id,
        "evt-journey-a",
        "camera-a",
        now,
        "car",
        0.9,
    )
    assert await identities.register_observation(identity, fingerprint)
    topology = CameraTopologyService(topology_repository, clock=lambda: now)
    await topology.create(TopologyCreate("a-b", "camera-a", "camera-b", 30, 120, 60))
    await topology.create(TopologyCreate("b-c", "camera-b", "camera-c", 30, 90, 60))
    journey_events = (
        replace(
            sample_event,
            id="evt-journey-c",
            track_id="camera-c:3",
            vehicle_id=identity.id,
            camera=CameraSnapshot("camera-c", "Loading Dock", "ZONE_C"),
            occurred_at=now + timedelta(seconds=200),
            created_at=now + timedelta(seconds=200),
        ),
        replace(
            sample_event,
            id="evt-journey-a",
            track_id="camera-a:1",
            vehicle_id=identity.id,
            camera=CameraSnapshot("camera-a", "Gate A", "ZONE_A"),
        ),
        replace(
            sample_event,
            id="evt-journey-b",
            track_id="camera-b:2",
            vehicle_id=identity.id,
            camera=CameraSnapshot("camera-b", "Warehouse", "ZONE_B"),
            occurred_at=now + timedelta(seconds=60),
            created_at=now + timedelta(seconds=60),
        ),
    )
    for event in journey_events:
        assert await events.save(event)
    service = VehicleJourneyService(
        identities,
        events,
        topology_repository,
        IdentityConfig(journey_event_limit=10),
    )

    journey = await service.journey(identity.id, limit=10)
    bounded = await service.journey(identity.id, limit=2)
    filtered = await service.timeline(
        identity.id,
        from_time=now + timedelta(seconds=30),
        to_time=now + timedelta(seconds=100),
    )

    assert [item.camera_id for item in journey.observations] == [
        "camera-a",
        "camera-b",
        "camera-c",
    ]
    assert journey.segments[0].feasible is True
    assert journey.segments[0].elapsed_seconds == 60
    assert journey.segments[1].feasible is False
    assert journey.segments[1].elapsed_seconds == 140
    assert bounded.truncated is True
    assert len(bounded.observations) == 2
    assert [event.id for event in filtered] == ["evt-journey-b"]
