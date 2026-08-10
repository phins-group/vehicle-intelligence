from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.topology import (
    CameraTopologyService,
    CrossCameraCandidateGenerator,
    TopologyCreate,
    TopologyUpdate,
)
from vehicle_intelligence.config import IdentityConfig
from vehicle_intelligence.domain import (
    CameraTopologyEdge,
    VehicleFingerprint,
    VehicleIdentity,
)
from vehicle_intelligence.exceptions import TopologyConflictError
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    InMemoryVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.topology_memory import (
    InMemoryCameraTopologyRepository,
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


def test_topology_rejects_self_loop_and_invalid_window() -> None:
    now = datetime(2026, 8, 10, tzinfo=UTC)
    with pytest.raises(ValueError, match="itself"):
        CameraTopologyEdge("edge", "a", "a", 10, 20, 15, True, now, now)
    with pytest.raises(ValueError, match="exceed"):
        CameraTopologyEdge("edge", "a", "b", 20, 20, 20, True, now, now)


async def test_topology_service_is_direction_unique_and_revision_safe() -> None:
    now = datetime(2026, 8, 10, tzinfo=UTC)
    repository = InMemoryCameraTopologyRepository()
    service = CameraTopologyService(repository, clock=lambda: now)
    command = TopologyCreate("a-b", "a", "b", 60, 600, 300)

    created = await service.create(command)
    assert created.revision == 1
    with pytest.raises(TopologyConflictError):
        await service.create(TopologyCreate("duplicate", "a", "b", 30, 500, 200))

    updated = await service.update(
        created.id,
        TopologyUpdate(1, "a", "b", 50, 700, 250, metadata={"road": "north"}),
    )
    assert updated.revision == 2
    assert updated.metadata == {"road": "north"}
    with pytest.raises(TopologyConflictError, match="revision"):
        await service.update(
            created.id,
            TopologyUpdate(1, "a", "b", 50, 700, 250),
        )


async def test_candidate_generation_is_directed_time_bounded_and_ranked() -> None:
    current_time = datetime(2026, 8, 10, 8, 30, tzinfo=UTC)
    identities = InMemoryVehicleIdentityRepository()
    topology = InMemoryCameraTopologyRepository()
    service = CameraTopologyService(topology, clock=lambda: current_time)
    await service.create(TopologyCreate("a-b", "camera-a", "camera-b", 60, 600, 300))
    await service.create(TopologyCreate("c-b", "camera-c", "camera-b", 30, 120, 60))

    observations = (
        _observation("fp-current", "veh-current", "camera-b", current_time),
        _observation(
            "fp-typical", "veh-typical", "camera-a", current_time - timedelta(seconds=300)
        ),
        _observation(
            "fp-boundary", "veh-boundary", "camera-a", current_time - timedelta(seconds=60)
        ),
        _observation("fp-too-old", "veh-old", "camera-a", current_time - timedelta(seconds=601)),
        _observation(
            "fp-wrong-camera", "veh-wrong", "camera-d", current_time - timedelta(seconds=300)
        ),
        _observation("fp-c", "veh-c", "camera-c", current_time - timedelta(seconds=60)),
    )
    for identity, fingerprint in observations:
        assert await identities.register_observation(identity, fingerprint)

    generator = CrossCameraCandidateGenerator(
        identities,
        topology,
        IdentityConfig(cross_camera_candidate_limit=10),
    )
    candidates = await generator.generate("fp-current", 10)

    assert {item.fingerprint_id for item in candidates[:2]} == {"fp-typical", "fp-c"}
    assert candidates[-1].fingerprint_id == "fp-boundary"
    assert candidates[0].time_score == candidates[1].time_score == 1
    assert candidates[-1].time_score == 0
    assert all(item.fingerprint_id != "fp-too-old" for item in candidates)
    assert await generator.generate("fp-typical", 10) == ()


async def test_candidate_limit_is_hard_bounded() -> None:
    identities = InMemoryVehicleIdentityRepository()
    topology = InMemoryCameraTopologyRepository()
    generator = CrossCameraCandidateGenerator(
        identities,
        topology,
        IdentityConfig(cross_camera_candidate_limit=2),
    )
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        await generator.generate("anything", 3)
