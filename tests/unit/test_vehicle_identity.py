from dataclasses import replace

import pytest

from vehicle_intelligence.application.identity import (
    VehicleIdentityProcessor,
    bootstrap_vehicle_id,
    fingerprint_id,
)
from vehicle_intelligence.application.ports import VectorSearchQuery
from vehicle_intelligence.config import IdentityConfig
from vehicle_intelligence.domain import EmbeddingModel, EmbeddingVector
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    InMemoryVectorRepository,
    InMemoryVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)


async def test_identity_bootstrap_is_idempotent_and_plate_is_only_a_signal(
    sample_event,
) -> None:
    events = InMemoryVehicleEventRepository()
    identities = InMemoryVehicleIdentityRepository()
    processor = VehicleIdentityProcessor(identities, events, IdentityConfig())
    assert await events.save(sample_event)
    await processor.initialize()

    await processor.process(sample_event)
    await processor.process(sample_event)

    vehicle_id = bootstrap_vehicle_id(sample_event.id)
    identity = await identities.get(vehicle_id)
    assert identity is not None
    assert identity.primary_plate == "51H-123.45"
    assert identity.observation_count == 1
    assert await identities.get_fingerprint(fingerprint_id(sample_event.id, 1)) is not None
    assert (await events.get(sample_event.id)).vehicle_id == vehicle_id

    another = replace(
        sample_event,
        id="evt_same_plate_different_observation",
        track_id="gate-02:track:2",
    )
    assert await events.save(another)
    await processor.process(another)
    second_id = bootstrap_vehicle_id(another.id)
    assert second_id != vehicle_id
    candidates = await identities.find_by_plate("51H-123.45")
    assert {item.id for item in candidates} == {vehicle_id, second_id}


async def test_vector_search_is_model_versioned_and_candidate_bounded(
    sample_event,
) -> None:
    repository = InMemoryVectorRepository()
    model_v1 = EmbeddingModel("vehicle-reid", "1", 3)
    model_v2 = EmbeddingModel("vehicle-reid", "2", 3)
    timestamp = sample_event.occurred_at
    assert await repository.put(EmbeddingVector("vec-a", model_v1, (1, 0, 0), timestamp))
    assert await repository.put(EmbeddingVector("vec-b", model_v1, (0.8, 0.2, 0), timestamp))
    assert await repository.put(EmbeddingVector("vec-v2", model_v2, (1, 0, 0), timestamp))

    neighbors = await repository.search(
        VectorSearchQuery(
            vector=(1, 0, 0),
            model=model_v1,
            candidate_ids=("vec-b", "vec-v2", "vec-a", "missing"),
            minimum_score=0.5,
        )
    )

    assert [item.vector_id for item in neighbors] == ["vec-a", "vec-b"]
    assert neighbors[0].score == pytest.approx(1)
    assert all(item.vector_id != "vec-v2" for item in neighbors)


def test_vector_search_rejects_unbounded_candidate_set() -> None:
    model = EmbeddingModel("vehicle-reid", "1", 2)
    with pytest.raises(ValueError, match="bounded"):
        VectorSearchQuery(
            vector=(1, 0),
            model=model,
            candidate_ids=tuple(f"vec-{index}" for index in range(5001)),
        )
