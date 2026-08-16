import os
import uuid
from dataclasses import replace

import pytest

from vehicle_intelligence.application.identity import (
    VehicleIdentityProcessor,
    bootstrap_vehicle_id,
    fingerprint_id,
)
from vehicle_intelligence.application.ports import VectorSearchQuery
from vehicle_intelligence.config import IdentityConfig, MongoConfig
from vehicle_intelligence.domain import EmbeddingModel, EmbeddingVector
from vehicle_intelligence.infrastructure.persistence.identity_mongo import (
    MongoVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.persistence.vector_mongo import MongoVectorRepository


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_identity_fingerprint_link_and_vector_candidates(sample_event) -> None:
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
    vectors = MongoVectorRepository(runtime, maximum_candidates=10)
    processor = VehicleIdentityProcessor(identities, events, IdentityConfig())
    event = replace(
        sample_event,
        id=f"evt-identity-{suffix}",
        track_id=f"gate-01:identity:{suffix}",
    )
    model_v1 = EmbeddingModel("vehicle-reid", "1", 3, "a" * 64)
    model_v2 = EmbeddingModel("vehicle-reid", "2", 3, "a" * 64)
    wrong_artifact = EmbeddingModel("vehicle-reid", "1", 3, "b" * 64)
    vector_ids = (
        f"vec-a-{suffix}",
        f"vec-b-{suffix}",
        f"vec-v2-{suffix}",
        f"vec-wrong-artifact-{suffix}",
    )
    try:
        await events.ensure_indexes()
        await processor.initialize()
        await vectors.ensure_indexes()
        assert await events.save(event)

        await processor.process(event)
        await processor.process(event)

        vehicle_id = bootstrap_vehicle_id(event.id)
        identity = await identities.get(vehicle_id)
        linked = await events.get(event.id)
        fingerprints = await identities.list_fingerprints(vehicle_id)
        assert identity is not None and identity.observation_count == 1
        assert linked is not None and linked.vehicle_id == vehicle_id
        assert [item.id for item in fingerprints] == [fingerprint_id(event.id, 1)]

        assert await vectors.put(
            EmbeddingVector(vector_ids[0], model_v1, (1, 0, 0), event.occurred_at)
        )
        assert await vectors.put(
            EmbeddingVector(vector_ids[1], model_v1, (0.9, 0.1, 0), event.occurred_at)
        )
        assert await vectors.put(
            EmbeddingVector(vector_ids[2], model_v2, (1, 0, 0), event.occurred_at)
        )
        assert await vectors.put(
            EmbeddingVector(vector_ids[3], wrong_artifact, (1, 0, 0), event.occurred_at)
        )
        neighbors = await vectors.search(
            VectorSearchQuery(
                vector=(1, 0, 0),
                model=model_v1,
                candidate_ids=vector_ids,
                minimum_score=0.5,
            )
        )
        assert [item.vector_id for item in neighbors] == list(vector_ids[:2])
    finally:
        await events._collection.delete_one({"_id": event.id})
        await identities._fingerprints.delete_many({"sourceEventId": event.id})
        await identities._identities.delete_many({"_id": bootstrap_vehicle_id(event.id)})
        await vectors._collection.delete_many({"_id": {"$in": list(vector_ids)}})
        await processor.close()
        await vectors.close()
        await events.close()
        await runtime.close()
