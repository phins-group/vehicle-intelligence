import os
from dataclasses import replace

import pytest

from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_repository_is_idempotent(sample_event) -> None:
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    repository = MongoVehicleEventRepository(config)
    await repository.ensure_indexes()
    event = replace(sample_event, id=f"{sample_event.id}_mongo")
    try:
        assert await repository.save(event) in (True, False)
        assert not await repository.save(event)
        restored = await repository.get(event.id)
        assert restored == event
    finally:
        await repository._collection.delete_one({"_id": event.id})
        await repository.close()
