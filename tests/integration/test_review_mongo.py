import os
import uuid
from dataclasses import replace

import pytest

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.ports import DatasetSampleQuery, EventQuery
from vehicle_intelligence.application.review import (
    HumanPlateReviewService,
    PlateReviewCommand,
)
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    AuthenticationMethod,
    EventStatus,
    Principal,
    UserRole,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.review_mongo import (
    MongoDatasetSampleRepository,
)


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_human_review_is_atomic_searchable_and_dataset_ready(sample_event) -> None:
    suffix = uuid.uuid4().hex
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    events = MongoVehicleEventRepository(config)
    samples = MongoDatasetSampleRepository(config)
    event = replace(
        sample_event,
        id=f"evt-review-{suffix}",
        track_id=f"gate-01:review:{suffix}",
        status=EventStatus.NEEDS_REVIEW,
        media=replace(sample_event.media, plate_crop_key=f"vehicles/{suffix}/plate.jpg"),
    )
    operator = Principal(
        id="operator-mongo",
        display_name="Mongo Operator",
        role=UserRole.OPERATOR,
        authentication_method=AuthenticationMethod.DEVELOPMENT,
    )
    service = HumanPlateReviewService(events, samples, VietnamPlateNormalizer())
    sample_id: str | None = None
    try:
        await events.ensure_indexes()
        await service.initialize()
        assert await events.save(event)

        result = await service.review(
            event.id,
            PlateReviewCommand("51H12346", 0, operator),
        )
        sample_id = result.dataset_sample_id
        corrected = await events.list(EventQuery(plate="51H-123.46"))
        original = await events.list(EventQuery(plate="51H-123.45"))
        dataset_page = await service.list_samples(DatasetSampleQuery(source_event_id=event.id))
        event_index_cursor = await events._collection.list_indexes()
        dataset_index_cursor = await samples._collection.list_indexes()
        event_indexes = {item["name"] async for item in event_index_cursor}
        dataset_indexes = {item["name"] async for item in dataset_index_cursor}

        assert result.event.plate.normalized == "51H-123.45"
        assert result.event.plate.final_normalized == "51H-123.46"
        assert [item.id for item in corrected.items] == [event.id]
        assert original.items == ()
        assert len(dataset_page.items) == 1
        assert dataset_page.items[0].id == sample_id
        assert "ix_plate_final_time" in event_indexes
        assert "ix_status_time_cursor" in event_indexes
        assert "uq_dataset_event_review" in dataset_indexes
    finally:
        await events._collection.delete_one({"_id": event.id})
        await samples._collection.delete_many({"sourceEventId": event.id})
        await service.close()
        await events.close()
