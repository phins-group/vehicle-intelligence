import os
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    AITrace,
    DatasetSample,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    EventStatus,
    ModelMetadata,
    OCRDatasetPrediction,
    PlateReview,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.quality_mongo import (
    MongoModelQualityRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_mongo import (
    MongoDatasetSampleRepository,
)


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_quality_aggregates_and_dataset_export_claims_are_atomic(
    sample_event,
) -> None:
    suffix = uuid.uuid4().hex
    timestamp = datetime(2099, 1, 1, 0, 0, 0, int(suffix[:6], 16) % 999_999, tzinfo=UTC)
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    events = MongoVehicleEventRepository(config)
    samples = MongoDatasetSampleRepository(config)
    quality = MongoModelQualityRepository(config)
    model = ModelMetadata(name="ocr-quality-test", version=suffix, hash="a" * 64)
    reviewed = replace(
        sample_event,
        id=f"evt-quality-{suffix}",
        track_id=f"gate-01:quality:{suffix}",
        schema_version=2,
        occurred_at=timestamp,
        created_at=timestamp,
        ai=AITrace(vehicle_detector=None, plate_detector=None, ocr=model),
        plate=replace(
            sample_event.plate,
            review=PlateReview(
                normalized="51H-123.46",
                revision=1,
                reviewed_at=timestamp,
                reviewed_by="operator",
                reviewer_display_name="Operator",
            ),
        ),
    )
    unknown = replace(
        sample_event,
        id=f"evt-quality-unknown-{suffix}",
        track_id=f"gate-01:quality-unknown:{suffix}",
        status=EventStatus.UNREADABLE,
        occurred_at=timestamp + timedelta(milliseconds=1),
        created_at=timestamp + timedelta(milliseconds=1),
        ai=AITrace(vehicle_detector=None, plate_detector=None, ocr=model),
        plate=None,
    )
    sample = DatasetSample(
        id=f"dss_quality_{suffix}",
        sample_type=DatasetSampleType.PLATE_OCR,
        status=DatasetSampleStatus.READY,
        source_event_id=reviewed.id,
        image_key=f"vehicles/{suffix}/plate.jpg",
        prediction=OCRDatasetPrediction(
            raw=reviewed.plate.raw,
            normalized=reviewed.plate.normalized,
            confidence=reviewed.plate.confidence,
            model=model,
        ),
        label=reviewed.plate.review.normalized,
        reason=DatasetSampleReason.HUMAN_CORRECTION,
        review_revision=1,
        reviewed_by="operator",
        reviewer_display_name="Operator",
        reviewed_at=timestamp,
        created_at=timestamp,
    )
    try:
        await events.ensure_indexes()
        await samples.ensure_indexes()
        assert await events.save(reviewed)
        assert await events.save(unknown)
        assert await samples.create(sample)

        claimed = await samples.claim_for_export(
            f"export-{suffix}",
            10,
            timestamp + timedelta(seconds=1),
            timestamp - timedelta(seconds=1),
        )
        report = await quality.summarize(
            timestamp - timedelta(seconds=1),
            timestamp + timedelta(seconds=1),
            timestamp + timedelta(seconds=2),
            10,
        )
        completed = await samples.mark_exported(
            (sample.id,),
            f"export-{suffix}",
            "b" * 64,
            timestamp + timedelta(seconds=2),
        )
        indexes = {item["name"] async for item in await samples._collection.list_indexes()}

        assert [item.id for item in claimed] == [sample.id]
        assert claimed[0].status is DatasetSampleStatus.EXPORTING
        assert report.totals.event_count == 2
        assert report.totals.ocr_success_rate == 0.5
        assert report.totals.unknown_plate_rate == 0.5
        assert report.totals.human_correction_rate == 1
        assert report.models[0].model.version == suffix
        assert report.feedback.exporting == 1
        assert completed == 1
        assert (await samples.get(sample.id)).status is DatasetSampleStatus.EXPORTED
        assert "ix_dataset_export_claim" in indexes
        assert "ix_dataset_export_resume" in indexes
    finally:
        await events._collection.delete_many({"_id": {"$in": [reviewed.id, unknown.id]}})
        await samples._collection.delete_one({"_id": sample.id})
        await quality.close()
        await samples.close()
        await events.close()
