from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.model_quality import ModelQualityService
from vehicle_intelligence.config import ModelQualityConfig
from vehicle_intelligence.domain import (
    DatasetSample,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    EventStatus,
    OCRDatasetPrediction,
    PlateReview,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.quality_memory import (
    InMemoryModelQualityRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_memory import (
    InMemoryDatasetSampleRepository,
)


@pytest.mark.asyncio
async def test_quality_report_tracks_ocr_status_review_and_model_version(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    reviewed = replace(
        sample_event,
        schema_version=2,
        plate=replace(
            sample_event.plate,
            review=PlateReview(
                normalized="51H-123.46",
                revision=1,
                reviewed_at=sample_event.occurred_at + timedelta(minutes=1),
                reviewed_by="operator",
                reviewer_display_name="Operator",
            ),
        ),
    )
    unknown = replace(
        sample_event,
        id="evt_unknown",
        track_id="gate-01:unknown",
        status=EventStatus.NO_PLATE,
        plate=None,
        occurred_at=sample_event.occurred_at + timedelta(hours=1),
        created_at=sample_event.created_at + timedelta(hours=1),
    )
    assert await events.save(reviewed)
    assert await events.save(unknown)
    assert await samples.create(
        DatasetSample(
            id="dss_quality",
            sample_type=DatasetSampleType.PLATE_OCR,
            status=DatasetSampleStatus.READY,
            source_event_id=reviewed.id,
            image_key="vehicles/test/plate.jpg",
            prediction=OCRDatasetPrediction(
                raw=reviewed.plate.raw,
                normalized=reviewed.plate.normalized,
                confidence=reviewed.plate.confidence,
                model=reviewed.ai.ocr,
            ),
            label=reviewed.plate.review.normalized,
            reason=DatasetSampleReason.HUMAN_CORRECTION,
            review_revision=1,
            reviewed_by="operator",
            reviewer_display_name="Operator",
            reviewed_at=reviewed.plate.review.reviewed_at,
            created_at=reviewed.plate.review.reviewed_at,
        )
    )
    generated = datetime(2026, 8, 10, tzinfo=UTC)
    service = ModelQualityService(
        InMemoryModelQualityRepository(events, samples),
        ModelQualityConfig(default_window_days=7),
        clock=lambda: generated,
    )

    report = await service.report()

    assert report.totals.event_count == 2
    assert report.totals.readable_plate_count == 1
    assert report.totals.ocr_success_rate == 0.5
    assert report.totals.unknown_plate_rate == 0.5
    assert report.totals.reviewed_count == 1
    assert report.totals.corrected_count == 1
    assert report.totals.human_correction_rate == 1
    assert report.models[0].model.version == "1"
    assert report.feedback.ready == 1
    assert report.feedback.corrections == 1
    assert not report.truncated


@pytest.mark.asyncio
async def test_quality_report_rejects_unbounded_or_naive_ranges(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    service = ModelQualityService(
        InMemoryModelQualityRepository(events, samples),
        ModelQualityConfig(default_window_days=7, maximum_window_days=30),
        clock=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="maximum"):
        await service.report(
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 8, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="timezone"):
        await service.report(datetime(2026, 8, 1), datetime(2026, 8, 2, tzinfo=UTC))
