from dataclasses import replace
from datetime import timedelta

import pytest

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.ports import DatasetSampleQuery
from vehicle_intelligence.application.review import (
    HumanPlateReviewService,
    PlateReviewCommand,
)
from vehicle_intelligence.domain import (
    AuthenticationMethod,
    DatasetSampleReason,
    EventStatus,
    Principal,
    UserRole,
)
from vehicle_intelligence.exceptions import (
    PlateReviewConflictError,
    PlateReviewValidationError,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_memory import (
    InMemoryDatasetSampleRepository,
)


def _operator() -> Principal:
    return Principal(
        id="operator-01",
        display_name="Gate Operator",
        role=UserRole.OPERATOR,
        authentication_method=AuthenticationMethod.DEVELOPMENT,
    )


class MissingMediaInspector:
    async def exists(self, _key: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_human_correction_preserves_prediction_and_creates_idempotent_sample(
    sample_event,
) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    event = replace(
        sample_event,
        status=EventStatus.NEEDS_REVIEW,
        media=replace(sample_event.media, plate_crop_key="vehicles/test/plate.jpg"),
    )
    await events.save(event)
    reviewed_at = event.occurred_at + timedelta(minutes=1)
    service = HumanPlateReviewService(
        events,
        samples,
        VietnamPlateNormalizer(),
        clock=lambda: reviewed_at,
    )
    command = PlateReviewCommand(
        text="51H12346",
        expected_revision=0,
        reviewer=_operator(),
        note="Operator checked the plate crop",
    )

    result = await service.review(event.id, command)
    retry = await service.review(event.id, command)
    page = await service.list_samples(DatasetSampleQuery())

    assert result.changed
    assert not retry.changed
    assert result.reason is DatasetSampleReason.HUMAN_CORRECTION
    assert result.event.schema_version == 2
    assert result.event.status is EventStatus.CONFIRMED
    assert result.event.plate.normalized == "51H-123.45"
    assert result.event.plate.final_normalized == "51H-123.46"
    assert result.event.plate.review.revision == 1
    assert result.event.plate.review.reviewed_by == "operator-01"
    assert retry.dataset_sample_id == result.dataset_sample_id
    assert len(page.items) == 1
    assert page.items[0].prediction.normalized == "51H-123.45"
    assert page.items[0].label == "51H-123.46"
    assert page.items[0].image_key == "vehicles/test/plate.jpg"
    assert page.items[0].prediction.model.name == "test-model"


@pytest.mark.asyncio
async def test_human_confirmation_and_stale_review_conflict(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    event = replace(
        sample_event,
        status=EventStatus.NEEDS_REVIEW,
        media=replace(sample_event.media, plate_crop_key="vehicles/test/plate.jpg"),
    )
    await events.save(event)
    service = HumanPlateReviewService(events, samples, VietnamPlateNormalizer())

    confirmed = await service.review(
        event.id,
        PlateReviewCommand("51H12345", 0, _operator()),
    )

    assert confirmed.reason is DatasetSampleReason.HUMAN_CONFIRMATION
    with pytest.raises(PlateReviewConflictError, match="expected 0, actual 1"):
        await service.review(
            event.id,
            PlateReviewCommand("51H12346", 0, _operator()),
        )


@pytest.mark.asyncio
async def test_human_review_rejects_invalid_or_missing_prediction(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    await events.save(sample_event)
    no_plate = replace(
        sample_event,
        id="evt-no-plate-review",
        track_id="gate-01:video-test:no-plate",
        plate=None,
        status=EventStatus.NO_PLATE,
    )
    await events.save(no_plate)
    service = HumanPlateReviewService(events, samples, VietnamPlateNormalizer())

    with pytest.raises(PlateReviewValidationError, match="invalid Vietnamese"):
        await service.review(
            sample_event.id,
            PlateReviewCommand("INVALID", 0, _operator()),
        )
    with pytest.raises(PlateReviewValidationError, match="no OCR plate"):
        await service.review(
            no_plate.id,
            PlateReviewCommand("51H12345", 0, _operator()),
        )


@pytest.mark.asyncio
async def test_human_review_skips_dataset_sample_when_crop_object_is_missing(
    sample_event,
) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    event = replace(
        sample_event,
        status=EventStatus.NEEDS_REVIEW,
        media=replace(sample_event.media, plate_crop_key="vehicles/missing/plate.jpg"),
    )
    await events.save(event)
    service = HumanPlateReviewService(
        events,
        samples,
        VietnamPlateNormalizer(),
        MissingMediaInspector(),
    )

    result = await service.review(
        event.id,
        PlateReviewCommand("51H12346", 0, _operator()),
    )

    assert result.dataset_sample_id is None
    assert (await service.list_samples(DatasetSampleQuery())).items == ()
