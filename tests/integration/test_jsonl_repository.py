from dataclasses import replace
from datetime import timedelta

from vehicle_intelligence.application.ports import EventQuery
from vehicle_intelligence.domain import EventStatus, PlateReview
from vehicle_intelligence.infrastructure.persistence.jsonl import JsonlVehicleEventRepository


async def test_jsonl_repository_deduplicates_and_cursor_paginates(tmp_path, sample_event) -> None:
    repository = JsonlVehicleEventRepository(tmp_path / "events.jsonl")
    await repository.ensure_indexes()
    second = replace(
        sample_event,
        id="evt_second",
        track_id="gate-01:video-test:13",
        occurred_at=sample_event.occurred_at + timedelta(seconds=1),
    )

    assert await repository.save(sample_event)
    assert not await repository.save(sample_event)
    assert await repository.save(second)

    first_page = await repository.list(EventQuery(limit=1))
    second_page = await repository.list(EventQuery(limit=1, cursor=first_page.next_cursor))

    assert [item.id for item in first_page.items] == ["evt_second"]
    assert first_page.next_cursor is not None
    assert [item.id for item in second_page.items] == ["evt_test"]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2

    reloaded = JsonlVehicleEventRepository(tmp_path / "events.jsonl")
    await reloaded.ensure_indexes()
    assert (await reloaded.get("evt_test")) == sample_event


async def test_jsonl_repository_durably_rewrites_one_reviewed_event(
    tmp_path,
    sample_event,
) -> None:
    path = tmp_path / "events.jsonl"
    repository = JsonlVehicleEventRepository(path)
    await repository.ensure_indexes()
    await repository.save(sample_event)
    review = PlateReview(
        normalized="51H-123.46",
        revision=1,
        reviewed_at=sample_event.occurred_at,
        reviewed_by="operator-01",
        reviewer_display_name="Gate Operator",
    )
    reviewed = replace(
        sample_event,
        schema_version=2,
        status=EventStatus.CONFIRMED,
        plate=replace(sample_event.plate, review=review),
    )

    assert await repository.update_plate_review(reviewed, expected_revision=0) == reviewed
    assert len(path.read_text().splitlines()) == 1

    reloaded = JsonlVehicleEventRepository(path)
    await reloaded.ensure_indexes()
    assert await reloaded.get(reviewed.id) == reviewed
