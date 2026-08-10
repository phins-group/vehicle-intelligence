from dataclasses import replace

from vehicle_intelligence.domain import EventStatus, PlateReview
from vehicle_intelligence.infrastructure.serialization import (
    document_to_event,
    event_to_document,
    event_to_jsonable,
)


def test_event_document_round_trip_preserves_domain_event(sample_event) -> None:
    document = event_to_document(sample_event)
    restored = document_to_event(document)

    assert restored == sample_event
    assert document["occurredAt"].tzinfo is not None
    assert event_to_jsonable(sample_event)["occurredAt"].endswith("Z")


def test_reviewed_event_v2_preserves_prediction_review_and_final_plate(sample_event) -> None:
    review = PlateReview(
        normalized="51H-123.46",
        revision=1,
        reviewed_at=sample_event.occurred_at,
        reviewed_by="operator-01",
        reviewer_display_name="Gate Operator",
        note="Checked crop",
    )
    event = replace(
        sample_event,
        schema_version=2,
        status=EventStatus.CONFIRMED,
        plate=replace(sample_event.plate, review=review),
    )

    document = event_to_document(event)
    public = event_to_jsonable(event)

    assert document_to_event(document) == event
    assert document["plate"]["prediction"]["normalized"] == "51H-123.45"
    assert document["plate"]["final"] == "51H-123.46"
    assert document["plate"]["review"]["reviewedAt"].tzinfo is not None
    assert public["plate"]["review"]["reviewedAt"].endswith("Z")
