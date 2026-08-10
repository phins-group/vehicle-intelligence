from vehicle_intelligence.domain import (
    DatasetSample,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    OCRDatasetPrediction,
)
from vehicle_intelligence.infrastructure.review_serialization import (
    dataset_sample_to_document,
    dataset_sample_to_jsonable,
    document_to_dataset_sample,
)


def test_dataset_sample_document_round_trip_preserves_feedback(sample_event) -> None:
    sample = DatasetSample(
        id="dss_test",
        sample_type=DatasetSampleType.PLATE_OCR,
        status=DatasetSampleStatus.READY,
        source_event_id=sample_event.id,
        image_key="vehicles/test/plate.jpg",
        prediction=OCRDatasetPrediction(
            raw=sample_event.plate.raw,
            normalized=sample_event.plate.normalized,
            confidence=sample_event.plate.confidence,
            model=sample_event.ai.ocr,
        ),
        label="51H-123.46",
        reason=DatasetSampleReason.HUMAN_CORRECTION,
        review_revision=1,
        reviewed_by="operator-01",
        reviewer_display_name="Gate Operator",
        reviewed_at=sample_event.occurred_at,
        created_at=sample_event.created_at,
    )

    document = dataset_sample_to_document(sample)

    assert document_to_dataset_sample(document) == sample
    assert document["createdAt"].tzinfo is not None
    assert dataset_sample_to_jsonable(sample)["createdAt"].endswith("Z")
