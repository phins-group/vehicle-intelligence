import json
from dataclasses import replace
from datetime import timedelta

import cv2
import numpy as np
import pytest

from vehicle_intelligence.application.dataset_evaluation import (
    evaluate_ocr_records,
    release_gates,
)
from vehicle_intelligence.application.dataset_export import (
    OCRDatasetExportService,
    verify_dataset_export,
)
from vehicle_intelligence.config import DatasetExportConfig
from vehicle_intelligence.domain import (
    DatasetSample,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    OCRDatasetPrediction,
)
from vehicle_intelligence.exceptions import DatasetExportError
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_memory import (
    InMemoryDatasetSampleRepository,
)
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage
from vehicle_intelligence.infrastructure.vision.opencv import OpenCVDatasetImageTranscoder


def _sample(sample_event, sample_id: str, event_id: str, image_key: str) -> DatasetSample:
    return DatasetSample(
        id=sample_id,
        sample_type=DatasetSampleType.PLATE_OCR,
        status=DatasetSampleStatus.READY,
        source_event_id=event_id,
        image_key=image_key,
        prediction=OCRDatasetPrediction(
            raw=sample_event.plate.raw,
            normalized=sample_event.plate.normalized,
            confidence=sample_event.plate.confidence,
            model=sample_event.ai.ocr,
        ),
        label="51H-123.46",
        reason=DatasetSampleReason.HUMAN_CORRECTION,
        review_revision=1,
        reviewed_by="operator",
        reviewer_display_name="Operator",
        reviewed_at=sample_event.occurred_at,
        created_at=sample_event.created_at,
    )


@pytest.mark.asyncio
async def test_dataset_export_is_atomic_verified_idempotent_and_camera_grouped(
    tmp_path,
    sample_event,
) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    media = LocalMediaStorage(tmp_path / "media")
    second = replace(
        sample_event,
        id="evt_second",
        track_id="gate-01:second",
        occurred_at=sample_event.occurred_at + timedelta(seconds=1),
        created_at=sample_event.created_at + timedelta(seconds=1),
    )
    missing_event = replace(
        sample_event,
        id="evt_missing",
        track_id="gate-01:missing",
        occurred_at=sample_event.occurred_at + timedelta(seconds=2),
        created_at=sample_event.created_at + timedelta(seconds=2),
    )
    assert await events.save(sample_event)
    assert await events.save(second)
    assert await events.save(missing_event)
    assert await samples.create(
        _sample(sample_event, "dss_good", sample_event.id, "plates/good.jpg")
    )
    assert await samples.create(_sample(sample_event, "dss_good_2", second.id, "plates/good-2.jpg"))
    assert await samples.create(
        _sample(sample_event, "dss_missing", missing_event.id, "plates/missing.jpg")
    )
    image = np.full((48, 160, 3), 180, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    await media.put("plates/good.jpg", bytes(buffer), "image/jpeg")
    await media.put("plates/good-2.jpg", bytes(buffer), "image/jpeg")
    config = DatasetExportConfig(
        output_directory=tmp_path / "exports",
        batch_size=10,
    )
    service = OCRDatasetExportService(
        config,
        samples,
        events,
        media,
        OpenCVDatasetImageTranscoder(),
        clock=lambda: sample_event.occurred_at + timedelta(hours=1),
    )

    result = await service.export("ocr-test")
    retry = await service.export("ocr-test")
    manifest, digest = verify_dataset_export(result.directory)

    assert result.exported_count == 2
    assert result.failed_count == 1
    assert result.manifest_sha256 == digest
    assert retry.reused
    assert manifest["sampleIds"] == ["dss_good", "dss_good_2"]
    records = [
        json.loads(line) for line in (result.directory / "labels.jsonl").read_text().splitlines()
    ]
    assert {record["cameraId"] for record in records} == {"gate-01"}
    assert len({record["split"] for record in records}) == 1
    assert records[0]["split"] in {"train", "validation", "test"}
    assert (await samples.get("dss_good")).status is DatasetSampleStatus.EXPORTED
    assert (await samples.get("dss_missing")).status is DatasetSampleStatus.EXPORT_FAILED
    image_path = result.directory / records[0]["imagePath"]
    image_path.write_bytes(b"tampered")
    with pytest.raises(DatasetExportError, match="verification failed"):
        verify_dataset_export(result.directory)


def test_ocr_feedback_evaluation_reports_accuracy_calibration_and_gates() -> None:
    records = [
        {
            "split": "train",
            "label": "51H-123.45",
            "prediction": {
                "normalized": "51H-123.45",
                "confidence": 0.9,
                "model": {"name": "ocr", "version": "v1"},
            },
        },
        {
            "split": "test",
            "label": "51H-123.45",
            "prediction": {
                "normalized": "51H-123.4S",
                "confidence": 0.8,
                "model": {"name": "ocr", "version": "v1"},
            },
        },
    ]

    evaluation = evaluate_ocr_records(records)

    assert evaluation.overall.exact_accuracy == 0.5
    assert evaluation.overall.character_accuracy > 0.9
    assert set(evaluation.by_split) == {"test", "train"}
    assert release_gates(evaluation, minimum_exact_accuracy=0.6) == ["exact_accuracy"]
    with pytest.raises(DatasetExportError, match=r"must be in \[0, 1\]"):
        release_gates(evaluation, maximum_ece=1.1)


def test_dataset_image_transcoder_preserves_bounded_failure_codes() -> None:
    transcoder = OpenCVDatasetImageTranscoder()
    image = np.full((4, 5, 3), 180, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded

    valid = transcoder.normalize_jpeg(
        bytes(buffer),
        maximum_pixels=20,
        jpeg_quality=95,
    )
    oversized = transcoder.normalize_jpeg(
        bytes(buffer),
        maximum_pixels=19,
        jpeg_quality=95,
    )
    invalid = transcoder.normalize_jpeg(
        b"not-an-image",
        maximum_pixels=20,
        jpeg_quality=95,
    )

    assert valid.jpeg is not None
    assert valid.error_code is None
    assert oversized.jpeg is None
    assert oversized.error_code == "MEDIA_DIMENSIONS_EXCEEDED"
    assert invalid.jpeg is None
    assert invalid.error_code == "MEDIA_INVALID"
