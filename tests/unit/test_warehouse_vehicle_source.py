from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.bootstrap.domain import (
    AcquiredDetectorSample,
    BootstrapSourceInfo,
)
from vehicle_intelligence.training.bootstrap.writer import (
    BootstrapSourceWriter,
    verify_bootstrap_source,
)
from vehicle_intelligence.training.domain import (
    DetectorAnnotation,
    DetectorRole,
    DetectorSample,
    TrainingBoundingBox,
)
from vehicle_intelligence.training.warehouse_vehicle import (
    VehicleClassPrediction,
    WarehouseVehicleImportOptions,
    WarehouseVehicleSourceBuilder,
    _find_blue_bbox,
    _remove_blue_overlay,
)


class _SequenceClassifier:
    model_name = "test-yolo.onnx"
    model_sha256 = "a" * 64

    def __init__(self, predictions: list[VehicleClassPrediction]) -> None:
        self._predictions = iter(predictions)

    def classify(self, _image: np.ndarray) -> VehicleClassPrediction:
        return next(self._predictions)


def _prediction(class_name: str) -> VehicleClassPrediction:
    return VehicleClassPrediction(
        class_name=class_name,
        confidence=0.90,
        class_margin=0.70,
        area_ratio=0.90,
        center_distance=0.01,
    )


def test_warehouse_import_excludes_exact_and_perceptual_duplicates(tmp_path: Path) -> None:
    base = _base_source(tmp_path)
    archive = tmp_path / "warehouse.tar.gz"
    accepted = _warehouse_jpeg(80)
    review = _warehouse_jpeg(120)
    rejected = _plain_jpeg()
    _archive(
        archive,
        [
            (_name("11111111-1111-1111-1111-111111111111", "a"), accepted),
            (_name("11111111-1111-1111-1111-111111111111", "b"), accepted),
            (_name("22222222-2222-2222-2222-222222222222", "a"), accepted + b"meta"),
            (_name("33333333-3333-3333-3333-333333333333", "a"), review),
            (_name("44444444-4444-4444-4444-444444444444", "a"), rejected),
        ],
    )
    output = tmp_path / "combined"
    result = WarehouseVehicleSourceBuilder(
        archive_path=archive,
        base_source_directory=base,
        output_directory=output,
        model_path=tmp_path / "unused.onnx",
        source_id="warehouse-test-v1",
        owner_namespace="phins-test",
        founder_id="tester",
        options=WarehouseVehicleImportOptions(
            minimum_brightness=0,
            minimum_contrast=0,
            minimum_sharpness=0,
        ),
        classifier=_SequenceClassifier(
            [_prediction("truck"), _prediction("truck"), _prediction("bus")]
        ),
    ).build()

    manifest, digest = verify_bootstrap_source(output)
    assert result.manifest_sha256 == digest
    assert result.base_sample_count == 1
    assert result.appended_sample_count == 1
    assert result.combined_sample_count == 2
    assert result.exact_duplicate_files_excluded == 1
    assert result.near_duplicate_images_excluded == 1
    assert result.review_queue_count == 1
    assert result.reject_count == 1
    assert manifest["sampleCount"] == 2

    duplicate_reasons = {
        json.loads(line)["reason"]
        for line in (output / "DUPLICATES.jsonl").read_text().splitlines()
    }
    assert duplicate_reasons == {
        "EXACT_SHA256_DUPLICATE",
        "PERCEPTUAL_NEAR_DUPLICATE",
    }
    review_record = json.loads((output / "REVIEW_QUEUE.jsonl").read_text())
    assert review_record["reason"] == "CLASS_REQUIRES_HUMAN_REVIEW"
    assert review_record["suggestion"]["className"] == "bus"
    reject_record = json.loads((output / "REJECTS.jsonl").read_text())
    assert reject_record["reason"] == "BURNED_IN_BOUNDING_BOX_NOT_RECOVERED"

    samples = [
        DetectorSample.model_validate_json(line)
        for line in (output / "annotations.jsonl").read_bytes().splitlines()
    ]
    warehouse = next(sample for sample in samples if sample.sample_id.startswith("phins-warehouse"))
    assert warehouse.annotations[0].class_name == "truck"
    assert warehouse.attributes["annotationReviewStatus"] == "MODEL_ASSISTED_UNREVIEWED"
    cleaned = cv2.imread(str(output / warehouse.image_path))
    assert cleaned is not None
    blue, green, red = cv2.split(cleaned)
    burned_blue = (blue > 200) & (green < 40) & (red < 40)
    assert int(np.count_nonzero(burned_blue)) < 20

    reused = WarehouseVehicleSourceBuilder(
        archive_path=archive,
        base_source_directory=base,
        output_directory=output,
        model_path=tmp_path / "unused.onnx",
        source_id="warehouse-test-v1",
        owner_namespace="phins-test",
        founder_id="tester",
        classifier=_SequenceClassifier([]),
    ).build()
    assert reused.reused is True
    assert reused.appended_sample_count == 1


def test_warehouse_import_rejects_unsafe_tar_paths(tmp_path: Path) -> None:
    base = _base_source(tmp_path)
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        data = _warehouse_jpeg(90)
        info = tarfile.TarInfo("../escape.jpg")
        info.size = len(data)
        stream.addfile(info, io.BytesIO(data))

    builder = WarehouseVehicleSourceBuilder(
        archive_path=archive,
        base_source_directory=base,
        output_directory=tmp_path / "combined",
        model_path=tmp_path / "unused.onnx",
        source_id="warehouse-test-v1",
        owner_namespace="phins-test",
        founder_id="tester",
        classifier=_SequenceClassifier([]),
    )
    with pytest.raises(DetectorDatasetError, match="unsafe"):
        builder.build()


def test_blue_overlay_cleanup_removes_nested_jpeg_rectangle_edges() -> None:
    image = np.full((240, 320, 3), 110, dtype=np.uint8)
    cv2.rectangle(image, (92, 12), (286, 218), (255, 0, 0), 2)
    cv2.rectangle(image, (100, 20), (278, 210), (255, 0, 0), 2)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    recovered = _find_blue_bbox(decoded)
    assert recovered is not None

    cleaned = _remove_blue_overlay(decoded, recovered[0])
    blue, green, red = cv2.split(cleaned)
    burned_blue = (
        (blue >= 155)
        & (green <= 75)
        & (red <= 75)
        & (blue.astype(np.int16) >= green.astype(np.int16) + 90)
        & (blue.astype(np.int16) >= red.astype(np.int16) + 90)
    )
    assert int(np.count_nonzero(burned_blue)) == 0


def _base_source(tmp_path: Path) -> Path:
    image = _plain_jpeg()
    sample = DetectorSample(
        sampleId="base-car",
        imagePath="images/base.jpg",
        groupId="base-car",
        cameraId="external",
        capturedAt=datetime(2024, 1, 1, tzinfo=UTC),
        attributes={"acceptanceEligible": False},
        annotations=(
            DetectorAnnotation(
                className="car",
                bbox=TrainingBoundingBox(x=10, y=10, width=50, height=40),
            ),
        ),
    )
    source = tmp_path / "base"
    BootstrapSourceWriter(DetectorRole.VEHICLE, source).write(
        BootstrapSourceInfo(
            source_id="base-test",
            dataset_url="https://example.invalid/base",
            revision="v1",
            annotation_license="test",
            image_license="test",
        ),
        [
            AcquiredDetectorSample(
                sample=sample,
                image_bytes=image,
                attribution={
                    "sample_id": sample.sample_id,
                    "source_dataset": "base-test",
                    "source_revision": "v1",
                    "license": "test",
                    "author": "tester",
                    "landing_url": "https://example.invalid/base",
                },
            )
        ],
    )
    return source


def _archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        directory = tarfile.TarInfo("Cameras")
        directory.type = tarfile.DIRTYPE
        stream.addfile(directory)
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))


def _name(group_id: str, nonce: str) -> str:
    return f"Cameras/front_image_{group_id}_1787698282859_{nonce}.jpg"


def _warehouse_jpeg(fill: int) -> bytes:
    image = np.full((90, 120, 3), fill, dtype=np.uint8)
    cv2.line(image, (25, 25), (90, 65), (240, 240, 240), 4)
    cv2.rectangle(image, (18, 12), (105, 78), (255, 0, 0), 3)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return encoded.tobytes()


def _plain_jpeg() -> bytes:
    image = np.full((90, 120, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
