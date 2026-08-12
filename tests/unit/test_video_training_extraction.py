from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_intelligence.domain import (
    BoundingBox,
    Detection,
    ModelMetadata,
    PlateDetection,
)
from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.domain import DetectorSample
from vehicle_intelligence.training.video_extraction import (
    VideoExtractionOptions,
    VideoTrainingImageExtractor,
)


class _VehicleDetector:
    def detect(self, image: np.ndarray) -> list[Detection]:
        return self._result(image)

    def detect_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        return [self._result(image) for image in images]

    @staticmethod
    def _result(image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        return [
            Detection(
                bbox=BoundingBox(10, 8, width - 10, height - 8),
                confidence=0.91,
                class_id=2,
                class_name="car",
                model=ModelMetadata("vehicle-test", "v1"),
            )
        ]


class _PlateDetector:
    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        height, width = image.shape[:2]
        return [
            PlateDetection(
                bbox=BoundingBox(
                    max(width // 3, 1),
                    max(height // 2, 1),
                    max(2 * width // 3, 2),
                    max(3 * height // 4, 2),
                ),
                confidence=0.83,
                model=ModelMetadata("plate-test", "v2"),
            )
        ]


def test_extracts_reviewable_vehicle_and_plate_images_with_lineage(tmp_path: Path) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    video = source / "traffic.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        1.0,
        (160, 120),
    )
    assert writer.isOpened()
    for index in range(3):
        image = np.full((120, 160, 3), 50 + index * 30, dtype=np.uint8)
        cv2.rectangle(image, (20 + index, 20), (140, 100), (200, 120, 80), -1)
        writer.write(image)
    writer.release()

    output = tmp_path / "extract"
    result = VideoTrainingImageExtractor(
        _VehicleDetector(),
        _PlateDetector(),
        VideoExtractionOptions(
            input_directory=source,
            output_directory=output,
            sample_interval_seconds=1.0,
            detector_frame_max_edge=640,
            plate_context_max_edge=640,
            vehicle_crop_max_edge=640,
            batch_size=2,
        ),
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).extract()

    assert result.videos_discovered == 1
    assert result.videos_processed == 1
    assert result.sampled_frames == 3
    assert result.vehicle_training_images == 3
    assert result.vehicle_crop_images == 3
    assert result.plate_training_images == 3
    assert result.plate_crop_images == 3
    assert result.vehicle_class_counts == {"car": 3}

    vehicle_lines = (output / "vehicle/annotations.auto.jsonl").read_text().splitlines()
    plate_lines = (output / "plate/annotations.auto.jsonl").read_text().splitlines()
    vehicle_records = [DetectorSample.model_validate_json(line) for line in vehicle_lines]
    plate_records = [DetectorSample.model_validate_json(line) for line in plate_lines]
    assert len({sample.group_id for sample in vehicle_records + plate_records}) == 1
    assert all(sample.attributes["reviewStatus"] == "PENDING_REVIEW" for sample in vehicle_records)
    assert all(sample.annotations[0].class_name == "license_plate" for sample in plate_records)
    assert len(list((output / "vehicle/crops/car").glob("*.jpg"))) == 3
    assert len(list((output / "plate/crops").glob("*.jpg"))) == 3

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["founderId"] == "duyhuynh"
    assert manifest["status"] == "COMPLETE"
    assert manifest["releaseEligible"] is False
    assert manifest["reviewStatus"] == "PENDING_REVIEW"


def test_refuses_to_mix_with_existing_extraction_files(tmp_path: Path) -> None:
    source = tmp_path / "videos"
    source.mkdir()
    (source / "empty.mp4").write_bytes(b"not-a-video")
    output = tmp_path / "extract"
    output.mkdir()
    (output / "keep.txt").write_text("operator-owned")

    extractor = VideoTrainingImageExtractor(
        _VehicleDetector(),
        _PlateDetector(),
        VideoExtractionOptions(input_directory=source, output_directory=output),
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    )
    with pytest.raises(DetectorDatasetError, match="not empty"):
        extractor.extract()
    assert (output / "keep.txt").read_text() == "operator-owned"
