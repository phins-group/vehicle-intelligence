from __future__ import annotations

import hashlib

import numpy as np
import pytest

from vehicle_intelligence.application.ports import Detector, PlateDetector, VehicleDetector
from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig
from vehicle_intelligence.domain import Detection, PlateDetection
from vehicle_intelligence.exceptions import ModelLoadError
from vehicle_intelligence.infrastructure.vision.ultralytics import (
    UltralyticsPlateDetector,
    YoloDetector,
)


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, coordinates, confidences, classes) -> None:
        self.xyxy = FakeTensor(coordinates)
        self.conf = FakeTensor(confidences)
        self.cls = FakeTensor(classes)


class FakeResult:
    def __init__(self, boxes, names) -> None:
        self.boxes = boxes
        self.names = names
        self.obb = None


class FakeModel:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result for _ in kwargs["source"]]


def test_yolo_provider_returns_mapped_clamped_canonical_detection(tmp_path) -> None:
    artifact = tmp_path / "vehicle.pt"
    artifact.write_bytes(b"vehicle-checkpoint")
    model = FakeModel(
        FakeResult(
            FakeBoxes([[-5.2, 2.1, 20.8, 14.4]], [0.9], [2]),
            {2: "truck"},
        )
    )
    detector = YoloDetector(
        VehicleDetectorConfig(
            provider="yolo",
            model_path=str(artifact),
            model_name="vehicle-yolo",
            model_version="1",
            confidence=0.4,
            iou=0.5,
            image_size=640,
            device="cpu",
            model_classes=["person", "bicycle", "car"],
            classes=["car"],
        ),
        model=model,
    )

    detections = detector.detect(np.zeros((10, 12, 3), dtype=np.uint8))

    assert isinstance(detector, Detector)
    assert isinstance(detector, VehicleDetector)
    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].bbox.as_xyxy() == (0, 2, 12, 10)
    assert detections[0].class_id == 2
    assert detections[0].class_name == "car"
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].model.hash == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert model.calls[0]["conf"] == 0.4
    assert model.calls[0]["iou"] == 0.5
    assert model.calls[0]["device"] == "cpu"


def test_yolo_plate_provider_returns_existing_canonical_plate_detection(tmp_path) -> None:
    artifact = tmp_path / "plate.pt"
    artifact.write_bytes(b"plate-checkpoint")
    model = FakeModel(
        FakeResult(
            FakeBoxes([[1.2, 2.2, 8.8, 6.8]], [0.75], [0]),
            {0: "license_plate"},
        )
    )
    detector = UltralyticsPlateDetector(
        DetectorConfig(
            provider="yolo",
            model_path=str(artifact),
            model_name="plate-yolo",
            model_version="1",
            confidence=0.4,
            iou=0.5,
        ),
        model=model,
    )

    detections = detector.detect(np.zeros((8, 10, 3), dtype=np.uint8))

    assert isinstance(detector, Detector)
    assert isinstance(detector, PlateDetector)
    assert len(detections) == 1
    assert isinstance(detections[0], PlateDetection)
    assert detections[0].bbox.as_xyxy() == (1, 2, 9, 7)
    assert detections[0].confidence == pytest.approx(0.75)

    batch = detector.detect_batch(
        [
            np.zeros((8, 10, 3), dtype=np.uint8),
            np.zeros((8, 10, 3), dtype=np.uint8),
        ]
    )
    assert len(batch) == 2
    assert all(len(items) == 1 for items in batch)
    assert len(model.calls[-1]["source"]) == 2


def test_ultralytics_rejects_a_checkpoint_with_the_wrong_configured_hash(tmp_path) -> None:
    artifact = tmp_path / "plate.pt"
    artifact.write_bytes(b"untrusted-checkpoint")

    with pytest.raises(ModelLoadError, match="SHA-256 mismatch"):
        UltralyticsPlateDetector(
            DetectorConfig(
                provider="yolo",
                model_path=str(artifact),
                model_name="plate-yolo",
                model_version="1",
                model_hash="0" * 64,
                confidence=0.4,
                iou=0.5,
            ),
            model=FakeModel(FakeResult(None, {})),
        )
