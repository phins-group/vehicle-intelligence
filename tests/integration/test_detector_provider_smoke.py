from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vehicle_intelligence.config import VehicleDetectorConfig
from vehicle_intelligence.domain import Detection
from vehicle_intelligence.infrastructure.vision.factory import create_vehicle_detector

ROOT = Path(__file__).resolve().parents[2]


def test_real_yolo_provider_loads_repository_artifact_and_returns_canonical_output() -> None:
    pytest.importorskip("ultralytics")
    model = ROOT / "models" / "yolo11n.pt"
    if not model.is_file():
        pytest.skip("repository YOLO smoke artifact is not available")
    detector = create_vehicle_detector(
        VehicleDetectorConfig(
            provider="yolo",
            model_path=str(model),
            model_name="vehicle-yolo-smoke",
            model_version="test",
            confidence=0.4,
            iou=0.7,
            image_size=64,
            device="cpu",
            classes=["car", "motorcycle", "bus", "truck"],
        )
    )

    detections = detector.detect(np.zeros((64, 96, 3), dtype=np.uint8))

    assert isinstance(detections, list)
    assert all(isinstance(item, Detection) for item in detections)
    assert all(0 <= item.confidence <= 1 for item in detections)
    assert all(0 <= item.bbox.x1 < item.bbox.x2 <= 96 for item in detections)
    assert all(0 <= item.bbox.y1 < item.bbox.y2 <= 64 for item in detections)


def test_real_picodet_smoke_requires_a_repository_artifact() -> None:
    model = ROOT / "models" / "picodet.onnx"
    if not model.is_file():
        pytest.skip("no evaluated PicoDet artifact exists in the repository")
    detector = create_vehicle_detector(
        VehicleDetectorConfig(
            provider="picodet",
            model_path=str(model),
            model_name="vehicle-picodet-smoke",
            model_version="test",
            confidence=0.4,
            iou=0.5,
            image_size=640,
            model_classes=["car", "motorcycle", "bus", "truck"],
            classes=["car", "motorcycle", "bus", "truck"],
        )
    )

    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert isinstance(detections, list)
    assert all(isinstance(item, Detection) for item in detections)
