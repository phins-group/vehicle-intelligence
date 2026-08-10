from __future__ import annotations

import hashlib

import numpy as np
import pytest

from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig
from vehicle_intelligence.exceptions import DependencyUnavailableError, ModelLoadError
from vehicle_intelligence.infrastructure.vision.onnx_runtime import (
    OnnxRuntimePlateDetector,
    OnnxRuntimeVehicleDetector,
    select_execution_providers,
)


class FakeInput:
    name = "images"
    type = "tensor(float)"


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.input_tensor: np.ndarray | None = None

    def get_inputs(self):
        return [FakeInput()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, input_feed):
        self.input_tensor = input_feed["images"]
        return [self.output]


def _artifact(tmp_path) -> tuple[str, str]:
    path = tmp_path / "detector.onnx"
    path.write_bytes(b"test-onnx-artifact")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_vehicle_raw_yolo_output_is_letterboxed_filtered_and_nms_applied(tmp_path) -> None:
    path, digest = _artifact(tmp_path)
    # Feature-first YOLO output: xywh + two class scores, two overlapping cars.
    output = np.asarray(
        [[[320, 322], [320, 321], [200, 200], [100, 100], [0.90, 0.80], [0.10, 0.20]]],
        dtype=np.float32,
    )
    session = FakeSession(output)
    detector = OnnxRuntimeVehicleDetector(
        VehicleDetectorConfig(
            provider="onnxruntime",
            model_path=path,
            model_name="vehicle-onnx",
            model_version="1",
            model_hash=digest,
            confidence=0.4,
            iou=0.5,
            image_size=640,
            onnx_output_format="raw",
            model_classes=["car", "person"],
            classes=["car"],
        ),
        session=session,
    )

    detections = detector.detect(np.zeros((320, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].class_name == "car"
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].bbox.as_xyxy() == (220, 110, 420, 210)
    assert detections[0].model.hash == digest
    assert session.input_tensor is not None
    assert session.input_tensor.shape == (1, 3, 640, 640)
    assert session.input_tensor.dtype == np.float32


def test_plate_post_nms_output_is_restored_to_source_coordinates(tmp_path) -> None:
    path, _ = _artifact(tmp_path)
    session = FakeSession(
        np.asarray([[[100, 200, 300, 400, 0.88, 0]]], dtype=np.float32)
    )
    detector = OnnxRuntimePlateDetector(
        DetectorConfig(
            provider="onnxruntime",
            model_path=path,
            model_name="plate-onnx",
            model_version="1",
            confidence=0.4,
            iou=0.5,
            image_size=640,
            onnx_output_format="auto",
        ),
        session=session,
    )

    detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].bbox.as_xyxy() == (100, 200, 300, 400)
    assert detections[0].confidence == pytest.approx(0.88)


def test_onnx_model_hash_mismatch_fails_before_inference(tmp_path) -> None:
    path, _ = _artifact(tmp_path)
    with pytest.raises(ModelLoadError, match="SHA-256"):
        OnnxRuntimePlateDetector(
            DetectorConfig(
                provider="onnxruntime",
                model_path=path,
                model_name="plate-onnx",
                model_version="1",
                model_hash="0" * 64,
                confidence=0.4,
                iou=0.5,
            ),
            session=FakeSession(np.zeros((1, 0, 6), dtype=np.float32)),
        )


def test_execution_provider_selection_requires_accelerator_and_adds_fallbacks() -> None:
    available = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert select_execution_providers(["tensorrt"], available) == (
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert select_execution_providers(["cuda"], available) == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    with pytest.raises(DependencyUnavailableError, match="unavailable"):
        select_execution_providers(["tensorrt"], ["CPUExecutionProvider"])
