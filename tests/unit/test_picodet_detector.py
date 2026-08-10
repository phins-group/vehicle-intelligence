from __future__ import annotations

import hashlib

import numpy as np
import pytest

from vehicle_intelligence.application.ports import Detector, PlateDetector, VehicleDetector
from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig
from vehicle_intelligence.domain import Detection, PlateDetection
from vehicle_intelligence.exceptions import InferenceError, ModelLoadError
from vehicle_intelligence.infrastructure.vision.picodet import (
    PicoDetDetector,
    PicoDetPlateDetector,
)


class FakeInput:
    def __init__(self, name: str, input_type: str = "tensor(float)") -> None:
        self.name = name
        self.type = input_type


class FakeSession:
    def __init__(self, outputs, input_names=("image", "scale_factor", "im_shape")) -> None:
        self.outputs = outputs
        self.inputs = [FakeInput(name) for name in input_names]
        self.feed = None
        self.runs = 0

    def get_inputs(self):
        return self.inputs

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, input_feed):
        self.feed = input_feed
        self.runs += 1
        return self.outputs


def artifact(tmp_path) -> tuple[str, str]:
    path = tmp_path / "picodet.onnx"
    path.write_bytes(b"test-picodet-onnx")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def raw_outputs() -> list[np.ndarray]:
    scores = np.full((1, 4, 2), 0.01, dtype=np.float32)
    scores[0, 0, 0] = 0.90
    scores[0, 3, 1] = 0.95
    distributions = np.zeros((1, 4, 8), dtype=np.float32)
    return [scores, distributions]


def test_picodet_vehicle_preprocesses_decodes_maps_and_clamps(tmp_path) -> None:
    path, digest = artifact(tmp_path)
    session = FakeSession(raw_outputs())
    detector = PicoDetDetector(
        VehicleDetectorConfig(
            provider="picodet",
            model_path=path,
            model_name="vehicle-picodet",
            model_version="1",
            model_hash=digest,
            confidence=0.4,
            iou=0.5,
            image_size=16,
            onnx_output_format="raw",
            model_classes=["car", "person"],
            classes=["car"],
            picodet={"strides": [8], "nms_top_k": 10, "keep_top_k": 10},
        ),
        session=session,
    )
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :, 2] = 255

    detections = detector.detect(image)

    assert isinstance(detector, Detector)
    assert isinstance(detector, VehicleDetector)
    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert detections[0].bbox.as_xyxy() == (0, 0, 8, 4)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].class_id == 0
    assert detections[0].class_name == "car"
    assert detections[0].model.hash == digest
    assert session.runs == 1
    assert session.feed is not None
    assert session.feed["image"].shape == (1, 3, 16, 16)
    assert session.feed["scale_factor"].tolist() == [[2.0, 1.0]]
    assert session.feed["im_shape"].tolist() == [[16.0, 16.0]]
    assert session.feed["image"][0, 0, 0, 0] == pytest.approx(
        (1.0 - 0.485) / 0.229,
        rel=1e-5,
    )


def test_picodet_plate_accepts_post_nms_rows_in_caller_coordinates(tmp_path) -> None:
    path, digest = artifact(tmp_path)
    # PaddleDetection post-NMS format: class, score, x1, y1, x2, y2.
    session = FakeSession(
        [
            np.asarray([[[0, 0.88, -2, 2, 20, 15]]], dtype=np.float32),
            np.asarray([1], dtype=np.int32),
        ],
        input_names=("image",),
    )
    detector = PicoDetPlateDetector(
        DetectorConfig(
            provider="picodet",
            model_path=path,
            model_name="plate-picodet",
            model_version="1",
            model_hash=digest,
            confidence=0.4,
            iou=0.5,
            image_size=16,
            onnx_output_format="nms",
        ),
        session=session,
    )

    detections = detector.detect(np.zeros((8, 16, 3), dtype=np.uint8))

    assert isinstance(detector, Detector)
    assert isinstance(detector, PlateDetector)
    assert len(detections) == 1
    assert isinstance(detections[0], PlateDetection)
    assert detections[0].bbox.as_xyxy() == (0, 1, 16, 8)
    assert detections[0].confidence == pytest.approx(0.88)
    assert detections[0].model.hash == digest


def test_picodet_post_nms_empty_sentinel_returns_no_detection(tmp_path) -> None:
    path, _ = artifact(tmp_path)
    session = FakeSession(
        [
            np.asarray([[[-1, 0, 0, 0, 0, 0]]], dtype=np.float32),
            np.asarray([0], dtype=np.int32),
        ],
        input_names=("image", "scale_factor"),
    )
    detector = PicoDetPlateDetector(
        DetectorConfig(
            provider="picodet",
            model_path=path,
            model_name="plate-picodet",
            model_version="1",
            confidence=0.4,
            iou=0.5,
            image_size=16,
            onnx_output_format="nms",
        ),
        session=session,
    )

    assert detector.detect(np.zeros((8, 16, 3), dtype=np.uint8)) == []


def test_picodet_requires_explicit_vehicle_class_mapping(tmp_path) -> None:
    path, _ = artifact(tmp_path)
    with pytest.raises(ModelLoadError, match="model_classes"):
        PicoDetDetector(
            VehicleDetectorConfig(
                provider="picodet",
                model_path=path,
                model_name="vehicle-picodet",
                model_version="1",
                confidence=0.4,
                iou=0.5,
            ),
            session=FakeSession(raw_outputs()),
        )


def test_picodet_wraps_backend_failure(tmp_path) -> None:
    path, _ = artifact(tmp_path)

    class FailingSession(FakeSession):
        def run(self, _output_names, _input_feed):
            raise RuntimeError("backend detail")

    detector = PicoDetDetector(
        VehicleDetectorConfig(
            provider="picodet",
            model_path=path,
            model_name="vehicle-picodet",
            model_version="1",
            confidence=0.4,
            iou=0.5,
            image_size=16,
            model_classes=["car", "person"],
            classes=["car"],
            picodet={"strides": [8]},
        ),
        session=FailingSession(raw_outputs()),
    )

    with pytest.raises(InferenceError, match="PicoDet inference failed"):
        detector.detect(np.zeros((8, 16, 3), dtype=np.uint8))
