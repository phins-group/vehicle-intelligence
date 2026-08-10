"""ONNX Runtime YOLO detector adapters with explicit execution-provider policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig
from vehicle_intelligence.domain import BoundingBox, Detection, ModelMetadata, PlateDetection
from vehicle_intelligence.exceptions import (
    DependencyUnavailableError,
    InferenceError,
    ModelLoadError,
)
from vehicle_intelligence.infrastructure.vision.model_artifact import validated_model_artifact
from vehicle_intelligence.infrastructure.vision.postprocessing import nms_indices

_COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


class _SessionInput(Protocol):
    name: str
    type: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_SessionInput]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(self, output_names: object, input_feed: dict[str, NDArray[Any]]) -> list[Any]: ...


def select_execution_providers(
    requested: Sequence[str], available: Sequence[str]
) -> tuple[str, ...]:
    """Resolve aliases and require the requested accelerator instead of falling back silently."""

    available_set = set(available)
    aliases = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
    }
    values = list(requested) or ["cpu"]
    primary = aliases.get(values[0].lower(), values[0])
    if primary not in available_set:
        raise DependencyUnavailableError(
            f"ONNX execution provider '{primary}' is unavailable; "
            f"available={sorted(available_set)}"
        )
    resolved: list[str] = []
    for value in values:
        provider = aliases.get(value.lower(), value)
        if provider not in available_set:
            raise DependencyUnavailableError(
                f"ONNX execution provider '{provider}' is unavailable; "
                f"available={sorted(available_set)}"
            )
        if provider not in resolved:
            resolved.append(provider)
    if primary == "TensorrtExecutionProvider":
        for fallback in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            if fallback in available_set and fallback not in resolved:
                resolved.append(fallback)
    elif primary != "CPUExecutionProvider" and "CPUExecutionProvider" in available_set:
        resolved.append("CPUExecutionProvider")
    return tuple(resolved)


def requested_execution_providers(config: DetectorConfig) -> tuple[str, ...]:
    """Translate detector device configuration into ONNX execution-provider aliases."""

    if config.execution_providers:
        return tuple(config.execution_providers)
    device = (config.device or "cpu").lower()
    if device in {"tensorrt", "cuda", "coreml", "cpu"}:
        return (device,)
    if device.isdigit() or device.startswith("cuda:"):
        return ("cuda",)
    return (device,)


class _OnnxRuntimeAdapter:
    def __init__(
        self,
        config: DetectorConfig,
        *,
        session: _InferenceSession | None = None,
    ) -> None:
        self._config = config
        path, artifact_hash = validated_model_artifact(config.model_path, config.model_hash)
        if path.suffix.lower() != ".onnx":
            raise ModelLoadError("ONNX Runtime detector requires a .onnx model artifact")
        self._metadata = ModelMetadata(
            name=config.model_name,
            version=config.model_version,
            hash=artifact_hash,
        )
        self._session = session or self._create_session(path)
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ModelLoadError("ONNX detector must expose exactly one image input")
        self._input_name = inputs[0].name
        self._input_dtype = np.float16 if inputs[0].type == "tensor(float16)" else np.float32

    @property
    def execution_providers(self) -> tuple[str, ...]:
        return tuple(self._session.get_providers())

    def _create_session(self, path: object) -> _InferenceSession:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyUnavailableError(
                "ONNX Runtime is not installed; install the 'optimization' extra"
            ) from exc
        providers = select_execution_providers(
            requested_execution_providers(self._config), ort.get_available_providers()
        )
        try:
            return ort.InferenceSession(str(path), providers=list(providers))
        except Exception as exc:
            raise ModelLoadError(f"cannot load ONNX detector model: {path}") from exc

    def _predict_rows(
        self, image: NDArray[np.uint8], class_count: int | None
    ) -> NDArray[np.float32]:
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise InferenceError("ONNX detector input must be a non-empty BGR image")
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise InferenceError(f"{self._config.model_name} ONNX inference failed") from exc
        if not outputs:
            raise InferenceError("ONNX detector produced no output")
        predictions = np.asarray(outputs[0], dtype=np.float32)
        rows, is_xyxy = self._decode_layout(predictions, class_count)
        if rows.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        if not is_xyxy:
            rows = self._decode_raw_rows(rows)
        rows = rows[rows[:, 4] >= self._config.confidence]
        if rows.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        keep = nms_indices(
            rows[:, :4],
            rows[:, 4],
            rows[:, 5].astype(np.int64),
            self._config.iou,
            class_agnostic=class_count == 1,
        )
        selected = rows[keep].copy()
        selected[:, [0, 2]] = (selected[:, [0, 2]] - pad_x) / scale
        selected[:, [1, 3]] = (selected[:, [1, 3]] - pad_y) / scale
        selected[:, [0, 2]] = np.clip(selected[:, [0, 2]], 0, image.shape[1])
        selected[:, [1, 3]] = np.clip(selected[:, [1, 3]], 0, image.shape[0])
        return selected

    def _preprocess(
        self, image: NDArray[np.uint8]
    ) -> tuple[NDArray[Any], float, float, float]:
        target = self._config.image_size
        height, width = image.shape[:2]
        scale = min(target / width, target / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (target - resized_width) / 2
        pad_y = (target - resized_height) / 2
        left, right = math.floor(pad_x), math.ceil(pad_x)
        top, bottom = math.floor(pad_y), math.ceil(pad_y)
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=self._input_dtype)
        tensor /= np.asarray(255.0, dtype=self._input_dtype)
        return tensor, scale, float(left), float(top)

    def _decode_layout(
        self, predictions: NDArray[np.float32], class_count: int | None
    ) -> tuple[NDArray[np.float32], bool]:
        while predictions.ndim > 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise InferenceError(
                f"unsupported ONNX detector output shape: {tuple(predictions.shape)}"
            )
        output_format = self._config.onnx_output_format
        if output_format == "nms" or (
            output_format == "auto" and predictions.shape[1] in {6, 7}
        ):
            if predictions.shape[1] < 6:
                raise InferenceError("NMS detector output must contain at least six columns")
            if predictions.shape[1] == 7:
                return predictions[:, 1:7], True
            return predictions[:, :6], True
        expected_features = 4 + class_count if class_count is not None else None
        if expected_features is not None and predictions.shape[0] == expected_features:
            predictions = predictions.T
        elif expected_features is not None and predictions.shape[1] == expected_features:
            pass
        elif predictions.shape[0] <= 512 and predictions.shape[1] > predictions.shape[0]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise InferenceError("raw YOLO output must contain box and class scores")
        return predictions, False

    @staticmethod
    def _decode_raw_rows(predictions: NDArray[np.float32]) -> NDArray[np.float32]:
        scores = predictions[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        centers = predictions[:, :4]
        rows = np.empty((predictions.shape[0], 6), dtype=np.float32)
        rows[:, 0] = centers[:, 0] - centers[:, 2] / 2
        rows[:, 1] = centers[:, 1] - centers[:, 3] / 2
        rows[:, 2] = centers[:, 0] + centers[:, 2] / 2
        rows[:, 3] = centers[:, 1] + centers[:, 3] / 2
        rows[:, 4] = confidences
        rows[:, 5] = class_ids
        return rows

    @staticmethod
    def _bbox(row: NDArray[np.float32]) -> BoundingBox | None:
        left, top = math.floor(float(row[0])), math.floor(float(row[1]))
        right, bottom = math.ceil(float(row[2])), math.ceil(float(row[3]))
        if right <= left or bottom <= top:
            return None
        return BoundingBox(left, top, right, bottom)


class OnnxRuntimeVehicleDetector(_OnnxRuntimeAdapter):
    def __init__(
        self,
        config: VehicleDetectorConfig,
        *,
        session: _InferenceSession | None = None,
    ) -> None:
        super().__init__(config, session=session)
        self._allowed_classes = set(config.classes)
        self._configured_class_names = (
            tuple(config.model_classes) if config.model_classes is not None else None
        )

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        class_names = self._configured_class_names or _COCO_CLASSES
        rows = self._predict_rows(image, len(class_names))
        detections: list[Detection] = []
        for row in rows:
            class_id = int(row[5])
            if class_id < 0 or class_id >= len(class_names):
                raise InferenceError(
                    f"ONNX class id {class_id} has no configured model_classes entry"
                )
            class_name = class_names[class_id]
            bbox = self._bbox(row)
            if bbox is None or class_name not in self._allowed_classes:
                continue
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=float(row[4]),
                    class_id=class_id,
                    class_name=class_name,
                    model=self._metadata,
                )
            )
        return detections


class OnnxRuntimePlateDetector(_OnnxRuntimeAdapter):
    def detect(self, image: NDArray[np.uint8]) -> list[PlateDetection]:
        rows = self._predict_rows(image, 1)
        detections: list[PlateDetection] = []
        for row in rows:
            bbox = self._bbox(row)
            if bbox is not None:
                detections.append(
                    PlateDetection(
                        bbox=bbox,
                        confidence=float(row[4]),
                        model=self._metadata,
                    )
                )
        return detections
