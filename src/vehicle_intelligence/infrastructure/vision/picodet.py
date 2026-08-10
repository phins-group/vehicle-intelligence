"""PicoDet ONNX provider adapters behind detector application ports."""

from __future__ import annotations

import logging
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
from vehicle_intelligence.infrastructure.vision.onnx_runtime import (
    requested_execution_providers,
    select_execution_providers,
)
from vehicle_intelligence.infrastructure.vision.postprocessing import nms_indices

logger = logging.getLogger(__name__)


class _SessionInput(Protocol):
    name: str
    type: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_SessionInput]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(self, output_names: object, input_feed: dict[str, NDArray[Any]]) -> list[Any]: ...


class _PicoDetOnnxAdapter:
    """Load once and convert PicoDet runtime outputs to canonical `[x1,y1,x2,y2]` rows."""

    def __init__(
        self,
        config: DetectorConfig,
        class_names: Sequence[str],
        *,
        session: _InferenceSession | None = None,
    ) -> None:
        if not class_names:
            raise ModelLoadError("PicoDet requires an explicit non-empty class mapping")
        self._config = config
        self._class_names = tuple(name.strip().lower() for name in class_names)
        if any(not name for name in self._class_names):
            raise ModelLoadError("PicoDet class mapping cannot contain empty names")
        path, artifact_hash = validated_model_artifact(config.model_path, config.model_hash)
        if path.suffix.lower() != ".onnx":
            raise ModelLoadError("PicoDet provider requires a .onnx model artifact")
        self._metadata = ModelMetadata(
            name=config.model_name,
            version=config.model_version,
            hash=artifact_hash,
        )
        self._session = session or self._create_session(path)
        inputs = tuple(self._session.get_inputs())
        if not inputs:
            raise ModelLoadError("PicoDet ONNX model exposes no inputs")
        input_names = tuple(item.name for item in inputs)
        if "image" in input_names:
            self._image_input_name = "image"
        elif "x" in input_names:
            self._image_input_name = "x"
        elif len(inputs) == 1:
            self._image_input_name = inputs[0].name
        else:
            raise ModelLoadError("PicoDet ONNX inputs must include 'image' or 'x'")
        supported = {self._image_input_name, "scale_factor", "im_shape"}
        unsupported = sorted(set(input_names) - supported)
        if unsupported:
            raise ModelLoadError(f"unsupported PicoDet ONNX inputs: {unsupported}")
        image_input = next(item for item in inputs if item.name == self._image_input_name)
        self._input_dtype = np.float16 if image_input.type == "tensor(float16)" else np.float32
        self._input_names = input_names
        logger.info(
            "detector_loaded",
            extra={
                "provider": "picodet",
                "model_name": config.model_name,
                "model_path": str(path),
                "device": config.device or "cpu",
            },
        )

    @property
    def execution_providers(self) -> tuple[str, ...]:
        return tuple(self._session.get_providers())

    def _create_session(self, path: object) -> _InferenceSession:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyUnavailableError(
                "ONNX Runtime is required by the PicoDet provider; install the optimization extra"
            ) from exc
        providers = select_execution_providers(
            requested_execution_providers(self._config),
            ort.get_available_providers(),
        )
        try:
            return ort.InferenceSession(str(path), providers=list(providers))
        except Exception as exc:
            logger.exception(
                "detector_load_failed",
                extra={
                    "provider": "picodet",
                    "model_name": self._config.model_name,
                    "model_path": str(path),
                    "device": self._config.device or "cpu",
                },
            )
            raise ModelLoadError(f"cannot load PicoDet ONNX model: {path}") from exc

    def _predict_rows(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        feed, scale_y, scale_x = self._preprocess(image)
        try:
            outputs = self._session.run(None, feed)
        except Exception as exc:
            logger.exception(
                "detector_inference_failed",
                extra={
                    "provider": "picodet",
                    "model_name": self._config.model_name,
                    "device": self._config.device or "cpu",
                },
            )
            raise InferenceError(f"{self._config.model_name} PicoDet inference failed") from exc
        if not outputs:
            raise InferenceError("PicoDet produced no output")
        output_format = self._config.onnx_output_format
        if output_format == "nms" or (
            output_format == "auto" and self._looks_postprocessed(outputs)
        ):
            rows = self._decode_postprocessed(outputs, image, scale_y, scale_x)
        else:
            rows = self._decode_raw(outputs, image, scale_y, scale_x)
        return rows

    def _preprocess(self, image: NDArray[np.uint8]) -> tuple[dict[str, NDArray[Any]], float, float]:
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise InferenceError("PicoDet input must be a non-empty BGR image")
        height, width = image.shape[:2]
        target = self._config.image_size
        resized = cv2.resize(image, (target, target), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        options = self._config.picodet
        rgb *= np.float32(options.scale)
        rgb -= np.asarray(options.mean, dtype=np.float32)
        rgb /= np.asarray(options.std, dtype=np.float32)
        tensor = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[None],
            dtype=self._input_dtype,
        )
        scale_y = target / height
        scale_x = target / width
        values: dict[str, NDArray[Any]] = {
            self._image_input_name: tensor,
            "scale_factor": np.asarray([[scale_y, scale_x]], dtype=np.float32),
            "im_shape": np.asarray([[target, target]], dtype=np.float32),
        }
        return {name: values[name] for name in self._input_names}, scale_y, scale_x

    def _looks_postprocessed(self, outputs: Sequence[Any]) -> bool:
        expected_raw_outputs = len(self._config.picodet.strides) * 2
        if len(outputs) >= expected_raw_outputs:
            return False
        rows = np.asarray(outputs[0])
        while rows.ndim > 2 and rows.shape[0] == 1:
            rows = rows[0]
        return rows.ndim == 2 and rows.shape[1] >= 6

    def _decode_postprocessed(
        self,
        outputs: Sequence[Any],
        image: NDArray[np.uint8],
        scale_y: float,
        scale_x: float,
    ) -> NDArray[np.float32]:
        predictions = np.asarray(outputs[0], dtype=np.float32)
        while predictions.ndim > 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            raise InferenceError(
                f"postprocessed PicoDet output must be [N,6], got {predictions.shape}"
            )
        if len(outputs) > 1:
            box_counts = np.asarray(outputs[1]).reshape(-1)
            if box_counts.size == 1 and np.issubdtype(box_counts.dtype, np.integer):
                count = int(box_counts[0])
                if count < 0 or count > predictions.shape[0]:
                    raise InferenceError("PicoDet bbox count is outside the output row range")
                predictions = predictions[:count]
        # PaddleDetection rows are [class_id, score, x1, y1, x2, y2].
        rows = np.empty((predictions.shape[0], 6), dtype=np.float32)
        rows[:, :4] = predictions[:, 2:6]
        rows[:, 4] = predictions[:, 1]
        rows[:, 5] = predictions[:, 0]
        if "scale_factor" not in self._input_names:
            rows[:, [0, 2]] /= scale_x
            rows[:, [1, 3]] /= scale_y
        return self._filter_and_clip(rows, image)

    def _decode_raw(
        self,
        outputs: Sequence[Any],
        image: NDArray[np.uint8],
        scale_y: float,
        scale_x: float,
    ) -> NDArray[np.float32]:
        strides = self._config.picodet.strides
        if len(outputs) != len(strides) * 2:
            raise InferenceError(
                "raw PicoDet output count must equal twice the configured stride count"
            )
        score_outputs = outputs[: len(strides)]
        box_outputs = outputs[len(strides) :]
        decoded_boxes: list[NDArray[np.float32]] = []
        decoded_scores: list[NDArray[np.float32]] = []
        target = self._config.image_size
        for stride, score_output, box_output in zip(
            strides, score_outputs, box_outputs, strict=True
        ):
            scores = self._single_batch_output(score_output, "scores")
            distributions = self._single_batch_output(box_output, "box distributions")
            if scores.shape[0] != distributions.shape[0]:
                raise InferenceError("PicoDet score/box location counts differ")
            if scores.shape[1] != len(self._class_names):
                raise InferenceError(
                    "PicoDet score class count does not match configured model_classes"
                )
            expected_locations = math.ceil(target / stride) ** 2
            if scores.shape[0] != expected_locations:
                raise InferenceError(
                    f"PicoDet stride {stride} expected {expected_locations} locations, "
                    f"got {scores.shape[0]}"
                )
            if distributions.shape[1] < 8 or distributions.shape[1] % 4:
                raise InferenceError("PicoDet box distribution width must be divisible by four")
            if not np.all(np.isfinite(scores)):
                raise InferenceError("PicoDet scores contain non-finite values")
            if not np.all(np.isfinite(distributions)):
                raise InferenceError("PicoDet box distributions contain non-finite values")
            if np.any(scores < -1e-5) or np.any(scores > 1 + 1e-5):
                raise InferenceError("PicoDet scores must be post-sigmoid probabilities")
            scores = np.clip(scores, 0, 1)
            level_boxes = self._decode_level(distributions, target, stride)
            top_k = min(self._config.picodet.nms_top_k, scores.shape[0])
            order = np.argsort(-scores.max(axis=1), kind="stable")[:top_k]
            decoded_boxes.append(level_boxes[order])
            decoded_scores.append(scores[order])
        boxes = np.concatenate(decoded_boxes, axis=0)
        scores = np.concatenate(decoded_scores, axis=0)
        rows: list[NDArray[np.float32]] = []
        for class_id in range(scores.shape[1]):
            class_scores = scores[:, class_id]
            mask = class_scores >= self._config.confidence
            if not np.any(mask):
                continue
            class_rows = np.empty((int(mask.sum()), 6), dtype=np.float32)
            class_rows[:, :4] = boxes[mask]
            class_rows[:, 4] = class_scores[mask]
            class_rows[:, 5] = class_id
            rows.append(class_rows)
        if not rows:
            return np.empty((0, 6), dtype=np.float32)
        combined = np.concatenate(rows, axis=0)
        keep = nms_indices(
            combined[:, :4],
            combined[:, 4],
            combined[:, 5].astype(np.int64),
            self._config.iou,
            class_agnostic=False,
        )
        selected = combined[keep]
        order = np.argsort(-selected[:, 4], kind="stable")
        selected = selected[order[: self._config.picodet.keep_top_k]].copy()
        selected[:, [0, 2]] /= scale_x
        selected[:, [1, 3]] /= scale_y
        return self._filter_and_clip(selected, image)

    @staticmethod
    def _single_batch_output(output: Any, label: str) -> NDArray[np.float32]:
        values = np.asarray(output, dtype=np.float32)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2:
            raise InferenceError(f"PicoDet {label} output must have one batch")
        return values

    @staticmethod
    def _decode_level(
        distributions: NDArray[np.float32], target: int, stride: int
    ) -> NDArray[np.float32]:
        reg_max = distributions.shape[1] // 4 - 1
        logits = distributions.reshape(-1, reg_max + 1)
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        distances = (probabilities @ np.arange(reg_max + 1, dtype=np.float32)).reshape(-1, 4)
        distances *= stride
        feature_size = math.ceil(target / stride)
        rows, columns = np.meshgrid(
            np.arange(feature_size, dtype=np.float32),
            np.arange(feature_size, dtype=np.float32),
            indexing="ij",
        )
        centers_x = (columns.reshape(-1) + 0.5) * stride
        centers_y = (rows.reshape(-1) + 0.5) * stride
        centers = np.stack((centers_x, centers_y, centers_x, centers_y), axis=1)
        return centers + np.asarray((-1, -1, 1, 1), dtype=np.float32) * distances

    def _filter_and_clip(
        self, rows: NDArray[np.float32], image: NDArray[np.uint8]
    ) -> NDArray[np.float32]:
        if rows.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        if not np.all(np.isfinite(rows)):
            raise InferenceError("PicoDet output contains non-finite values")
        scores = rows[:, 4]
        if np.any(scores < -1e-5) or np.any(scores > 1 + 1e-5):
            raise InferenceError("PicoDet returned an invalid confidence")
        filtered = rows[scores >= self._config.confidence].copy()
        if filtered.size == 0:
            return np.empty((0, 6), dtype=np.float32)
        filtered[:, 4] = np.clip(filtered[:, 4], 0, 1)
        class_ids = filtered[:, 5]
        if np.any(class_ids != np.floor(class_ids)):
            raise InferenceError("PicoDet class ids must be integers")
        if np.any(class_ids < 0) or np.any(class_ids >= len(self._class_names)):
            raise InferenceError("PicoDet class id has no configured model_classes entry")
        height, width = image.shape[:2]
        filtered[:, [0, 2]] = np.clip(filtered[:, [0, 2]], 0, width)
        filtered[:, [1, 3]] = np.clip(filtered[:, [1, 3]], 0, height)
        valid = (filtered[:, 2] > filtered[:, 0]) & (filtered[:, 3] > filtered[:, 1])
        return filtered[valid]

    @staticmethod
    def _bbox(row: NDArray[np.float32]) -> BoundingBox | None:
        left, top = math.floor(float(row[0])), math.floor(float(row[1]))
        right, bottom = math.ceil(float(row[2])), math.ceil(float(row[3]))
        if right <= left or bottom <= top:
            return None
        return BoundingBox(left, top, right, bottom)


class PicoDetDetector(_PicoDetOnnxAdapter):
    """Vehicle/object PicoDet provider returning canonical `Detection` values."""

    def __init__(
        self,
        config: VehicleDetectorConfig,
        *,
        session: _InferenceSession | None = None,
    ) -> None:
        if config.model_classes is None:
            raise ModelLoadError("PicoDet vehicle detector requires model_classes")
        super().__init__(config, config.model_classes, session=session)
        self._allowed_classes = set(config.classes)

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        detections: list[Detection] = []
        for row in self._predict_rows(image):
            class_id = int(row[5])
            class_name = self._class_names[class_id]
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


class PicoDetPlateDetector(_PicoDetOnnxAdapter):
    """Plate PicoDet provider returning the existing canonical plate model."""

    def __init__(
        self,
        config: DetectorConfig,
        *,
        session: _InferenceSession | None = None,
    ) -> None:
        class_names = config.model_classes or ("license_plate",)
        super().__init__(config, class_names, session=session)

    def detect(self, image: NDArray[np.uint8]) -> list[PlateDetection]:
        detections: list[PlateDetection] = []
        for row in self._predict_rows(image):
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


PicoDetVehicleDetector = PicoDetDetector
