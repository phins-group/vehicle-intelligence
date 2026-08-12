"""Ultralytics detector adapters behind domain ports."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig
from vehicle_intelligence.domain import (
    BoundingBox,
    Detection,
    ModelMetadata,
    PlateDetection,
    Point,
)
from vehicle_intelligence.exceptions import (
    DependencyUnavailableError,
    InferenceError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)


class _UltralyticsAdapter:
    def __init__(self, config: DetectorConfig, *, model: Any | None = None) -> None:
        if not config.model_path:
            raise ModelLoadError(
                f"model_path is required for {config.model_name}; supply a trained checkpoint"
            )
        self._config = config
        self._metadata = ModelMetadata(
            name=config.model_name,
            version=config.model_version,
            hash=config.model_hash,
        )
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise DependencyUnavailableError(
                    "Ultralytics is not installed; install the 'vision' extra"
                ) from exc
            try:
                model = YOLO(config.model_path)
            except Exception as exc:
                logger.exception(
                    "detector_load_failed",
                    extra={
                        "provider": "ultralytics",
                        "model_name": config.model_name,
                        "model_path": config.model_path,
                        "device": config.device or "auto",
                    },
                )
                raise ModelLoadError(
                    f"cannot load Ultralytics model '{config.model_path}' for {config.model_name}"
                ) from exc
        self._model = model
        logger.info(
            "detector_loaded",
            extra={
                "provider": "ultralytics",
                "model_name": config.model_name,
                "model_path": config.model_path,
                "device": config.device or "auto",
            },
        )

    def _predict(self, image: NDArray[np.uint8]) -> Any:
        results = self._predict_many([image])
        return results[0] if results else None

    def _predict_many(self, images: Sequence[NDArray[np.uint8]]) -> list[Any]:
        if not images:
            return []
        if any(image.ndim != 3 or image.shape[2] != 3 or image.size == 0 for image in images):
            raise InferenceError("Ultralytics detector input must be a non-empty BGR image")
        kwargs: dict[str, object] = {
            "source": list(images),
            "conf": self._config.confidence,
            "iou": self._config.iou,
            "imgsz": self._config.image_size,
            "verbose": False,
        }
        if self._config.device:
            kwargs["device"] = self._config.device
        try:
            results = self._model.predict(**kwargs)
        except Exception as exc:
            logger.exception(
                "detector_inference_failed",
                extra={
                    "provider": "ultralytics",
                    "model_name": self._config.model_name,
                    "device": self._config.device or "auto",
                },
            )
            raise InferenceError(f"{self._config.model_name} inference failed") from exc
        return list(results)

    @staticmethod
    def _name(names: object, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _bbox(values: Iterable[float], image: NDArray[np.uint8]) -> BoundingBox | None:
        x1, y1, x2, y2 = (float(value) for value in values)
        height, width = image.shape[:2]
        left = min(max(math.floor(x1), 0), width)
        top = min(max(math.floor(y1), 0), height)
        right = min(max(math.ceil(x2), 0), width)
        bottom = min(max(math.ceil(y2), 0), height)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(left, top, right, bottom)

    @staticmethod
    def _confidence(value: object) -> float:
        confidence = float(value)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise InferenceError("Ultralytics detector returned an invalid confidence")
        return confidence


class UltralyticsVehicleDetector(_UltralyticsAdapter):
    def __init__(self, config: VehicleDetectorConfig, *, model: Any | None = None) -> None:
        super().__init__(config, model=model)
        self._allowed_classes = set(config.classes)
        self._configured_class_names = (
            tuple(config.model_classes) if config.model_classes is not None else None
        )

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        return self._detections(self._predict(image), image)

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        results = self._predict_many(images)
        if len(results) != len(images):
            raise InferenceError("Ultralytics result count does not match input count")
        return [
            self._detections(result, image) for result, image in zip(results, images, strict=True)
        ]

    def _detections(self, result: Any, image: NDArray[np.uint8]) -> list[Detection]:
        if result is None or result.boxes is None:
            return []
        coordinates = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        detections: list[Detection] = []
        for values, confidence, class_id in zip(coordinates, confidences, classes, strict=True):
            raw_class_id = int(class_id)
            class_name = self._class_name(result.names, raw_class_id)
            score = self._confidence(confidence)
            bbox = self._bbox(values, image)
            if (
                bbox is None
                or score < self._config.confidence
                or class_name not in self._allowed_classes
            ):
                continue
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=score,
                    class_id=raw_class_id,
                    class_name=class_name,
                    model=self._metadata,
                )
            )
        return detections

    def _class_name(self, names: object, class_id: int) -> str:
        if self._configured_class_names is not None:
            if class_id < 0 or class_id >= len(self._configured_class_names):
                raise InferenceError(
                    f"Ultralytics class id {class_id} has no configured model_classes entry"
                )
            return self._configured_class_names[class_id]
        return self._name(names, class_id).lower()


class UltralyticsPlateDetector(_UltralyticsAdapter):
    def detect(self, image: NDArray[np.uint8]) -> list[PlateDetection]:
        return self._detections(self._predict(image), image)

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[PlateDetection]]:
        results = self._predict_many(images)
        if len(results) != len(images):
            raise InferenceError("Ultralytics result count does not match input count")
        return [
            self._detections(result, image) for result, image in zip(results, images, strict=True)
        ]

    def _detections(self, result: Any, image: NDArray[np.uint8]) -> list[PlateDetection]:
        if result is None:
            return []
        if getattr(result, "obb", None) is not None:
            return self._oriented_detections(result.obb, image)
        if result.boxes is None:
            return []
        coordinates = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        detections: list[PlateDetection] = []
        for values, confidence in zip(coordinates, confidences, strict=True):
            score = self._confidence(confidence)
            bbox = self._bbox(values, image)
            if bbox is not None and score >= self._config.confidence:
                detections.append(PlateDetection(bbox=bbox, confidence=score, model=self._metadata))
        return detections

    def _oriented_detections(self, boxes: Any, image: NDArray[np.uint8]) -> list[PlateDetection]:
        polygons = boxes.xyxyxyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        detections: list[PlateDetection] = []
        for polygon, confidence in zip(polygons, confidences, strict=True):
            score = self._confidence(confidence)
            if score < self._config.confidence:
                continue
            xs = polygon[:, 0]
            ys = polygon[:, 1]
            bbox = self._bbox((xs.min(), ys.min(), xs.max(), ys.max()), image)
            if bbox is None:
                continue
            height, width = image.shape[:2]
            points = tuple(
                Point(
                    min(max(float(x), 0), width),
                    min(max(float(y), 0), height),
                )
                for x, y in polygon
            )
            if len(points) != 4:
                continue
            corners = (points[0], points[1], points[2], points[3])
            detections.append(
                PlateDetection(
                    bbox=bbox,
                    confidence=score,
                    model=self._metadata,
                    corners=corners,
                )
            )
        return detections


YoloDetector = UltralyticsVehicleDetector
YoloPlateDetector = UltralyticsPlateDetector
