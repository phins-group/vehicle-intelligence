"""Pure NumPy plate quality evaluation."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.config import PlateQualityConfig
from vehicle_intelligence.domain import PlateDetection, PlateQuality


def grayscale(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    if image.ndim == 2:
        return image.astype(np.float32)
    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    return (0.114 * blue + 0.587 * green + 0.299 * red).astype(np.float32)


def sharpness_variance(image: NDArray[np.uint8]) -> float:
    gray = grayscale(image)
    if min(gray.shape[:2]) < 3:
        return 0.0
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(laplacian))


class PlateQualityEvaluator:
    def __init__(self, config: PlateQualityConfig) -> None:
        self._config = config

    def evaluate(
        self,
        image: NDArray[np.uint8],
        detection: PlateDetection,
    ) -> PlateQuality:
        gray = grayscale(image)
        height, width = gray.shape[:2]
        config = self._config
        sharpness = min(sharpness_variance(image) / config.blur_reference, 1.0)
        mean_brightness = float(np.mean(gray))
        brightness = self._brightness_score(mean_brightness)
        contrast = min(float(np.std(gray)) / config.contrast_reference, 1.0)
        resolution = min(width / config.target_width, height / config.target_height, 1.0)
        angle = self._angle_score(detection, width, height)
        values = {
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "resolution": resolution,
            "angle": angle,
            "detector": detection.confidence,
        }
        weights = config.weights.model_dump()
        weight_total = sum(weights.values())
        total = sum(values[name] * weight for name, weight in weights.items()) / weight_total
        dimensions_ok = width >= config.min_width and height >= config.min_height
        return PlateQuality(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            resolution_score=resolution,
            angle_score=angle,
            detector_score=detection.confidence,
            total_score=max(0.0, min(total, 1.0)),
            eligible=dimensions_ok and total >= config.minimum,
        )

    def _brightness_score(self, value: float) -> float:
        low = self._config.brightness_min
        high = self._config.brightness_max
        middle = (low + high) / 2
        half_range = (high - low) / 2
        return max(0.0, 1.0 - abs(value - middle) / half_range)

    def _angle_score(self, detection: PlateDetection, width: int, height: int) -> float:
        if detection.corners:
            top_left, top_right, bottom_right, _ = detection.corners
            horizontal_angle = abs(
                math.atan2(top_right.y - top_left.y, top_right.x - top_left.x)
            )
            vertical_angle = abs(
                math.atan2(bottom_right.x - top_right.x, bottom_right.y - top_right.y)
            )
            return max(0.0, 1.0 - (horizontal_angle + vertical_angle) / math.pi)
        ratio = width / max(height, 1)
        config = self._config
        if ratio <= config.aspect_ratio_min or ratio >= config.aspect_ratio_max:
            return 0.0
        if ratio <= config.aspect_ratio_ideal:
            return (ratio - config.aspect_ratio_min) / (
                config.aspect_ratio_ideal - config.aspect_ratio_min
            )
        return (config.aspect_ratio_max - ratio) / (
            config.aspect_ratio_max - config.aspect_ratio_ideal
        )
