"""Best-frame scoring strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.application.quality import sharpness_variance
from vehicle_intelligence.config import SnapshotSelectionConfig
from vehicle_intelligence.domain import BoundingBox, PlateQuality


class BestFrameSelector:
    def __init__(self, config: SnapshotSelectionConfig) -> None:
        self._config = config

    def vehicle_score(
        self,
        crop: NDArray[np.uint8],
        bbox: BoundingBox,
        frame_width: int,
        frame_height: int,
        detector_confidence: float,
    ) -> float:
        config = self._config
        area = min(bbox.area / max(frame_width * frame_height, 1), 1.0)
        # Square root avoids making medium, useful vehicles score near zero.
        area = area**0.5
        sharpness = min(sharpness_variance(crop) / config.sharpness_reference, 1.0)
        components = (
            (area, config.vehicle_area_weight),
            (sharpness, config.sharpness_weight),
            (detector_confidence, config.detector_confidence_weight),
        )
        return self._weighted(components)

    def plate_score(
        self, quality: PlateQuality, ocr_confidence: float, detector_confidence: float
    ) -> float:
        config = self._config
        return self._weighted(
            (
                (quality.total_score, config.plate_quality_weight),
                (ocr_confidence, config.plate_ocr_weight),
                (detector_confidence, config.plate_detector_weight),
            )
        )

    @staticmethod
    def _weighted(components: tuple[tuple[float, float], ...]) -> float:
        weight_total = sum(weight for _, weight in components)
        result = sum(value * weight for value, weight in components) / weight_total
        return min(max(result, 0.0), 1.0)
