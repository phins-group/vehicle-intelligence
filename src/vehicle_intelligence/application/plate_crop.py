"""Config-driven plate crop expansion before quality evaluation and OCR."""

from __future__ import annotations

import math
from dataclasses import replace

from vehicle_intelligence.config import PlateCropConfig
from vehicle_intelligence.domain import BoundingBox, PlateDetection


def expanded_plate_detection(
    detection: PlateDetection,
    *,
    image_width: int,
    image_height: int,
    vehicle_type: str,
    config: PlateCropConfig,
) -> PlateDetection | None:
    """Return a clipped detection whose crop includes contextual plate rows."""

    clipped = detection.bbox.clip(image_width, image_height)
    if clipped is None:
        return None
    horizontal = math.ceil(clipped.width * config.horizontal_padding_ratio)
    vertical = math.ceil(clipped.height * config.vertical_padding_ratio)
    extra_top = 0
    if vehicle_type.strip().lower() in config.two_line_vehicle_classes:
        extra_top = math.ceil(clipped.height * config.two_line_top_expansion_ratio)
    expanded = BoundingBox(
        max(0, clipped.x1 - horizontal),
        max(0, clipped.y1 - vertical - extra_top),
        min(image_width, clipped.x2 + horizontal),
        min(image_height, clipped.y2 + vertical),
    )
    return replace(detection, bbox=expanded)
