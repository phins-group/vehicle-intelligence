import numpy as np

from vehicle_intelligence.application.quality import PlateQualityEvaluator
from vehicle_intelligence.config import PlateQualityConfig
from vehicle_intelligence.domain import BoundingBox, ModelMetadata, PlateDetection


def detection(width: int, height: int) -> PlateDetection:
    return PlateDetection(
        bbox=BoundingBox(0, 0, width, height),
        confidence=0.95,
        model=ModelMetadata("plate", "1"),
    )


def test_accepts_sharp_well_exposed_plate() -> None:
    image = np.indices((40, 120)).sum(axis=0) % 2 * 255
    image = np.repeat(image[..., None], 3, axis=2).astype(np.uint8)

    quality = PlateQualityEvaluator(PlateQualityConfig()).evaluate(image, detection(120, 40))

    assert quality.eligible
    assert quality.sharpness > 0.9
    assert quality.contrast > 0.9


def test_rejects_tiny_flat_plate_before_ocr() -> None:
    image = np.full((8, 20, 3), 127, dtype=np.uint8)

    quality = PlateQualityEvaluator(PlateQualityConfig()).evaluate(image, detection(20, 8))

    assert not quality.eligible
    assert quality.sharpness == 0
