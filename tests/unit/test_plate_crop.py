from vehicle_intelligence.application.plate_crop import expanded_plate_detection
from vehicle_intelligence.config import PlateCropConfig
from vehicle_intelligence.domain import BoundingBox, ModelMetadata, PlateDetection


def test_motorcycle_plate_crop_recovers_an_upper_text_row() -> None:
    detection = PlateDetection(
        BoundingBox(50, 100, 120, 140),
        0.8,
        ModelMetadata("plate", "1"),
    )

    expanded = expanded_plate_detection(
        detection,
        image_width=200,
        image_height=180,
        vehicle_type="motorcycle",
        config=PlateCropConfig(),
    )

    assert expanded is not None
    assert expanded.bbox == BoundingBox(44, 56, 126, 144)
    assert expanded.confidence == detection.confidence


def test_non_two_line_vehicle_only_receives_regular_padding() -> None:
    detection = PlateDetection(
        BoundingBox(50, 100, 120, 140),
        0.8,
        ModelMetadata("plate", "1"),
    )

    expanded = expanded_plate_detection(
        detection,
        image_width=200,
        image_height=180,
        vehicle_type="car",
        config=PlateCropConfig(),
    )

    assert expanded is not None
    assert expanded.bbox == BoundingBox(44, 96, 126, 144)


def test_expanded_crop_is_clamped_to_image_boundaries() -> None:
    detection = PlateDetection(
        BoundingBox(1, 2, 40, 20),
        0.8,
        ModelMetadata("plate", "1"),
    )

    expanded = expanded_plate_detection(
        detection,
        image_width=100,
        image_height=80,
        vehicle_type="motorcycle",
        config=PlateCropConfig(),
    )

    assert expanded is not None
    assert expanded.bbox.x1 == 0
    assert expanded.bbox.y1 == 0
