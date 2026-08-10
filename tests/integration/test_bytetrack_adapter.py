import pytest

pytest.importorskip("supervision")

import numpy as np

from vehicle_intelligence.config import TrackingConfig
from vehicle_intelligence.domain import BoundingBox, Detection, ModelMetadata
from vehicle_intelligence.infrastructure.vision.bytetrack import ByteTrackVehicleTracker


def test_real_bytetrack_adapter_keeps_id_across_frames() -> None:
    tracker = ByteTrackVehicleTracker(
        TrackingConfig(
            activation_threshold=0.25,
            minimum_consecutive_frames=1,
            minimum_matching_threshold=0.8,
        ),
        frame_rate=10,
    )
    model = ModelMetadata("detector", "1")
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    track_ids: list[int] = []
    for offset in (0, 3, 6):
        detections = [
            Detection(
                BoundingBox(20 + offset, 30, 140 + offset, 130),
                0.9,
                2,
                "car",
                model,
            )
        ]
        tracked = tracker.update(detections, image)
        if tracked:
            track_ids.append(tracked[0].track_id)

    assert len(track_ids) >= 2
    assert len(set(track_ids)) == 1


def test_minimum_one_emits_first_observation_after_empty_frames() -> None:
    tracker = ByteTrackVehicleTracker(
        TrackingConfig(
            activation_threshold=0.25,
            minimum_consecutive_frames=1,
            minimum_matching_threshold=0.8,
        ),
        frame_rate=10,
    )
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    for _ in range(10):
        assert tracker.update([], image) == []
    detection = Detection(
        BoundingBox(30, 40, 150, 140),
        0.9,
        0,
        "car",
        ModelMetadata("detector", "1"),
    )

    tracked = tracker.update([detection], image)

    assert len(tracked) == 1
    assert tracked[0].detection.class_name == detection.class_name
    assert tracked[0].detection.confidence == detection.confidence
    observed_box = tracked[0].detection.bbox
    assert abs(observed_box.x1 - detection.bbox.x1) <= 1
    assert abs(observed_box.y1 - detection.bbox.y1) <= 1
    assert abs(observed_box.x2 - detection.bbox.x2) <= 1
    assert abs(observed_box.y2 - detection.bbox.y2) <= 1


@pytest.mark.parametrize("provider", ("yolo", "picodet"))
def test_tracker_contract_depends_only_on_canonical_detection(provider: str) -> None:
    tracker = ByteTrackVehicleTracker(
        TrackingConfig(
            activation_threshold=0.25,
            minimum_consecutive_frames=1,
            minimum_matching_threshold=0.8,
        ),
        frame_rate=10,
    )
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    model = ModelMetadata(f"{provider}-vehicle", "1")
    observed = []
    for offset in (0, 2, 4):
        detection = Detection(
            BoundingBox(30 + offset, 40, 150 + offset, 140),
            0.9,
            0,
            "car",
            model,
        )
        observed.extend(tracker.update([detection], image))

    assert observed
    assert all(item.detection.model.name == f"{provider}-vehicle" for item in observed)
