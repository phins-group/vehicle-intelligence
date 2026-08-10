import copy
from datetime import UTC, datetime

import numpy as np

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.finalization import VehicleEventFinalizer
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.ports import EventQuery
from vehicle_intelligence.application.voting import PlateCandidateAggregator
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.domain import (
    BoundingBox,
    Detection,
    Direction,
    ImageCandidate,
    ModelMetadata,
    PlateObservation,
    TrackedDetection,
    VehicleTrack,
)
from vehicle_intelligence.infrastructure.messaging.direct import RepositoryEventPublisher
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)


class MemoryMediaStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        assert content_type == "image/jpeg"
        self.objects[key] = data
        return key


class TestEncoder:
    def encode_jpeg(self, image) -> bytes:
        return b"jpeg:" + bytes([int(image.mean())])


async def test_finalizer_emits_once_and_keeps_unknown_plate_event() -> None:
    settings = load_settings()
    repository = InMemoryVehicleEventRepository()
    media = MemoryMediaStorage()
    direction = DirectionEstimator(None, Direction.ENTER, "ENTRY")
    finalizer = VehicleEventFinalizer(
        camera=settings.camera,
        events=settings.events,
        storage_config=settings.storage,
        aggregator=PlateCandidateAggregator(settings.voting, VietnamPlateNormalizer()),
        direction_estimator=direction,
        media_storage=media,
        image_encoder=TestEncoder(),
        publisher=RepositoryEventPublisher(repository),
    )
    timestamp = datetime(2026, 8, 8, tzinfo=UTC)
    track = VehicleTrack(
        camera_id=settings.camera.id,
        session_id="unit-test",
        local_track_id=7,
        first_seen=timestamp,
        last_seen=timestamp,
        max_trajectory_points=16,
        max_plate_observations=8,
    )
    model = ModelMetadata("vehicle", "1")
    tracked = TrackedDetection(
        7,
        Detection(BoundingBox(0, 0, 100, 60), 0.9, 2, "car", model),
    )
    track.update(tracked, 1, timestamp)
    track.best_snapshot = ImageCandidate(1, timestamp, 0.8, np.full((20, 20, 3), 80, np.uint8))
    same_logical_track = copy.deepcopy(track)

    event = await finalizer.finalize(track)
    duplicate = await finalizer.finalize(track)
    repository_duplicate = await finalizer.finalize(same_logical_track)
    page = await repository.list(EventQuery())

    assert event is not None
    assert event.plate is None
    assert event.status.value == "NO_PLATE"
    assert event.direction is Direction.ENTER
    assert duplicate is None
    assert repository_duplicate is None
    assert len(page.items) == 1
    assert len(media.objects) == 1


async def test_finalizer_routes_partial_plate_to_human_review() -> None:
    settings = load_settings()
    repository = InMemoryVehicleEventRepository()
    direction = DirectionEstimator(None, Direction.ENTER, "ENTRY")
    normalizer = VietnamPlateNormalizer(allow_partial=True)
    finalizer = VehicleEventFinalizer(
        camera=settings.camera,
        events=settings.events,
        storage_config=settings.storage,
        aggregator=PlateCandidateAggregator(settings.voting, normalizer),
        direction_estimator=direction,
        media_storage=MemoryMediaStorage(),
        image_encoder=TestEncoder(),
        publisher=RepositoryEventPublisher(repository),
    )
    timestamp = datetime(2026, 8, 8, tzinfo=UTC)
    track = VehicleTrack(
        camera_id=settings.camera.id,
        session_id="partial-test",
        local_track_id=8,
        first_seen=timestamp,
        last_seen=timestamp,
        max_trajectory_points=16,
        max_plate_observations=8,
    )
    model = ModelMetadata("test", "1")
    track.update(
        TrackedDetection(
            8,
            Detection(BoundingBox(0, 0, 100, 60), 0.9, 2, "motorcycle", model),
        ),
        1,
        timestamp,
    )
    normalized = normalizer.normalize("006.05")
    track.plate_detections_seen = 1
    track.add_plate_observation(
        PlateObservation(
            frame_id=1,
            timestamp=timestamp,
            raw_text="006.05",
            normalized_text=normalized.normalized,
            compact_text=normalized.compact,
            ocr_confidence=0.99,
            detection_confidence=0.65,
            quality_score=0.72,
            corrections=(),
            plate_model=model,
            ocr_model=model,
            partial=True,
        )
    )

    event = await finalizer.finalize(track)

    assert event is not None
    assert event.plate is not None
    assert event.plate.normalized == "006.05"
    assert event.plate.partial
    assert event.status.value == "NEEDS_REVIEW"
