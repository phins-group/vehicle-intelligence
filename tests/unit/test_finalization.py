import asyncio
import copy
from datetime import UTC, datetime

import numpy as np
import pytest

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.finalization import VehicleEventFinalizer
from vehicle_intelligence.application.finalization_outbox import FinalizationMediaObject
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
    VehicleEvent,
    VehicleTrack,
)
from vehicle_intelligence.exceptions import FinalizationOutboxRetryableError
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


class RecordingOutbox:
    def __init__(self) -> None:
        self.staged: list[tuple[VehicleEvent, tuple[FinalizationMediaObject, ...]]] = []

    async def initialize(self) -> None:
        return None

    async def stage(
        self,
        event: VehicleEvent,
        media: tuple[FinalizationMediaObject, ...],
    ) -> None:
        self.staged.append((event, media))

    async def close(self) -> None:
        return None


class BlockingOutbox(RecordingOutbox):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stage(
        self,
        event: VehicleEvent,
        media: tuple[FinalizationMediaObject, ...],
    ) -> None:
        self.started.set()
        await self.release.wait()
        await super().stage(event, media)


class FailOnceOutbox(RecordingOutbox):
    async def stage(
        self,
        event: VehicleEvent,
        media: tuple[FinalizationMediaObject, ...],
    ) -> None:
        await super().stage(event, media)
        if len(self.staged) == 1:
            raise FinalizationOutboxRetryableError("injected post-rename uncertainty")


def _durable_finalizer_and_track(outbox, local_track_id: int):
    settings = load_settings()
    timestamp = datetime(2026, 8, 8, tzinfo=UTC)
    track = VehicleTrack(
        camera_id=settings.camera.id,
        session_id=f"durable-stage-{local_track_id}",
        local_track_id=local_track_id,
        first_seen=timestamp,
        last_seen=timestamp,
        max_trajectory_points=16,
        max_plate_observations=8,
    )
    model = ModelMetadata("vehicle", "1")
    track.update(
        TrackedDetection(
            local_track_id,
            Detection(BoundingBox(0, 0, 100, 60), 0.9, 2, "car", model),
        ),
        1,
        timestamp,
    )
    track.best_snapshot = ImageCandidate(
        1,
        timestamp,
        0.8,
        np.full((20, 20, 3), 80, np.uint8),
    )
    finalizer = VehicleEventFinalizer(
        camera=settings.camera,
        events=settings.events,
        storage_config=settings.storage,
        aggregator=PlateCandidateAggregator(settings.voting, VietnamPlateNormalizer()),
        direction_estimator=DirectionEstimator(None, Direction.ENTER, "ENTRY"),
        media_storage=MemoryMediaStorage(),
        image_encoder=TestEncoder(),
        publisher=RepositoryEventPublisher(InMemoryVehicleEventRepository()),
        outbox=outbox,
    )
    return finalizer, track


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


async def test_finalizer_marks_track_only_after_durable_outbox_stage() -> None:
    outbox = RecordingOutbox()
    finalizer, track = _durable_finalizer_and_track(outbox, 9)

    event = await finalizer.finalize(track)

    assert event is not None
    assert track.status.value == "FINALIZED"
    assert len(outbox.staged) == 1
    staged_event, staged_media = outbox.staged[0]
    assert staged_event == event
    assert staged_media[0].key == event.media.snapshot_key


async def test_cancellation_after_stage_commit_marks_track_before_propagating() -> None:
    outbox = BlockingOutbox()
    finalizer, track = _durable_finalizer_and_track(outbox, 10)
    finalization = asyncio.create_task(finalizer.finalize(track))
    await asyncio.wait_for(outbox.started.wait(), timeout=1)

    finalization.cancel()
    await asyncio.sleep(0)
    assert not finalization.done()
    outbox.release.set()

    with pytest.raises(asyncio.CancelledError):
        await finalization
    assert track.status.value == "FINALIZED"
    assert len(outbox.staged) == 1


async def test_retry_reuses_identical_event_and_media_after_uncertain_commit() -> None:
    outbox = FailOnceOutbox()
    finalizer, track = _durable_finalizer_and_track(outbox, 11)

    with pytest.raises(FinalizationOutboxRetryableError):
        await finalizer.finalize(track)
    event = await finalizer.finalize(track)

    assert event is not None
    assert track.status.value == "FINALIZED"
    assert len(outbox.staged) == 2
    assert outbox.staged[0] == outbox.staged[1]
    assert event == outbox.staged[0][0]
