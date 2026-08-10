"""Exactly-once track finalization and event persistence."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.ports import (
    ImageEncoder,
    MediaStorage,
    VehicleEventPublisher,
)
from vehicle_intelligence.application.voting import PlateCandidateAggregator
from vehicle_intelligence.config import CameraConfig, EventConfig, StorageConfig
from vehicle_intelligence.domain import (
    AITrace,
    CameraSnapshot,
    Direction,
    EventStatus,
    EventType,
    MediaReferences,
    PlateEvidence,
    TrackStatus,
    VehicleEvent,
    VehicleEvidence,
    VehicleTrack,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def deterministic_event_id(track: VehicleTrack, event_type: EventType) -> str:
    identity = f"{track.logical_id}|{event_type.value}|{track.first_seen.isoformat()}"
    return f"evt_{uuid.uuid5(uuid.NAMESPACE_URL, identity).hex}"


class VehicleEventFinalizer:
    def __init__(
        self,
        camera: CameraConfig,
        events: EventConfig,
        storage_config: StorageConfig,
        aggregator: PlateCandidateAggregator,
        direction_estimator: DirectionEstimator,
        media_storage: MediaStorage,
        image_encoder: ImageEncoder,
        publisher: VehicleEventPublisher,
        config_version: str | None = None,
        clock: Callable[[], datetime] = utc_now,
        event_id_factory: Callable[[VehicleTrack, EventType], str] = deterministic_event_id,
    ) -> None:
        self._camera = camera
        self._events = events
        self._storage_config = storage_config
        self._aggregator = aggregator
        self._direction_estimator = direction_estimator
        self._media_storage = media_storage
        self._image_encoder = image_encoder
        self._publisher = publisher
        self._config_version = config_version
        self._clock = clock
        self._event_id_factory = event_id_factory
        self._lock = asyncio.Lock()

    async def finalize(self, track: VehicleTrack) -> VehicleEvent | None:
        async with self._lock:
            if track.status is TrackStatus.FINALIZED:
                return None
            direction = self._direction_estimator.estimate(tuple(track.trajectory))
            track.direction = direction
            event_type = self._event_type(direction)
            event_id = self._event_id_factory(track, event_type)
            candidate = self._aggregator.aggregate(tuple(track.plate_observations))
            plate = (
                PlateEvidence(
                    raw=candidate.raw_text,
                    normalized=candidate.normalized_text,
                    confidence=candidate.confidence,
                    observation_count=candidate.observation_count,
                    corrections=candidate.corrections,
                    partial=candidate.partial,
                )
                if candidate is not None
                else None
            )
            status = self._event_status(track, plate)
            media = await self._persist_media(track, event_id)
            event = VehicleEvent(
                id=event_id,
                schema_version=1,
                camera=CameraSnapshot(
                    id=self._camera.id,
                    name=self._camera.name,
                    zone=self._camera.zone,
                ),
                track_id=track.logical_id,
                event_type=event_type,
                occurred_at=track.last_seen.astimezone(UTC),
                created_at=self._clock().astimezone(UTC),
                direction=direction,
                status=status,
                vehicle=VehicleEvidence(
                    type=track.vehicle_type,
                    confidence=min(max(track.vehicle_confidence, 0.0), 1.0),
                ),
                plate=plate,
                media=media,
                ai=AITrace(
                    vehicle_detector=track.vehicle_model,
                    plate_detector=track.plate_model,
                    ocr=track.ocr_model,
                    config_version=self._config_version,
                ),
                metadata={
                    "stats": {
                        "frames": track.frames_seen,
                        "plateDetections": track.plate_detections_seen,
                        "plateObservations": len(track.plate_observations),
                    }
                },
            )
            created = await self._publisher.publish(event)
            track.mark_finalized()
            return event if created else None

    async def _persist_media(self, track: VehicleTrack, event_id: str) -> MediaReferences:
        occurred = track.last_seen.astimezone(UTC)
        prefix = f"vehicles/{occurred:%Y/%m/%d}/{self._camera.id}/{event_id}"
        entries = (
            (
                "snapshot_key",
                self._storage_config.snapshots,
                track.best_snapshot,
                f"{prefix}/snapshot.jpg",
            ),
            (
                "vehicle_crop_key",
                self._storage_config.vehicle_crops,
                track.best_vehicle_crop,
                f"{prefix}/vehicle.jpg",
            ),
            (
                "plate_crop_key",
                self._storage_config.plate_crops,
                track.best_plate_crop,
                f"{prefix}/plate.jpg",
            ),
        )
        references: dict[str, str | None] = {
            "snapshot_key": None,
            "vehicle_crop_key": None,
            "plate_crop_key": None,
        }
        pending: list[tuple[str, str, Awaitable[str]]] = []
        for field_name, enabled, candidate, key in entries:
            if enabled and candidate is not None:
                encoded = self._image_encoder.encode_jpeg(candidate.image)
                pending.append(
                    (field_name, key, self._media_storage.put(key, encoded, "image/jpeg"))
                )
        if pending:
            stored = await asyncio.gather(*(operation for _, _, operation in pending))
            for (field_name, _, _), stored_key in zip(pending, stored, strict=True):
                references[field_name] = stored_key
        return MediaReferences(**references)

    def _event_status(self, track: VehicleTrack, plate: PlateEvidence | None) -> EventStatus:
        if plate is None:
            return (
                EventStatus.NO_PLATE if track.plate_detections_seen == 0 else EventStatus.UNREADABLE
            )
        if plate.partial:
            return EventStatus.NEEDS_REVIEW
        if plate.confidence >= self._events.review_plate_confidence:
            return EventStatus.CONFIRMED
        if plate.confidence >= self._events.minimum_plate_confidence:
            return EventStatus.NEEDS_REVIEW
        return EventStatus.LOW_CONFIDENCE

    @staticmethod
    def _event_type(direction: Direction) -> EventType:
        if direction is Direction.ENTER:
            return EventType.VEHICLE_ENTER
        if direction is Direction.EXIT:
            return EventType.VEHICLE_EXIT
        return EventType.VEHICLE_DETECTED
