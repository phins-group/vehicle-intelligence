"""Exactly-once track finalization and event persistence."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.finalization_outbox import (
    FinalizationMediaObject,
    FinalizationOutbox,
    MediaReferenceField,
)
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
from vehicle_intelligence.exceptions import MediaStorageError


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
        outbox: FinalizationOutbox | None = None,
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
        self._outbox = outbox
        self._pending_outbox: dict[
            str,
            tuple[VehicleEvent, tuple[FinalizationMediaObject, ...]],
        ] = {}
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
            media, media_objects = self._prepare_media(track, event_id)
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
            if self._outbox is not None:
                pending = self._pending_outbox.get(event.id)
                if pending is None:
                    self._pending_outbox[event.id] = (event, media_objects)
                else:
                    event, media_objects = pending
                return await self._stage_and_finalize(track, event, media_objects)
            await self._persist_media(media_objects)
            created = await self._publisher.publish(event)
            track.mark_finalized()
            return event if created else None

    def _prepare_media(
        self,
        track: VehicleTrack,
        event_id: str,
    ) -> tuple[MediaReferences, tuple[FinalizationMediaObject, ...]]:
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
        media_objects: list[FinalizationMediaObject] = []
        for field_name, enabled, candidate, key in entries:
            if enabled and candidate is not None:
                encoded = self._image_encoder.encode_jpeg(candidate.image)
                references[field_name] = key
                media_objects.append(
                    FinalizationMediaObject(
                        reference_field=cast(MediaReferenceField, field_name),
                        key=key,
                        data=encoded,
                    )
                )
        return MediaReferences(**references), tuple(media_objects)

    async def _persist_media(self, media: tuple[FinalizationMediaObject, ...]) -> None:
        stored = await asyncio.gather(
            *(self._media_storage.put(item.key, item.data, item.content_type) for item in media)
        )
        if any(stored_key != item.key for item, stored_key in zip(media, stored, strict=True)):
            raise MediaStorageError("media storage returned a non-deterministic object key")

    async def _stage_and_finalize(
        self,
        track: VehicleTrack,
        event: VehicleEvent,
        media: tuple[FinalizationMediaObject, ...],
    ) -> VehicleEvent:
        if self._outbox is None:
            raise AssertionError("durable stage called without an outbox")
        stage_task = asyncio.create_task(self._outbox.stage(event, media))
        caller_cancellation: asyncio.CancelledError | None = None
        while not stage_task.done():
            try:
                await asyncio.shield(stage_task)
            except asyncio.CancelledError as exc:
                caller_cancellation = exc
            except BaseException:
                break
        stage_error: BaseException | None = None
        try:
            stage_task.result()
        except BaseException as exc:
            stage_error = exc
        if stage_error is not None:
            if caller_cancellation is not None:
                raise caller_cancellation from stage_error
            raise stage_error
        track.mark_finalized()
        self._pending_outbox.pop(event.id, None)
        if caller_cancellation is not None:
            raise caller_cancellation
        return event

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
