"""Human OCR review and deterministic retraining-feedback orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.ports import (
    DatasetSamplePage,
    DatasetSampleQuery,
    DatasetSampleRepository,
    MediaObjectInspector,
    VehicleEventRepository,
)
from vehicle_intelligence.domain import (
    DatasetSample,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    EventStatus,
    OCRDatasetPrediction,
    PlateReview,
    Principal,
    VehicleEvent,
)
from vehicle_intelligence.exceptions import (
    PlateReviewConflictError,
    PlateReviewValidationError,
    VehicleEventNotFoundError,
)


@dataclass(frozen=True, slots=True)
class PlateReviewCommand:
    text: str
    expected_revision: int
    reviewer: Principal
    note: str | None = None


@dataclass(frozen=True, slots=True)
class PlateReviewResult:
    before: VehicleEvent
    event: VehicleEvent
    dataset_sample_id: str | None
    reason: DatasetSampleReason
    changed: bool


class HumanPlateReviewService:
    def __init__(
        self,
        events: VehicleEventRepository,
        samples: DatasetSampleRepository,
        normalizer: VietnamPlateNormalizer,
        media: MediaObjectInspector | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._events = events
        self._samples = samples
        self._normalizer = normalizer
        self._media = media
        self._clock = clock

    async def initialize(self) -> None:
        await self._samples.ensure_indexes()

    async def close(self) -> None:
        await self._samples.close()

    async def review(
        self,
        event_id: str,
        command: PlateReviewCommand,
    ) -> PlateReviewResult:
        if command.expected_revision < 0:
            raise PlateReviewValidationError("expected review revision cannot be negative")
        normalized = self._normalizer.normalize(command.text)
        if not normalized.valid or normalized.normalized is None:
            raise PlateReviewValidationError("invalid Vietnamese plate format")
        note = command.note.strip() if command.note is not None else None
        note = note or None
        if note is not None and len(note) > 500:
            raise PlateReviewValidationError("plate review note is too long")

        current = await self._required_event(event_id)
        if current.plate is None:
            raise PlateReviewValidationError("event has no OCR plate prediction to review")

        if current.plate.review_revision != command.expected_revision:
            return await self._resolve_retry(current, normalized.normalized, command, note)

        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("human review clock must be timezone-aware")
        review = PlateReview(
            normalized=normalized.normalized,
            revision=command.expected_revision + 1,
            reviewed_at=now.astimezone(UTC),
            reviewed_by=command.reviewer.id,
            reviewer_display_name=command.reviewer.display_name,
            note=note,
        )
        updated = replace(
            current,
            schema_version=max(2, current.schema_version),
            status=EventStatus.CONFIRMED,
            plate=replace(current.plate, review=review),
        )
        persisted = await self._events.update_plate_review(updated, command.expected_revision)
        if persisted is None:
            latest = await self._required_event(event_id)
            return await self._resolve_retry(latest, normalized.normalized, command, note)
        sample_id, reason = await self._persist_sample(persisted)
        return PlateReviewResult(
            before=current,
            event=persisted,
            dataset_sample_id=sample_id,
            reason=reason,
            changed=True,
        )

    async def list_samples(self, query: DatasetSampleQuery) -> DatasetSamplePage:
        return await self._samples.list(query)

    async def _required_event(self, event_id: str) -> VehicleEvent:
        event = await self._events.get(event_id)
        if event is None:
            raise VehicleEventNotFoundError(f"vehicle event not found: {event_id}")
        return event

    async def _resolve_retry(
        self,
        current: VehicleEvent,
        normalized: str,
        command: PlateReviewCommand,
        note: str | None,
    ) -> PlateReviewResult:
        review = current.plate.review if current.plate is not None else None
        is_idempotent_retry = (
            review is not None
            and review.revision == command.expected_revision + 1
            and review.normalized == normalized
            and review.reviewed_by == command.reviewer.id
            and review.note == note
        )
        if not is_idempotent_retry:
            actual = current.plate.review_revision if current.plate is not None else 0
            raise PlateReviewConflictError(
                "plate review revision conflict: "
                f"expected {command.expected_revision}, actual {actual}"
            )
        sample_id, reason = await self._persist_sample(current)
        return PlateReviewResult(
            before=current,
            event=current,
            dataset_sample_id=sample_id,
            reason=reason,
            changed=False,
        )

    async def _persist_sample(
        self,
        event: VehicleEvent,
    ) -> tuple[str | None, DatasetSampleReason]:
        if event.plate is None or event.plate.review is None:
            raise PlateReviewValidationError("reviewed event is missing plate review evidence")
        reason = (
            DatasetSampleReason.HUMAN_CONFIRMATION
            if event.plate.review.normalized == event.plate.normalized
            else DatasetSampleReason.HUMAN_CORRECTION
        )
        if event.media.plate_crop_key is None:
            return None, reason
        if self._media is not None and not await self._media.exists(event.media.plate_crop_key):
            return None, reason
        sample_id = _dataset_sample_id(event.id, event.plate.review.revision)
        sample = DatasetSample(
            id=sample_id,
            sample_type=DatasetSampleType.PLATE_OCR,
            status=DatasetSampleStatus.READY,
            source_event_id=event.id,
            image_key=event.media.plate_crop_key,
            prediction=OCRDatasetPrediction(
                raw=event.plate.raw,
                normalized=event.plate.normalized,
                confidence=event.plate.confidence,
                model=event.ai.ocr,
            ),
            label=event.plate.review.normalized,
            reason=reason,
            review_revision=event.plate.review.revision,
            reviewed_by=event.plate.review.reviewed_by,
            reviewer_display_name=event.plate.review.reviewer_display_name,
            reviewed_at=event.plate.review.reviewed_at,
            created_at=event.plate.review.reviewed_at,
        )
        await self._samples.create(sample)
        return sample_id, reason


def _dataset_sample_id(event_id: str, review_revision: int) -> str:
    digest = hashlib.sha256(f"{event_id}:{review_revision}".encode()).hexdigest()[:24]
    return f"dss_{digest}"
