"""Use cases for human-labeling detector review queues."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from vehicle_intelligence.domain import Principal
from vehicle_intelligence.domain.dataset_review import (
    DetectorPromotionJob,
    DetectorReviewAction,
    DetectorReviewAnnotation,
    DetectorReviewDecision,
    DetectorReviewImage,
    DetectorReviewItem,
    DetectorReviewPage,
    DetectorReviewSourceSummary,
    DetectorReviewStatus,
)
from vehicle_intelligence.exceptions import DatasetReviewValidationError


@dataclass(frozen=True, slots=True)
class DetectorReviewQuery:
    source_id: str
    limit: int = 50
    cursor: str | None = None
    status: DetectorReviewStatus | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DetectorReviewCommand:
    action: DetectorReviewAction
    expected_revision: int
    reviewer: Principal
    annotations: tuple[DetectorReviewAnnotation, ...] = ()
    note: str | None = None


class DetectorReviewRepository(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def list_sources(self) -> tuple[DetectorReviewSourceSummary, ...]: ...

    async def list_items(self, query: DetectorReviewQuery) -> DetectorReviewPage: ...

    async def get_item(self, source_id: str, review_id: str) -> DetectorReviewItem: ...

    async def get_image(self, source_id: str, review_id: str) -> DetectorReviewImage: ...

    async def save_decision(
        self,
        source_id: str,
        review_id: str,
        decision: DetectorReviewDecision,
        expected_revision: int,
    ) -> DetectorReviewItem: ...

    async def decision_history(
        self,
        source_id: str,
        review_id: str,
    ) -> tuple[DetectorReviewDecision, ...]: ...

    async def create_promotion_job(
        self,
        source_id: str,
        target_source_id: str,
        requested_by: str,
    ) -> DetectorPromotionJob: ...

    async def run_promotion_job(self, job_id: str) -> None: ...

    async def get_promotion_job(self, job_id: str) -> DetectorPromotionJob: ...


class DetectorDatasetReviewService:
    def __init__(
        self,
        repository: DetectorReviewRepository,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._tasks: set[asyncio.Task[None]] = set()

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._repository.close()

    async def list_sources(self) -> tuple[DetectorReviewSourceSummary, ...]:
        return await self._repository.list_sources()

    async def list_items(self, query: DetectorReviewQuery) -> DetectorReviewPage:
        return await self._repository.list_items(query)

    async def get_item(self, source_id: str, review_id: str) -> DetectorReviewItem:
        return await self._repository.get_item(source_id, review_id)

    async def get_image(self, source_id: str, review_id: str) -> DetectorReviewImage:
        return await self._repository.get_image(source_id, review_id)

    async def review(
        self,
        source_id: str,
        review_id: str,
        command: DetectorReviewCommand,
    ) -> DetectorReviewItem:
        item = await self._repository.get_item(source_id, review_id)
        annotations = self._annotations_for(command, item)
        reviewed_at = self._clock()
        if reviewed_at.tzinfo is None:
            raise DatasetReviewValidationError("dataset review clock must be timezone-aware")
        decision = DetectorReviewDecision(
            action=command.action,
            status=_status_for_action(command.action),
            annotations=annotations,
            revision=command.expected_revision + 1,
            reviewed_by=command.reviewer.id,
            reviewer_display_name=command.reviewer.display_name,
            reviewed_at=reviewed_at.astimezone(UTC),
            note=_normalized_note(command.note),
        )
        return await self._repository.save_decision(
            source_id,
            review_id,
            decision,
            command.expected_revision,
        )

    async def history(
        self,
        source_id: str,
        review_id: str,
    ) -> tuple[DetectorReviewDecision, ...]:
        return await self._repository.decision_history(source_id, review_id)

    async def start_promotion(
        self,
        source_id: str,
        target_source_id: str,
        principal: Principal,
    ) -> DetectorPromotionJob:
        job = await self._repository.create_promotion_job(
            source_id,
            target_source_id,
            principal.id,
        )
        task = asyncio.create_task(
            self._repository.run_promotion_job(job.id),
            name=f"detector-promotion-{job.id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def get_promotion(self, job_id: str) -> DetectorPromotionJob:
        return await self._repository.get_promotion_job(job_id)

    @staticmethod
    def _annotations_for(
        command: DetectorReviewCommand,
        item: DetectorReviewItem,
    ) -> tuple[DetectorReviewAnnotation, ...]:
        if command.expected_revision < 0:
            raise DatasetReviewValidationError("expected review revision cannot be negative")
        if command.action is DetectorReviewAction.APPROVE:
            if not item.suggestions:
                raise DatasetReviewValidationError(
                    "an item without model suggestions cannot be approved"
                )
            if command.annotations:
                raise DatasetReviewValidationError(
                    "APPROVE uses the original model suggestions without replacements"
                )
            annotations = item.suggestions
        elif command.action is DetectorReviewAction.CORRECT:
            annotations = command.annotations
            if not annotations:
                raise DatasetReviewValidationError(
                    "CORRECT requires at least one license-plate bounding box"
                )
        else:
            if command.annotations:
                raise DatasetReviewValidationError(
                    "negative and rejected samples cannot contain annotations"
                )
            annotations = ()
            if command.action is DetectorReviewAction.REJECT and not _normalized_note(
                command.note
            ):
                raise DatasetReviewValidationError("REJECT requires a review note")
        if len(annotations) > 16:
            raise DatasetReviewValidationError("one review image cannot contain over 16 plates")
        if item.image_width is None or item.image_height is None:
            raise DatasetReviewValidationError("review image dimensions are unavailable")
        for annotation in annotations:
            box = annotation.bbox
            if box.x + box.width > item.image_width or box.y + box.height > item.image_height:
                raise DatasetReviewValidationError(
                    "review bounding box must remain inside the source image"
                )
        return annotations


def _status_for_action(action: DetectorReviewAction) -> DetectorReviewStatus:
    return {
        DetectorReviewAction.APPROVE: DetectorReviewStatus.APPROVED,
        DetectorReviewAction.CORRECT: DetectorReviewStatus.CORRECTED,
        DetectorReviewAction.MARK_NEGATIVE: DetectorReviewStatus.NEGATIVE,
        DetectorReviewAction.REJECT: DetectorReviewStatus.REJECTED,
    }[action]


def _normalized_note(note: str | None) -> str | None:
    if note is None:
        return None
    normalized = note.strip()
    if len(normalized) > 1000:
        raise DatasetReviewValidationError("review note cannot exceed 1000 characters")
    return normalized or None
