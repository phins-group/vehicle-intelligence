"""Use cases for human-labeling detector review queues."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from vehicle_intelligence.domain import AuditLog, Principal
from vehicle_intelligence.domain.dataset_review import (
    DetectorPromotionJob,
    DetectorPromotionStatus,
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

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class PreparedDetectorReview:
    source_id: str
    review_id: str
    expected_revision: int
    before: DetectorReviewItem
    after: DetectorReviewItem
    decision: DetectorReviewDecision


class AuditEntrySink(Protocol):
    async def persist(self, entry: AuditLog) -> AuditLog: ...


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
        audit_entry: AuditLog | None = None,
    ) -> DetectorReviewItem: ...

    async def pending_audits(self, limit: int = 100) -> tuple[AuditLog, ...]: ...

    async def mark_audit_delivered(self, entry_id: str) -> None: ...

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

    async def fail_queued_promotion_job(self, job_id: str, error_code: str) -> None: ...


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
        self._audit_relay_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def close(self) -> None:
        if self._audit_relay_task is not None:
            self._audit_relay_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._audit_relay_task
            self._audit_relay_task = None
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
        prepared = await self.prepare_review(source_id, review_id, command)
        return await self.commit_review(prepared)

    async def prepare_review(
        self,
        source_id: str,
        review_id: str,
        command: DetectorReviewCommand,
    ) -> PreparedDetectorReview:
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
        after = replace(
            item,
            status=decision.status,
            revision=decision.revision,
            decision=decision,
        )
        return PreparedDetectorReview(
            source_id=source_id,
            review_id=review_id,
            expected_revision=command.expected_revision,
            before=item,
            after=after,
            decision=decision,
        )

    async def commit_review(
        self,
        prepared: PreparedDetectorReview,
        audit_entry: AuditLog | None = None,
    ) -> DetectorReviewItem:
        return await self._repository.save_decision(
            prepared.source_id,
            prepared.review_id,
            prepared.decision,
            prepared.expected_revision,
            audit_entry,
        )

    async def deliver_audit(self, entry: AuditLog, audits: AuditEntrySink) -> bool:
        try:
            await audits.persist(entry)
            await self._repository.mark_audit_delivered(entry.id)
        except Exception:
            logger.exception(
                "detector review audit delivery deferred",
                extra={"audit_id": entry.id},
            )
            return False
        return True

    async def flush_pending_audits(
        self,
        audits: AuditEntrySink,
        *,
        limit: int = 100,
    ) -> int:
        delivered = 0
        for entry in await self._repository.pending_audits(limit):
            if not await self.deliver_audit(entry, audits):
                break
            delivered += 1
        return delivered

    def start_audit_relay(
        self,
        audits: AuditEntrySink,
        *,
        retry_seconds: float = 5.0,
    ) -> None:
        if retry_seconds <= 0:
            raise ValueError("audit outbox retry interval must be positive")
        if self._audit_relay_task is not None and not self._audit_relay_task.done():
            return
        self._audit_relay_task = asyncio.create_task(
            self._run_audit_relay(audits, retry_seconds),
            name="detector-review-audit-relay",
        )

    async def _run_audit_relay(
        self,
        audits: AuditEntrySink,
        retry_seconds: float,
    ) -> None:
        while True:
            try:
                await self.flush_pending_audits(audits)
            except Exception:
                logger.exception("detector review audit relay failed; retry scheduled")
            await asyncio.sleep(retry_seconds)

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
        job = await self.prepare_promotion(source_id, target_source_id, principal)
        self.dispatch_promotion(job)
        return job

    async def prepare_promotion(
        self,
        source_id: str,
        target_source_id: str,
        principal: Principal,
    ) -> DetectorPromotionJob:
        """Persist a validated promotion without starting the background task."""

        return await self._repository.create_promotion_job(
            source_id,
            target_source_id,
            principal.id,
        )

    def dispatch_promotion(self, job: DetectorPromotionJob) -> None:
        """Start a queued promotion after its audit record is durable."""

        name = f"detector-promotion-{job.id}"
        if job.status is not DetectorPromotionStatus.QUEUED or any(
            task.get_name() == name for task in self._tasks
        ):
            return
        task = asyncio.create_task(
            self._repository.run_promotion_job(job.id),
            name=name,
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def fail_prepared_promotion(self, job_id: str, error_code: str) -> None:
        await self._repository.fail_queued_promotion_job(job_id, error_code)

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
            if command.action is DetectorReviewAction.REJECT and not _normalized_note(command.note):
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
