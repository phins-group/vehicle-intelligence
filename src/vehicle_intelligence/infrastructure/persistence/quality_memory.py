"""Bounded quality aggregation for local/embedded repositories."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from vehicle_intelligence.application.ports import (
    DatasetSampleQuery,
    DatasetSampleRepository,
    EventQuery,
    VehicleEventRepository,
)
from vehicle_intelligence.domain import (
    DailyQualityPoint,
    DatasetFeedbackMetrics,
    DatasetSampleReason,
    DatasetSampleStatus,
    EventStatus,
    ModelMetadata,
    ModelQualityReport,
    ModelQualitySlice,
    VehicleEvent,
)
from vehicle_intelligence.infrastructure.persistence.quality_common import QualityCounts


class InMemoryModelQualityRepository:
    def __init__(
        self,
        events: VehicleEventRepository,
        samples: DatasetSampleRepository,
        maximum_scan: int = 100_000,
    ) -> None:
        self._events = events
        self._samples = samples
        self._maximum_scan = maximum_scan

    async def summarize(
        self,
        from_time: datetime,
        to_time: datetime,
        generated_at: datetime,
        maximum_models: int,
    ) -> ModelQualityReport:
        total = QualityCounts()
        models: dict[tuple[str, str, str | None] | None, QualityCounts] = defaultdict(
            QualityCounts
        )
        daily: dict[str, QualityCounts] = defaultdict(QualityCounts)
        cursor: str | None = None
        scanned = 0
        truncated = False
        while scanned < self._maximum_scan:
            page = await self._events.list(
                EventQuery(
                    limit=min(200, self._maximum_scan - scanned),
                    cursor=cursor,
                    from_time=from_time,
                    to_time=to_time,
                )
            )
            for event in page.items:
                if not from_time <= event.occurred_at < to_time:
                    continue
                counts = _event_counts(event)
                total.add(counts)
                models[_model_key(event.ai.ocr)].add(counts)
                daily[event.occurred_at.date().isoformat()].add(counts)
            scanned += len(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        if cursor is not None:
            truncated = True

        feedback, samples_truncated = await self._feedback(from_time, to_time)
        ordered_models = sorted(
            models.items(),
            key=lambda item: (-item[1].event_count, str(item[0])),
        )[:maximum_models]
        return ModelQualityReport(
            from_time=from_time,
            to_time=to_time,
            generated_at=generated_at,
            totals=total.metrics(),
            models=tuple(
                ModelQualitySlice(model=_model(key), metrics=counts.metrics())
                for key, counts in ordered_models
            ),
            daily=tuple(
                DailyQualityPoint(day=day, metrics=counts.metrics())
                for day, counts in sorted(daily.items())
            ),
            feedback=feedback,
            truncated=truncated or samples_truncated,
        )

    async def _feedback(
        self,
        from_time: datetime,
        to_time: datetime,
    ) -> tuple[DatasetFeedbackMetrics, bool]:
        statuses: dict[DatasetSampleStatus, int] = defaultdict(int)
        reasons: dict[DatasetSampleReason, int] = defaultdict(int)
        cursor: str | None = None
        scanned = 0
        while scanned < self._maximum_scan:
            page = await self._samples.list(
                DatasetSampleQuery(
                    limit=min(200, self._maximum_scan - scanned),
                    cursor=cursor,
                    from_time=from_time,
                    to_time=to_time,
                )
            )
            for sample in page.items:
                statuses[sample.status] += 1
                reasons[sample.reason] += 1
            scanned += len(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
        total = sum(statuses.values())
        return (
            DatasetFeedbackMetrics(
                total=total,
                ready=statuses[DatasetSampleStatus.READY],
                exporting=statuses[DatasetSampleStatus.EXPORTING],
                exported=statuses[DatasetSampleStatus.EXPORTED],
                export_failed=statuses[DatasetSampleStatus.EXPORT_FAILED],
                corrections=reasons[DatasetSampleReason.HUMAN_CORRECTION],
                confirmations=reasons[DatasetSampleReason.HUMAN_CONFIRMATION],
            ),
            cursor is not None,
        )

    async def close(self) -> None:
        return None


def _event_counts(event: VehicleEvent) -> QualityCounts:
    reviewed = event.plate is not None and event.plate.review is not None
    corrected = (
        reviewed
        and event.plate is not None
        and event.plate.review is not None
        and event.plate.review.normalized != event.plate.normalized
    )
    confidence = event.plate.confidence if event.plate is not None else None
    return QualityCounts(
        event_count=1,
        readable_plate_count=int(event.plate is not None),
        confirmed_count=int(event.status is EventStatus.CONFIRMED),
        needs_review_count=int(event.status is EventStatus.NEEDS_REVIEW),
        no_plate_count=int(event.status is EventStatus.NO_PLATE),
        unreadable_count=int(event.status is EventStatus.UNREADABLE),
        reviewed_count=int(reviewed),
        corrected_count=int(corrected),
        confidence_sum=confidence or 0.0,
        confidence_count=int(confidence is not None),
    )


def _model_key(model: ModelMetadata | None) -> tuple[str, str, str | None] | None:
    return (model.name, model.version, model.hash) if model is not None else None


def _model(key: tuple[str, str, str | None] | None) -> ModelMetadata | None:
    return ModelMetadata(name=key[0], version=key[1], hash=key[2]) if key else None
