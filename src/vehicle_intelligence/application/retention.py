"""Coordinated, leased media/event retention independent of MongoDB and MinIO."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from vehicle_intelligence.application.ports import (
    MediaLifecycleManager,
    MediaObjectCleaner,
    RetentionRepository,
)
from vehicle_intelligence.config import RetentionConfig
from vehicle_intelligence.domain import LifecycleReconcileResult, MediaKind
from vehicle_intelligence.exceptions import MediaStorageError


class RetentionMetrics(Protocol):
    def observe_retention(
        self,
        *,
        deleted_by_kind: dict[str, int],
        failed_by_kind: dict[str, int],
        events_deleted: int,
        duration_seconds: float,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class RetentionStats:
    claimed_by_kind: dict[str, int] = field(default_factory=dict)
    deleted_by_kind: dict[str, int] = field(default_factory=dict)
    failed_by_kind: dict[str, int] = field(default_factory=dict)
    events_deleted: int = 0
    lifecycle: LifecycleReconcileResult | None = None
    duration_seconds: float = 0.0


class RetentionService:
    def __init__(
        self,
        config: RetentionConfig,
        repository: RetentionRepository,
        media: MediaObjectCleaner,
        lifecycle: MediaLifecycleManager | None = None,
        metrics: RetentionMetrics | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._repository = repository
        self._media = media
        self._lifecycle = lifecycle
        self._metrics = metrics
        self._clock = clock
        self._initialized = False
        self._lifecycle_result: LifecycleReconcileResult | None = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        await self._repository.ensure_indexes()
        self._initialized = True

    async def run_once(self) -> RetentionStats:
        await self.initialize()
        started = time.perf_counter()
        if self._lifecycle is not None and self._config.minio_lifecycle_enabled:
            self._lifecycle_result = await self._lifecycle.reconcile_lifecycle(
                self._config.debug_images_days,
            )
        now = self._now()
        stale_before = now - timedelta(seconds=self._config.claim_stale_seconds)
        claimed: dict[str, int] = {}
        deleted: dict[str, int] = {}
        failed: dict[str, int] = {}
        windows = {
            MediaKind.SNAPSHOT: self._config.snapshots_days,
            MediaKind.VEHICLE_CROP: self._config.vehicle_crops_days,
            MediaKind.PLATE_CROP: self._config.plate_crops_days,
            MediaKind.EVENT_CLIP: self._config.event_clips_days,
        }
        for kind, days in windows.items():
            lease_id = f"ret_{uuid.uuid4().hex}"
            claims = await self._repository.claim_media(
                kind,
                now - timedelta(days=days),
                stale_before,
                lease_id,
                self._config.batch_size,
            )
            claimed[kind.value] = len(claims)
            for claim in claims:
                try:
                    await self._media.remove(claim.key)
                except MediaStorageError as exc:
                    failed[kind.value] = failed.get(kind.value, 0) + 1
                    await self._repository.mark_media_failed(
                        claim,
                        type(exc).__name__,
                        self._now(),
                    )
                else:
                    await self._repository.mark_media_deleted(claim, self._now())
                    deleted[kind.value] = deleted.get(kind.value, 0) + 1

        events_deleted = await self._repository.delete_expired_events(
            now - timedelta(days=self._config.vehicle_events_days),
            self._config.batch_size,
        )
        duration = time.perf_counter() - started
        stats = RetentionStats(
            claimed_by_kind=claimed,
            deleted_by_kind=deleted,
            failed_by_kind=failed,
            events_deleted=events_deleted,
            lifecycle=self._lifecycle_result,
            duration_seconds=duration,
        )
        if self._metrics is not None:
            self._metrics.observe_retention(
                deleted_by_kind=deleted,
                failed_by_kind=failed,
                events_deleted=events_deleted,
                duration_seconds=duration,
            )
        return stats

    async def close(self) -> None:
        await self._repository.close()

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("retention clock must be timezone-aware")
        return value.astimezone(UTC)
