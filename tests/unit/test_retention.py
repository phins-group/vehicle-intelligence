from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.retention import RetentionService
from vehicle_intelligence.config import RetentionConfig
from vehicle_intelligence.domain import (
    LifecycleReconcileResult,
    MediaKind,
    MediaRetentionClaim,
)
from vehicle_intelligence.exceptions import MediaStorageError


class FakeRepository:
    def __init__(self) -> None:
        now = datetime(2026, 8, 1, tzinfo=UTC)
        self.claims = {
            MediaKind.SNAPSHOT: [
                MediaRetentionClaim("evt-ok", MediaKind.SNAPSHOT, "vehicles/ok.jpg", "seed", now),
                MediaRetentionClaim(
                    "evt-fail", MediaKind.SNAPSHOT, "vehicles/fail.jpg", "seed", now
                ),
            ]
        }
        self.claim_calls = []
        self.deleted = []
        self.failed = []
        self.initialized = 0
        self.closed = 0

    async def ensure_indexes(self) -> None:
        self.initialized += 1

    async def claim_media(self, kind, older_than, stale_before, lease_id, limit):
        self.claim_calls.append((kind, older_than, stale_before, lease_id, limit))
        return [
            MediaRetentionClaim(
                claim.event_id,
                claim.kind,
                claim.key,
                lease_id,
                claim.occurred_at,
            )
            for claim in self.claims.get(kind, [])
        ]

    async def mark_media_deleted(self, claim, deleted_at) -> None:
        self.deleted.append((claim, deleted_at))

    async def mark_media_failed(self, claim, error_code, failed_at) -> None:
        self.failed.append((claim, error_code, failed_at))

    async def delete_expired_events(self, older_than, limit) -> int:
        self.event_cutoff = older_than
        self.event_limit = limit
        return 2

    async def close(self) -> None:
        self.closed += 1


class FakeMedia:
    async def remove(self, key: str) -> None:
        if key.endswith("fail.jpg"):
            raise MediaStorageError("simulated failure")


class FakeLifecycle:
    def __init__(self) -> None:
        self.calls = []

    async def reconcile_lifecycle(self, debug_expiry_days):
        self.calls.append(debug_expiry_days)
        return LifecycleReconcileResult(True, 2, 1)


class FakeMetrics:
    def observe_retention(self, **values) -> None:
        self.values = values


@pytest.mark.asyncio
async def test_retention_service_is_bounded_leased_resilient_and_instrumented() -> None:
    repository = FakeRepository()
    lifecycle = FakeLifecycle()
    metrics = FakeMetrics()
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    config = RetentionConfig(
        enabled=True,
        batch_size=7,
        claim_stale_seconds=120,
        vehicle_events_days=2,
        snapshots_days=1,
        vehicle_crops_days=1,
        plate_crops_days=1,
        event_clips_days=1,
        debug_images_days=3,
    )
    service = RetentionService(
        config,
        repository,
        FakeMedia(),
        lifecycle=lifecycle,
        metrics=metrics,
        clock=lambda: now,
    )

    stats = await service.run_once()
    await service.close()

    assert repository.initialized == 1
    assert lifecycle.calls == [3]
    assert stats.claimed_by_kind["snapshot"] == 2
    assert len(repository.deleted) == 1
    assert repository.failed[0][1] == "MediaStorageError"
    assert stats.events_deleted == 2
    assert repository.event_limit == 7
    assert repository.event_cutoff == datetime(2026, 8, 7, 12, tzinfo=UTC)
    assert metrics.values["events_deleted"] == 2
    assert repository.closed == 1
