import asyncio
from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.pipeline import PipelineStats, VideoVehiclePipeline
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.domain import VehicleTrack
from vehicle_intelligence.exceptions import FinalizationOutboxFullError


class _FullOutboxFinalizer:
    async def finalize(self, track: VehicleTrack):
        del track
        raise FinalizationOutboxFullError("injected full outbox")


class _Tracker:
    def reset(self) -> None:
        raise RuntimeError("injected tracker cleanup failure")


class _Source:
    def __init__(self, *, cancel: bool) -> None:
        self.cancel = cancel

    def frames(self):
        if not self.cancel:
            return iter(())

        def cancelled():
            raise asyncio.CancelledError
            yield

        return cancelled()

    def close(self) -> None:
        raise RuntimeError("injected source cleanup failure")


async def test_failed_durable_stage_keeps_track_active_until_worker_fails() -> None:
    timestamp = datetime(2026, 8, 8, tzinfo=UTC)
    track = VehicleTrack(
        camera_id="gate-01",
        session_id="retry",
        local_track_id=1,
        first_seen=timestamp,
        last_seen=timestamp,
        max_trajectory_points=8,
        max_plate_observations=8,
    )
    key = (0, 1)
    pipeline = object.__new__(VideoVehiclePipeline)
    pipeline._active = {key: track}
    pipeline._completed_track_ids = {}
    pipeline._finalizer = _FullOutboxFinalizer()
    pipeline._stats = PipelineStats()

    with pytest.raises(FinalizationOutboxFullError, match="full outbox"):
        await pipeline._finalize_track(key)

    assert pipeline._active[key] is track
    assert key not in pipeline._completed_track_ids
    assert pipeline._stats.finalized_tracks == 0


@pytest.mark.parametrize(
    ("cancel", "expected_error"),
    (
        (True, asyncio.CancelledError),
        (False, FinalizationOutboxFullError),
    ),
)
async def test_shutdown_finalization_does_not_mask_active_cancellation(
    cancel,
    expected_error,
) -> None:
    pipeline = object.__new__(VideoVehiclePipeline)
    pipeline._settings = load_settings()
    pipeline._source = _Source(cancel=cancel)
    pipeline._tracker = _Tracker()

    async def fail_finalization():
        raise FinalizationOutboxFullError("injected shutdown failure")

    async def no_health_report(*, force=False):
        del force
        raise RuntimeError("injected health cleanup failure")

    pipeline._finalize_all = fail_finalization
    pipeline._report_health = no_health_report

    with pytest.raises(expected_error):
        await pipeline.run()
