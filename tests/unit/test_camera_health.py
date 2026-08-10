from dataclasses import replace
from datetime import UTC, datetime

from vehicle_intelligence.application.health import CameraHealthReporter
from vehicle_intelligence.domain import CameraHealth, CameraStatus
from vehicle_intelligence.infrastructure.persistence.camera_memory import (
    InMemoryCameraHealthRepository,
)


async def test_health_reporter_throttles_and_keeps_latest_state_only() -> None:
    repository = InMemoryCameraHealthRepository()
    monotonic = [0.0]
    reporter = CameraHealthReporter(repository, 5.0, monotonic_clock=lambda: monotonic[0])
    await reporter.initialize()
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    online = CameraHealth(
        camera_id="gate-01",
        status=CameraStatus.ONLINE,
        source_fps=25,
        decode_fps=24,
        queue_size=1,
        dropped_frames=2,
        reconnect_count=0,
        connection_failures=0,
        stream_epoch=0,
        last_frame_at=timestamp,
        updated_at=timestamp,
    )

    assert await reporter.report(online)
    monotonic[0] = 1
    assert not await reporter.report(online)
    monotonic[0] = 5
    offline = replace(online, status=CameraStatus.OFFLINE)
    assert await reporter.report(offline)

    assert len(await repository.list()) == 1
    assert (await repository.get("gate-01")).status is CameraStatus.OFFLINE
    await reporter.close()
