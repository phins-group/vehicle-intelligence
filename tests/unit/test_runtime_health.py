import asyncio

from vehicle_intelligence.application.runtime_health import (
    RuntimeCheckStatus,
    RuntimeDependency,
    RuntimeHealthService,
)


async def test_runtime_health_only_blocks_on_required_dependencies() -> None:
    async def available() -> bool:
        return True

    async def unavailable() -> bool:
        return False

    health = RuntimeHealthService(
        (
            RuntimeDependency("eventStore", required=True, probe=available),
            RuntimeDependency("minio", required=False, probe=unavailable),
            RuntimeDependency("realtime", required=False, probe=None),
        ),
        cache_seconds=0,
    )

    before_start = await health.assess()
    health.start()
    running = await health.assess()
    await health.stop()
    after_stop = await health.assess()

    assert not before_start.ready
    assert before_start.checks["application"].status is RuntimeCheckStatus.UNAVAILABLE
    assert running.ready
    assert running.checks["eventStore"].status is RuntimeCheckStatus.READY
    assert running.checks["minio"].status is RuntimeCheckStatus.DEGRADED
    assert running.checks["realtime"].status is RuntimeCheckStatus.DISABLED
    assert not after_stop.ready


async def test_runtime_health_returns_not_ready_for_required_failure() -> None:
    async def unavailable() -> bool:
        return False

    health = RuntimeHealthService(
        (RuntimeDependency("eventStore", required=True, probe=unavailable),),
        cache_seconds=0,
    )
    health.start()

    snapshot = await health.assess()
    await health.stop()

    assert not snapshot.ready
    assert snapshot.checks["eventStore"].status is RuntimeCheckStatus.UNAVAILABLE


async def test_runtime_health_coalesces_timed_out_probe() -> None:
    release = asyncio.Event()
    calls = 0

    async def blocked() -> bool:
        nonlocal calls
        calls += 1
        await release.wait()
        return True

    health = RuntimeHealthService(
        (RuntimeDependency("minio", required=False, probe=blocked),),
        probe_timeout_seconds=0.01,
        cache_seconds=0,
    )
    health.start()

    first = await health.assess()
    second = await health.assess()
    release.set()
    await health.stop()

    assert first.ready and second.ready
    assert first.checks["minio"].status is RuntimeCheckStatus.DEGRADED
    assert calls == 1
