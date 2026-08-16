from vehicle_intelligence.config import (
    IdentityConfig,
    MongoConfig,
    RealtimeConfig,
    RedisConfig,
    RuleEngineConfig,
)
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.interfaces.event_worker_composition import build_event_worker


async def test_event_worker_composition_exposes_enabled_production_processors() -> None:
    runtime = MongoRuntime(MongoConfig())
    components = build_event_worker(
        runtime,
        RedisConfig(),
        IdentityConfig(enabled=True),
        RuleEngineConfig(enabled=True),
        RealtimeConfig(enabled=True),
        "test-worker",
    )

    assert components.identities is not None
    assert components.rules is not None
    assert components.action_executions is not None

    await components.worker.close()
    await runtime.close()


async def test_event_worker_composition_omits_disabled_processors() -> None:
    runtime = MongoRuntime(MongoConfig())
    components = build_event_worker(
        runtime,
        RedisConfig(),
        IdentityConfig(enabled=False),
        RuleEngineConfig(enabled=False),
        RealtimeConfig(enabled=False),
        "test-worker",
    )

    assert components.identities is None
    assert components.rules is None
    assert components.action_executions is None

    await components.worker.close()
    await runtime.close()
