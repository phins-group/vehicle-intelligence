"""Shared production composition for the durable vehicle-event worker."""

from __future__ import annotations

from dataclasses import dataclass

from vehicle_intelligence.application.actions import (
    ActionEngine,
    AlertActionHandler,
    HttpActionHandler,
    LogActionHandler,
)
from vehicle_intelligence.application.event_policy import VehicleEventPolicyProcessor
from vehicle_intelligence.application.event_worker import VehicleEventWorker
from vehicle_intelligence.application.identity import (
    CompositeVehicleEventPostProcessor,
    VehicleIdentityProcessor,
)
from vehicle_intelligence.application.ports import VehicleEventCodec, VehicleEventPostProcessor
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.config import (
    IdentityConfig,
    RealtimeConfig,
    RedisConfig,
    RuleEngineConfig,
)
from vehicle_intelligence.domain import RuleActionType
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.realtime_redis import (
    RedisRealtimeEventPublisher,
)
from vehicle_intelligence.infrastructure.messaging.redis_streams import RedisStreamEventConsumer
from vehicle_intelligence.infrastructure.persistence.identity_mongo import (
    MongoVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.persistence.policy_mongo import (
    MongoActionExecutionRepository,
    MongoAlertRepository,
    MongoRuleRepository,
    MongoWatchlistRepository,
)


@dataclass(frozen=True, slots=True)
class EventWorkerComponents:
    worker: VehicleEventWorker
    events: MongoVehicleEventRepository
    identities: MongoVehicleIdentityRepository | None
    rules: MongoRuleRepository | None
    action_executions: MongoActionExecutionRepository | None


def build_event_worker(
    runtime: MongoRuntime,
    redis: RedisConfig,
    identity: IdentityConfig,
    rule_engine: RuleEngineConfig,
    realtime: RealtimeConfig,
    consumer_name: str,
    codec: VehicleEventCodec | None = None,
) -> EventWorkerComponents:
    event_codec = codec or JsonEventEnvelopeCodec()
    consumer = RedisStreamEventConsumer(redis, consumer_name)
    events = MongoVehicleEventRepository(runtime)
    identities: MongoVehicleIdentityRepository | None = None
    rules: MongoRuleRepository | None = None
    action_executions: MongoActionExecutionRepository | None = None
    processors: list[VehicleEventPostProcessor] = []

    if identity.enabled:
        identities = MongoVehicleIdentityRepository(runtime)
        processors.append(VehicleIdentityProcessor(identities, events, identity))

    if rule_engine.enabled:
        rules = MongoRuleRepository(runtime)
        action_executions = MongoActionExecutionRepository(runtime)
        alert_handler = AlertActionHandler(MongoAlertRepository(runtime))
        http_handler = HttpActionHandler(rule_engine)
        actions = ActionEngine(
            action_executions,
            {
                RuleActionType.CREATE_ALERT: alert_handler,
                RuleActionType.LOG: LogActionHandler(),
                RuleActionType.OPEN_BARRIER: http_handler,
                RuleActionType.WEBHOOK: http_handler,
                RuleActionType.HTTP_REQUEST: http_handler,
                RuleActionType.NOTIFICATION: http_handler,
            },
            rule_engine,
        )
        processors.append(
            VehicleEventPolicyProcessor(
                MongoWatchlistRepository(runtime),
                rules,
                RuleEvaluator(),
                actions,
                rule_engine.evaluation_max_rules,
                rule_engine.rule_cache_ttl_seconds,
            )
        )

    post_processor = _combine_processors(processors)
    publisher = (
        RedisRealtimeEventPublisher(redis, realtime, event_codec) if realtime.enabled else None
    )
    worker = VehicleEventWorker(
        consumer,
        events,
        event_codec,
        retry_delay_seconds=redis.retry_delay_seconds,
        post_processor=post_processor,
        realtime_publisher=publisher,
        maximum_concurrency=redis.worker_concurrency,
        reclaim_interval_seconds=redis.reclaim_interval_ms / 1000.0,
    )
    return EventWorkerComponents(worker, events, identities, rules, action_executions)


def _combine_processors(
    processors: list[VehicleEventPostProcessor],
) -> VehicleEventPostProcessor | None:
    if not processors:
        return None
    if len(processors) == 1:
        return processors[0]
    return CompositeVehicleEventPostProcessor(tuple(processors))
