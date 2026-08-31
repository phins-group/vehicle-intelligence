import asyncio
import os
import uuid
from dataclasses import replace

import httpx
import pytest
from redis.asyncio import Redis

from vehicle_intelligence.application.actions import (
    ActionEngine,
    AlertActionHandler,
    HttpActionHandler,
    action_execution_id,
)
from vehicle_intelligence.application.event_policy import VehicleEventPolicyProcessor
from vehicle_intelligence.application.event_worker import VehicleEventWorker
from vehicle_intelligence.application.ports import AlertQuery
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.config import MongoConfig, RedisConfig, RuleEngineConfig
from vehicle_intelligence.domain import (
    ActionExecutionStatus,
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    WatchlistEntry,
    WatchlistType,
)
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.redis_streams import (
    RedisStreamEventConsumer,
    RedisStreamEventPublisher,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.policy_mongo import (
    MongoActionExecutionRepository,
    MongoAlertRepository,
    MongoRuleRepository,
    MongoWatchlistRepository,
)


@pytest.mark.skipif(
    not os.getenv("TEST_REDIS_URL") or not os.getenv("TEST_MONGODB_URI"),
    reason="TEST_REDIS_URL and TEST_MONGODB_URI are not configured",
)
async def test_redis_event_worker_persists_one_alert_for_duplicate_delivery(
    sample_event,
) -> None:
    suffix = uuid.uuid4().hex
    stream = f"vehicle.events.policy-test.{suffix}"
    dead_letter = f"vehicle.events.policy-test.dlq.{suffix}"
    redis_config = RedisConfig(
        url=os.environ["TEST_REDIS_URL"],
        stream=stream,
        dead_letter_stream=dead_letter,
        consumer_group=f"policy-processors-{suffix}",
        max_length=100,
        dead_letter_max_length=10,
        batch_size=10,
        block_ms=50,
        claim_idle_ms=1000,
    )
    mongo_config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    event = replace(
        sample_event,
        id=f"evt-policy-{suffix}",
        track_id=f"gate-01:policy:{suffix}",
    )
    entry_id = f"wle-{suffix}"
    rule_id = f"rule-{suffix}"
    action_id = f"alert-{suffix}"
    execution_id = action_execution_id(event.id, rule_id, action_id)
    codec = JsonEventEnvelopeCodec()
    publisher = RedisStreamEventPublisher(redis_config, codec)
    consumer = RedisStreamEventConsumer(redis_config, f"worker-{suffix}")
    event_repository = MongoVehicleEventRepository(mongo_config)
    watchlists = MongoWatchlistRepository(mongo_config)
    rules = MongoRuleRepository(mongo_config)
    executions = MongoActionExecutionRepository(mongo_config)
    action_alerts = MongoAlertRepository(mongo_config)
    inspection_alerts = MongoAlertRepository(mongo_config)
    engine = ActionEngine(
        executions,
        {RuleActionType.CREATE_ALERT: AlertActionHandler(action_alerts)},
        RuleEngineConfig(),
    )
    processor = VehicleEventPolicyProcessor(watchlists, rules, RuleEvaluator(), engine)
    worker = VehicleEventWorker(consumer, event_repository, codec, post_processor=processor)
    admin = Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    try:
        await publisher.initialize()
        await worker.initialize()
        await inspection_alerts.ensure_indexes()
        assert await watchlists.create(
            WatchlistEntry(
                id=entry_id,
                plate=event.plate.normalized,
                list_type=WatchlistType.BLACKLIST,
                enabled=True,
                created_at=event.occurred_at,
                updated_at=event.occurred_at,
            )
        )
        assert await rules.create(
            Rule(
                id=rule_id,
                name="Redis blacklist alert",
                enabled=True,
                priority=100,
                conditions=(
                    RuleCondition(
                        "watchlist",
                        RuleConditionOperator.CONTAINS,
                        "BLACKLIST",
                    ),
                ),
                actions=(
                    RuleAction(
                        action_id,
                        RuleActionType.CREATE_ALERT,
                        {"severity": "HIGH"},
                    ),
                ),
                created_at=event.occurred_at,
                updated_at=event.occurred_at,
            )
        )
        assert await publisher.publish(event)
        assert await publisher.publish(event)

        assert await worker.run_once() == 2

        assert worker.stats.events_persisted == 1
        assert worker.stats.duplicate_events == 1
        assert worker.stats.matched_rules == 2
        assert worker.stats.actions_succeeded == 1
        assert worker.stats.actions_skipped == 1
        page = await inspection_alerts.list(AlertQuery(rule_id=rule_id))
        assert len(page.items) == 1
        pending = await admin.xpending(stream, redis_config.consumer_group)
        assert pending["pending"] == 0
        assert await admin.xlen(stream) == 0
    finally:
        await event_repository._collection.delete_one({"_id": event.id})
        await watchlists._collection.delete_one({"_id": entry_id})
        await rules._collection.delete_one({"_id": rule_id})
        await executions._collection.delete_one({"_id": execution_id})
        await inspection_alerts._collection.delete_many({"rule.id": rule_id})
        await worker.close()
        await publisher.close()
        await inspection_alerts.close()
        await admin.delete(stream, dead_letter)
        await admin.aclose()


@pytest.mark.skipif(
    not os.getenv("TEST_REDIS_URL") or not os.getenv("TEST_MONGODB_URI"),
    reason="TEST_REDIS_URL and TEST_MONGODB_URI are not configured",
)
async def test_partial_policy_failure_reclaims_without_repeating_completed_action(
    sample_event,
) -> None:
    suffix = uuid.uuid4().hex
    redis_config = RedisConfig(
        url=os.environ["TEST_REDIS_URL"],
        stream=f"vehicle.events.policy-retry.{suffix}",
        dead_letter_stream=f"vehicle.events.policy-retry.dlq.{suffix}",
        consumer_group=f"policy-retry-{suffix}",
        max_length=100,
        dead_letter_max_length=10,
        batch_size=10,
        block_ms=50,
        claim_idle_ms=1000,
        reclaim_interval_ms=100,
    )
    mongo_config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    calls = 0

    def receiver(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 204)

    client = httpx.AsyncClient(transport=httpx.MockTransport(receiver))
    action_config = RuleEngineConfig(
        action_max_attempts=3,
        external_actions_enabled=True,
        external_allowed_hosts=["receiver.test"],
    )
    codec = JsonEventEnvelopeCodec()
    publisher = RedisStreamEventPublisher(redis_config, codec)
    consumer = RedisStreamEventConsumer(redis_config, f"worker-{suffix}")
    events = MongoVehicleEventRepository(mongo_config)
    watchlists = MongoWatchlistRepository(mongo_config)
    rules = MongoRuleRepository(mongo_config)
    executions = MongoActionExecutionRepository(mongo_config)
    action_alerts = MongoAlertRepository(mongo_config)
    inspect_alerts = MongoAlertRepository(mongo_config)
    alert_action = RuleAction(f"alert-{suffix}", RuleActionType.CREATE_ALERT)
    webhook_action = RuleAction(
        f"webhook-{suffix}",
        RuleActionType.WEBHOOK,
        {"url": "https://receiver.test/events"},
    )
    rule = Rule(
        id=f"rule-retry-{suffix}",
        name="Retry without duplicate side effect",
        enabled=True,
        priority=100,
        conditions=(RuleCondition("camera.id", RuleConditionOperator.EQ, sample_event.camera.id),),
        actions=(alert_action, webhook_action),
        created_at=sample_event.occurred_at,
        updated_at=sample_event.occurred_at,
    )
    event = replace(
        sample_event,
        id=f"evt-policy-retry-{suffix}",
        track_id=f"gate-01:policy-retry:{suffix}",
    )
    engine = ActionEngine(
        executions,
        {
            RuleActionType.CREATE_ALERT: AlertActionHandler(action_alerts),
            RuleActionType.WEBHOOK: HttpActionHandler(action_config, client),
        },
        action_config,
    )
    processor = VehicleEventPolicyProcessor(
        watchlists,
        rules,
        RuleEvaluator(),
        engine,
    )
    worker = VehicleEventWorker(
        consumer,
        events,
        codec,
        post_processor=processor,
        reclaim_interval_seconds=redis_config.reclaim_interval_ms / 1000.0,
    )
    admin = Redis.from_url(os.environ["TEST_REDIS_URL"], decode_responses=True)
    execution_ids = [action_execution_id(event.id, rule.id, action.id) for action in rule.actions]
    try:
        await publisher.initialize()
        await worker.initialize()
        await inspect_alerts.ensure_indexes()
        assert await rules.create(rule)
        assert await publisher.publish(event)

        assert await worker.run_once() == 1
        assert worker.stats.policy_failures == 1
        assert (await admin.xpending(redis_config.stream, redis_config.consumer_group))[
            "pending"
        ] == 1

        await asyncio.sleep(1.05)
        assert await worker.run_once() == 1
        assert calls == 2
        assert (await admin.xpending(redis_config.stream, redis_config.consumer_group))[
            "pending"
        ] == 0
        page = await inspect_alerts.list(AlertQuery(rule_id=rule.id))
        assert len(page.items) == 1
        first_execution = await executions.get(execution_ids[0])
        second_execution = await executions.get(execution_ids[1])
        assert first_execution.status is ActionExecutionStatus.SUCCEEDED
        assert first_execution.attempt_count == 1
        assert second_execution.status is ActionExecutionStatus.SUCCEEDED
        assert second_execution.attempt_count == 2
    finally:
        await events._collection.delete_one({"_id": event.id})
        await rules._collection.delete_one({"_id": rule.id})
        await executions._collection.delete_many({"_id": {"$in": execution_ids}})
        await inspect_alerts._collection.delete_many({"rule.id": rule.id})
        await worker.close()
        await publisher.close()
        await inspect_alerts.close()
        await client.aclose()
        await admin.delete(redis_config.stream, redis_config.dead_letter_stream)
        await admin.aclose()
