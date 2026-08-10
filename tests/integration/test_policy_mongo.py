import os
import uuid
from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.actions import (
    ActionEngine,
    AlertActionHandler,
    action_execution_id,
)
from vehicle_intelligence.application.event_policy import VehicleEventPolicyProcessor
from vehicle_intelligence.application.ports import AlertQuery
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.config import MongoConfig, RuleEngineConfig
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
from vehicle_intelligence.infrastructure.persistence.policy_mongo import (
    MongoActionExecutionRepository,
    MongoAlertRepository,
    MongoRuleRepository,
    MongoWatchlistRepository,
)


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_policy_processor_is_idempotent_and_indexes_queries(sample_event) -> None:
    suffix = uuid.uuid4().hex
    entry_id = f"wle-{suffix}"
    rule_id = f"rule-{suffix}"
    action_id = f"alert-{suffix}"
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    watchlists = MongoWatchlistRepository(config)
    rules = MongoRuleRepository(config)
    executions = MongoActionExecutionRepository(config)
    action_alerts = MongoAlertRepository(config)
    inspection_alerts = MongoAlertRepository(config)
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    entry = WatchlistEntry(
        id=entry_id,
        plate=sample_event.plate.normalized,
        list_type=WatchlistType.BLACKLIST,
        enabled=True,
        created_at=timestamp,
        updated_at=timestamp,
    )
    action = RuleAction(
        action_id,
        RuleActionType.CREATE_ALERT,
        {"severity": "CRITICAL", "message": "Mongo blacklist alert"},
    )
    rule = Rule(
        id=rule_id,
        name="Mongo blacklist rule",
        enabled=True,
        priority=100,
        conditions=(
            RuleCondition(
                "watchlist",
                RuleConditionOperator.CONTAINS,
                "BLACKLIST",
            ),
        ),
        actions=(action,),
        created_at=timestamp,
        updated_at=timestamp,
    )
    engine = ActionEngine(
        executions,
        {RuleActionType.CREATE_ALERT: AlertActionHandler(action_alerts, clock=lambda: timestamp)},
        RuleEngineConfig(),
        clock=lambda: timestamp,
    )
    processor = VehicleEventPolicyProcessor(watchlists, rules, RuleEvaluator(), engine)
    execution_id = action_execution_id(sample_event.id, rule.id, action.id)
    try:
        await processor.initialize()
        await inspection_alerts.ensure_indexes()
        assert await watchlists.create(entry)
        assert await rules.create(rule)

        first = await processor.process(sample_event)
        second = await processor.process(sample_event)

        assert first.actions_succeeded == 1
        assert second.actions_skipped == 1
        execution = await executions.get(execution_id)
        assert execution.status is ActionExecutionStatus.SUCCEEDED
        assert execution.attempt_count == 1
        alert_page = await inspection_alerts.list(
            AlertQuery(rule_id=rule_id, plate=sample_event.plate.normalized)
        )
        assert len(alert_page.items) == 1
        assert alert_page.items[0].source.execution_id == execution_id
        assert alert_page.items[0].severity.value == "CRITICAL"

        watchlist_index_cursor = await watchlists._collection.list_indexes()
        action_index_cursor = await executions._collection.list_indexes()
        watchlist_indexes = {item["name"] async for item in watchlist_index_cursor}
        action_indexes = {item["name"] async for item in action_index_cursor}
        assert "ix_watchlist_plate_active" in watchlist_indexes
        assert "ix_action_status_updated" in action_indexes
    finally:
        await watchlists._collection.delete_one({"_id": entry_id})
        await rules._collection.delete_one({"_id": rule_id})
        await executions._collection.delete_one({"_id": execution_id})
        await inspection_alerts._collection.delete_many({"rule.id": rule_id})
        await processor.close()
        await inspection_alerts.close()
