from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.domain import (
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    WatchlistEntry,
    WatchlistType,
)
from vehicle_intelligence.exceptions import RuleValidationError


def rule(timestamp: datetime, field: str = "watchlist") -> Rule:
    return Rule(
        id="rule-blacklist",
        name="Blacklist entry alert",
        enabled=True,
        priority=100,
        conditions=(
            RuleCondition(field, RuleConditionOperator.CONTAINS, "BLACKLIST"),
            RuleCondition("camera.id", RuleConditionOperator.EQ, "gate-01"),
            RuleCondition("direction", RuleConditionOperator.EQ, "ENTER"),
        ),
        actions=(RuleAction("create-alert", RuleActionType.CREATE_ALERT),),
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_rule_evaluator_matches_watchlist_camera_and_direction(sample_event) -> None:
    timestamp = sample_event.occurred_at
    entry = WatchlistEntry(
        id="wle-blacklist",
        plate="51H-123.45",
        list_type=WatchlistType.BLACKLIST,
        enabled=True,
        valid_from=timestamp - timedelta(days=1),
        valid_until=timestamp + timedelta(days=1),
        created_at=timestamp,
        updated_at=timestamp,
    )
    evaluator = RuleEvaluator()
    configured = rule(timestamp)

    evaluator.validate(configured)
    assert evaluator.matches(configured, sample_event, (entry,))
    assert not evaluator.matches(configured, sample_event, ())
    assert not evaluator.matches(
        configured,
        replace(sample_event, camera=replace(sample_event.camera, id="gate-02")),
        (entry,),
    )


def test_rule_evaluator_rejects_unknown_fields_and_unsafe_actions() -> None:
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    evaluator = RuleEvaluator()
    with pytest.raises(RuleValidationError, match="unsupported rule field"):
        evaluator.validate(rule(timestamp, "metadata.arbitrary"))

    external = replace(
        rule(timestamp),
        actions=(
            RuleAction(
                "webhook",
                RuleActionType.WEBHOOK,
                {"url": "file:///etc/passwd"},
            ),
        ),
    )
    with pytest.raises(RuleValidationError, match="http or https"):
        evaluator.validate(external)


def test_watchlist_temporal_validity_is_inclusive_and_timezone_aware() -> None:
    timestamp = datetime(2026, 8, 9, 12, tzinfo=UTC)
    entry = WatchlistEntry(
        id="wle-temporal",
        plate="51H-123.45",
        list_type=WatchlistType.VIP,
        enabled=True,
        valid_from=timestamp,
        valid_until=timestamp + timedelta(hours=1),
        created_at=timestamp,
        updated_at=timestamp,
    )

    assert entry.is_active_at(timestamp)
    assert entry.is_active_at(timestamp + timedelta(hours=1))
    assert not entry.is_active_at(timestamp - timedelta(microseconds=1))
    assert not entry.is_active_at(timestamp + timedelta(hours=1, microseconds=1))
