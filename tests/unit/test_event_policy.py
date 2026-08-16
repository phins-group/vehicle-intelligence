import asyncio

from vehicle_intelligence.application.actions import ActionEngine, AlertActionHandler
from vehicle_intelligence.application.event_policy import VehicleEventPolicyProcessor
from vehicle_intelligence.application.ports import AlertQuery
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.config import RuleEngineConfig
from vehicle_intelligence.domain import (
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    WatchlistEntry,
    WatchlistType,
)
from vehicle_intelligence.infrastructure.persistence.policy_memory import (
    InMemoryActionExecutionRepository,
    InMemoryAlertRepository,
    InMemoryRuleRepository,
    InMemoryWatchlistRepository,
)


class CountingRuleRepository(InMemoryRuleRepository):
    def __init__(self) -> None:
        super().__init__()
        self.list_calls = 0

    async def list(self, enabled_only: bool = False, limit: int = 200) -> list[Rule]:
        self.list_calls += 1
        return await super().list(enabled_only=enabled_only, limit=limit)


async def test_event_policy_matches_watchlist_and_creates_one_idempotent_alert(
    sample_event,
) -> None:
    watchlists = InMemoryWatchlistRepository()
    rules = CountingRuleRepository()
    alerts = InMemoryAlertRepository()
    executions = InMemoryActionExecutionRepository()
    timestamp = sample_event.occurred_at
    await watchlists.create(
        WatchlistEntry(
            id="wle-blacklist",
            plate=sample_event.plate.normalized,
            list_type=WatchlistType.BLACKLIST,
            enabled=True,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    await rules.create(
        Rule(
            id="rule-blacklist",
            name="Blacklist alert",
            enabled=True,
            priority=100,
            conditions=(
                RuleCondition(
                    "watchlist",
                    RuleConditionOperator.CONTAINS,
                    "BLACKLIST",
                ),
                RuleCondition("direction", RuleConditionOperator.EQ, "ENTER"),
            ),
            actions=(
                RuleAction(
                    "create-alert",
                    RuleActionType.CREATE_ALERT,
                    {"severity": "CRITICAL"},
                ),
            ),
            created_at=timestamp,
            updated_at=timestamp,
        )
    )
    engine = ActionEngine(
        executions,
        {RuleActionType.CREATE_ALERT: AlertActionHandler(alerts)},
        RuleEngineConfig(),
        clock=lambda: timestamp,
    )
    now = [0.0]
    processor = VehicleEventPolicyProcessor(
        watchlists,
        rules,
        RuleEvaluator(),
        engine,
        cache_ttl_seconds=2.0,
        monotonic_clock=lambda: now[0],
    )
    await processor.initialize()

    first = await processor.process(sample_event)
    second = await processor.process(sample_event)

    assert first.matched_rules == 1
    assert first.actions_succeeded == 1
    assert second.matched_rules == 1
    assert second.actions_skipped == 1
    assert rules.list_calls == 1
    now[0] = 2.0
    await asyncio.gather(*(processor._active_rules() for _ in range(8)))
    assert rules.list_calls == 2
    page = await alerts.list(AlertQuery())
    assert len(page.items) == 1
    assert page.items[0].plate == "51H-123.45"
    await processor.close()
