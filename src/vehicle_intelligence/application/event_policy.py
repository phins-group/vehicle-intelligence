"""Apply active watchlists and rules to one canonical vehicle event."""

from __future__ import annotations

from vehicle_intelligence.application.actions import ActionEngine
from vehicle_intelligence.application.ports import (
    EventPolicyResult,
    RuleRepository,
    WatchlistRepository,
)
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.domain import VehicleEvent, WatchlistEntry
from vehicle_intelligence.exceptions import RuleValidationError


class VehicleEventPolicyProcessor:
    def __init__(
        self,
        watchlists: WatchlistRepository,
        rules: RuleRepository,
        evaluator: RuleEvaluator,
        actions: ActionEngine,
        maximum_rules: int = 1000,
    ) -> None:
        if maximum_rules < 1:
            raise ValueError("event policy maximum_rules must be positive")
        self._watchlists = watchlists
        self._rules = rules
        self._evaluator = evaluator
        self._actions = actions
        self._maximum_rules = maximum_rules

    async def initialize(self) -> None:
        await self._watchlists.ensure_indexes()
        await self._rules.ensure_indexes()
        await self._actions.initialize()

    async def close(self) -> None:
        try:
            await self._watchlists.close()
        finally:
            try:
                await self._rules.close()
            finally:
                await self._actions.close()

    async def process(self, event: VehicleEvent) -> EventPolicyResult:
        watchlists: tuple[WatchlistEntry, ...] = ()
        if event.plate is not None:
            watchlists = tuple(
                await self._watchlists.find_active_by_plate(
                    event.plate.final_normalized,
                    event.occurred_at,
                )
            )
        rules = await self._rules.list(
            enabled_only=True,
            limit=self._maximum_rules + 1,
        )
        if len(rules) > self._maximum_rules:
            raise RuleValidationError("active rule count exceeds configured evaluation limit")
        matched_rules = 0
        actions_succeeded = 0
        actions_skipped = 0
        for rule in rules:
            self._evaluator.validate(rule)
            if not self._evaluator.matches(rule, event, watchlists):
                continue
            matched_rules += 1
            for action in rule.actions:
                executed = await self._actions.execute(event, rule, action, watchlists)
                if executed:
                    actions_succeeded += 1
                else:
                    actions_skipped += 1
        return EventPolicyResult(
            matched_rules=matched_rules,
            actions_succeeded=actions_succeeded,
            actions_skipped=actions_skipped,
        )
