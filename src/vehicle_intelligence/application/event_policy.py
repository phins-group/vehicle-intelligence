"""Apply active watchlists and rules to one canonical vehicle event."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from vehicle_intelligence.application.actions import ActionEngine
from vehicle_intelligence.application.ports import (
    EventPolicyResult,
    RuleRepository,
    WatchlistRepository,
)
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.domain import Rule, VehicleEvent, WatchlistEntry
from vehicle_intelligence.exceptions import RuleValidationError


class VehicleEventPolicyProcessor:
    def __init__(
        self,
        watchlists: WatchlistRepository,
        rules: RuleRepository,
        evaluator: RuleEvaluator,
        actions: ActionEngine,
        maximum_rules: int = 1000,
        cache_ttl_seconds: float = 2.0,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if maximum_rules < 1:
            raise ValueError("event policy maximum_rules must be positive")
        if cache_ttl_seconds <= 0:
            raise ValueError("event policy cache TTL must be positive")
        self._watchlists = watchlists
        self._rules = rules
        self._evaluator = evaluator
        self._actions = actions
        self._maximum_rules = maximum_rules
        self._cache_ttl_seconds = cache_ttl_seconds
        self._monotonic = monotonic_clock
        self._cached_rules: tuple[Rule, ...] | None = None
        self._rules_expire_at = 0.0
        self._rules_lock = asyncio.Lock()

    async def initialize(self) -> None:
        await self._watchlists.ensure_indexes()
        await self._rules.ensure_indexes()
        await self._actions.initialize()

    async def close(self) -> None:
        self._cached_rules = None
        self._rules_expire_at = 0.0
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
        rules = await self._active_rules()
        matched_rules = 0
        actions_succeeded = 0
        actions_skipped = 0
        for rule in rules:
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

    async def _active_rules(self) -> tuple[Rule, ...]:
        now = float(self._monotonic())
        if self._cached_rules is not None and now < self._rules_expire_at:
            return self._cached_rules
        async with self._rules_lock:
            now = float(self._monotonic())
            if self._cached_rules is not None and now < self._rules_expire_at:
                return self._cached_rules
            loaded = await self._rules.list(
                enabled_only=True,
                limit=self._maximum_rules + 1,
            )
            if len(loaded) > self._maximum_rules:
                raise RuleValidationError("active rule count exceeds configured evaluation limit")
            rules = tuple(loaded)
            for rule in rules:
                self._evaluator.validate(rule)
            self._cached_rules = rules
            self._rules_expire_at = now + self._cache_ttl_seconds
            return rules
