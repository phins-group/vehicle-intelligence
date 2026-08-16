"""CRUD application services for watchlists, rules, and alerts."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.ports import (
    AlertPage,
    AlertQuery,
    AlertRepository,
    RuleRepository,
    WatchlistRepository,
)
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.domain import (
    Alert,
    AlertStatus,
    Rule,
    RuleAction,
    RuleCondition,
    WatchlistEntry,
    WatchlistType,
)
from vehicle_intelligence.exceptions import PolicyConflictError, PolicyNotFoundError


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class WatchlistCreate:
    plate: str
    list_type: WatchlistType
    enabled: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True, slots=True)
class WatchlistUpdate:
    revision: int
    plate: str
    list_type: WatchlistType
    enabled: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RuleCreate:
    name: str
    enabled: bool
    priority: int
    conditions: tuple[RuleCondition, ...]
    actions: tuple[RuleAction, ...]
    metadata: dict[str, object] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True, slots=True)
class RuleUpdate:
    revision: int
    name: str
    enabled: bool
    priority: int
    conditions: tuple[RuleCondition, ...]
    actions: tuple[RuleAction, ...]
    metadata: dict[str, object] = field(default_factory=dict)


class WatchlistService:
    def __init__(
        self,
        repository: WatchlistRepository,
        normalizer: VietnamPlateNormalizer,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        id_factory: Callable[[], str] = lambda: _identifier("wle"),
    ) -> None:
        self._repository = repository
        self._normalizer = normalizer
        self._clock = clock
        self._id_factory = id_factory

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def close(self) -> None:
        await self._repository.close()

    async def create(self, command: WatchlistCreate) -> WatchlistEntry:
        now = self._now()
        entry = WatchlistEntry(
            id=(command.id or self._id_factory()).strip(),
            plate=self._plate(command.plate),
            list_type=command.list_type,
            enabled=command.enabled,
            valid_from=command.valid_from,
            valid_until=command.valid_until,
            metadata=dict(command.metadata),
            created_at=now,
            updated_at=now,
        )
        if not await self._repository.create(entry):
            raise PolicyConflictError(f"watchlist entry already exists: {entry.id}")
        return entry

    async def update(self, entry_id: str, command: WatchlistUpdate) -> WatchlistEntry:
        current = await self._required(entry_id)
        if current.revision != command.revision:
            raise PolicyConflictError(
                f"watchlist revision conflict: expected {command.revision}, "
                f"current {current.revision}"
            )
        updated = WatchlistEntry(
            id=current.id,
            plate=self._plate(command.plate),
            list_type=command.list_type,
            enabled=command.enabled,
            valid_from=command.valid_from,
            valid_until=command.valid_until,
            metadata=dict(command.metadata),
            schema_version=current.schema_version,
            revision=current.revision + 1,
            created_at=current.created_at,
            updated_at=self._now(),
        )
        if not await self._repository.replace(updated, current.revision):
            raise PolicyConflictError(f"watchlist entry was concurrently updated: {entry_id}")
        return updated

    async def get(self, entry_id: str) -> WatchlistEntry:
        return await self._required(entry_id)

    async def list(
        self,
        list_type: WatchlistType | None = None,
        enabled: bool | None = None,
        limit: int = 200,
    ) -> list[WatchlistEntry]:
        return await self._repository.list(list_type, enabled, limit)

    async def delete(self, entry_id: str) -> None:
        if not await self._repository.delete(entry_id):
            raise PolicyNotFoundError(f"watchlist entry not found: {entry_id}")

    async def _required(self, entry_id: str) -> WatchlistEntry:
        entry = await self._repository.get(entry_id)
        if entry is None:
            raise PolicyNotFoundError(f"watchlist entry not found: {entry_id}")
        return entry

    def _plate(self, raw: str) -> str:
        result = self._normalizer.normalize(raw)
        if not result.valid or result.normalized is None:
            raise ValueError("invalid Vietnamese plate format")
        return result.normalized

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("watchlist service clock must be timezone-aware")
        return value.astimezone(UTC)


class RuleService:
    def __init__(
        self,
        repository: RuleRepository,
        evaluator: RuleEvaluator,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        id_factory: Callable[[], str] = lambda: _identifier("rule"),
    ) -> None:
        self._repository = repository
        self._evaluator = evaluator
        self._clock = clock
        self._id_factory = id_factory

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def close(self) -> None:
        await self._repository.close()

    async def create(self, command: RuleCreate) -> Rule:
        now = self._now()
        rule = Rule(
            id=(command.id or self._id_factory()).strip(),
            name=command.name.strip(),
            enabled=command.enabled,
            priority=command.priority,
            conditions=command.conditions,
            actions=command.actions,
            metadata=dict(command.metadata),
            created_at=now,
            updated_at=now,
        )
        self._evaluator.validate(rule)
        if not await self._repository.create(rule):
            raise PolicyConflictError(f"rule already exists: {rule.id}")
        return rule

    async def update(self, rule_id: str, command: RuleUpdate) -> Rule:
        current = await self._required(rule_id)
        if current.revision != command.revision:
            raise PolicyConflictError(
                f"rule revision conflict: expected {command.revision}, current {current.revision}"
            )
        updated = Rule(
            id=current.id,
            name=command.name.strip(),
            enabled=command.enabled,
            priority=command.priority,
            conditions=command.conditions,
            actions=command.actions,
            metadata=dict(command.metadata),
            schema_version=current.schema_version,
            revision=current.revision + 1,
            created_at=current.created_at,
            updated_at=self._now(),
        )
        self._evaluator.validate(updated)
        if not await self._repository.replace(updated, current.revision):
            raise PolicyConflictError(f"rule was concurrently updated: {rule_id}")
        return updated

    async def get(self, rule_id: str) -> Rule:
        return await self._required(rule_id)

    async def list(self, enabled_only: bool = False, limit: int = 200) -> list[Rule]:
        return await self._repository.list(enabled_only, limit)

    async def delete(self, rule_id: str) -> None:
        if not await self._repository.delete(rule_id):
            raise PolicyNotFoundError(f"rule not found: {rule_id}")

    async def _required(self, rule_id: str) -> Rule:
        rule = await self._repository.get(rule_id)
        if rule is None:
            raise PolicyNotFoundError(f"rule not found: {rule_id}")
        return rule

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("rule service clock must be timezone-aware")
        return value.astimezone(UTC)


class AlertService:
    def __init__(
        self,
        repository: AlertRepository,
        normalizer: VietnamPlateNormalizer,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._normalizer = normalizer
        self._clock = clock

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def close(self) -> None:
        await self._repository.close()

    async def get(self, alert_id: str) -> Alert:
        alert = await self._repository.get(alert_id)
        if alert is None:
            raise PolicyNotFoundError(f"alert not found: {alert_id}")
        return alert

    async def list(self, query: AlertQuery) -> AlertPage:
        normalized = query.plate
        if normalized is not None:
            result = self._normalizer.normalize(normalized)
            if not result.valid or result.normalized is None:
                raise ValueError("invalid Vietnamese plate format")
            normalized = result.normalized
        return await self._repository.list(replace(query, plate=normalized))

    async def acknowledge(self, alert_id: str, actor_id: str) -> Alert:
        return await self._transition(alert_id, AlertStatus.ACKNOWLEDGED, actor_id)

    async def resolve(self, alert_id: str, actor_id: str) -> Alert:
        return await self._transition(alert_id, AlertStatus.RESOLVED, actor_id)

    async def _transition(self, alert_id: str, target: AlertStatus, actor_id: str) -> Alert:
        actor = actor_id.strip()
        if not actor:
            raise ValueError("alert transition actor is required")
        current = await self.get(alert_id)
        if current.status is target:
            return current
        if current.status is AlertStatus.RESOLVED:
            raise PolicyConflictError("resolved alert cannot transition")
        now = self._now()
        updated = replace(
            current,
            status=target,
            acknowledged_at=(
                now if target is AlertStatus.ACKNOWLEDGED else current.acknowledged_at
            ),
            acknowledged_by=(
                actor if target is AlertStatus.ACKNOWLEDGED else current.acknowledged_by
            ),
            resolved_at=now if target is AlertStatus.RESOLVED else None,
            resolved_by=actor if target is AlertStatus.RESOLVED else None,
            revision=current.revision + 1,
            updated_at=now,
        )
        if not await self._repository.replace(updated, current.revision):
            raise PolicyConflictError(f"alert was concurrently updated: {alert_id}")
        return updated

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("alert service clock must be timezone-aware")
        return value.astimezone(UTC)


@dataclass(slots=True)
class PolicyServices:
    watchlists: WatchlistService
    rules: RuleService
    alerts: AlertService

    async def initialize(self) -> None:
        await self.watchlists.initialize()
        await self.rules.initialize()
        await self.alerts.initialize()

    async def close(self) -> None:
        try:
            await self.watchlists.close()
        finally:
            try:
                await self.rules.close()
            finally:
                await self.alerts.close()
