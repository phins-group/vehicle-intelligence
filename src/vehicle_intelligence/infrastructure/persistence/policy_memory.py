"""In-memory policy repositories for deterministic tests and local API use."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime

from vehicle_intelligence.application.ports import AlertPage, AlertQuery
from vehicle_intelligence.domain import (
    ActionExecution,
    ActionExecutionStatus,
    Alert,
    Rule,
    WatchlistEntry,
    WatchlistType,
)
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor


class InMemoryWatchlistRepository:
    def __init__(self) -> None:
        self._entries: dict[str, WatchlistEntry] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def create(self, entry: WatchlistEntry) -> bool:
        async with self._lock:
            if entry.id in self._entries:
                return False
            self._entries[entry.id] = entry
            return True

    async def replace(self, entry: WatchlistEntry, expected_revision: int) -> bool:
        if entry.revision != expected_revision + 1:
            raise ValueError("replacement watchlist revision must increment by one")
        async with self._lock:
            current = self._entries.get(entry.id)
            if current is None or current.revision != expected_revision:
                return False
            self._entries[entry.id] = entry
            return True

    async def get(self, entry_id: str) -> WatchlistEntry | None:
        return self._entries.get(entry_id)

    async def list(
        self,
        list_type: WatchlistType | None = None,
        enabled: bool | None = None,
        limit: int = 200,
    ) -> list[WatchlistEntry]:
        entries = [
            entry
            for entry in self._entries.values()
            if (list_type is None or entry.list_type is list_type)
            and (enabled is None or entry.enabled is enabled)
        ]
        entries.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
        return entries[:limit]

    async def find_active_by_plate(
        self, plate: str, timestamp: datetime
    ) -> list[WatchlistEntry]:
        entries = [
            entry
            for entry in self._entries.values()
            if entry.plate == plate and entry.is_active_at(timestamp)
        ]
        entries.sort(key=lambda item: (item.list_type.value, item.id))
        return entries

    async def delete(self, entry_id: str) -> bool:
        async with self._lock:
            return self._entries.pop(entry_id, None) is not None

    async def close(self) -> None:
        return None


class InMemoryRuleRepository:
    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def create(self, rule: Rule) -> bool:
        async with self._lock:
            if rule.id in self._rules:
                return False
            self._rules[rule.id] = rule
            return True

    async def replace(self, rule: Rule, expected_revision: int) -> bool:
        if rule.revision != expected_revision + 1:
            raise ValueError("replacement rule revision must increment by one")
        async with self._lock:
            current = self._rules.get(rule.id)
            if current is None or current.revision != expected_revision:
                return False
            self._rules[rule.id] = rule
            return True

    async def get(self, rule_id: str) -> Rule | None:
        return self._rules.get(rule_id)

    async def list(self, enabled_only: bool = False, limit: int = 200) -> list[Rule]:
        rules = [rule for rule in self._rules.values() if not enabled_only or rule.enabled]
        rules.sort(key=lambda item: (-item.priority, item.name.casefold(), item.id))
        return rules[:limit]

    async def delete(self, rule_id: str) -> bool:
        async with self._lock:
            return self._rules.pop(rule_id, None) is not None

    async def close(self) -> None:
        return None


class InMemoryAlertRepository:
    def __init__(self) -> None:
        self._alerts: dict[str, Alert] = {}
        self._execution_ids: set[str] = set()
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def create(self, alert: Alert) -> bool:
        async with self._lock:
            if alert.id in self._alerts or alert.source.execution_id in self._execution_ids:
                return False
            self._alerts[alert.id] = alert
            self._execution_ids.add(alert.source.execution_id)
            return True

    async def replace(self, alert: Alert, expected_revision: int) -> bool:
        if alert.revision != expected_revision + 1:
            raise ValueError("replacement alert revision must increment by one")
        async with self._lock:
            current = self._alerts.get(alert.id)
            if current is None or current.revision != expected_revision:
                return False
            self._alerts[alert.id] = alert
            return True

    async def get(self, alert_id: str) -> Alert | None:
        return self._alerts.get(alert_id)

    async def list(self, query: AlertQuery) -> AlertPage:
        alerts = [alert for alert in self._alerts.values() if self._matches(alert, query)]
        alerts.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            alerts = [
                alert
                for alert in alerts
                if (alert.created_at, alert.id) < (cursor_time, cursor_id)
            ]
        page = alerts[: query.limit + 1]
        has_more = len(page) > query.limit
        page = page[: query.limit]
        next_cursor = (
            encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None
        )
        return AlertPage(tuple(page), next_cursor)

    async def close(self) -> None:
        return None

    @staticmethod
    def _matches(alert: Alert, query: AlertQuery) -> bool:
        return not any(
            (
                query.status is not None and alert.status is not query.status,
                query.plate is not None and alert.plate != query.plate,
                query.camera_id is not None and alert.camera.id != query.camera_id,
                query.rule_id is not None and alert.rule.id != query.rule_id,
            )
        )


class InMemoryActionExecutionRepository:
    def __init__(self) -> None:
        self._executions: dict[str, ActionExecution] = {}
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def claim(
        self,
        execution: ActionExecution,
        stale_before: datetime,
        maximum_attempts: int,
    ) -> ActionExecution | None:
        async with self._lock:
            current = self._executions.get(execution.id)
            if current is None:
                self._executions[execution.id] = execution
                return execution
            if current.status is ActionExecutionStatus.SUCCEEDED:
                return None
            if current.attempt_count >= maximum_attempts:
                return None
            if (
                current.status is ActionExecutionStatus.RUNNING
                and current.updated_at >= stale_before
            ):
                return None
            claimed = replace(
                current,
                status=ActionExecutionStatus.RUNNING,
                attempt_count=current.attempt_count + 1,
                updated_at=execution.updated_at,
                completed_at=None,
                error_code=None,
            )
            self._executions[execution.id] = claimed
            return claimed

    async def get(self, execution_id: str) -> ActionExecution | None:
        return self._executions.get(execution_id)

    async def mark_succeeded(self, execution_id: str, timestamp: datetime) -> None:
        async with self._lock:
            current = self._required(execution_id)
            self._executions[execution_id] = replace(
                current,
                status=ActionExecutionStatus.SUCCEEDED,
                completed_at=timestamp,
                updated_at=timestamp,
                error_code=None,
            )

    async def mark_failed(
        self,
        execution_id: str,
        error_code: str,
        timestamp: datetime,
        *,
        terminal: bool,
        maximum_attempts: int,
        consume_attempt: bool = True,
    ) -> None:
        async with self._lock:
            current = self._required(execution_id)
            self._executions[execution_id] = replace(
                current,
                status=ActionExecutionStatus.FAILED,
                attempt_count=(
                    maximum_attempts
                    if terminal
                    else max(0, current.attempt_count - (0 if consume_attempt else 1))
                ),
                completed_at=timestamp,
                updated_at=timestamp,
                error_code=error_code,
            )

    async def close(self) -> None:
        return None

    def _required(self, execution_id: str) -> ActionExecution:
        execution = self._executions.get(execution_id)
        if execution is None:
            raise PersistenceError(f"action execution not found: {execution_id}")
        return execution
