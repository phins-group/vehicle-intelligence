from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vehicle_intelligence.domain.enums import (
    ActionExecutionStatus,
    AlertSeverity,
    AlertStatus,
    Direction,
    EventType,
    RuleActionType,
    RuleConditionOperator,
    WatchlistType,
)
from vehicle_intelligence.domain.events import CameraSnapshot


def _require_aware(value: datetime | None, field_name: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class WatchlistEntry:
    id: str
    plate: str
    list_type: WatchlistType
    enabled: bool
    created_at: datetime
    updated_at: datetime
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.plate.strip():
            raise ValueError("watchlist id and plate are required")
        for name, value in (
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
            ("valid_from", self.valid_from),
            ("valid_until", self.valid_until),
        ):
            _require_aware(value, name)
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError("watchlist valid_until must be after valid_from")
        if self.schema_version < 1 or self.revision < 1:
            raise ValueError("watchlist schema version and revision must be positive")

    def is_active_at(self, timestamp: datetime) -> bool:
        _require_aware(timestamp, "watchlist evaluation timestamp")
        return bool(
            self.enabled
            and (self.valid_from is None or self.valid_from <= timestamp)
            and (self.valid_until is None or timestamp <= self.valid_until)
        )


@dataclass(frozen=True, slots=True)
class RuleCondition:
    field: str
    operator: RuleConditionOperator
    value: object

    def __post_init__(self) -> None:
        if not self.field.strip():
            raise ValueError("rule condition field is required")


@dataclass(frozen=True, slots=True)
class RuleAction:
    id: str
    type: RuleActionType
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("rule action id is required")


@dataclass(frozen=True, slots=True)
class Rule:
    id: str
    name: str
    enabled: bool
    priority: int
    conditions: tuple[RuleCondition, ...]
    actions: tuple[RuleAction, ...]
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("rule id and name are required")
        if not self.conditions or not self.actions:
            raise ValueError("rule requires at least one condition and one action")
        action_ids = [action.id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("rule action ids must be unique within a rule")
        _require_aware(self.created_at, "rule created_at")
        _require_aware(self.updated_at, "rule updated_at")
        if self.schema_version < 1 or self.revision < 1:
            raise ValueError("rule schema version and revision must be positive")


@dataclass(frozen=True, slots=True)
class AlertSource:
    event_id: str
    execution_id: str
    action_id: str

    def __post_init__(self) -> None:
        if not self.event_id or not self.execution_id or not self.action_id:
            raise ValueError("alert source identifiers are required")


@dataclass(frozen=True, slots=True)
class RuleSnapshot:
    id: str
    name: str


@dataclass(frozen=True, slots=True)
class Alert:
    id: str
    source: AlertSource
    rule: RuleSnapshot
    camera: CameraSnapshot
    event_type: EventType
    direction: Direction
    severity: AlertSeverity
    status: AlertStatus
    message: str
    occurred_at: datetime
    created_at: datetime
    updated_at: datetime
    plate: str | None = None
    vehicle_type: str | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.message.strip():
            raise ValueError("alert id and message are required")
        for name, value in (
            ("occurred_at", self.occurred_at),
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
            ("acknowledged_at", self.acknowledged_at),
            ("resolved_at", self.resolved_at),
        ):
            _require_aware(value, name)
        if self.status is AlertStatus.ACKNOWLEDGED and (
            not self.acknowledged_by or self.acknowledged_at is None
        ):
            raise ValueError("acknowledged alert requires an actor and timestamp")
        if self.status is AlertStatus.RESOLVED and (
            not self.resolved_by or self.resolved_at is None
        ):
            raise ValueError("resolved alert requires an actor and timestamp")
        if self.schema_version < 1 or self.revision < 1:
            raise ValueError("alert schema version and revision must be positive")


@dataclass(frozen=True, slots=True)
class ActionExecution:
    id: str
    event_id: str
    rule_id: str
    action_id: str
    action_type: RuleActionType
    status: ActionExecutionStatus
    attempt_count: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifiers = (self.id, self.event_id, self.rule_id, self.action_id)
        if any(not value.strip() for value in identifiers):
            raise ValueError("action execution identifiers are required")
        if self.attempt_count < 1:
            raise ValueError("action execution attempt_count must be positive")
        _require_aware(self.created_at, "action execution created_at")
        _require_aware(self.updated_at, "action execution updated_at")
        _require_aware(self.completed_at, "action execution completed_at")
        if self.status is ActionExecutionStatus.SUCCEEDED and self.completed_at is None:
            raise ValueError("succeeded action execution requires completed_at")
        if self.schema_version < 1:
            raise ValueError("action execution schema version must be positive")
