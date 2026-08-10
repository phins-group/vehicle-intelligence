"""Pydantic request/response schemas for policy and alert APIs."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vehicle_intelligence.application.policies import (
    RuleCreate,
    RuleUpdate,
    WatchlistCreate,
    WatchlistUpdate,
)
from vehicle_intelligence.domain import (
    Alert,
    AlertSeverity,
    AlertStatus,
    Direction,
    EventType,
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    WatchlistEntry,
    WatchlistType,
)


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


def _action_id() -> str:
    return f"action_{uuid.uuid4().hex}"


class WatchlistCreateRequest(APIModel):
    id: str | None = Field(default=None, min_length=1, max_length=128)
    plate: str = Field(min_length=4, max_length=32)
    list_type: WatchlistType = Field(alias="listType")
    enabled: bool = True
    valid_from: datetime | None = Field(default=None, alias="validFrom")
    valid_until: datetime | None = Field(default=None, alias="validUntil")
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("valid_from", "valid_until")
    @classmethod
    def validate_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("watchlist validity timestamps must include a timezone")
        return value

    def to_command(self) -> WatchlistCreate:
        return WatchlistCreate(
            id=self.id,
            plate=self.plate,
            list_type=self.list_type,
            enabled=self.enabled,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            metadata=self.metadata,
        )


class WatchlistUpdateRequest(WatchlistCreateRequest):
    id: None = Field(default=None, exclude=True)
    revision: int = Field(ge=1)

    def to_command(self) -> WatchlistUpdate:
        return WatchlistUpdate(
            revision=self.revision,
            plate=self.plate,
            list_type=self.list_type,
            enabled=self.enabled,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            metadata=self.metadata,
        )


class WatchlistPublic(APIModel):
    id: str
    schema_version: int = Field(alias="schemaVersion")
    revision: int
    plate: str
    list_type: WatchlistType = Field(alias="listType")
    enabled: bool
    valid_from: datetime | None = Field(alias="validFrom")
    valid_until: datetime | None = Field(alias="validUntil")
    metadata: dict[str, object]
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, entry: WatchlistEntry) -> WatchlistPublic:
        return cls(
            id=entry.id,
            schemaVersion=entry.schema_version,
            revision=entry.revision,
            plate=entry.plate,
            listType=entry.list_type,
            enabled=entry.enabled,
            validFrom=entry.valid_from,
            validUntil=entry.valid_until,
            metadata=entry.metadata,
            createdAt=entry.created_at,
            updatedAt=entry.updated_at,
        )


class WatchlistListPublic(APIModel):
    items: list[WatchlistPublic]


class RuleConditionInput(APIModel):
    field: str = Field(min_length=1, max_length=64)
    operator: RuleConditionOperator
    value: object

    def to_domain(self) -> RuleCondition:
        return RuleCondition(field=self.field, operator=self.operator, value=self.value)


class RuleActionInput(APIModel):
    id: str = Field(default_factory=_action_id, min_length=1, max_length=128)
    type: RuleActionType
    parameters: dict[str, object] = Field(default_factory=dict)

    def to_domain(self) -> RuleAction:
        return RuleAction(id=self.id, type=self.type, parameters=self.parameters)


class RuleCreateRequest(APIModel):
    id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    enabled: bool = True
    priority: int = Field(default=0, ge=-10000, le=10000)
    conditions: list[RuleConditionInput] = Field(min_length=1, max_length=32)
    actions: list[RuleActionInput] = Field(min_length=1, max_length=16)
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_command(self) -> RuleCreate:
        return RuleCreate(
            id=self.id,
            name=self.name,
            enabled=self.enabled,
            priority=self.priority,
            conditions=tuple(item.to_domain() for item in self.conditions),
            actions=tuple(item.to_domain() for item in self.actions),
            metadata=self.metadata,
        )


class RuleUpdateRequest(RuleCreateRequest):
    id: None = Field(default=None, exclude=True)
    revision: int = Field(ge=1)

    def to_command(self) -> RuleUpdate:
        return RuleUpdate(
            revision=self.revision,
            name=self.name,
            enabled=self.enabled,
            priority=self.priority,
            conditions=tuple(item.to_domain() for item in self.conditions),
            actions=tuple(item.to_domain() for item in self.actions),
            metadata=self.metadata,
        )


class RulePublic(APIModel):
    id: str
    schema_version: int = Field(alias="schemaVersion")
    revision: int
    name: str
    enabled: bool
    priority: int
    conditions: list[RuleConditionInput]
    actions: list[RuleActionInput]
    metadata: dict[str, object]
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, rule: Rule) -> RulePublic:
        return cls(
            id=rule.id,
            schemaVersion=rule.schema_version,
            revision=rule.revision,
            name=rule.name,
            enabled=rule.enabled,
            priority=rule.priority,
            conditions=[
                RuleConditionInput(
                    field=item.field,
                    operator=item.operator,
                    value=item.value,
                )
                for item in rule.conditions
            ],
            actions=[
                RuleActionInput(
                    id=item.id,
                    type=item.type,
                    parameters=item.parameters,
                )
                for item in rule.actions
            ],
            metadata=rule.metadata,
            createdAt=rule.created_at,
            updatedAt=rule.updated_at,
        )


class RuleListPublic(APIModel):
    items: list[RulePublic]


class AlertSourcePublic(APIModel):
    event_id: str = Field(alias="eventId")
    execution_id: str = Field(alias="executionId")
    action_id: str = Field(alias="actionId")


class AlertRulePublic(APIModel):
    id: str
    name: str


class AlertCameraPublic(APIModel):
    id: str
    name: str
    zone: str | None


class AlertPublic(APIModel):
    id: str
    schema_version: int = Field(alias="schemaVersion")
    revision: int
    source: AlertSourcePublic
    rule: AlertRulePublic
    camera: AlertCameraPublic
    event_type: EventType = Field(alias="eventType")
    direction: Direction
    severity: AlertSeverity
    status: AlertStatus
    message: str
    plate: str | None
    vehicle_type: str | None = Field(alias="vehicleType")
    occurred_at: datetime = Field(alias="occurredAt")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    acknowledged_at: datetime | None = Field(alias="acknowledgedAt")
    acknowledged_by: str | None = Field(alias="acknowledgedBy")
    resolved_at: datetime | None = Field(alias="resolvedAt")
    resolved_by: str | None = Field(alias="resolvedBy")
    metadata: dict[str, object]

    @classmethod
    def from_domain(cls, alert: Alert) -> AlertPublic:
        return cls(
            id=alert.id,
            schemaVersion=alert.schema_version,
            revision=alert.revision,
            source=AlertSourcePublic(
                eventId=alert.source.event_id,
                executionId=alert.source.execution_id,
                actionId=alert.source.action_id,
            ),
            rule=AlertRulePublic(id=alert.rule.id, name=alert.rule.name),
            camera=AlertCameraPublic(
                id=alert.camera.id,
                name=alert.camera.name,
                zone=alert.camera.zone,
            ),
            eventType=alert.event_type,
            direction=alert.direction,
            severity=alert.severity,
            status=alert.status,
            message=alert.message,
            plate=alert.plate,
            vehicleType=alert.vehicle_type,
            occurredAt=alert.occurred_at,
            createdAt=alert.created_at,
            updatedAt=alert.updated_at,
            acknowledgedAt=alert.acknowledged_at,
            acknowledgedBy=alert.acknowledged_by,
            resolvedAt=alert.resolved_at,
            resolvedBy=alert.resolved_by,
            metadata=alert.metadata,
        )


class AlertListPublic(APIModel):
    items: list[AlertPublic]
    next_cursor: str | None = Field(alias="nextCursor")


class AlertTransitionRequest(APIModel):
    actor_id: str | None = Field(
        default=None,
        alias="actorId",
        min_length=1,
        max_length=128,
        description="Deprecated consistency check; identity comes from the Bearer principal.",
    )
