"""Portable JSON representation for durable audit outbox entries."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite

from vehicle_intelligence.domain import (
    AuditAction,
    AuditActor,
    AuditLog,
    AuditResourceType,
    AuthenticationMethod,
    UserRole,
)


def audit_log_to_json(entry: AuditLog) -> dict[str, object]:
    return {
        "schemaVersion": entry.schema_version,
        "id": entry.id,
        "actor": {
            "id": entry.actor.id,
            "displayName": entry.actor.display_name,
            "role": entry.actor.role.value,
            "authenticationMethod": entry.actor.authentication_method.value,
        },
        "action": entry.action.value,
        "resource": {
            "type": entry.resource_type.value,
            "id": entry.resource_id,
        },
        "occurredAt": entry.occurred_at.astimezone(UTC).isoformat(),
        "requestId": entry.request_id,
        "before": _json_mapping(entry.before, "before") if entry.before is not None else None,
        "after": _json_mapping(entry.after, "after") if entry.after is not None else None,
        "metadata": _json_mapping(entry.metadata, "metadata"),
    }


def audit_log_from_json(value: object) -> AuditLog:
    if (
        not isinstance(value, dict)
        or isinstance(value.get("schemaVersion"), bool)
        or value.get("schemaVersion") != 1
    ):
        raise ValueError("audit outbox contract is invalid")
    actor = _mapping(value.get("actor"), "actor")
    resource = _mapping(value.get("resource"), "resource")
    occurred_at = datetime.fromisoformat(str(value.get("occurredAt", "")))
    if occurred_at.tzinfo is None:
        raise ValueError("audit outbox timestamp must be timezone-aware")
    before_value = value.get("before")
    after_value = value.get("after")
    return AuditLog(
        id=_required_string(value, "id"),
        schema_version=1,
        actor=AuditActor(
            id=_required_string(actor, "id"),
            display_name=_required_string(actor, "displayName"),
            role=UserRole(_required_string(actor, "role")),
            authentication_method=AuthenticationMethod(
                _required_string(actor, "authenticationMethod")
            ),
        ),
        action=AuditAction(_required_string(value, "action")),
        resource_type=AuditResourceType(_required_string(resource, "type")),
        resource_id=_required_string(resource, "id"),
        occurred_at=occurred_at.astimezone(UTC),
        request_id=_required_string(value, "requestId"),
        before=(
            _json_mapping(_mapping(before_value, "before"), "before")
            if before_value is not None
            else None
        ),
        after=(
            _json_mapping(_mapping(after_value, "after"), "after")
            if after_value is not None
            else None
        ),
        metadata=_json_mapping(_mapping(value.get("metadata"), "metadata"), "metadata"),
    )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"audit outbox {name} must be an object")
    return value


def _required_string(value: dict[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"audit outbox {key} must be a non-empty string")
    return item


def _json_mapping(value: dict[str, object], name: str) -> dict[str, object]:
    _validate_json(value, name)
    return value


def _validate_json(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if isfinite(value):
            return
        raise ValueError(f"audit outbox {name} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _validate_json(item, name)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json(item, name)
        return
    raise ValueError(f"audit outbox {name} is not JSON-compatible")
