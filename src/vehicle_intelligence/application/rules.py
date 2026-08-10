"""Pure declarative rule validation and evaluation."""

from __future__ import annotations

from urllib.parse import urlsplit

from vehicle_intelligence.domain import (
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    VehicleEvent,
    WatchlistEntry,
)
from vehicle_intelligence.exceptions import RuleValidationError

SUPPORTED_RULE_FIELDS = frozenset(
    {
        "watchlist",
        "camera.id",
        "camera.zone",
        "direction",
        "eventType",
        "status",
        "plate.normalized",
        "vehicle.type",
        "vehicle.color",
    }
)
EXTERNAL_ACTIONS = frozenset(
    {
        RuleActionType.OPEN_BARRIER,
        RuleActionType.WEBHOOK,
        RuleActionType.HTTP_REQUEST,
        RuleActionType.NOTIFICATION,
    }
)


class RuleEvaluator:
    def validate(self, rule: Rule) -> None:
        for condition in rule.conditions:
            self._validate_condition(condition)
        for action in rule.actions:
            self._validate_action(action)

    def matches(
        self,
        rule: Rule,
        event: VehicleEvent,
        watchlists: tuple[WatchlistEntry, ...],
    ) -> bool:
        if not rule.enabled:
            return False
        context: dict[str, object] = {
            "watchlist": tuple(entry.list_type.value for entry in watchlists),
            "camera.id": event.camera.id,
            "camera.zone": event.camera.zone,
            "direction": event.direction.value,
            "eventType": event.event_type.value,
            "status": event.status.value,
            "plate.normalized": event.plate.final_normalized if event.plate else None,
            "vehicle.type": event.vehicle.type,
            "vehicle.color": event.vehicle.color,
        }
        return all(
            self._matches_condition(context[condition.field], condition)
            for condition in rule.conditions
        )

    @staticmethod
    def _matches_condition(actual: object, condition: RuleCondition) -> bool:
        expected = condition.value
        operator = condition.operator
        if operator is RuleConditionOperator.EQ:
            return actual == expected
        if operator is RuleConditionOperator.NEQ:
            return actual != expected
        if operator is RuleConditionOperator.IN:
            return actual in expected  # type: ignore[operator]
        if operator is RuleConditionOperator.NOT_IN:
            return actual not in expected  # type: ignore[operator]
        if operator is RuleConditionOperator.CONTAINS:
            return expected in actual  # type: ignore[operator]
        if operator is RuleConditionOperator.EXISTS:
            return (actual is not None) is expected
        raise RuleValidationError(f"unsupported rule operator: {operator}")

    @staticmethod
    def _validate_condition(condition: RuleCondition) -> None:
        if condition.field not in SUPPORTED_RULE_FIELDS:
            raise RuleValidationError(f"unsupported rule field: {condition.field}")
        if condition.operator in {
            RuleConditionOperator.IN,
            RuleConditionOperator.NOT_IN,
        } and (not isinstance(condition.value, (list, tuple)) or not condition.value):
            raise RuleValidationError(f"{condition.operator} requires a non-empty list")
        if condition.operator is RuleConditionOperator.CONTAINS:
            if condition.field != "watchlist":
                raise RuleValidationError("CONTAINS is only supported for watchlist")
            if not isinstance(condition.value, str):
                raise RuleValidationError("watchlist CONTAINS requires a string value")
        if condition.operator is RuleConditionOperator.EXISTS and not isinstance(
            condition.value, bool
        ):
            raise RuleValidationError("EXISTS requires a boolean value")

    @staticmethod
    def _validate_action(action: RuleAction) -> None:
        parameters = action.parameters
        if action.type in EXTERNAL_ACTIONS:
            url = parameters.get("url")
            if not isinstance(url, str):
                raise RuleValidationError(f"{action.type} requires parameters.url")
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise RuleValidationError("external action URL must use http or https")
            if parsed.username is not None or parsed.password is not None:
                raise RuleValidationError("external action URL cannot contain credentials")
            method = parameters.get("method", "POST")
            if method not in {"GET", "POST", "PUT", "PATCH"}:
                raise RuleValidationError("external action method is not allowed")
        if action.type is RuleActionType.CREATE_ALERT:
            message = parameters.get("message")
            if message is not None and (not isinstance(message, str) or not message.strip()):
                raise RuleValidationError("CREATE_ALERT message must be a non-empty string")
            severity = parameters.get("severity")
            if severity is not None and severity not in {
                "INFO",
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL",
            }:
                raise RuleValidationError("CREATE_ALERT severity is invalid")
