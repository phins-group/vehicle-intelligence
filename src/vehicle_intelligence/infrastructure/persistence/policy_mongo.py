"""MongoDB repositories for watchlists, rules, alerts, and action idempotency."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.application.ports import AlertPage, AlertQuery
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    ActionExecution,
    ActionExecutionStatus,
    Alert,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    CameraSnapshot,
    Direction,
    EventType,
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    RuleSnapshot,
    WatchlistEntry,
    WatchlistType,
)
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import (
    ACTION_EXECUTIONS,
    ALERTS,
    RULES,
    WATCHLISTS,
)
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class _MongoRepository:
    def __init__(self, config: MongoConfig | MongoRuntime, collection: str) -> None:
        binding = bind_mongo(config)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[collection]

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()


class MongoWatchlistRepository(_MongoRepository):
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        super().__init__(config, WATCHLISTS)

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [
                            ("plate", ASCENDING),
                            ("enabled", ASCENDING),
                            ("validFrom", ASCENDING),
                            ("validUntil", ASCENDING),
                        ],
                        name="ix_watchlist_plate_active",
                    ),
                    IndexModel(
                        [
                            ("listType", ASCENDING),
                            ("enabled", ASCENDING),
                            ("updatedAt", DESCENDING),
                        ],
                        name="ix_watchlist_type_enabled",
                    ),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB watchlist indexes") from exc

    async def create(self, entry: WatchlistEntry) -> bool:
        try:
            await self._collection.insert_one(_watchlist_to_document(entry))
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot create watchlist entry: {entry.id}") from exc

    async def replace(self, entry: WatchlistEntry, expected_revision: int) -> bool:
        if entry.revision != expected_revision + 1:
            raise ValueError("replacement watchlist revision must increment by one")
        try:
            result = await self._collection.replace_one(
                {"_id": entry.id, "revision": expected_revision},
                _watchlist_to_document(entry),
            )
            return result.matched_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot update watchlist entry: {entry.id}") from exc

    async def get(self, entry_id: str) -> WatchlistEntry | None:
        try:
            document = await self._collection.find_one({"_id": entry_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read watchlist entry: {entry_id}") from exc
        return _document_to_watchlist(document) if document is not None else None

    async def list(
        self,
        list_type: WatchlistType | None = None,
        enabled: bool | None = None,
        limit: int = 200,
    ) -> list[WatchlistEntry]:
        query: dict[str, object] = {}
        if list_type is not None:
            query["listType"] = list_type.value
        if enabled is not None:
            query["enabled"] = enabled
        try:
            cursor = (
                self._collection.find(query)
                .sort([("updatedAt", DESCENDING), ("_id", DESCENDING)])
                .limit(limit)
            )
            return [_document_to_watchlist(document) async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot list watchlist entries") from exc

    async def find_active_by_plate(self, plate: str, timestamp: datetime) -> list[WatchlistEntry]:
        query = {
            "plate": plate,
            "enabled": True,
            "$and": [
                {"$or": [{"validFrom": None}, {"validFrom": {"$lte": timestamp}}]},
                {"$or": [{"validUntil": None}, {"validUntil": {"$gte": timestamp}}]},
            ],
        }
        try:
            cursor = self._collection.find(query).sort(
                [("listType", ASCENDING), ("_id", ASCENDING)]
            )
            return [_document_to_watchlist(document) async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError(f"cannot match watchlists for plate: {plate}") from exc

    async def delete(self, entry_id: str) -> bool:
        try:
            result = await self._collection.delete_one({"_id": entry_id})
            return result.deleted_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot delete watchlist entry: {entry_id}") from exc


class MongoRuleRepository(_MongoRepository):
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        super().__init__(config, RULES)

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [
                            ("enabled", ASCENDING),
                            ("priority", DESCENDING),
                            ("name", ASCENDING),
                            ("_id", ASCENDING),
                        ],
                        name="ix_rules_enabled_priority",
                    ),
                    IndexModel([("updatedAt", DESCENDING)], name="ix_rules_updated"),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB rule indexes") from exc

    async def create(self, rule: Rule) -> bool:
        try:
            await self._collection.insert_one(_rule_to_document(rule))
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot create rule: {rule.id}") from exc

    async def replace(self, rule: Rule, expected_revision: int) -> bool:
        if rule.revision != expected_revision + 1:
            raise ValueError("replacement rule revision must increment by one")
        try:
            result = await self._collection.replace_one(
                {"_id": rule.id, "revision": expected_revision},
                _rule_to_document(rule),
            )
            return result.matched_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot update rule: {rule.id}") from exc

    async def get(self, rule_id: str) -> Rule | None:
        try:
            document = await self._collection.find_one({"_id": rule_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read rule: {rule_id}") from exc
        return _document_to_rule(document) if document is not None else None

    async def list(self, enabled_only: bool = False, limit: int = 200) -> list[Rule]:
        query = {"enabled": True} if enabled_only else {}
        try:
            cursor = (
                self._collection.find(query)
                .sort(
                    [
                        ("priority", DESCENDING),
                        ("name", ASCENDING),
                        ("_id", ASCENDING),
                    ]
                )
                .limit(limit)
            )
            return [_document_to_rule(document) async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot list rules") from exc

    async def delete(self, rule_id: str) -> bool:
        try:
            result = await self._collection.delete_one({"_id": rule_id})
            return result.deleted_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot delete rule: {rule_id}") from exc


class MongoAlertRepository(_MongoRepository):
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        super().__init__(config, ALERTS)

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [("source.executionId", ASCENDING)],
                        unique=True,
                        name="uq_alert_execution",
                    ),
                    IndexModel(
                        [("status", ASCENDING), ("createdAt", DESCENDING), ("_id", DESCENDING)],
                        name="ix_alert_status_cursor",
                    ),
                    IndexModel(
                        [("plate", ASCENDING), ("createdAt", DESCENDING)],
                        name="ix_alert_plate_time",
                        partialFilterExpression={"plate": {"$type": "string"}},
                    ),
                    IndexModel(
                        [("camera.id", ASCENDING), ("createdAt", DESCENDING)],
                        name="ix_alert_camera_time",
                    ),
                    IndexModel(
                        [("rule.id", ASCENDING), ("createdAt", DESCENDING)],
                        name="ix_alert_rule_time",
                    ),
                    IndexModel(
                        [("createdAt", DESCENDING), ("_id", DESCENDING)],
                        name="ix_alert_cursor",
                    ),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB alert indexes") from exc

    async def create(self, alert: Alert) -> bool:
        try:
            await self._collection.insert_one(_alert_to_document(alert))
            return True
        except DuplicateKeyError:
            return False
        except PyMongoError as exc:
            raise PersistenceError(f"cannot create alert: {alert.id}") from exc

    async def replace(self, alert: Alert, expected_revision: int) -> bool:
        if alert.revision != expected_revision + 1:
            raise ValueError("replacement alert revision must increment by one")
        try:
            result = await self._collection.replace_one(
                {"_id": alert.id, "revision": expected_revision},
                _alert_to_document(alert),
            )
            return result.matched_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot update alert: {alert.id}") from exc

    async def get(self, alert_id: str) -> Alert | None:
        try:
            document = await self._collection.find_one({"_id": alert_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read alert: {alert_id}") from exc
        return _document_to_alert(document) if document is not None else None

    async def list(self, query: AlertQuery) -> AlertPage:
        filters: dict[str, Any] = {}
        if query.status is not None:
            filters["status"] = query.status.value
        if query.plate is not None:
            filters["plate"] = query.plate
        if query.camera_id is not None:
            filters["camera.id"] = query.camera_id
        if query.rule_id is not None:
            filters["rule.id"] = query.rule_id
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            filters["$or"] = [
                {"createdAt": {"$lt": cursor_time}},
                {"createdAt": cursor_time, "_id": {"$lt": cursor_id}},
            ]
        try:
            cursor = (
                self._collection.find(filters)
                .sort([("createdAt", DESCENDING), ("_id", DESCENDING)])
                .limit(query.limit + 1)
            )
            documents = [document async for document in cursor]
        except PyMongoError as exc:
            raise PersistenceError("cannot list alerts") from exc
        has_more = len(documents) > query.limit
        alerts = tuple(_document_to_alert(document) for document in documents[: query.limit])
        next_cursor = (
            encode_cursor(alerts[-1].created_at, alerts[-1].id) if has_more and alerts else None
        )
        return AlertPage(alerts, next_cursor)


class MongoActionExecutionRepository(_MongoRepository):
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        super().__init__(config, ACTION_EXECUTIONS)

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [("status", ASCENDING), ("updatedAt", ASCENDING)],
                        name="ix_action_status_updated",
                    ),
                    IndexModel(
                        [("eventId", ASCENDING), ("ruleId", ASCENDING)],
                        name="ix_action_event_rule",
                    ),
                ]
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize action-execution indexes") from exc

    async def claim(
        self,
        execution: ActionExecution,
        stale_before: datetime,
        maximum_attempts: int,
    ) -> ActionExecution | None:
        try:
            await self._collection.insert_one(_action_execution_to_document(execution))
            return execution
        except DuplicateKeyError:
            pass
        except PyMongoError as exc:
            raise PersistenceError(f"cannot claim action execution: {execution.id}") from exc

        claim_filter = {
            "_id": execution.id,
            "attemptCount": {"$lt": maximum_attempts},
            "$or": [
                {"status": ActionExecutionStatus.FAILED.value},
                {
                    "status": ActionExecutionStatus.RUNNING.value,
                    "updatedAt": {"$lt": stale_before},
                },
            ],
        }
        try:
            document = await self._collection.find_one_and_update(
                claim_filter,
                {
                    "$set": {
                        "status": ActionExecutionStatus.RUNNING.value,
                        "updatedAt": execution.updated_at,
                        "completedAt": None,
                        "errorCode": None,
                    },
                    "$inc": {"attemptCount": 1},
                },
                return_document=ReturnDocument.AFTER,
            )
        except PyMongoError as exc:
            raise PersistenceError(f"cannot reclaim action execution: {execution.id}") from exc
        return _document_to_action_execution(document) if document is not None else None

    async def get(self, execution_id: str) -> ActionExecution | None:
        try:
            document = await self._collection.find_one({"_id": execution_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read action execution: {execution_id}") from exc
        return _document_to_action_execution(document) if document is not None else None

    async def mark_succeeded(self, execution_id: str, timestamp: datetime) -> None:
        try:
            result = await self._collection.update_one(
                {"_id": execution_id, "status": ActionExecutionStatus.RUNNING.value},
                {
                    "$set": {
                        "status": ActionExecutionStatus.SUCCEEDED.value,
                        "completedAt": timestamp,
                        "updatedAt": timestamp,
                        "errorCode": None,
                    }
                },
            )
        except PyMongoError as exc:
            raise PersistenceError(f"cannot complete action execution: {execution_id}") from exc
        if result.matched_count != 1:
            raise PersistenceError(f"running action execution not found: {execution_id}")

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
        update: dict[str, Any] = {
            "$set": {
                "status": ActionExecutionStatus.FAILED.value,
                "completedAt": timestamp,
                "updatedAt": timestamp,
                "errorCode": error_code,
            }
        }
        if terminal:
            update["$set"]["attemptCount"] = maximum_attempts
        elif not consume_attempt:
            update["$inc"] = {"attemptCount": -1}
        try:
            result = await self._collection.update_one(
                {"_id": execution_id, "status": ActionExecutionStatus.RUNNING.value},
                update,
            )
        except PyMongoError as exc:
            raise PersistenceError(f"cannot fail action execution: {execution_id}") from exc
        if result.matched_count != 1:
            raise PersistenceError(f"running action execution not found: {execution_id}")


def _watchlist_to_document(entry: WatchlistEntry) -> dict[str, Any]:
    return {
        "_id": entry.id,
        "schemaVersion": entry.schema_version,
        "revision": entry.revision,
        "plate": entry.plate,
        "listType": entry.list_type.value,
        "enabled": entry.enabled,
        "validFrom": entry.valid_from.astimezone(UTC) if entry.valid_from else None,
        "validUntil": entry.valid_until.astimezone(UTC) if entry.valid_until else None,
        "metadata": entry.metadata,
        "createdAt": entry.created_at.astimezone(UTC),
        "updatedAt": entry.updated_at.astimezone(UTC),
    }


def _document_to_watchlist(document: dict[str, Any]) -> WatchlistEntry:
    return WatchlistEntry(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        revision=int(document["revision"]),
        plate=str(document["plate"]),
        list_type=WatchlistType(document["listType"]),
        enabled=bool(document["enabled"]),
        valid_from=_aware(document["validFrom"]) if document.get("validFrom") else None,
        valid_until=_aware(document["validUntil"]) if document.get("validUntil") else None,
        metadata=document.get("metadata") or {},
        created_at=_aware(document["createdAt"]),
        updated_at=_aware(document["updatedAt"]),
    )


def _rule_to_document(rule: Rule) -> dict[str, Any]:
    return {
        "_id": rule.id,
        "schemaVersion": rule.schema_version,
        "revision": rule.revision,
        "name": rule.name,
        "enabled": rule.enabled,
        "priority": rule.priority,
        "conditions": [
            {
                "field": condition.field,
                "operator": condition.operator.value,
                "value": condition.value,
            }
            for condition in rule.conditions
        ],
        "actions": [
            {
                "id": action.id,
                "type": action.type.value,
                "parameters": action.parameters,
            }
            for action in rule.actions
        ],
        "metadata": rule.metadata,
        "createdAt": rule.created_at.astimezone(UTC),
        "updatedAt": rule.updated_at.astimezone(UTC),
    }


def _document_to_rule(document: dict[str, Any]) -> Rule:
    return Rule(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        revision=int(document["revision"]),
        name=str(document["name"]),
        enabled=bool(document["enabled"]),
        priority=int(document.get("priority", 0)),
        conditions=tuple(
            RuleCondition(
                field=str(item["field"]),
                operator=RuleConditionOperator(item["operator"]),
                value=item.get("value"),
            )
            for item in document["conditions"]
        ),
        actions=tuple(
            RuleAction(
                id=str(item["id"]),
                type=RuleActionType(item["type"]),
                parameters=item.get("parameters") or {},
            )
            for item in document["actions"]
        ),
        metadata=document.get("metadata") or {},
        created_at=_aware(document["createdAt"]),
        updated_at=_aware(document["updatedAt"]),
    )


def _alert_to_document(alert: Alert) -> dict[str, Any]:
    return {
        "_id": alert.id,
        "schemaVersion": alert.schema_version,
        "revision": alert.revision,
        "source": {
            "eventId": alert.source.event_id,
            "executionId": alert.source.execution_id,
            "actionId": alert.source.action_id,
        },
        "rule": {"id": alert.rule.id, "name": alert.rule.name},
        "camera": {
            "id": alert.camera.id,
            "name": alert.camera.name,
            "zone": alert.camera.zone,
        },
        "eventType": alert.event_type.value,
        "direction": alert.direction.value,
        "severity": alert.severity.value,
        "status": alert.status.value,
        "message": alert.message,
        "plate": alert.plate,
        "vehicleType": alert.vehicle_type,
        "occurredAt": alert.occurred_at.astimezone(UTC),
        "createdAt": alert.created_at.astimezone(UTC),
        "updatedAt": alert.updated_at.astimezone(UTC),
        "acknowledgedAt": (
            alert.acknowledged_at.astimezone(UTC) if alert.acknowledged_at else None
        ),
        "acknowledgedBy": alert.acknowledged_by,
        "resolvedAt": alert.resolved_at.astimezone(UTC) if alert.resolved_at else None,
        "resolvedBy": alert.resolved_by,
        "metadata": alert.metadata,
    }


def _document_to_alert(document: dict[str, Any]) -> Alert:
    source = document["source"]
    rule = document["rule"]
    camera = document["camera"]
    return Alert(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        revision=int(document["revision"]),
        source=AlertSource(
            event_id=str(source["eventId"]),
            execution_id=str(source["executionId"]),
            action_id=str(source["actionId"]),
        ),
        rule=RuleSnapshot(id=str(rule["id"]), name=str(rule["name"])),
        camera=CameraSnapshot(
            id=str(camera["id"]),
            name=str(camera["name"]),
            zone=camera.get("zone"),
        ),
        event_type=EventType(document["eventType"]),
        direction=Direction(document["direction"]),
        severity=AlertSeverity(document["severity"]),
        status=AlertStatus(document["status"]),
        message=str(document["message"]),
        plate=document.get("plate"),
        vehicle_type=document.get("vehicleType"),
        occurred_at=_aware(document["occurredAt"]),
        created_at=_aware(document["createdAt"]),
        updated_at=_aware(document["updatedAt"]),
        acknowledged_at=(
            _aware(document["acknowledgedAt"]) if document.get("acknowledgedAt") else None
        ),
        acknowledged_by=document.get("acknowledgedBy"),
        resolved_at=(_aware(document["resolvedAt"]) if document.get("resolvedAt") else None),
        resolved_by=document.get("resolvedBy"),
        metadata=document.get("metadata") or {},
    )


def _action_execution_to_document(execution: ActionExecution) -> dict[str, Any]:
    return {
        "_id": execution.id,
        "schemaVersion": execution.schema_version,
        "eventId": execution.event_id,
        "ruleId": execution.rule_id,
        "actionId": execution.action_id,
        "actionType": execution.action_type.value,
        "status": execution.status.value,
        "attemptCount": execution.attempt_count,
        "errorCode": execution.error_code,
        "createdAt": execution.created_at.astimezone(UTC),
        "updatedAt": execution.updated_at.astimezone(UTC),
        "completedAt": (execution.completed_at.astimezone(UTC) if execution.completed_at else None),
    }


def _document_to_action_execution(document: dict[str, Any]) -> ActionExecution:
    return ActionExecution(
        id=str(document["_id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        event_id=str(document["eventId"]),
        rule_id=str(document["ruleId"]),
        action_id=str(document["actionId"]),
        action_type=RuleActionType(document["actionType"]),
        status=ActionExecutionStatus(document["status"]),
        attempt_count=int(document["attemptCount"]),
        error_code=document.get("errorCode"),
        created_at=_aware(document["createdAt"]),
        updated_at=_aware(document["updatedAt"]),
        completed_at=(_aware(document["completedAt"]) if document.get("completedAt") else None),
    )
