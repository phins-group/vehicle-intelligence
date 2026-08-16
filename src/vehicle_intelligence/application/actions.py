"""Durable idempotent rule-action execution and concrete handlers."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import httpx

from vehicle_intelligence.application.ports import (
    ActionContext,
    ActionExecutionRepository,
    ActionHandler,
    AlertRepository,
)
from vehicle_intelligence.config import ExternalActionTargetConfig, RuleEngineConfig
from vehicle_intelligence.domain import (
    ActionExecution,
    ActionExecutionStatus,
    Alert,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    Rule,
    RuleAction,
    RuleActionType,
    RuleSnapshot,
    VehicleEvent,
    WatchlistEntry,
)
from vehicle_intelligence.exceptions import ActionExecutionError, ActionHandlerError

logger = logging.getLogger(__name__)
_SAFE_CODE = re.compile(r"[^A-Z0-9_]+")


def action_execution_id(event_id: str, rule_id: str, action_id: str) -> str:
    value = f"{event_id}|{rule_id}|{action_id}".encode()
    return f"act_{hashlib.sha256(value).hexdigest()[:40]}"


def _alert_id(execution_id: str) -> str:
    return f"alr_{hashlib.sha256(execution_id.encode()).hexdigest()[:40]}"


def _safe_code(value: str) -> str:
    normalized = _SAFE_CODE.sub("_", value.upper()).strip("_")
    return (normalized or "ACTION_FAILED")[:64]


class ActionEngine:
    def __init__(
        self,
        repository: ActionExecutionRepository,
        handlers: Mapping[RuleActionType, ActionHandler],
        config: RuleEngineConfig,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._handlers = dict(handlers)
        self._config = config
        self._clock = clock

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()
        for handler in self._unique_handlers():
            await handler.initialize()

    async def close(self) -> None:
        try:
            for handler in self._unique_handlers():
                await handler.close()
        finally:
            await self._repository.close()

    async def execute(
        self,
        event: VehicleEvent,
        rule: Rule,
        action: RuleAction,
        watchlists: tuple[WatchlistEntry, ...],
    ) -> bool:
        now = self._now()
        execution_id = action_execution_id(event.id, rule.id, action.id)
        candidate = ActionExecution(
            id=execution_id,
            event_id=event.id,
            rule_id=rule.id,
            action_id=action.id,
            action_type=action.type,
            status=ActionExecutionStatus.RUNNING,
            attempt_count=1,
            created_at=now,
            updated_at=now,
        )
        claimed = await self._repository.claim(
            candidate,
            now - timedelta(seconds=self._config.action_claim_stale_seconds),
            self._config.action_max_attempts,
        )
        if claimed is None:
            current = await self._repository.get(execution_id)
            if current is None:
                raise ActionExecutionError("ACTION_CLAIM_LOST")
            if current.status is ActionExecutionStatus.SUCCEEDED:
                return False
            if (
                current.status is ActionExecutionStatus.FAILED
                and current.attempt_count >= self._config.action_max_attempts
            ):
                return False
            raise ActionExecutionError("ACTION_ALREADY_RUNNING")

        handler = self._handlers.get(action.type)
        if handler is None:
            error = ActionHandlerError("ACTION_HANDLER_NOT_CONFIGURED", retryable=False)
            await self._record_failure(claimed.id, error)
            raise ActionExecutionError(error.code)

        context = ActionContext(
            execution_id=claimed.id,
            event=event,
            rule=rule,
            action=action,
            watchlist_types=tuple(
                sorted({entry.list_type for entry in watchlists}, key=lambda item: item.value)
            ),
        )
        try:
            async with asyncio.timeout(self._config.action_timeout_seconds):
                await handler.execute(context)
        except TimeoutError:
            error = ActionHandlerError("ACTION_TIMEOUT")
            await self._record_failure(claimed.id, error)
            raise ActionExecutionError(error.code) from None
        except ActionHandlerError as error:
            await self._record_failure(claimed.id, error)
            raise ActionExecutionError(error.code) from error
        except Exception as exc:
            error = ActionHandlerError("ACTION_HANDLER_ERROR")
            await self._record_failure(claimed.id, error)
            logger.exception(
                "rule action handler failed",
                extra={
                    "event_id": event.id,
                    "rule_id": rule.id,
                    "action_id": action.id,
                    "action_type": action.type.value,
                },
            )
            raise ActionExecutionError(error.code) from exc

        await self._repository.mark_succeeded(claimed.id, self._now())
        return True

    async def _record_failure(self, execution_id: str, error: ActionHandlerError) -> None:
        await self._repository.mark_failed(
            execution_id,
            _safe_code(error.code),
            self._now(),
            terminal=not error.retryable,
            maximum_attempts=self._config.action_max_attempts,
            consume_attempt=error.consume_attempt,
        )

    def _unique_handlers(self) -> tuple[ActionHandler, ...]:
        unique: dict[int, ActionHandler] = {}
        for handler in self._handlers.values():
            unique[id(handler)] = handler
        return tuple(unique.values())

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("action engine clock must be timezone-aware")
        return value.astimezone(UTC)


class AlertActionHandler:
    def __init__(
        self,
        repository: AlertRepository,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._clock = clock

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def close(self) -> None:
        await self._repository.close()

    async def execute(self, context: ActionContext) -> None:
        parameters = context.action.parameters
        severity = AlertSeverity(str(parameters.get("severity", "HIGH")))
        plate = context.event.plate.final_normalized if context.event.plate else None
        message_value = parameters.get("message")
        message = (
            str(message_value).strip()
            if message_value is not None
            else f"Rule {context.rule.name} matched vehicle {plate or 'UNREADABLE'}"
        )
        metadata = parameters.get("metadata")
        now = self._now()
        alert = Alert(
            id=_alert_id(context.execution_id),
            source=AlertSource(
                event_id=context.event.id,
                execution_id=context.execution_id,
                action_id=context.action.id,
            ),
            rule=RuleSnapshot(id=context.rule.id, name=context.rule.name),
            camera=context.event.camera,
            event_type=context.event.event_type,
            direction=context.event.direction,
            severity=severity,
            status=AlertStatus.OPEN,
            message=message,
            plate=plate,
            vehicle_type=context.event.vehicle.type,
            occurred_at=context.event.occurred_at,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )
        await self._repository.create(alert)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("alert action clock must be timezone-aware")
        return value.astimezone(UTC)


class LogActionHandler:
    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def execute(self, context: ActionContext) -> None:
        message = context.action.parameters.get("message", "vehicle rule matched")
        logger.info(
            str(message),
            extra={
                "event_id": context.event.id,
                "camera_id": context.event.camera.id,
                "track_id": context.event.track_id,
                "rule_id": context.rule.id,
                "action_id": context.action.id,
                "action_execution_id": context.execution_id,
            },
        )


@dataclass(slots=True)
class _CircuitState:
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


class _TargetCircuitBreaker:
    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._states: dict[str, _CircuitState] = {}
        self._lock = asyncio.Lock()

    async def before_request(self, target: ExternalActionTargetConfig) -> None:
        async with self._lock:
            state = self._states.setdefault(target.host, _CircuitState())
            if state.opened_at is None:
                return
            elapsed = self._monotonic() - state.opened_at
            if elapsed < target.circuit_recovery_seconds or state.probe_in_flight:
                raise ActionHandlerError(
                    "EXTERNAL_CIRCUIT_OPEN",
                    consume_attempt=False,
                )
            state.probe_in_flight = True

    async def succeeded(self, target: ExternalActionTargetConfig) -> None:
        async with self._lock:
            self._states[target.host] = _CircuitState()

    async def failed(self, target: ExternalActionTargetConfig) -> None:
        async with self._lock:
            state = self._states.setdefault(target.host, _CircuitState())
            state.probe_in_flight = False
            state.failures += 1
            if state.failures >= target.circuit_failure_threshold:
                state.opened_at = self._monotonic()


class HttpActionHandler:
    def __init__(
        self,
        config: RuleEngineConfig,
        client: httpx.AsyncClient | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._circuit = _TargetCircuitBreaker(monotonic)
        self._clock = clock
        self._targets = {target.host: target for target in config.external_targets}

    async def initialize(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.action_timeout_seconds,
                follow_redirects=False,
            )

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def execute(self, context: ActionContext) -> None:
        if not self._config.external_actions_enabled:
            raise ActionHandlerError("EXTERNAL_ACTIONS_DISABLED", retryable=False)
        url = str(context.action.parameters["url"])
        if len(url) > self._config.external_maximum_url_length:
            raise ActionHandlerError("EXTERNAL_URL_INVALID", retryable=False)
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        target = self._targets.get(hostname)
        if target is None and hostname in set(self._config.external_allowed_hosts):
            target = ExternalActionTargetConfig(host=hostname, require_https=False)
        try:
            parsed_port = parsed.port
        except ValueError as exc:
            raise ActionHandlerError("EXTERNAL_URL_INVALID", retryable=False) from exc
        if (
            target is None
            or parsed.scheme not in {"http", "https"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or not parsed.path.startswith("/")
            or parsed_port is not None
            and not 1 <= parsed_port <= 65_535
            or target.require_https
            and parsed.scheme != "https"
        ):
            raise ActionHandlerError("EXTERNAL_HOST_NOT_ALLOWED", retryable=False)
        method = str(context.action.parameters.get("method", "POST")).upper()
        if method not in {"GET", "POST", "PUT", "PATCH"}:
            raise ActionHandlerError("EXTERNAL_METHOD_NOT_ALLOWED", retryable=False)
        if self._client is None:
            raise ActionHandlerError("HTTP_CLIENT_NOT_INITIALIZED")
        await self._circuit.before_request(target)
        headers = {
            "Idempotency-Key": context.execution_id,
            "X-Vehicle-Event-Id": context.event.id,
            "X-Vehicle-Rule-Id": context.rule.id,
        }
        payload = b""
        if method == "GET":
            request_url = httpx.URL(url).copy_merge_params(
                {"eventId": context.event.id, "ruleId": context.rule.id}
            )
        else:
            request_url = httpx.URL(url)
            payload = json.dumps(
                {
                    "eventId": context.event.id,
                    "ruleId": context.rule.id,
                    "actionId": context.action.id,
                    "cameraId": context.event.camera.id,
                    "plate": (
                        context.event.plate.final_normalized if context.event.plate else None
                    ),
                    "direction": context.event.direction.value,
                    "data": context.action.parameters.get("body", {}),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
        self._authenticate(target, method, request_url, payload, headers, context.execution_id)
        try:
            response = await self._client.request(
                method,
                request_url,
                headers=headers,
                content=payload or None,
            )
        except httpx.RequestError as exc:
            await self._circuit.failed(target)
            raise ActionHandlerError("HTTP_TRANSPORT_ERROR") from exc
        if 200 <= response.status_code < 300:
            await self._circuit.succeeded(target)
            return
        retryable = response.status_code >= 500 or response.status_code in {408, 429}
        if retryable:
            await self._circuit.failed(target)
        else:
            await self._circuit.succeeded(target)
        raise ActionHandlerError(
            f"HTTP_STATUS_{response.status_code}",
            retryable=retryable,
        )

    def _authenticate(
        self,
        target: ExternalActionTargetConfig,
        method: str,
        url: httpx.URL,
        body: bytes,
        headers: dict[str, str],
        idempotency_key: str,
    ) -> None:
        if target.authentication == "bearer":
            assert target.bearer_token is not None
            headers["Authorization"] = f"Bearer {target.bearer_token.get_secret_value()}"
            return
        if target.authentication != "hmac_sha256":
            return
        assert target.hmac_secret is not None and target.hmac_key_id is not None
        now = self._clock()
        if now.tzinfo is None:
            raise ActionHandlerError("EXTERNAL_CLOCK_INVALID", retryable=False)
        timestamp = str(int(now.timestamp()))
        body_hash = hashlib.sha256(body).hexdigest()
        request_target = url.raw_path.decode("ascii")
        canonical = "\n".join(
            (timestamp, idempotency_key, method, request_target, body_hash)
        ).encode("utf-8")
        signature = hmac.new(
            target.hmac_secret.get_secret_value().encode("utf-8"),
            canonical,
            hashlib.sha256,
        ).hexdigest()
        headers.update(
            {
                "X-Signature-Algorithm": "hmac-sha256",
                "X-Signature-Key-Id": target.hmac_key_id,
                "X-Signature-Timestamp": timestamp,
                "X-Signature": signature,
            }
        )
