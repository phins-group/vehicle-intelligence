from datetime import datetime

import httpx
import pytest

from vehicle_intelligence.application.actions import (
    ActionEngine,
    AlertActionHandler,
    HttpActionHandler,
    action_execution_id,
)
from vehicle_intelligence.application.ports import AlertQuery
from vehicle_intelligence.config import ExternalActionTargetConfig, RuleEngineConfig
from vehicle_intelligence.domain import (
    ActionExecutionStatus,
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
)
from vehicle_intelligence.exceptions import ActionExecutionError, ActionHandlerError
from vehicle_intelligence.infrastructure.persistence.policy_memory import (
    InMemoryActionExecutionRepository,
    InMemoryAlertRepository,
)


class FlakyHandler:
    def __init__(self) -> None:
        self.calls = 0

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def execute(self, _context) -> None:
        self.calls += 1
        if self.calls == 1:
            raise ActionHandlerError("TEMPORARY_FAILURE")


def configured_rule(timestamp: datetime, action: RuleAction) -> Rule:
    return Rule(
        id="rule-test",
        name="Test rule",
        enabled=True,
        priority=10,
        conditions=(
            RuleCondition("camera.id", RuleConditionOperator.EQ, "gate-01"),
        ),
        actions=(action,),
        created_at=timestamp,
        updated_at=timestamp,
    )


async def test_action_engine_creates_one_alert_and_skips_completed_redelivery(
    sample_event,
) -> None:
    executions = InMemoryActionExecutionRepository()
    alerts = InMemoryAlertRepository()
    action = RuleAction(
        "create-alert",
        RuleActionType.CREATE_ALERT,
        {"severity": "CRITICAL", "message": "Blacklist match"},
    )
    rule = configured_rule(sample_event.occurred_at, action)
    engine = ActionEngine(
        executions,
        {RuleActionType.CREATE_ALERT: AlertActionHandler(alerts)},
        RuleEngineConfig(),
        clock=lambda: sample_event.occurred_at,
    )
    await engine.initialize()

    assert await engine.execute(sample_event, rule, action, ())
    assert not await engine.execute(sample_event, rule, action, ())
    page = await alerts.list(AlertQuery())
    assert len(page.items) == 1
    assert page.items[0].severity.value == "CRITICAL"
    execution = await executions.get(
        action_execution_id(sample_event.id, rule.id, action.id)
    )
    assert execution.status is ActionExecutionStatus.SUCCEEDED
    assert execution.attempt_count == 1
    await engine.close()


async def test_action_engine_retries_failed_execution_then_succeeds(sample_event) -> None:
    executions = InMemoryActionExecutionRepository()
    handler = FlakyHandler()
    action = RuleAction("log", RuleActionType.LOG)
    rule = configured_rule(sample_event.occurred_at, action)
    engine = ActionEngine(
        executions,
        {RuleActionType.LOG: handler},
        RuleEngineConfig(action_max_attempts=3),
        clock=lambda: sample_event.occurred_at,
    )
    await engine.initialize()

    with pytest.raises(ActionExecutionError, match="TEMPORARY_FAILURE"):
        await engine.execute(sample_event, rule, action, ())
    assert await engine.execute(sample_event, rule, action, ())
    assert not await engine.execute(sample_event, rule, action, ())
    execution = await executions.get(
        action_execution_id(sample_event.id, rule.id, action.id)
    )
    assert execution.status is ActionExecutionStatus.SUCCEEDED
    assert execution.attempt_count == 2


async def test_http_action_uses_allowlist_and_idempotency_header(sample_event) -> None:
    captured: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["idempotency"] = request.headers["Idempotency-Key"]
        captured["body"] = request.content
        return httpx.Response(202)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    config = RuleEngineConfig(
        external_actions_enabled=True,
        external_allowed_hosts=["hooks.example"],
    )
    handler = HttpActionHandler(config, client)
    executions = InMemoryActionExecutionRepository()
    action = RuleAction(
        "webhook",
        RuleActionType.WEBHOOK,
        {"url": "https://hooks.example/vehicle", "body": {"source": "rule"}},
    )
    rule = configured_rule(sample_event.occurred_at, action)
    engine = ActionEngine(
        executions,
        {RuleActionType.WEBHOOK: handler},
        config,
        clock=lambda: sample_event.occurred_at,
    )
    await engine.initialize()

    assert await engine.execute(sample_event, rule, action, ())
    assert captured["idempotency"] == action_execution_id(
        sample_event.id, rule.id, action.id
    )
    assert b'"eventId":"evt_test"' in captured["body"]
    await engine.close()
    await client.aclose()


async def test_http_action_uses_server_managed_hmac_signature(sample_event) -> None:
    captured: dict[str, str] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        for name in (
            "X-Signature-Algorithm",
            "X-Signature-Key-Id",
            "X-Signature-Timestamp",
            "X-Signature",
        ):
            captured[name] = request.headers[name]
        return httpx.Response(204)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    target = ExternalActionTargetConfig(
        host="hooks.example",
        authentication="hmac_sha256",
        hmac_secret="server-secret",
        hmac_key_id="hook-v1",
    )
    config = RuleEngineConfig(
        external_actions_enabled=True,
        external_targets=[target],
    )
    handler = HttpActionHandler(
        config,
        client,
        clock=lambda: sample_event.occurred_at,
    )
    action = RuleAction(
        "webhook",
        RuleActionType.WEBHOOK,
        {"url": "https://hooks.example/vehicle", "body": {"source": "rule"}},
    )
    rule = configured_rule(sample_event.occurred_at, action)
    engine = ActionEngine(
        InMemoryActionExecutionRepository(),
        {RuleActionType.WEBHOOK: handler},
        config,
        clock=lambda: sample_event.occurred_at,
    )
    await engine.initialize()

    assert await engine.execute(sample_event, rule, action, ())
    assert captured["X-Signature-Algorithm"] == "hmac-sha256"
    assert captured["X-Signature-Key-Id"] == "hook-v1"
    assert len(captured["X-Signature"]) == 64
    await engine.close()
    await client.aclose()


async def test_http_action_circuit_open_does_not_consume_retry_attempt(sample_event) -> None:
    request_count = 0
    monotonic = [0.0]

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(503 if request_count <= 2 else 204)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    config = RuleEngineConfig(
        external_actions_enabled=True,
        action_max_attempts=5,
        external_targets=[
            ExternalActionTargetConfig(
                host="hooks.example",
                circuit_failure_threshold=2,
                circuit_recovery_seconds=30,
            )
        ],
    )
    executions = InMemoryActionExecutionRepository()
    handler = HttpActionHandler(config, client, monotonic=lambda: monotonic[0])
    action = RuleAction(
        "webhook",
        RuleActionType.WEBHOOK,
        {"url": "https://hooks.example/vehicle"},
    )
    rule = configured_rule(sample_event.occurred_at, action)
    engine = ActionEngine(
        executions,
        {RuleActionType.WEBHOOK: handler},
        config,
        clock=lambda: sample_event.occurred_at,
    )
    await engine.initialize()

    for _ in range(2):
        with pytest.raises(ActionExecutionError, match="HTTP_STATUS_503"):
            await engine.execute(sample_event, rule, action, ())
    with pytest.raises(ActionExecutionError, match="EXTERNAL_CIRCUIT_OPEN"):
        await engine.execute(sample_event, rule, action, ())
    execution_id = action_execution_id(sample_event.id, rule.id, action.id)
    assert (await executions.get(execution_id)).attempt_count == 2
    assert request_count == 2

    monotonic[0] = 31
    assert await engine.execute(sample_event, rule, action, ())
    assert request_count == 3
    await engine.close()
    await client.aclose()
