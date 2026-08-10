from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.policies import (
    AlertService,
    RuleCreate,
    RuleService,
    WatchlistCreate,
    WatchlistService,
    WatchlistUpdate,
)
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.domain import (
    Alert,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    CameraSnapshot,
    Direction,
    EventType,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    RuleSnapshot,
    WatchlistType,
)
from vehicle_intelligence.exceptions import PolicyConflictError, RuleValidationError
from vehicle_intelligence.infrastructure.persistence.policy_memory import (
    InMemoryAlertRepository,
    InMemoryRuleRepository,
    InMemoryWatchlistRepository,
)


async def test_watchlist_service_normalizes_plate_and_enforces_revision() -> None:
    now = datetime(2026, 8, 9, 12, tzinfo=UTC)
    repository = InMemoryWatchlistRepository()
    service = WatchlistService(
        repository,
        VietnamPlateNormalizer(),
        clock=lambda: now,
        id_factory=lambda: "wle-test",
    )
    created = await service.create(
        WatchlistCreate(
            plate="51H 12345",
            list_type=WatchlistType.WHITELIST,
            valid_until=now + timedelta(days=1),
        )
    )
    assert created.plate == "51H-123.45"
    assert [item.id for item in await repository.find_active_by_plate(created.plate, now)] == [
        "wle-test"
    ]

    updated = await service.update(
        created.id,
        WatchlistUpdate(
            revision=1,
            plate=created.plate,
            list_type=WatchlistType.STAFF,
            enabled=False,
        ),
    )
    assert updated.revision == 2
    assert not updated.enabled
    with pytest.raises(PolicyConflictError, match="revision conflict"):
        await service.update(
            created.id,
            WatchlistUpdate(
                revision=1,
                plate=created.plate,
                list_type=WatchlistType.STAFF,
            ),
        )


async def test_rule_service_rejects_unknown_context_field() -> None:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    service = RuleService(
        InMemoryRuleRepository(),
        RuleEvaluator(),
        clock=lambda: now,
        id_factory=lambda: "rule-test",
    )
    with pytest.raises(RuleValidationError, match="unsupported rule field"):
        await service.create(
            RuleCreate(
                name="Unsafe",
                enabled=True,
                priority=0,
                conditions=(
                    RuleCondition(
                        "metadata.userInput",
                        RuleConditionOperator.EQ,
                        "anything",
                    ),
                ),
                actions=(RuleAction("log", RuleActionType.LOG),),
            )
        )


async def test_alert_service_acknowledges_and_resolves_with_optimistic_revision() -> None:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    repository = InMemoryAlertRepository()
    service = AlertService(repository, VietnamPlateNormalizer(), clock=lambda: now)
    alert = Alert(
        id="alr-test",
        source=AlertSource("evt-test", "act-test", "create-alert"),
        rule=RuleSnapshot("rule-test", "Test rule"),
        camera=CameraSnapshot("gate-01", "Main Gate"),
        event_type=EventType.VEHICLE_ENTER,
        direction=Direction.ENTER,
        severity=AlertSeverity.HIGH,
        status=AlertStatus.OPEN,
        message="Blacklist vehicle",
        plate="51H-123.45",
        occurred_at=now,
        created_at=now,
        updated_at=now,
    )
    assert await repository.create(alert)

    acknowledged = await service.acknowledge(alert.id, "operator-01")
    assert acknowledged.status is AlertStatus.ACKNOWLEDGED
    assert acknowledged.revision == 2
    resolved = await service.resolve(alert.id, "operator-02")
    assert resolved.status is AlertStatus.RESOLVED
    assert resolved.revision == 3
    with pytest.raises(PolicyConflictError, match="cannot transition"):
        await service.acknowledge(alert.id, "operator-01")
