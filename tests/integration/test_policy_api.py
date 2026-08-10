import asyncio
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from vehicle_intelligence.application.audit import AuditService
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.policies import (
    AlertService,
    PolicyServices,
    RuleService,
    WatchlistService,
)
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.domain import (
    Alert,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    CameraSnapshot,
    Direction,
    EventType,
    RuleSnapshot,
)
from vehicle_intelligence.infrastructure.persistence.audit_memory import (
    InMemoryAuditLogRepository,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.policy_memory import (
    InMemoryAlertRepository,
    InMemoryRuleRepository,
    InMemoryWatchlistRepository,
)
from vehicle_intelligence.interfaces.api import create_app


def policy_services(alerts: InMemoryAlertRepository) -> PolicyServices:
    normalizer = VietnamPlateNormalizer()
    return PolicyServices(
        watchlists=WatchlistService(InMemoryWatchlistRepository(), normalizer),
        rules=RuleService(InMemoryRuleRepository(), RuleEvaluator()),
        alerts=AlertService(alerts, normalizer),
    )


def alert(alert_id: str, timestamp: datetime) -> Alert:
    return Alert(
        id=alert_id,
        source=AlertSource(f"evt-{alert_id}", f"act-{alert_id}", "create-alert"),
        rule=RuleSnapshot("rule-blacklist", "Blacklist alert"),
        camera=CameraSnapshot("gate-01", "Main Gate", "ZONE_A"),
        event_type=EventType.VEHICLE_ENTER,
        direction=Direction.ENTER,
        severity=AlertSeverity.CRITICAL,
        status=AlertStatus.OPEN,
        message="Blacklist vehicle detected",
        plate="51H-123.45",
        vehicle_type="car",
        occurred_at=timestamp,
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_watchlist_and_rule_api_crud_with_validation() -> None:
    alerts = InMemoryAlertRepository()
    audit_repository = InMemoryAuditLogRepository()
    app = create_app(
        load_settings(),
        InMemoryVehicleEventRepository(),
        policy_services=policy_services(alerts),
        audit_service=AuditService(audit_repository),
    )
    watchlist_payload = {
        "id": "wle-api",
        "plate": "51H 12345",
        "listType": "BLACKLIST",
        "enabled": True,
        "metadata": {"reason": "security"},
    }
    rule_payload = {
        "id": "rule-api",
        "name": "Alert for blacklisted entry",
        "enabled": True,
        "priority": 100,
        "conditions": [
            {"field": "watchlist", "operator": "CONTAINS", "value": "BLACKLIST"},
            {"field": "direction", "operator": "EQ", "value": "ENTER"},
        ],
        "actions": [
            {
                "id": "create-alert",
                "type": "CREATE_ALERT",
                "parameters": {"severity": "CRITICAL"},
            }
        ],
    }

    with TestClient(app) as client:
        created_watchlist = client.post("/api/watchlists", json=watchlist_payload)
        listed_watchlists = client.get(
            "/api/watchlists", params={"listType": "BLACKLIST", "enabled": True}
        )
        update_payload = dict(watchlist_payload)
        update_payload.pop("id")
        update_payload["revision"] = 1
        update_payload["listType"] = "VIP"
        updated_watchlist = client.put("/api/watchlists/wle-api", json=update_payload)

        created_rule = client.post("/api/rules", json=rule_payload)
        listed_rules = client.get("/api/rules", params={"enabledOnly": True})
        unsafe_rule = dict(rule_payload)
        unsafe_rule["id"] = "rule-unsafe"
        unsafe_rule["conditions"] = [
            {"field": "metadata.any", "operator": "EQ", "value": "x"}
        ]
        rejected_rule = client.post("/api/rules", json=unsafe_rule)
        audit_logs = client.get("/api/audit-logs", params={"limit": 20})

    assert created_watchlist.status_code == 201
    assert created_watchlist.json()["plate"] == "51H-123.45"
    assert listed_watchlists.json()["items"][0]["id"] == "wle-api"
    assert updated_watchlist.json()["listType"] == "VIP"
    assert updated_watchlist.json()["revision"] == 2
    assert created_rule.status_code == 201
    assert created_rule.json()["actions"][0]["id"] == "create-alert"
    assert listed_rules.json()["items"][0]["id"] == "rule-api"
    assert rejected_rule.status_code == 422
    assert {item["action"] for item in audit_logs.json()["items"]} == {
        "WATCHLIST_CREATED",
        "WATCHLIST_UPDATED",
        "RULE_CREATED",
    }


def test_alert_api_cursor_filter_acknowledge_and_resolve() -> None:
    timestamp = datetime(2026, 8, 9, 12, tzinfo=UTC)
    alerts = InMemoryAlertRepository()
    asyncio.run(alerts.create(alert("alr-old", timestamp)))
    asyncio.run(alerts.create(alert("alr-new", timestamp + timedelta(minutes=1))))
    app = create_app(
        load_settings(),
        InMemoryVehicleEventRepository(),
        policy_services=policy_services(alerts),
    )

    with TestClient(app) as client:
        first_page = client.get(
            "/api/alerts",
            params={"limit": 1, "plate": "51H12345", "status": "OPEN"},
        )
        cursor = first_page.json()["nextCursor"]
        second_page = client.get("/api/alerts", params={"limit": 1, "cursor": cursor})
        acknowledged = client.post(
            "/api/alerts/alr-new/acknowledge",
            json={},
        )
        resolved = client.post(
            "/api/alerts/alr-new/resolve",
            json={},
        )
        detail = client.get("/api/alerts/alr-new")

    assert first_page.status_code == 200
    assert first_page.json()["items"][0]["id"] == "alr-new"
    assert cursor is not None
    assert second_page.json()["items"][0]["id"] == "alr-old"
    assert acknowledged.json()["status"] == "ACKNOWLEDGED"
    assert acknowledged.json()["acknowledgedBy"] == "development-admin"
    assert resolved.json()["status"] == "RESOLVED"
    assert detail.json()["revision"] == 3
