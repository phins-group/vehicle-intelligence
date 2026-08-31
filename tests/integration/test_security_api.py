import asyncio
import hashlib
import json
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from vehicle_intelligence.application.audit import AuditService
from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.discovery import OnvifDiscoveryService
from vehicle_intelligence.application.media_access import VehicleEventMediaService
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.policies import (
    AlertService,
    PolicyServices,
    RuleService,
    WatchlistService,
)
from vehicle_intelligence.application.ports import CameraConnectionTestResult
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.config import AuthConfig, AuthPrincipalConfig, load_settings
from vehicle_intelligence.domain import (
    Alert,
    AlertSeverity,
    AlertSource,
    AlertStatus,
    CameraSnapshot,
    Direction,
    EventType,
    OnvifDiscoveredDevice,
    RuleSnapshot,
)
from vehicle_intelligence.infrastructure.persistence.audit_memory import (
    InMemoryAuditLogRepository,
)
from vehicle_intelligence.infrastructure.persistence.camera_memory import (
    InMemoryCameraHealthRepository,
    InMemoryCameraRepository,
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

ADMIN_TOKEN = "admin-api-key-" + "a" * 40
OPERATOR_TOKEN = "operator-api-key-" + "o" * 40
VIEWER_TOKEN = "viewer-api-key-" + "v" * 40


class ConnectedTester:
    async def test(self, camera):
        return CameraConnectionTestResult(True, 3.5, camera.updated_at)


class OneDiscoveredDevice:
    async def discover(self):
        return [
            OnvifDiscoveredDevice(
                endpoint_reference="urn:uuid:secure-discovery",
                xaddrs=("http://192.0.2.90/onvif/device_service",),
                types=("tds:Device",),
                scopes=(),
                discovered_at=datetime.now().astimezone(),
            )
        ]


class FakeMediaUrlSigner:
    async def presign_get(self, key: str, expires: timedelta) -> str | None:
        del expires
        return f"https://media.example/{key}?signature=rbac"


def key_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def settings_with_auth():
    base = load_settings()
    auth = AuthConfig(
        enabled=True,
        principals=[
            AuthPrincipalConfig(
                id="admin-01",
                display_name="Platform Admin",
                role="ADMIN",
                key_sha256=key_hash(ADMIN_TOKEN),
            ),
            AuthPrincipalConfig(
                id="operator-01",
                display_name="Gate Operator",
                role="OPERATOR",
                key_sha256=key_hash(OPERATOR_TOKEN),
            ),
            AuthPrincipalConfig(
                id="viewer-01",
                display_name="Security Viewer",
                role="VIEWER",
                key_sha256=key_hash(VIEWER_TOKEN),
            ),
        ],
    )
    return base.model_copy(update={"auth": auth})


def authorization(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def camera_payload() -> dict[str, object]:
    return {
        "id": "gate-auth",
        "name": "Authenticated Gate",
        "stream": {
            "rtspUrl": "rtsp://admin:camera-secret@camera.example/live",
            "fpsLimit": 6,
        },
        "direction": "BOTH",
        "enabled": True,
    }


def alert(timestamp: datetime) -> Alert:
    return Alert(
        id="alr-auth",
        source=AlertSource("evt-auth", "act-auth", "create-alert"),
        rule=RuleSnapshot("rule-auth", "Authentication alert"),
        camera=CameraSnapshot("gate-auth", "Authenticated Gate", "ZONE_A"),
        event_type=EventType.VEHICLE_ENTER,
        direction=Direction.ENTER,
        severity=AlertSeverity.HIGH,
        status=AlertStatus.OPEN,
        message="Operator review required",
        plate="51H-123.45",
        vehicle_type="car",
        occurred_at=timestamp,
        created_at=timestamp,
        updated_at=timestamp,
    )


def policy_services(alerts: InMemoryAlertRepository) -> PolicyServices:
    normalizer = VietnamPlateNormalizer()
    return PolicyServices(
        watchlists=WatchlistService(InMemoryWatchlistRepository(), normalizer),
        rules=RuleService(InMemoryRuleRepository(), RuleEvaluator()),
        alerts=AlertService(alerts, normalizer),
    )


def test_onvif_discovery_allows_operator_but_batch_import_requires_admin() -> None:
    app = create_app(
        settings_with_auth(),
        InMemoryVehicleEventRepository(),
        CameraService(
            InMemoryCameraRepository(),
            InMemoryCameraHealthRepository(),
            ConnectedTester(),
        ),
        onvif_discovery_service=OnvifDiscoveryService(
            OneDiscoveredDevice(),
            maximum_results=10,
        ),
    )

    with TestClient(app) as client:
        missing = client.post("/api/cameras/discover")
        viewer = client.post(
            "/api/cameras/discover",
            headers=authorization(VIEWER_TOKEN),
        )
        operator = client.post(
            "/api/cameras/discover",
            headers=authorization(OPERATOR_TOKEN),
        )
        operator_batch = client.post(
            "/api/cameras/batch",
            json={"items": [camera_payload()]},
            headers=authorization(OPERATOR_TOKEN),
        )
        admin_batch = client.post(
            "/api/cameras/batch",
            json={"items": [camera_payload()]},
            headers=authorization(ADMIN_TOKEN),
        )

    assert missing.status_code == 401
    assert viewer.status_code == 403
    assert operator.status_code == 200
    assert operator.json()["count"] == 1
    assert operator_batch.status_code == 403
    assert admin_batch.status_code == 200
    assert admin_batch.json()["createdCount"] == 1


def test_api_authentication_rbac_and_actor_audit(sample_event) -> None:
    event_repository = InMemoryVehicleEventRepository()
    asyncio.run(event_repository.save(sample_event))
    alerts = InMemoryAlertRepository()
    asyncio.run(alerts.create(alert(sample_event.occurred_at)))
    audit_repository = InMemoryAuditLogRepository()
    camera_service = CameraService(
        InMemoryCameraRepository(),
        InMemoryCameraHealthRepository(),
        ConnectedTester(),
    )
    app = create_app(
        settings_with_auth(),
        event_repository,
        camera_service,
        policy_services(alerts),
        AuditService(audit_repository),
        media_access_service=VehicleEventMediaService(
            event_repository,
            FakeMediaUrlSigner(),
            120,
        ),
    )

    with TestClient(app) as client:
        health = client.get("/api/system/health")
        auth_configuration = client.get("/api/auth/config")
        missing = client.get("/api/events")
        invalid = client.get(
            "/api/events",
            headers=authorization("invalid-api-key-" + "x" * 40),
        )
        viewer_events = client.get("/api/events", headers=authorization(VIEWER_TOKEN))
        viewer_vehicle_search = client.get(
            "/api/vehicles/search",
            params={"plate": "51H12345"},
            headers=authorization(VIEWER_TOKEN),
        )
        missing_media_auth = client.get(f"/api/events/{sample_event.id}/media")
        viewer_media = client.get(
            f"/api/events/{sample_event.id}/media",
            headers=authorization(VIEWER_TOKEN),
        )
        viewer_me = client.get("/api/auth/me", headers=authorization(VIEWER_TOKEN))
        viewer_create = client.post(
            "/api/cameras",
            json=camera_payload(),
            headers=authorization(VIEWER_TOKEN),
        )
        viewer_audit = client.get(
            "/api/audit-logs",
            headers=authorization(VIEWER_TOKEN),
        )
        viewer_review = client.put(
            f"/api/events/{sample_event.id}/plate-review",
            json={"text": "51H12346", "expectedRevision": 0},
            headers=authorization(VIEWER_TOKEN),
        )
        viewer_dataset = client.get(
            "/api/dataset-samples",
            headers=authorization(VIEWER_TOKEN),
        )
        operator_review = client.put(
            f"/api/events/{sample_event.id}/plate-review",
            json={"text": "51H12346", "expectedRevision": 0},
            headers=authorization(OPERATOR_TOKEN),
        )
        admin_headers = {
            **authorization(ADMIN_TOKEN),
            "X-Request-ID": "req-security-create",
        }
        created = client.post("/api/cameras", json=camera_payload(), headers=admin_headers)
        operator_test = client.post(
            "/api/cameras/gate-auth/test-connection",
            headers=authorization(OPERATOR_TOKEN),
        )
        operator_policy = client.post(
            "/api/watchlists",
            json={"plate": "51H12345", "listType": "BLACKLIST"},
            headers=authorization(OPERATOR_TOKEN),
        )
        mismatched_actor = client.post(
            "/api/alerts/alr-auth/acknowledge",
            json={"actorId": "admin-01"},
            headers=authorization(OPERATOR_TOKEN),
        )
        acknowledged = client.post(
            "/api/alerts/alr-auth/acknowledge",
            json={},
            headers=authorization(OPERATOR_TOKEN),
        )
        first_audit_page = client.get(
            "/api/audit-logs",
            params={"resourceType": "CAMERA", "resourceId": "gate-auth", "limit": 1},
            headers=authorization(ADMIN_TOKEN),
        )
        second_audit_page = client.get(
            "/api/audit-logs",
            params={
                "resourceType": "CAMERA",
                "resourceId": "gate-auth",
                "cursor": first_audit_page.json()["nextCursor"],
                "limit": 1,
            },
            headers=authorization(ADMIN_TOKEN),
        )
        invalid_audit_cursor = client.get(
            "/api/audit-logs",
            params={"cursor": "not-a-valid-cursor"},
            headers=authorization(ADMIN_TOKEN),
        )

    assert health.status_code == 200
    assert health.json()["authentication"] == "enabled"
    assert auth_configuration.status_code == 200
    assert auth_configuration.headers["cache-control"] == "no-store"
    assert auth_configuration.json() == {
        "enabled": True,
        "provider": "api_key",
        "oidc": None,
    }
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"].startswith("Bearer")
    assert invalid.status_code == 401
    assert "invalid-api-key" not in invalid.text
    assert viewer_events.status_code == 200
    assert viewer_vehicle_search.status_code == 200
    assert viewer_vehicle_search.json()["query"] == "51H-123.45"
    assert missing_media_auth.status_code == 401
    assert viewer_media.status_code == 200
    assert viewer_media.json()["media"]["snapshot"]["status"] == "AVAILABLE"
    assert viewer_me.json()["role"] == "VIEWER"
    assert viewer_create.status_code == 403
    assert viewer_audit.status_code == 403
    assert viewer_review.status_code == 403
    assert viewer_dataset.status_code == 403
    assert operator_review.status_code == 200
    assert operator_review.json()["event"]["plate"]["final"] == "51H-123.46"
    assert created.status_code == 201
    assert created.headers["x-request-id"] == "req-security-create"
    assert operator_test.status_code == 200
    assert operator_policy.status_code == 403
    assert mismatched_actor.status_code == 403
    assert acknowledged.status_code == 200
    assert acknowledged.json()["acknowledgedBy"] == "operator-01"
    assert first_audit_page.status_code == 200
    assert invalid_audit_cursor.status_code == 400
    assert first_audit_page.json()["nextCursor"] is not None
    audit_items = first_audit_page.json()["items"] + second_audit_page.json()["items"]
    assert {item["action"] for item in audit_items} == {
        "CAMERA_CREATED",
        "CAMERA_CONNECTION_TESTED",
    }
    created_audit = next(item for item in audit_items if item["action"] == "CAMERA_CREATED")
    assert created_audit["actor"]["id"] == "admin-01"
    assert created_audit["requestId"] == "req-security-create"
    serialized = json.dumps(audit_items)
    assert "camera-secret" not in serialized
    assert "rtspUrl" not in serialized
