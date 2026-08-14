import asyncio
import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from vehicle_intelligence.application.audit import AuditService
from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.discovery import OnvifDiscoveryService
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.policies import (
    AlertService,
    PolicyServices,
    RuleService,
    WatchlistService,
)
from vehicle_intelligence.application.ports import CameraConnectionTestResult
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.config import OnvifDiscoveryConfig, SecurityConfig, load_settings
from vehicle_intelligence.domain import (
    CameraHealth,
    CameraStatus,
    OnvifDiscoveredDevice,
)
from vehicle_intelligence.infrastructure.observability.metrics import PrometheusMetrics
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


class ConnectedTester:
    async def test(self, camera):
        return CameraConnectionTestResult(True, 4.2, camera.updated_at)


class DiscoveredDevices:
    async def discover(self):
        return [
            OnvifDiscoveredDevice(
                endpoint_reference="urn:uuid:onvif-gate",
                xaddrs=("http://192.0.2.55/onvif/device_service",),
                types=("tds:Device", "dn:NetworkVideoTransmitter"),
                scopes=("onvif://www.onvif.org/name/ONVIF%20Gate",),
                remote_address="192.0.2.55",
                name="ONVIF Gate",
                hardware="IPC-55",
                discovered_at=datetime(2026, 8, 9, tzinfo=UTC),
            )
        ]


def in_memory_policy_services() -> PolicyServices:
    normalizer = VietnamPlateNormalizer()
    return PolicyServices(
        watchlists=WatchlistService(InMemoryWatchlistRepository(), normalizer),
        rules=RuleService(InMemoryRuleRepository(), RuleEvaluator()),
        alerts=AlertService(InMemoryAlertRepository(), normalizer),
    )


def camera_payload(secret: str = "top-secret") -> dict[str, object]:
    return {
        "id": "gate-01",
        "name": "Main Gate",
        "stream": {
            "rtspUrl": f"rtsp://admin:{secret}@camera.example/live",
            "fpsLimit": 6,
        },
        "location": {"name": "Entrance", "zone": "ZONE_A"},
        "direction": "BOTH",
        "vision": {"vehicleConfidence": 0.4, "plateConfidence": 0.45},
        "geometry": {
            "vehicleRoi": [[0, 0], [100, 0], [100, 100]],
            "crossingLine": [[0, 50], [100, 50]],
            "crossingPositiveToNegative": "ENTER",
            "finalizeOnCrossing": True,
        },
        "enabled": True,
        "metadata": {"lane": 1},
    }


def test_camera_api_crud_never_returns_rtsp_credentials() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    service = CameraService(cameras, health, ConnectedTester())
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    asyncio.run(
        health.save(
            CameraHealth(
                camera_id="gate-01",
                status=CameraStatus.ONLINE,
                source_fps=25,
                decode_fps=24,
                queue_size=1,
                dropped_frames=0,
                reconnect_count=0,
                connection_failures=0,
                stream_epoch=0,
                last_frame_at=timestamp,
                updated_at=timestamp,
            )
        )
    )
    app = create_app(load_settings(), InMemoryVehicleEventRepository(), service)

    with TestClient(app) as client:
        created = client.post("/api/cameras", json=camera_payload())
        listed = client.get("/api/cameras")
        detail = client.get("/api/cameras/gate-01")
        health_response = client.get("/api/cameras/gate-01/health")
        connection = client.post("/api/cameras/gate-01/test-connection")
        updated_payload = camera_payload()
        updated_payload["revision"] = 1
        updated_payload["name"] = "Updated Gate"
        updated_payload["stream"] = {"fpsLimit": 8}
        updated_payload.pop("id")
        updated = client.put("/api/cameras/gate-01", json=updated_payload)
        disabled = client.post("/api/cameras/gate-01/disable")
        stale = client.put("/api/cameras/gate-01", json=updated_payload)

    assert created.status_code == 201
    serialized = json.dumps(
        [created.json(), listed.json(), detail.json(), updated.json(), disabled.json()]
    )
    assert "top-secret" not in serialized
    assert "rtspUrl" not in serialized
    assert created.json()["stream"] == {
        "fpsLimit": 6.0,
        "credentialsConfigured": True,
    }
    assert health_response.json()["status"] == "ONLINE"
    assert connection.json()["connected"] is True
    assert updated.json()["revision"] == 2
    assert disabled.json()["enabled"] is False
    assert stale.status_code == 409


def test_onvif_discovery_and_batch_camera_import_are_bounded_and_audited() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    service = CameraService(
        cameras,
        health,
        ConnectedTester(),
        maximum_cameras=2,
        batch_create_limit=3,
    )
    audits = InMemoryAuditLogRepository()
    app = create_app(
        load_settings(),
        InMemoryVehicleEventRepository(),
        service,
        audit_service=AuditService(audits),
        onvif_discovery_service=OnvifDiscoveryService(
            DiscoveredDevices(),
            maximum_results=10,
        ),
    )

    first = camera_payload("first-secret")
    second = camera_payload("second-secret")
    second["id"] = "gate-02"
    second["name"] = "Gate 02"
    third = camera_payload("third-secret")
    third["id"] = "gate-03"
    third["name"] = "Gate 03"

    with TestClient(app) as client:
        assert client.post("/api/cameras", json=first).status_code == 201
        discovery = client.post("/api/cameras/discover")
        batch = client.post("/api/cameras/batch", json={"items": [first, second, third]})
        audit_page = client.get("/api/audit-logs", params={"action": "CAMERA_DISCOVERY_RUN"})

    assert discovery.status_code == 200
    assert discovery.json()["count"] == 1
    assert discovery.json()["items"][0]["name"] == "ONVIF Gate"
    assert discovery.json()["items"][0]["serviceAddresses"] == [
        "http://192.0.2.55/onvif/device_service"
    ]
    assert batch.status_code == 200
    assert [item["status"] for item in batch.json()["items"]] == [
        "CONFLICT",
        "CREATED",
        "CAPACITY_REACHED",
    ]
    serialized = json.dumps([discovery.json(), batch.json(), audit_page.json()])
    assert "first-secret" not in serialized
    assert "second-secret" not in serialized
    assert "third-secret" not in serialized
    assert "rtspUrl" not in serialized
    assert audit_page.status_code == 200
    assert audit_page.json()["items"][0]["metadata"]["resultCount"] == 1


def test_prometheus_endpoint_exports_normalized_http_and_latest_camera_metrics() -> None:
    cameras = InMemoryCameraRepository()
    health = InMemoryCameraHealthRepository()
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    asyncio.run(
        health.save(
            CameraHealth(
                camera_id="gate-metrics",
                status=CameraStatus.ONLINE,
                source_fps=25,
                decode_fps=24,
                queue_size=1,
                dropped_frames=3,
                reconnect_count=1,
                connection_failures=0,
                stream_epoch=1,
                last_frame_at=timestamp,
                updated_at=timestamp,
                decoded_frames=120,
                sampled_frames=40,
                vehicle_detections=9,
                ocr_requests=4,
                ocr_success=3,
            )
        )
    )
    metrics = PrometheusMetrics(include_process_metrics=False)
    app = create_app(
        load_settings(),
        InMemoryVehicleEventRepository(),
        CameraService(cameras, health, ConnectedTester()),
        prometheus_metrics=metrics,
    )

    with TestClient(app) as client:
        assert client.get("/api/events/not-a-real-event").status_code == 404
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "text/plain" in response.headers["content-type"]
    assert (
        'http_requests_total{method="GET",route="/api/events/{event_id}",status="404"} 1.0'
        in response.text
    )
    assert 'camera_frames_total{camera_id="gate-metrics",kind="decoded"} 120.0' in response.text
    assert "not-a-real-event" not in response.text


def test_disabled_onvif_discovery_fails_without_opening_network_socket() -> None:
    base = load_settings()
    settings = base.model_copy(update={"onvif_discovery": OnvifDiscoveryConfig(enabled=False)})
    app = create_app(settings, InMemoryVehicleEventRepository())

    with TestClient(app) as client:
        response = client.post("/api/cameras/discover")

    assert response.status_code == 503
    assert response.json()["detail"] == "ONVIF discovery is disabled"


def test_invalid_camera_url_validation_does_not_echo_secret() -> None:
    app = create_app(load_settings(), InMemoryVehicleEventRepository())
    payload = camera_payload("validation-leak")
    payload["stream"] = {"rtspUrl": "rtsp://admin:validation-leak@", "fpsLimit": 6}

    with TestClient(app) as client:
        response = client.post("/api/cameras", json=payload)

    assert response.status_code == 422
    assert "validation-leak" not in response.text


def test_persistent_camera_api_fails_closed_without_encryption_key() -> None:
    base = load_settings()
    settings = base.model_copy(
        update={
            "mongodb": base.mongodb.model_copy(update={"enabled": True}),
            "security": SecurityConfig(),
        }
    )
    app = create_app(
        settings,
        InMemoryVehicleEventRepository(),
        policy_services=in_memory_policy_services(),
        audit_service=AuditService(InMemoryAuditLogRepository()),
    )

    with TestClient(app) as client:
        health_response = client.get("/api/system/health")
        response = client.post("/api/cameras", json=camera_payload())

    assert health_response.json()["cameraManagement"] == "unavailable"
    assert response.status_code == 503


def test_persistent_camera_api_fails_closed_with_invalid_encryption_key() -> None:
    base = load_settings()
    settings = base.model_copy(
        update={
            "mongodb": base.mongodb.model_copy(update={"enabled": True}),
            "security": SecurityConfig(camera_credential_key="invalid-base64"),
        }
    )
    app = create_app(
        settings,
        InMemoryVehicleEventRepository(),
        policy_services=in_memory_policy_services(),
        audit_service=AuditService(InMemoryAuditLogRepository()),
    )

    with TestClient(app) as client:
        health_response = client.get("/api/system/health")
        response = client.post("/api/cameras", json=camera_payload())

    assert health_response.json()["cameraManagement"] == "unavailable"
    assert response.status_code == 503
