from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.live_monitor import LiveMonitorService
from vehicle_intelligence.application.ports import CameraConnectionTestResult
from vehicle_intelligence.config import (
    AuthConfig,
    AuthPrincipalConfig,
    LiveMonitorConfig,
    load_settings,
)
from vehicle_intelligence.domain import (
    BoundingBox,
    Direction,
    LiveFrameMetadata,
    LiveFramePacket,
    LiveVehicleOverlay,
)
from vehicle_intelligence.infrastructure.persistence.camera_memory import (
    InMemoryCameraHealthRepository,
    InMemoryCameraRepository,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.interfaces.api import create_app


class ConnectedTester:
    async def test(self, camera):
        return CameraConnectionTestResult(True, 2.5, camera.updated_at)


ADMIN_TOKEN = "live-admin-key-" + "a" * 40
VIEWER_TOKEN = "live-viewer-key-" + "v" * 40


def authenticated_settings():
    base = load_settings()
    auth = AuthConfig(
        enabled=True,
        principals=[
            AuthPrincipalConfig(
                id="live-admin",
                role="ADMIN",
                key_sha256=hashlib.sha256(ADMIN_TOKEN.encode()).hexdigest(),
            ),
            AuthPrincipalConfig(
                id="live-viewer",
                role="VIEWER",
                key_sha256=hashlib.sha256(VIEWER_TOKEN.encode()).hexdigest(),
            ),
        ],
    )
    return base.model_copy(update={"auth": auth})


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def camera_payload() -> dict[str, object]:
    return {
        "id": "gate-live",
        "name": "Live Gate",
        "stream": {"rtspUrl": "rtsp://user:secret@camera.example/live", "fpsLimit": 6},
        "location": {"name": "Entrance", "zone": "ZONE_A"},
        "direction": "BOTH",
        "enabled": True,
    }


def packet() -> LiveFramePacket:
    timestamp = datetime(2026, 8, 9, 15, 0, tzinfo=UTC)
    return LiveFramePacket(
        metadata=LiveFrameMetadata(
            camera_id="gate-live",
            frame_id=7,
            stream_epoch=1,
            captured_at=timestamp,
            source_width=640,
            source_height=360,
            vehicles=(
                LiveVehicleOverlay(
                    track_id="gate-live:session:7",
                    bbox=BoundingBox(10, 20, 200, 300),
                    confidence=0.97,
                    vehicle_type="car",
                    direction=Direction.ENTER,
                ),
            ),
        ),
        jpeg=b"\xff\xd8api-preview\xff\xd9",
        preview_width=640,
        preview_height=360,
    )


def test_live_monitor_api_synchronizes_metadata_and_exact_jpeg() -> None:
    camera_service = CameraService(
        InMemoryCameraRepository(),
        InMemoryCameraHealthRepository(),
        ConnectedTester(),
    )
    live = LiveMonitorService(LiveMonitorConfig(enabled=True))
    app = create_app(
        load_settings(),
        InMemoryVehicleEventRepository(),
        camera_service,
        live_monitor_service=live,
    )

    with TestClient(app) as client:
        created = client.post("/api/cameras", json=camera_payload())
        buffered = live.ingest(packet())
        state = client.get("/api/cameras/gate-live/live")
        frame = client.get(
            "/api/cameras/gate-live/live/frame",
            params={"sequence": buffered.sequence},
        )
        expired = client.get(
            "/api/cameras/gate-live/live/frame",
            params={"sequence": buffered.sequence + 99},
        )
        health = client.get("/api/live-monitor/health")
        client.post("/api/cameras/gate-live/disable")
        disabled = client.get("/api/cameras/gate-live/live")

    assert created.status_code == 201
    assert state.status_code == 200
    body = state.json()
    assert body["status"] == "LIVE"
    assert body["latest"]["sequence"] == buffered.sequence
    assert body["latest"]["vehicles"][0]["trackId"] == "gate-live:session:7"
    assert "jpeg" not in state.text.lower()
    assert "secret" not in state.text
    assert state.headers["cache-control"] == "no-store"
    assert frame.status_code == 200
    assert frame.content == packet().jpeg
    assert frame.headers["x-live-sequence"] == str(buffered.sequence)
    assert frame.headers["cache-control"] == "no-store"
    assert expired.status_code == 410
    assert health.json()["framesReceived"] == 1
    assert disabled.json()["status"] == "DISABLED"
    assert disabled.json()["latest"] is None


def test_live_monitor_metadata_and_jpeg_require_read_permission() -> None:
    camera_service = CameraService(
        InMemoryCameraRepository(),
        InMemoryCameraHealthRepository(),
        ConnectedTester(),
    )
    live = LiveMonitorService(LiveMonitorConfig(enabled=True))
    app = create_app(
        authenticated_settings(),
        InMemoryVehicleEventRepository(),
        camera_service,
        live_monitor_service=live,
    )

    with TestClient(app) as client:
        created = client.post(
            "/api/cameras",
            json=camera_payload(),
            headers=bearer(ADMIN_TOKEN),
        )
        buffered = live.ingest(packet())
        missing = client.get("/api/cameras/gate-live/live")
        state = client.get(
            "/api/cameras/gate-live/live",
            headers=bearer(VIEWER_TOKEN),
        )
        frame = client.get(
            "/api/cameras/gate-live/live/frame",
            params={"sequence": buffered.sequence},
            headers=bearer(VIEWER_TOKEN),
        )

    assert created.status_code == 201
    assert missing.status_code == 401
    assert state.status_code == 200
    assert frame.status_code == 200
