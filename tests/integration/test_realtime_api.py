import hashlib
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vehicle_intelligence.application.realtime import RealtimeEventService
from vehicle_intelligence.config import (
    AuthConfig,
    AuthPrincipalConfig,
    RealtimeConfig,
    load_settings,
)
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.interfaces.api import create_app

VIEWER_TOKEN = "realtime-viewer-key-" + "r" * 40


def realtime_settings():
    base = load_settings()
    auth = AuthConfig(
        enabled=True,
        principals=[
            AuthPrincipalConfig(
                id="realtime-viewer",
                role="VIEWER",
                key_sha256=hashlib.sha256(VIEWER_TOKEN.encode()).hexdigest(),
            ),
            AuthPrincipalConfig(
                id="realtime-admin",
                role="ADMIN",
                key_sha256=hashlib.sha256((VIEWER_TOKEN + "-admin").encode()).hexdigest(),
            ),
        ],
    )
    realtime = RealtimeConfig(
        enabled=True,
        client_queue_size=4,
        replay_size=8,
        heartbeat_seconds=0.05,
        websocket_auth_timeout_seconds=0.05,
    )
    return base.model_copy(update={"auth": auth, "realtime": realtime})


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {VIEWER_TOKEN}"}


def test_realtime_websocket_auth_delivery_replay_and_gap(sample_event) -> None:
    settings = realtime_settings()
    service = RealtimeEventService(settings.realtime)
    app = create_app(
        settings,
        InMemoryVehicleEventRepository(),
        realtime_service=service,
    )
    codec = JsonEventEnvelopeCodec()

    with TestClient(app) as client:
        assert client.get("/api/events/stream").status_code == 401
        health = client.get("/api/realtime/health", headers=auth_headers())
        assert health.json()["status"] == "ONLINE"

        with client.websocket_connect("/ws/events", headers=auth_headers()) as websocket:
            assert websocket.receive_json()["type"] == "system.realtime.ready"
            client.portal.call(service.publish, sample_event)
            delivered = websocket.receive_text()
            assert codec.decode(delivered) == sample_event

        replay_one = replace(sample_event, id="evt-realtime-replay-1")
        replay_two = replace(sample_event, id="evt-realtime-replay-2")
        client.portal.call(service.publish, replay_one)
        client.portal.call(service.publish, replay_two)
        replay_headers = {**auth_headers(), "Last-Event-ID": replay_one.id}
        with client.websocket_connect("/ws/events", headers=replay_headers) as websocket:
            assert websocket.receive_json()["type"] == "system.realtime.ready"
            assert codec.decode(websocket.receive_text()) == replay_two

        with client.websocket_connect(
            "/ws/events?lastEventId=evt-expired",
            headers=auth_headers(),
        ) as websocket:
            assert websocket.receive_json()["type"] == "system.realtime.ready"
            gap = websocket.receive_json()
            assert gap["type"] == "system.realtime.gap"
            assert gap["data"]["reason"] == "replay_unavailable"


def test_realtime_websocket_browser_message_auth_and_rejection(sample_event) -> None:
    settings = realtime_settings()
    service = RealtimeEventService(settings.realtime)
    app = create_app(
        settings,
        InMemoryVehicleEventRepository(),
        realtime_service=service,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            websocket.send_json({"type": "authenticate", "token": VIEWER_TOKEN})
            assert websocket.receive_json()["type"] == "system.realtime.ready"
            client.portal.call(service.publish, sample_event)
            assert websocket.receive_json()["id"] == sample_event.id

        with client.websocket_connect("/ws/events") as websocket:
            websocket.send_json({"type": "authenticate", "token": "invalid"})
            with pytest.raises(WebSocketDisconnect) as caught:
                websocket.receive_json()
            assert caught.value.code == 4401

        with client.websocket_connect("/ws/events") as websocket:
            websocket.send_bytes(b"not-an-authentication-message")
            with pytest.raises(WebSocketDisconnect) as caught:
                websocket.receive_json()
            assert caught.value.code == 4401

        with client.websocket_connect("/ws/events") as websocket:
            with pytest.raises(WebSocketDisconnect) as caught:
                websocket.receive_json()
            assert caught.value.code == 4401
