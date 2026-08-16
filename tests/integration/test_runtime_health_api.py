from fastapi.testclient import TestClient

from vehicle_intelligence.application.runtime_health import (
    RuntimeDependency,
    RuntimeHealthService,
)
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.interfaces.api import create_app


def local_settings():
    settings = load_settings()
    return settings.model_copy(
        update={
            "mongodb": settings.mongodb.model_copy(update={"enabled": False}),
            "storage": settings.storage.model_copy(update={"backend": "local"}),
            "realtime": settings.realtime.model_copy(update={"enabled": False}),
            "live_monitor": settings.live_monitor.model_copy(update={"enabled": False}),
        }
    )


def test_runtime_health_endpoints_keep_compatibility_summary() -> None:
    app = create_app(local_settings(), InMemoryVehicleEventRepository())

    with TestClient(app) as client:
        liveness = client.get("/livez")
        readiness = client.get("/readyz")
        compatibility = client.get("/api/system/health")

    assert liveness.status_code == 200
    assert liveness.json() == {"status": "alive"}
    assert liveness.headers["cache-control"] == "no-store"
    assert readiness.status_code == 200
    assert readiness.headers["cache-control"] == "no-store"
    assert readiness.json() == {
        "status": "ready",
        "checks": {
            "application": {"status": "ready", "required": True},
            "eventStore": {"status": "ready", "required": True},
            "cameraManagement": {"status": "ready", "required": False},
            "minio": {"status": "disabled", "required": False},
            "realtime": {"status": "disabled", "required": False},
            "liveMonitor": {"status": "disabled", "required": False},
        },
    }
    assert compatibility.status_code == 200
    assert compatibility.json()["status"] == "ok"
    openapi_paths = client.app.openapi()["paths"]
    assert "/livez" not in openapi_paths
    assert "/readyz" not in openapi_paths


def test_readiness_returns_503_without_leaking_probe_failure() -> None:
    async def unavailable() -> bool:
        raise RuntimeError("secret dependency detail")

    health = RuntimeHealthService(
        (RuntimeDependency("eventStore", required=True, probe=unavailable),),
        cache_seconds=0,
    )
    app = create_app(
        local_settings(),
        InMemoryVehicleEventRepository(),
        runtime_health_service=health,
    )

    with TestClient(app) as client:
        readiness = client.get("/readyz")
        compatibility = client.get("/api/system/health")

    assert readiness.status_code == 503
    assert readiness.json() == {
        "status": "not_ready",
        "checks": {
            "application": {"status": "ready", "required": True},
            "eventStore": {"status": "unavailable", "required": True},
        },
    }
    assert "secret dependency detail" not in readiness.text
    assert readiness.headers["cache-control"] == "no-store"
    assert compatibility.status_code == 200
