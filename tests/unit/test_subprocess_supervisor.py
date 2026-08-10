from datetime import UTC, datetime

from vehicle_intelligence.domain import Camera, CameraDirection, Point, SecretUri
from vehicle_intelligence.infrastructure.supervision.subprocess import (
    ENCRYPTION_KEY_ENV,
    RTSP_SECRET_ENV,
    SubprocessCameraWorkerLauncher,
)


class FakeProcess:
    returncode = None


async def test_subprocess_launcher_passes_rtsp_only_in_redacted_child_environment(
    monkeypatch,
) -> None:
    captured = {}

    async def spawn(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setenv(ENCRYPTION_KEY_ENV, "must-not-reach-child")
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    secret = "rtsp://admin:camera-secret@camera.example/live"
    configured = Camera(
        id="gate-01",
        name="Main Gate",
        rtsp_url=SecretUri(secret),
        fps_limit=8,
        direction=CameraDirection.ENTRY,
        enabled=True,
        vehicle_confidence=0.5,
        plate_confidence=0.6,
        roi=(Point(0, 0), Point(100, 0), Point(100, 100)),
        created_at=timestamp,
        updated_at=timestamp,
    )
    launcher = SubprocessCameraWorkerLauncher(
        ["vehicle-camera"],
        "configs/default.yaml",
        10,
        spawn=spawn,
    )

    handle = await launcher.start(configured)

    assert handle.running
    assert secret not in " ".join(captured["args"])
    assert captured["env"][RTSP_SECRET_ENV] == secret
    assert captured["env"][ENCRYPTION_KEY_ENV] == ""
    assert captured["env"]["VIP_CAMERA__DIRECTION"] == "ENTRY"
    assert "camera-secret" not in repr(configured)
