import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vehicle_intelligence.config import GPUSchedulerConfig
from vehicle_intelligence.domain import Camera, CameraDirection, Point, SecretUri
from vehicle_intelligence.exceptions import InferenceError
from vehicle_intelligence.infrastructure.inference.protocol import (
    INFERENCE_TOKEN_FD_ENV,
    derive_camera_token,
)
from vehicle_intelligence.infrastructure.inference.socket_path import SocketIdentity
from vehicle_intelligence.infrastructure.supervision import subprocess as subprocess_module
from vehicle_intelligence.infrastructure.supervision.subprocess import (
    ENCRYPTION_KEY_ENV,
    RTSP_SECRET_ENV,
    SubprocessCameraWorkerLauncher,
    SubprocessInferenceServiceHandle,
    SubprocessInferenceServiceLauncher,
)


class FakeProcess:
    returncode = None


class ManagedFakeProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.terminate_count = 0

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


class DelayedStopProcess(ManagedFakeProcess):
    def __init__(self) -> None:
        super().__init__()
        self.terminated = asyncio.Event()
        self.release = asyncio.Event()

    def terminate(self) -> None:
        self.terminate_count += 1
        self.terminated.set()

    async def wait(self) -> int:
        await self.release.wait()
        self.returncode = 0
        return 0


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


async def test_camera_launcher_passes_only_camera_bound_capability_via_fd(monkeypatch) -> None:
    captured = {}
    master = "m" * 32
    inference_config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=2,
        maximum_batch_size=2,
        socket_path=Path("/private/shared/inference.sock"),
        maximum_images_per_request=2,
        maximum_isolation_attempts=3,
        camera_failure_threshold=4,
        camera_quarantine_seconds=45,
        provider_failure_threshold=5,
        provider_failure_minimum_cameras=2,
    )

    async def spawn(*args, **kwargs):
        descriptor = kwargs["pass_fds"][0]
        captured["token"] = __import__("os").read(descriptor, 256).decode()
        captured["env"] = kwargs["env"]
        captured["args"] = args
        return FakeProcess()

    monkeypatch.setenv("UNRELATED_API_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("VIP_SECURITY__JWT_SECRET", "must-not-reach-child")
    configured = Camera(
        id="gate-01",
        name="Main Gate",
        rtsp_url=SecretUri("rtsp://camera.example/live"),
        fps_limit=8,
        direction=CameraDirection.ENTRY,
        enabled=True,
        vehicle_confidence=0.5,
        plate_confidence=0.6,
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        updated_at=datetime(2026, 8, 9, tzinfo=UTC),
    )
    launcher = SubprocessCameraWorkerLauncher(
        ["vehicle-camera"],
        "configs/default.yaml",
        10,
        spawn=spawn,
        inference_token=master,
        inference_config=inference_config,
    )

    await launcher.start(configured)

    assert captured["token"] == derive_camera_token(master, "gate-01")
    assert INFERENCE_TOKEN_FD_ENV in captured["env"]
    assert "UNRELATED_API_TOKEN" not in captured["env"]
    assert "VIP_SECURITY__JWT_SECRET" not in captured["env"]
    assert captured["env"]["VIP_GPU_SCHEDULER__MAXIMUM_ISOLATION_ATTEMPTS"] == "3"
    assert captured["env"]["VIP_GPU_SCHEDULER__CAMERA_FAILURE_THRESHOLD"] == "4"
    assert captured["env"]["VIP_GPU_SCHEDULER__CAMERA_QUARANTINE_SECONDS"] == "45.0"
    assert captured["env"]["VIP_GPU_SCHEDULER__PROVIDER_FAILURE_THRESHOLD"] == "5"
    assert captured["env"]["VIP_GPU_SCHEDULER__PROVIDER_FAILURE_MINIMUM_CAMERAS"] == "2"
    assert master not in repr(captured)


async def test_inference_launcher_cleans_process_when_startup_is_cancelled(monkeypatch) -> None:
    process = ManagedFakeProcess()
    spawned = asyncio.Event()

    async def spawn(*_args, **_kwargs):
        spawned.set()
        return process

    monkeypatch.setattr(subprocess_module, "prepare_socket_path", lambda _path: None)
    monkeypatch.setattr(
        subprocess_module,
        "socket_identity",
        lambda _path: SocketIdentity(1, 2),
    )
    monkeypatch.setattr(subprocess_module, "unlink_owned_socket", lambda *_args: True)
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=1,
        maximum_batch_size=1,
        socket_path=Path("/private/shared/inference.sock"),
        startup_timeout_seconds=10,
        maximum_images_per_request=1,
        provider_failure_minimum_cameras=1,
    )
    launcher = SubprocessInferenceServiceLauncher(
        config,
        "configs/default.yaml",
        "m" * 32,
        spawn=spawn,
        probe=lambda: (_ for _ in ()).throw(InferenceError("not ready")),
    )

    task = asyncio.create_task(launcher.start())
    await spawned.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.terminate_count == 1


async def test_inference_handle_waits_for_exit_before_unlink_when_stop_is_cancelled(
    monkeypatch,
) -> None:
    process = DelayedStopProcess()
    unlinked = []
    monkeypatch.setattr(
        subprocess_module,
        "unlink_owned_socket",
        lambda *args: unlinked.append(args) or True,
    )
    handle = SubprocessInferenceServiceHandle(
        process,
        shutdown_seconds=1,
        socket_path=Path("/private/shared/inference.sock"),
    )
    handle.claim_socket(SocketIdentity(1, 2))

    stop = asyncio.create_task(handle.stop())
    await process.terminated.wait()
    stop.cancel()
    await asyncio.sleep(0)
    assert unlinked == []

    process.release.set()
    with pytest.raises(asyncio.CancelledError):
        await stop
    assert len(unlinked) == 1
    assert process.returncode == 0
