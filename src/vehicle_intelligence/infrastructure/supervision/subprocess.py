"""One subprocess per camera for inference and decoder failure isolation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from vehicle_intelligence.domain import Camera
from vehicle_intelligence.exceptions import CameraWorkerError

RTSP_SECRET_ENV = "VIP_MANAGED_CAMERA_RTSP_URL"
ENCRYPTION_KEY_ENV = "VIP_SECURITY__CAMERA_CREDENTIAL_KEY"


class SubprocessCameraWorkerHandle:
    def __init__(self, process: asyncio.subprocess.Process, shutdown_seconds: float) -> None:
        self._process = process
        self._shutdown_seconds = shutdown_seconds

    @property
    def running(self) -> bool:
        return self._process.returncode is None

    @property
    def return_code(self) -> int | None:
        return self._process.returncode

    async def stop(self) -> None:
        if self._process.returncode is not None:
            return
        try:
            self._process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self._process.wait(), timeout=self._shutdown_seconds)
        except TimeoutError:
            self._process.kill()
            await self._process.wait()


class SubprocessCameraWorkerLauncher:
    def __init__(
        self,
        command: list[str],
        config_path: str | Path,
        shutdown_seconds: float,
        spawn: Any = asyncio.create_subprocess_exec,
    ) -> None:
        if not command:
            raise ValueError("camera worker command cannot be empty")
        self._command = tuple(command)
        self._config_path = str(config_path)
        self._shutdown_seconds = shutdown_seconds
        self._spawn = spawn

    async def start(self, camera: Camera) -> SubprocessCameraWorkerHandle:
        env = self._worker_environment(camera)
        args = (
            *self._command,
            "--config",
            self._config_path,
            "--camera",
            camera.id,
            "--camera-name",
            camera.name,
            "--fps-limit",
            str(camera.fps_limit),
            "--rtsp-env",
            RTSP_SECRET_ENV,
        )
        try:
            process = await self._spawn(*args, env=env)
        except OSError as exc:
            raise CameraWorkerError(f"cannot start camera worker: {camera.id}") from exc
        return SubprocessCameraWorkerHandle(process, self._shutdown_seconds)

    @staticmethod
    def _worker_environment(camera: Camera) -> dict[str, str]:
        env = dict(os.environ)
        # An explicit empty value overrides a possible .env key in the child.
        env[ENCRYPTION_KEY_ENV] = ""
        optional_keys = (
            "VIP_CAMERA__ZONE",
            "VIP_CAMERA__ROI",
            "VIP_CAMERA__CROSSING_LINE",
        )
        for key in optional_keys:
            env.pop(key, None)
        env.update(
            {
                RTSP_SECRET_ENV: camera.rtsp_url.reveal(),
                "VIP_CAMERA__DIRECTION": camera.direction.value,
                "VIP_CAMERA__CROSSING_POSITIVE_TO_NEGATIVE": (
                    camera.crossing_positive_to_negative.value
                ),
                "VIP_CAMERA__FINALIZE_ON_CROSSING": json.dumps(
                    camera.finalize_on_crossing
                ),
                "VIP_VISION__VEHICLE_DETECTION__CONFIDENCE": str(
                    camera.vehicle_confidence
                ),
                "VIP_VISION__PLATE_DETECTION__CONFIDENCE": str(camera.plate_confidence),
                "VIP_MONGODB__ENABLED": "true",
            }
        )
        if camera.zone is not None:
            env["VIP_CAMERA__ZONE"] = camera.zone
        if camera.roi is not None:
            env["VIP_CAMERA__ROI"] = json.dumps([[point.x, point.y] for point in camera.roi])
        if camera.crossing_line is not None:
            env["VIP_CAMERA__CROSSING_LINE"] = json.dumps(
                [[point.x, point.y] for point in camera.crossing_line]
            )
        return env
