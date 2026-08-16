"""One subprocess per camera for inference and decoder failure isolation."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from vehicle_intelligence.config import GPUSchedulerConfig
from vehicle_intelligence.domain import Camera
from vehicle_intelligence.exceptions import CameraWorkerError, InferenceError
from vehicle_intelligence.infrastructure.inference.protocol import (
    INFERENCE_CAMERA_ENV,
    INFERENCE_SOCKET_ENV,
    INFERENCE_TOKEN_ENV,
    INFERENCE_TOKEN_FD_ENV,
    derive_camera_token,
    derive_supervisor_token,
    validate_inference_token,
)
from vehicle_intelligence.infrastructure.inference.socket_path import (
    SocketIdentity,
    prepare_socket_path,
    socket_identity,
    unlink_owned_socket,
)
from vehicle_intelligence.infrastructure.vision.remote import UnixInferenceClient

RTSP_SECRET_ENV = "VIP_MANAGED_CAMERA_RTSP_URL"
ENCRYPTION_KEY_ENV = "VIP_SECURITY__CAMERA_CREDENTIAL_KEY"

_SYSTEM_ENVIRONMENT_KEYS = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LANG",
        "LC_ALL",
        "TZ",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    }
)
_CAMERA_ENVIRONMENT_PREFIXES = (
    "VIP_APP__",
    "VIP_CAMERA_MANAGER__",
    "VIP_EVENT_BUS__",
    "VIP_EVENTS__",
    "VIP_LIVE_MONITOR__",
    "VIP_MINIO__",
    "VIP_MONGODB__",
    "VIP_OBSERVABILITY__",
    "VIP_REDIS__",
    "VIP_STORAGE__",
    "VIP_TRACKING__",
    "VIP_VISION__",
)
_SERVICE_ENVIRONMENT_PREFIXES = (
    "VIP_VISION__VEHICLE_DETECTION__",
    "VIP_VISION__PLATE_DETECTION__",
)


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
        *,
        inference_socket_path: str | Path | None = None,
        inference_token: str | None = None,
        inference_config: GPUSchedulerConfig | None = None,
    ) -> None:
        if not command:
            raise ValueError("camera worker command cannot be empty")
        if inference_config is not None:
            if not inference_config.enabled:
                raise ValueError("shared inference config must be enabled")
            if (
                inference_socket_path is not None
                and Path(inference_socket_path) != inference_config.socket_path
            ):
                raise ValueError("shared inference socket does not match config")
            inference_socket_path = inference_config.socket_path
        if (inference_socket_path is None) != (inference_token is None):
            raise ValueError("shared inference socket and token must be configured together")
        if inference_token is not None:
            try:
                validate_inference_token(inference_token)
            except InferenceError as exc:
                raise ValueError("shared inference token is invalid") from exc
        self._command = tuple(command)
        self._config_path = str(config_path)
        self._shutdown_seconds = shutdown_seconds
        self._spawn = spawn
        self._inference_socket_path = (
            str(inference_socket_path) if inference_socket_path is not None else None
        )
        self._inference_token = inference_token
        self._inference_config = inference_config

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
            if self._inference_token is None:
                process = await self._spawn(*args, env=env)
            else:
                process = await _spawn_with_token(
                    self._spawn,
                    args,
                    env,
                    derive_camera_token(self._inference_token, camera.id),
                )
        except OSError as exc:
            raise CameraWorkerError(f"cannot start camera worker: {camera.id}") from exc
        return SubprocessCameraWorkerHandle(process, self._shutdown_seconds)

    def _worker_environment(self, camera: Camera) -> dict[str, str]:
        env = _sanitized_environment(_CAMERA_ENVIRONMENT_PREFIXES)
        # An explicit empty value overrides a possible .env key in the child.
        env[ENCRYPTION_KEY_ENV] = ""
        optional_keys = (
            "VIP_CAMERA__ZONE",
            "VIP_CAMERA__ROI",
            "VIP_CAMERA__CROSSING_LINE",
        )
        for key in optional_keys:
            env.pop(key, None)
        for key in (
            INFERENCE_SOCKET_ENV,
            INFERENCE_TOKEN_ENV,
            INFERENCE_TOKEN_FD_ENV,
            INFERENCE_CAMERA_ENV,
        ):
            env.pop(key, None)
        env.update(
            {
                RTSP_SECRET_ENV: camera.rtsp_url.reveal(),
                "VIP_CAMERA__DIRECTION": camera.direction.value,
                "VIP_CAMERA__CROSSING_POSITIVE_TO_NEGATIVE": (
                    camera.crossing_positive_to_negative.value
                ),
                "VIP_CAMERA__FINALIZE_ON_CROSSING": json.dumps(camera.finalize_on_crossing),
                "VIP_VISION__VEHICLE_DETECTION__CONFIDENCE": str(camera.vehicle_confidence),
                "VIP_VISION__PLATE_DETECTION__CONFIDENCE": str(camera.plate_confidence),
                "VIP_MONGODB__ENABLED": "true",
                "VIP_GPU_SCHEDULER__ENABLED": (
                    "true" if self._inference_socket_path is not None else "false"
                ),
            }
        )
        if self._inference_socket_path is not None and self._inference_token is not None:
            env.update(
                {
                    INFERENCE_SOCKET_ENV: self._inference_socket_path,
                    INFERENCE_CAMERA_ENV: camera.id,
                    "VIP_GPU_SCHEDULER__SOCKET_PATH": self._inference_socket_path,
                }
            )
            if self._inference_config is not None:
                env.update(_scheduler_environment(self._inference_config))
        if camera.zone is not None:
            env["VIP_CAMERA__ZONE"] = camera.zone
        if camera.roi is not None:
            env["VIP_CAMERA__ROI"] = json.dumps([[point.x, point.y] for point in camera.roi])
        if camera.crossing_line is not None:
            env["VIP_CAMERA__CROSSING_LINE"] = json.dumps(
                [[point.x, point.y] for point in camera.crossing_line]
            )
        return env


class SubprocessInferenceServiceHandle(SubprocessCameraWorkerHandle):
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        shutdown_seconds: float,
        socket_path: Path,
    ) -> None:
        super().__init__(process, shutdown_seconds)
        self._socket_path = socket_path
        self._socket_identity: SocketIdentity | None = None

    def claim_socket(self, identity: SocketIdentity) -> None:
        self._socket_identity = identity

    async def stop(self) -> None:
        stop_task = asyncio.create_task(super().stop())
        try:
            await asyncio.shield(stop_task)
        except asyncio.CancelledError:
            await asyncio.shield(stop_task)
            raise
        finally:
            if not self.running:
                with suppress(InferenceError):
                    unlink_owned_socket(self._socket_path, self._socket_identity)


class SubprocessInferenceServiceLauncher:
    def __init__(
        self,
        config: GPUSchedulerConfig,
        config_path: str | Path,
        token: str,
        spawn: Any = asyncio.create_subprocess_exec,
        probe: Callable[[], None] | None = None,
    ) -> None:
        try:
            validate_inference_token(token)
        except InferenceError as exc:
            raise ValueError("shared inference token is invalid") from exc
        self._config = config
        self._config_path = str(config_path)
        self._token = token
        self._spawn = spawn
        self._probe = probe or self._ping

    async def start(self) -> SubprocessInferenceServiceHandle:
        await asyncio.to_thread(prepare_socket_path, self._config.socket_path)
        args = (
            *self._config.service_command,
            "--config",
            self._config_path,
            "--socket",
            str(self._config.socket_path),
        )
        try:
            process = await _spawn_with_token(
                self._spawn,
                args,
                self._service_environment(),
                self._token,
            )
        except OSError as exc:
            raise CameraWorkerError("cannot spawn shared inference service") from exc
        handle = SubprocessInferenceServiceHandle(
            process,
            self._config.shutdown_timeout_seconds,
            self._config.socket_path,
        )
        try:
            return await self._await_ready(handle)
        except BaseException:
            await asyncio.shield(handle.stop())
            raise

    async def _await_ready(
        self,
        handle: SubprocessInferenceServiceHandle,
    ) -> SubprocessInferenceServiceHandle:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.startup_timeout_seconds
        while handle.running:
            try:
                await asyncio.to_thread(self._probe)
                identity = socket_identity(self._config.socket_path)
                if identity is None:
                    raise InferenceError("shared inference socket is missing after startup")
                handle.claim_socket(identity)
                return handle
            except InferenceError:
                if loop.time() >= deadline:
                    break
                await asyncio.sleep(min(0.05, max(0, deadline - loop.time())))
        if handle.return_code is None:
            raise CameraWorkerError("shared inference service startup timed out")
        raise CameraWorkerError(
            f"shared inference service exited during startup with code {handle.return_code}"
        )

    def _ping(self) -> None:
        UnixInferenceClient(
            self._config.socket_path,
            "supervisor-probe",
            derive_supervisor_token(self._token),
            timeout_seconds=min(1.0, self._config.request_timeout_seconds),
            maximum_payload_bytes=self._config.maximum_payload_bytes,
            maximum_images=self._config.maximum_images_per_request,
        ).ping()

    def _service_environment(self) -> dict[str, str]:
        env = _sanitized_environment(_SERVICE_ENVIRONMENT_PREFIXES)
        for key in (RTSP_SECRET_ENV, ENCRYPTION_KEY_ENV, INFERENCE_CAMERA_ENV):
            env.pop(key, None)
        env.update(
            {
                INFERENCE_SOCKET_ENV: str(self._config.socket_path),
            }
        )
        env.update(_scheduler_environment(self._config))
        return env


def _sanitized_environment(prefixes: tuple[str, ...]) -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key in _SYSTEM_ENVIRONMENT_KEYS or key.startswith(prefixes)
    }


def _scheduler_environment(config: GPUSchedulerConfig) -> dict[str, str]:
    return {
        "VIP_GPU_SCHEDULER__ENABLED": "true",
        "VIP_GPU_SCHEDULER__SOCKET_PATH": str(config.socket_path),
        "VIP_GPU_SCHEDULER__MAXIMUM_CAMERAS": str(config.maximum_cameras),
        "VIP_GPU_SCHEDULER__MAXIMUM_CLIENTS": str(config.maximum_clients),
        "VIP_GPU_SCHEDULER__MAXIMUM_BATCH_SIZE": str(config.maximum_batch_size),
        "VIP_GPU_SCHEDULER__PER_CAMERA_QUEUE_SIZE": str(config.per_camera_queue_size),
        "VIP_GPU_SCHEDULER__MAXIMUM_FRAME_AGE_MS": str(config.maximum_frame_age_ms),
        "VIP_GPU_SCHEDULER__BATCH_WAIT_MS": str(config.batch_wait_ms),
        "VIP_GPU_SCHEDULER__REQUEST_TIMEOUT_SECONDS": str(config.request_timeout_seconds),
        "VIP_GPU_SCHEDULER__MAXIMUM_PAYLOAD_BYTES": str(config.maximum_payload_bytes),
        "VIP_GPU_SCHEDULER__MAXIMUM_INFLIGHT_PAYLOAD_BYTES": str(
            config.maximum_inflight_payload_bytes
        ),
        "VIP_GPU_SCHEDULER__MAXIMUM_IMAGES_PER_REQUEST": str(config.maximum_images_per_request),
        "VIP_GPU_SCHEDULER__MAXIMUM_ISOLATION_ATTEMPTS": str(config.maximum_isolation_attempts),
        "VIP_GPU_SCHEDULER__CAMERA_FAILURE_THRESHOLD": str(config.camera_failure_threshold),
        "VIP_GPU_SCHEDULER__CAMERA_QUARANTINE_SECONDS": str(config.camera_quarantine_seconds),
        "VIP_GPU_SCHEDULER__PROVIDER_FAILURE_THRESHOLD": str(config.provider_failure_threshold),
        "VIP_GPU_SCHEDULER__PROVIDER_FAILURE_MINIMUM_CAMERAS": str(
            config.provider_failure_minimum_cameras
        ),
    }


async def _spawn_with_token(
    spawn: Any,
    args: tuple[str, ...],
    environment: dict[str, str],
    token: str,
) -> asyncio.subprocess.Process:
    read_descriptor, write_descriptor = os.pipe()
    try:
        try:
            encoded = validate_inference_token(token).encode("ascii")
            if os.write(write_descriptor, encoded) != len(encoded):
                raise OSError("cannot write shared inference token pipe")
        finally:
            os.close(write_descriptor)
        environment[INFERENCE_TOKEN_FD_ENV] = str(read_descriptor)
        return await spawn(*args, env=environment, pass_fds=(read_descriptor,))
    finally:
        with suppress(OSError):
            os.close(read_descriptor)
