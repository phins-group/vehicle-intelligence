import asyncio
import hashlib
import hmac
import json
import os
import socket
import stat
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from numpy.typing import NDArray

from vehicle_intelligence.config import GPUSchedulerConfig
from vehicle_intelligence.domain import Detection, PlateDetection
from vehicle_intelligence.exceptions import InferenceError, InferenceProtocolError
from vehicle_intelligence.infrastructure.inference.protocol import (
    INFERENCE_TOKEN_ENV,
    INFERENCE_TOKEN_FD_ENV,
    decode_request,
    derive_camera_token,
    derive_supervisor_token,
    encode_detect_request,
    read_inference_token,
)
from vehicle_intelligence.infrastructure.inference.service import SharedInferenceService
from vehicle_intelligence.infrastructure.inference.socket_path import (
    prepare_socket_path,
    socket_identity,
    unlink_owned_socket,
)
from vehicle_intelligence.infrastructure.vision.remote import (
    RemoteVehicleDetector,
    UnixInferenceClient,
)

TOKEN = "hardening-token-" + "x" * 32
MAXIMUM_PAYLOAD_BYTES = 1_048_576


class EmptyVehicleDetector:
    def detect(self, _image: NDArray[np.uint8]) -> list[Detection]:
        return []


class EmptyPlateDetector:
    def detect(self, _image: NDArray[np.uint8]) -> list[PlateDetection]:
        return []


class SelectiveFailureVehicleDetector:
    def __init__(self, failing_marker: int) -> None:
        self._failing_marker = failing_marker
        self.calls: list[int] = []

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        marker = int(image[0, 0, 0])
        self.calls.append(marker)
        if marker == self._failing_marker:
            raise RuntimeError("provider rejected image")
        return []


class CancelledVehicleDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _image: NDArray[np.uint8]) -> list[Detection]:
        self.calls += 1
        if self.calls == 1:
            raise asyncio.CancelledError
        return []


@pytest.fixture
def ipc_socket_path() -> Iterator[Path]:
    with TemporaryDirectory(prefix="vi-hardening-", dir="/tmp") as directory:
        yield Path(directory) / "inference.sock"


def _scheduler(socket_path: Path, **overrides: object) -> GPUSchedulerConfig:
    values: dict[str, object] = {
        "enabled": True,
        "maximum_cameras": 3,
        "maximum_clients": 6,
        "maximum_batch_size": 1,
        "per_camera_queue_size": 1,
        "maximum_frame_age_ms": 500,
        "batch_wait_ms": 0,
        "socket_path": socket_path,
        "request_timeout_seconds": 2,
        "maximum_payload_bytes": MAXIMUM_PAYLOAD_BYTES,
        "maximum_inflight_payload_bytes": MAXIMUM_PAYLOAD_BYTES * 2,
        "maximum_images_per_request": 1,
    }
    values.update(overrides)
    return GPUSchedulerConfig.model_validate(values)


def _client(config: GPUSchedulerConfig, camera_id: str) -> UnixInferenceClient:
    return UnixInferenceClient(
        config.socket_path,
        camera_id,
        derive_camera_token(TOKEN, camera_id),
        timeout_seconds=config.request_timeout_seconds,
        maximum_payload_bytes=config.maximum_payload_bytes,
        maximum_images=config.maximum_images_per_request,
    )


def _supervisor_client(config: GPUSchedulerConfig) -> UnixInferenceClient:
    return UnixInferenceClient(
        config.socket_path,
        "supervisor-probe",
        derive_supervisor_token(TOKEN),
        timeout_seconds=config.request_timeout_seconds,
        maximum_payload_bytes=config.maximum_payload_bytes,
        maximum_images=config.maximum_images_per_request,
    )


async def _open_authenticated_header(
    socket_path: Path,
    payload: bytes,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    header_length = int.from_bytes(payload[:4], "big")
    header_end = 4 + header_length
    writer.write(len(payload).to_bytes(4, "big") + payload[:header_end])
    await writer.drain()
    return reader, writer


async def _wait_until(predicate: Callable[[], bool], timeout: float = 1) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


async def _close_connection(
    connection: tuple[asyncio.StreamReader, asyncio.StreamWriter],
) -> None:
    _, writer = connection
    writer.close()
    with suppress(ConnectionError, OSError):
        await writer.wait_closed()


@pytest.mark.parametrize(
    ("large_camera_ids", "retry_camera_id", "scheduler_overrides"),
    [
        (("gate-a", "gate-b"), "gate-c", {}),
        (
            ("gate-a", "gate-a"),
            "gate-a",
            {
                "maximum_cameras": 1,
                "per_camera_queue_size": 3,
                "maximum_inflight_payload_bytes": MAXIMUM_PAYLOAD_BYTES * 3,
                "provider_failure_minimum_cameras": 1,
            },
        ),
    ],
    ids=("global-budget", "per-camera-budget"),
)
async def test_payload_budget_rejects_excess_and_releases_disconnected_owner(
    ipc_socket_path: Path,
    large_camera_ids: tuple[str, str],
    retry_camera_id: str,
    scheduler_overrides: dict[str, object],
) -> None:
    config = _scheduler(ipc_socket_path, **scheduler_overrides)
    service = SharedInferenceService(
        config,
        EmptyVehicleDetector(),
        EmptyPlateDetector(),
        TOKEN,
    )
    large_image = np.zeros((591, 591, 3), dtype=np.uint8)
    large_payloads = tuple(
        encode_detect_request(
            request_id * 32,
            camera_id,
            "vehicle",
            (large_image,),
            derive_camera_token(TOKEN, camera_id),
            config.maximum_payload_bytes,
            config.maximum_images_per_request,
        )
        for request_id, camera_id in zip(("a", "b"), large_camera_ids, strict=True)
    )
    assert sum(map(len, large_payloads)) < config.maximum_inflight_payload_bytes
    retry_image = np.zeros((16, 16, 3), dtype=np.uint8)
    retry_payload = encode_detect_request(
        "c" * 32,
        retry_camera_id,
        "vehicle",
        (retry_image,),
        derive_camera_token(TOKEN, retry_camera_id),
        config.maximum_payload_bytes,
        config.maximum_images_per_request,
    )
    applicable_budget = min(
        config.maximum_inflight_payload_bytes,
        config.maximum_payload_bytes * 2,
    )
    assert sum(map(len, large_payloads)) + len(retry_payload) > applicable_budget
    if large_camera_ids[0] == large_camera_ids[1]:
        assert (
            sum(map(len, large_payloads)) + len(retry_payload)
            <= config.maximum_inflight_payload_bytes
        )

    connections: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
    await service.start()
    try:
        for payload in large_payloads:
            connections.append(await _open_authenticated_header(config.socket_path, payload))
        await _wait_until(lambda: service._inflight_payload_bytes == sum(map(len, large_payloads)))

        with pytest.raises(InferenceError, match="request_failed"):
            await asyncio.to_thread(
                RemoteVehicleDetector(_client(config, retry_camera_id)).detect,
                retry_image,
            )

        await _close_connection(connections[0])
        await _wait_until(lambda: service._inflight_payload_bytes == len(large_payloads[1]))
        assert (
            await asyncio.to_thread(
                RemoteVehicleDetector(_client(config, retry_camera_id)).detect,
                retry_image,
            )
            == []
        )
    finally:
        await asyncio.gather(
            *(_close_connection(connection) for connection in connections),
            return_exceptions=True,
        )
        await service.close()

    assert service._inflight_payload_bytes == 0


async def test_slowloris_clients_time_out_and_release_all_client_slots(
    ipc_socket_path: Path,
) -> None:
    config = _scheduler(
        ipc_socket_path,
        maximum_cameras=1,
        maximum_clients=4,
        request_timeout_seconds=0.3,
        maximum_inflight_payload_bytes=MAXIMUM_PAYLOAD_BYTES,
        provider_failure_minimum_cameras=1,
    )
    service = SharedInferenceService(
        config,
        EmptyVehicleDetector(),
        EmptyPlateDetector(),
        TOKEN,
    )
    slow_connections: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
    await service.start()
    try:
        for _ in range(config.maximum_clients):
            reader, writer = await asyncio.open_unix_connection(config.socket_path)
            writer.write(b"\x00")
            await writer.drain()
            slow_connections.append((reader, writer))
        await _wait_until(lambda: service._active_clients == config.maximum_clients)

        with pytest.raises(InferenceError, match="service_capacity_exhausted"):
            await asyncio.to_thread(_supervisor_client(config).ping)

        await _wait_until(lambda: service._active_clients == 0)
        await asyncio.to_thread(_supervisor_client(config).ping)
        assert service.stats.rejected >= config.maximum_clients + 1
    finally:
        await asyncio.gather(
            *(_close_connection(connection) for connection in slow_connections),
            return_exceptions=True,
        )
        await service.close()


async def test_quarantine_rejects_before_admission_and_expiry_restores_camera(
    ipc_socket_path: Path,
) -> None:
    config = _scheduler(
        ipc_socket_path,
        maximum_cameras=1,
        maximum_inflight_payload_bytes=MAXIMUM_PAYLOAD_BYTES,
        camera_failure_threshold=1,
        camera_quarantine_seconds=60,
        provider_failure_threshold=100,
        provider_failure_minimum_cameras=1,
    )
    detector = SelectiveFailureVehicleDetector(failing_marker=99)
    service = SharedInferenceService(config, detector, EmptyPlateDetector(), TOKEN)
    remote = RemoteVehicleDetector(_client(config, "gate-a"))
    await service.start()
    try:
        with pytest.raises(InferenceError, match="request_failed"):
            await asyncio.to_thread(remote.detect, np.full((8, 8, 3), 99, dtype=np.uint8))
        await _wait_until(lambda: service._inflight_payload_bytes == 0)
        assert service._camera_admissions == {}
        assert service._camera_payload_bytes == {}

        with pytest.raises(InferenceError, match="request_failed"):
            await asyncio.to_thread(remote.detect, np.full((8, 8, 3), 1, dtype=np.uint8))
        assert detector.calls == [99]
        assert service._camera_admissions == {}
        assert service._camera_payload_bytes == {}

        quarantine_key = ("gate-a", "vehicle")
        service._camera_quarantine_until[quarantine_key] = asyncio.get_running_loop().time() - 1
        assert (
            await asyncio.to_thread(
                remote.detect,
                np.full((8, 8, 3), 1, dtype=np.uint8),
            )
            == []
        )
        await _wait_until(lambda: service._inflight_payload_bytes == 0)
        assert quarantine_key not in service._camera_quarantine_until
        assert detector.calls == [99, 1]
        assert service.stats.camera_quarantines == 1
        assert service.stats.quarantined_requests == 1
    finally:
        await service.close()


async def test_success_resets_provider_breaker_before_cross_camera_trip(
    ipc_socket_path: Path,
) -> None:
    config = _scheduler(
        ipc_socket_path,
        maximum_cameras=2,
        camera_failure_threshold=100,
        provider_failure_threshold=2,
        provider_failure_minimum_cameras=2,
    )
    detector = SelectiveFailureVehicleDetector(failing_marker=99)
    service = SharedInferenceService(config, detector, EmptyPlateDetector(), TOKEN)
    camera_a = RemoteVehicleDetector(_client(config, "gate-a"))
    camera_b = RemoteVehicleDetector(_client(config, "gate-b"))
    poison = np.full((8, 8, 3), 99, dtype=np.uint8)
    healthy = np.full((8, 8, 3), 1, dtype=np.uint8)
    await service.start()
    try:
        with pytest.raises(InferenceError):
            await asyncio.to_thread(camera_a.detect, poison)
        assert service.running

        assert await asyncio.to_thread(camera_a.detect, healthy) == []

        with pytest.raises(InferenceError):
            await asyncio.to_thread(camera_b.detect, poison)
        assert service.running

        with pytest.raises(InferenceError):
            await asyncio.to_thread(camera_a.detect, poison)
        with pytest.raises(InferenceError, match="circuit breaker"):
            await asyncio.wait_for(service.wait(), timeout=0.5)
        assert service.stats.circuit_breaker_trips == 1
        assert not service.running
    finally:
        await service.close()


async def test_provider_cancelled_error_is_data_failure_without_killing_dispatcher(
    ipc_socket_path: Path,
) -> None:
    config = _scheduler(
        ipc_socket_path,
        maximum_cameras=1,
        request_timeout_seconds=0.1,
        maximum_inflight_payload_bytes=MAXIMUM_PAYLOAD_BYTES,
        provider_failure_minimum_cameras=1,
    )
    detector = CancelledVehicleDetector()
    service = SharedInferenceService(
        config,
        detector,
        EmptyPlateDetector(),
        TOKEN,
    )
    await service.start()
    try:
        result = await asyncio.gather(
            asyncio.to_thread(
                RemoteVehicleDetector(_client(config, "gate-a")).detect,
                np.zeros((8, 8, 3), dtype=np.uint8),
            ),
            return_exceptions=True,
        )
        assert isinstance(result[0], InferenceError)
        assert service.running
        assert (
            await asyncio.to_thread(
                RemoteVehicleDetector(_client(config, "gate-a")).detect,
                np.zeros((8, 8, 3), dtype=np.uint8),
            )
            == []
        )
        assert detector.calls == 2
        assert service.stats.provider_failures == 1
        assert service.stats.circuit_breaker_trips == 0
    finally:
        await service.close()


async def test_service_socket_and_parent_are_private(ipc_socket_path: Path) -> None:
    config = _scheduler(ipc_socket_path)
    service = SharedInferenceService(
        config,
        EmptyVehicleDetector(),
        EmptyPlateDetector(),
        TOKEN,
    )

    await service.start()
    try:
        assert stat.S_IMODE(config.socket_path.lstat().st_mode) == 0o600
        assert stat.S_IMODE(config.socket_path.parent.lstat().st_mode) == 0o700
    finally:
        await service.close()


def test_prepare_socket_path_preserves_live_listener(ipc_socket_path: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(ipc_socket_path))
        listener.listen()

        with pytest.raises(InferenceError, match="already serving"):
            prepare_socket_path(ipc_socket_path)

        assert socket_identity(ipc_socket_path) is not None


def test_prepare_socket_path_removes_stale_listener(ipc_socket_path: Path) -> None:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(ipc_socket_path))
    listener.close()
    assert socket_identity(ipc_socket_path) is not None

    prepare_socket_path(ipc_socket_path)

    assert socket_identity(ipc_socket_path) is None


def test_owned_socket_cleanup_preserves_inode_replacement(ipc_socket_path: Path) -> None:
    replacement_path = ipc_socket_path.with_name("replacement.sock")
    with (
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as original,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as replacement,
    ):
        original.bind(str(ipc_socket_path))
        old_identity = socket_identity(ipc_socket_path)
        assert old_identity is not None
        replacement.bind(str(replacement_path))
        os.replace(replacement_path, ipc_socket_path)
        new_identity = socket_identity(ipc_socket_path)
        assert new_identity is not None and new_identity != old_identity

        assert unlink_owned_socket(ipc_socket_path, old_identity) is False
        assert socket_identity(ipc_socket_path) == new_identity
        assert unlink_owned_socket(ipc_socket_path, new_identity) is True


@pytest.mark.parametrize(
    ("image_specification", "body", "error"),
    [
        ({"shape": [2, 2], "byteLength": 4}, b"\0" * 4, "specification"),
        ({"shape": [2, 2, 4], "byteLength": 16}, b"\0" * 16, "channel count"),
        ({"shape": [2, 2, 3], "byteLength": 11}, b"\0" * 11, "byte length"),
        ({"shape": [2, True, 3], "byteLength": 6}, b"\0" * 6, "specification"),
        ({"shape": [2, 2, 3], "byteLength": True}, b"\0", "specification"),
    ],
)
def test_signed_raw_request_rejects_malformed_image_header(
    image_specification: dict[str, object],
    body: bytes,
    error: str,
) -> None:
    camera_token = derive_camera_token(TOKEN, "gate-a")
    document: dict[str, object] = {
        "schemaVersion": 1,
        "type": "detect",
        "requestId": "d" * 32,
        "cameraId": "gate-a",
        "detector": "vehicle",
        "images": [image_specification],
        "bodySha256": hashlib.sha256(body).hexdigest(),
    }
    canonical = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    document["signature"] = hmac.new(
        camera_token.encode(),
        canonical,
        hashlib.sha256,
    ).hexdigest()
    header = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    payload = len(header).to_bytes(4, "big") + header + body

    with pytest.raises(InferenceProtocolError, match=error):
        decode_request(payload, TOKEN, maximum_images=1)


def test_token_fd_is_closed_and_all_token_environment_entries_are_scrubbed() -> None:
    read_descriptor, write_descriptor = os.pipe()
    token = "f" * 32
    os.write(write_descriptor, token.encode())
    os.close(write_descriptor)
    environment = {
        INFERENCE_TOKEN_FD_ENV: str(read_descriptor),
        INFERENCE_TOKEN_ENV: "legacy-token-" + "l" * 32,
        "KEEP_ME": "yes",
    }

    assert read_inference_token(environment) == token
    assert environment == {"KEEP_ME": "yes"}
    with pytest.raises(OSError):
        os.fstat(read_descriptor)


def test_invalid_token_fd_payload_is_closed_and_environment_is_scrubbed() -> None:
    read_descriptor, write_descriptor = os.pipe()
    os.write(write_descriptor, b"\xff" * 32)
    os.close(write_descriptor)
    environment = {
        INFERENCE_TOKEN_FD_ENV: str(read_descriptor),
        INFERENCE_TOKEN_ENV: "legacy-token-" + "l" * 32,
    }

    with pytest.raises(InferenceProtocolError, match="invalid"):
        read_inference_token(environment)
    assert environment == {}
    with pytest.raises(OSError):
        os.fstat(read_descriptor)
