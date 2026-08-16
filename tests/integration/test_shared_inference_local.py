import asyncio
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from numpy.typing import NDArray

from vehicle_intelligence.config import GPUSchedulerConfig
from vehicle_intelligence.domain import BoundingBox, Detection, ModelMetadata, PlateDetection
from vehicle_intelligence.exceptions import InferenceError
from vehicle_intelligence.infrastructure.inference.protocol import derive_camera_token
from vehicle_intelligence.infrastructure.inference.service import SharedInferenceService
from vehicle_intelligence.infrastructure.vision.remote import (
    RemotePlateDetector,
    RemoteVehicleDetector,
    UnixInferenceClient,
)

TOKEN = "local-integration-token-" + "x" * 32
MODEL = ModelMetadata("fake", "test")


def _marker(image: NDArray[np.uint8]) -> int:
    return int(image[0, 0, 0])


def _image(marker: int) -> NDArray[np.uint8]:
    return np.full((8, 8, 3), marker, dtype=np.uint8)


class RecordingVehicleDetector:
    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        return self.detect_batch((image,))[0]

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        markers = [_marker(image) for image in images]
        self.batches.append(markers)
        return [
            [Detection(BoundingBox(0, 0, 4, 4), 0.9, marker, f"class-{marker}", MODEL)]
            for marker in markers
        ]


class RecordingPlateDetector:
    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def detect(self, image: NDArray[np.uint8]) -> list[PlateDetection]:
        return self.detect_batch((image,))[0]

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[PlateDetection]]:
        markers = [_marker(image) for image in images]
        self.batches.append(markers)
        return [
            [PlateDetection(BoundingBox(marker, 0, marker + 2, 2), 0.8, MODEL)]
            for marker in markers
        ]


class SlowVehicleDetector(RecordingVehicleDetector):
    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        time.sleep(0.06)
        return super().detect_batch(images)


class HangingVehicleDetector(RecordingVehicleDetector):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        self.started.set()
        threading.Event().wait(1)
        return super().detect_batch(images)


class MalformedVehicleDetector(RecordingVehicleDetector):
    def detect_batch(self, images: Sequence[NDArray[np.uint8]]):
        return [None for _ in images]


class PoisonVehicleDetector(RecordingVehicleDetector):
    def __init__(self, poison_markers: set[int]) -> None:
        super().__init__()
        self.poison_markers = poison_markers

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        markers = [_marker(image) for image in images]
        self.batches.append(markers)
        if self.poison_markers.intersection(markers):
            raise RuntimeError("provider rejected poison image")
        return [
            [Detection(BoundingBox(0, 0, 4, 4), 0.9, marker, f"class-{marker}", MODEL)]
            for marker in markers
        ]


class AlwaysFailVehicleDetector(RecordingVehicleDetector):
    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        self.batches.append([_marker(image) for image in images])
        raise RuntimeError("provider unavailable")


class MultiImageFailVehicleDetector(RecordingVehicleDetector):
    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        if len(images) > 1:
            self.batches.append([_marker(image) for image in images])
            raise RuntimeError("provider rejected multi-image call")
        return super().detect_batch(images)


def _client(config: GPUSchedulerConfig, camera_id: str, token: str = TOKEN):
    return UnixInferenceClient(
        config.socket_path,
        camera_id,
        derive_camera_token(token, camera_id),
        timeout_seconds=config.request_timeout_seconds,
        maximum_payload_bytes=config.maximum_payload_bytes,
        maximum_images=config.maximum_images_per_request,
    )


@pytest.fixture
def short_socket_path() -> Iterator[Path]:
    with TemporaryDirectory(prefix="vi-", dir="/tmp") as directory:
        yield Path(directory) / "inference.sock"


async def test_service_multiplexes_two_cameras_and_preserves_batch_mapping(
    short_socket_path: Path,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=2,
        maximum_batch_size=4,
        per_camera_queue_size=2,
        maximum_frame_age_ms=5000,
        batch_wait_ms=50,
        socket_path=short_socket_path,
        request_timeout_seconds=3,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=8,
    )
    vehicles = RecordingVehicleDetector()
    plates = RecordingPlateDetector()
    service = SharedInferenceService(config, vehicles, plates, TOKEN)
    await service.start()
    camera_a = _client(config, "gate-a")
    camera_b = _client(config, "gate-b")

    try:
        first, second = await asyncio.gather(
            asyncio.to_thread(RemoteVehicleDetector(camera_a).detect, _image(11)),
            asyncio.to_thread(RemoteVehicleDetector(camera_b).detect, _image(22)),
        )
        assert first[0].class_id == 11
        assert second[0].class_id == 22
        assert any(set(batch) == {11, 22} for batch in vehicles.batches)

        first_plates, second_plates = await asyncio.gather(
            asyncio.to_thread(
                RemotePlateDetector(camera_a).detect_batch,
                [_image(31), _image(32)],
            ),
            asyncio.to_thread(RemotePlateDetector(camera_b).detect_batch, [_image(41)]),
        )
        assert [detections[0].bbox.x1 for detections in first_plates] == [31, 32]
        assert [detections[0].bbox.x1 for detections in second_plates] == [41]
        assert any(set(batch) == {31, 32, 41} for batch in plates.batches)
        assert service.stats.vehicle_images == 2
        assert service.stats.plate_images == 3

        image_count_bounded = UnixInferenceClient(
            config.socket_path,
            "gate-a",
            derive_camera_token(TOKEN, "gate-a"),
            timeout_seconds=3,
            maximum_payload_bytes=config.maximum_payload_bytes,
            maximum_images=2,
        )
        bounded_results = await asyncio.to_thread(
            RemotePlateDetector(image_count_bounded).detect_batch,
            [_image(marker) for marker in range(51, 56)],
        )
        assert [detections[0].bbox.x1 for detections in bounded_results] == list(range(51, 56))
        assert [len(batch) for batch in plates.batches[-3:]] == [2, 2, 1]

        payload_bounded = UnixInferenceClient(
            config.socket_path,
            "gate-b",
            derive_camera_token(TOKEN, "gate-b"),
            timeout_seconds=3,
            maximum_payload_bytes=config.maximum_payload_bytes,
            maximum_images=8,
        )
        payload_results = await asyncio.to_thread(
            RemotePlateDetector(payload_bounded).detect_batch,
            [np.full((400, 400, 3), marker, dtype=np.uint8) for marker in range(61, 64)],
        )
        assert [detections[0].bbox.x1 for detections in payload_results] == [61, 62, 63]
        assert [len(batch) for batch in plates.batches[-2:]] == [2, 1]

        with pytest.raises(InferenceError):
            await asyncio.to_thread(_client(config, "gate-a", "z" * 32).ping)
    finally:
        await service.close()

    assert not config.socket_path.exists()


async def test_multi_batch_request_uses_request_timeout_not_frame_age(
    short_socket_path: Path,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=1,
        maximum_batch_size=1,
        per_camera_queue_size=1,
        maximum_frame_age_ms=25,
        batch_wait_ms=0,
        socket_path=short_socket_path,
        request_timeout_seconds=1,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=4,
        provider_failure_minimum_cameras=1,
    )
    detector = SlowVehicleDetector()
    service = SharedInferenceService(config, detector, RecordingPlateDetector(), TOKEN)
    await service.start()

    try:
        results = await asyncio.to_thread(
            RemoteVehicleDetector(_client(config, "gate-a")).detect_batch,
            [_image(71), _image(72)],
        )
        assert [detections[0].class_id for detections in results] == [71, 72]
        assert detector.batches == [[71], [72]]
    finally:
        await service.close()


async def test_detector_deadline_marks_service_unhealthy_without_second_gpu_call(
    short_socket_path: Path,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=2,
        maximum_batch_size=1,
        per_camera_queue_size=1,
        maximum_frame_age_ms=500,
        batch_wait_ms=0,
        socket_path=short_socket_path,
        request_timeout_seconds=0.1,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=2,
    )
    vehicles = HangingVehicleDetector()
    plates = RecordingPlateDetector()
    service = SharedInferenceService(config, vehicles, plates, TOKEN)
    await service.start()

    vehicle_call = asyncio.create_task(
        asyncio.to_thread(RemoteVehicleDetector(_client(config, "gate-a")).detect, _image(81))
    )
    try:
        assert await asyncio.to_thread(vehicles.started.wait, 0.5)
        plate_call = asyncio.create_task(
            asyncio.to_thread(RemotePlateDetector(_client(config, "gate-b")).detect, _image(82))
        )
        await asyncio.gather(vehicle_call, plate_call, return_exceptions=True)
        with pytest.raises(InferenceError, match="deadline"):
            await asyncio.wait_for(service.wait(), timeout=0.5)
        assert plates.batches == []
    finally:
        vehicle_call.cancel()
        await service.close()


async def test_malformed_detector_result_cannot_silently_kill_dispatcher(
    short_socket_path: Path,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=1,
        maximum_batch_size=1,
        per_camera_queue_size=1,
        maximum_frame_age_ms=500,
        batch_wait_ms=0,
        socket_path=short_socket_path,
        request_timeout_seconds=1,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=1,
        provider_failure_minimum_cameras=1,
    )
    service = SharedInferenceService(
        config,
        MalformedVehicleDetector(),
        RecordingPlateDetector(),
        TOKEN,
    )
    await service.start()
    try:
        result = await asyncio.gather(
            asyncio.to_thread(RemoteVehicleDetector(_client(config, "gate-a")).detect, _image(91)),
            return_exceptions=True,
        )
        assert isinstance(result[0], InferenceError)
        with pytest.raises(InferenceError, match="malformed"):
            await asyncio.wait_for(service.wait(), timeout=0.5)
    finally:
        await service.close()


async def test_provider_error_isolates_and_quarantines_only_poison_camera(
    short_socket_path: Path,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=2,
        maximum_batch_size=2,
        per_camera_queue_size=1,
        maximum_frame_age_ms=5000,
        batch_wait_ms=50,
        socket_path=short_socket_path,
        request_timeout_seconds=2,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=2,
        maximum_isolation_attempts=3,
        camera_failure_threshold=2,
        camera_quarantine_seconds=60,
        provider_failure_threshold=2,
        provider_failure_minimum_cameras=2,
    )
    detector = PoisonVehicleDetector({99})
    service = SharedInferenceService(config, detector, RecordingPlateDetector(), TOKEN)
    await service.start()
    poison = RemoteVehicleDetector(_client(config, "gate-poison"))
    healthy = RemoteVehicleDetector(_client(config, "gate-healthy"))

    try:
        for healthy_marker in (21, 22):
            poison_result, healthy_result = await asyncio.gather(
                asyncio.to_thread(poison.detect, _image(99)),
                asyncio.to_thread(healthy.detect, _image(healthy_marker)),
                return_exceptions=True,
            )
            assert isinstance(poison_result, InferenceError)
            assert not isinstance(healthy_result, BaseException)
            assert healthy_result[0].class_id == healthy_marker

        assert any(set(batch) == {99, 21} for batch in detector.batches)
        assert any(set(batch) == {99, 22} for batch in detector.batches)
        calls_before_quarantine_probe = len(detector.batches)
        with pytest.raises(InferenceError):
            await asyncio.to_thread(poison.detect, _image(99))
        assert len(detector.batches) == calls_before_quarantine_probe

        healthy_after_quarantine = await asyncio.to_thread(healthy.detect, _image(23))
        assert healthy_after_quarantine[0].class_id == 23
        assert service.running
        assert service.stats.provider_failures >= 4
        assert service.stats.isolation_retries >= 4
        assert service.stats.isolated_image_failures == 2
        assert service.stats.camera_quarantines == 1
        assert service.stats.quarantined_requests == 1
        assert service.stats.circuit_breaker_trips == 0
    finally:
        await service.close()


async def test_isolation_attempt_budget_is_bounded_and_service_recovers(
    short_socket_path: Path,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=1,
        maximum_batch_size=4,
        per_camera_queue_size=1,
        maximum_frame_age_ms=5000,
        batch_wait_ms=0,
        socket_path=short_socket_path,
        request_timeout_seconds=2,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=4,
        maximum_isolation_attempts=5,
        provider_failure_threshold=2,
        provider_failure_minimum_cameras=1,
    )
    detector = PoisonVehicleDetector({10, 30})
    service = SharedInferenceService(config, detector, RecordingPlateDetector(), TOKEN)
    await service.start()
    remote = RemoteVehicleDetector(_client(config, "gate-a"))

    try:
        with pytest.raises(InferenceError):
            await asyncio.to_thread(
                remote.detect_batch,
                [_image(marker) for marker in (10, 20, 30, 40)],
            )
        assert detector.batches == [
            [10, 20, 30, 40],
            [10, 20],
            [10],
            [20],
            [30, 40],
        ]
        assert service.stats.isolation_retries == 4
        assert service.running

        recovered = await asyncio.to_thread(remote.detect, _image(50))
        assert recovered[0].class_id == 50
        assert service.stats.circuit_breaker_trips == 0
    finally:
        await service.close()


async def test_fully_recovered_provider_error_is_observable(
    short_socket_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=1,
        maximum_batch_size=2,
        per_camera_queue_size=1,
        maximum_frame_age_ms=5000,
        batch_wait_ms=0,
        socket_path=short_socket_path,
        request_timeout_seconds=2,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=2,
        maximum_isolation_attempts=3,
        provider_failure_minimum_cameras=1,
    )
    detector = MultiImageFailVehicleDetector()
    service = SharedInferenceService(config, detector, RecordingPlateDetector(), TOKEN)
    await service.start()
    caplog.set_level(logging.WARNING)

    try:
        results = await asyncio.to_thread(
            RemoteVehicleDetector(_client(config, "gate-a")).detect_batch,
            [_image(51), _image(52)],
        )
        assert [detections[0].class_id for detections in results] == [51, 52]
        assert detector.batches == [[51, 52], [51], [52]]
        assert service.stats.provider_failures == 1
        assert service.stats.isolation_retries == 2
        assert service.stats.isolated_image_failures == 0
        assert service.stats.circuit_breaker_trips == 0
        records = [
            record
            for record in caplog.records
            if record.message == "shared_inference_images_isolated"
        ]
        assert len(records) == 1
        assert records[0].recovered is True
    finally:
        await service.close()


async def test_consecutive_cross_camera_provider_failures_open_circuit_breaker(
    short_socket_path: Path,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=2,
        maximum_batch_size=2,
        per_camera_queue_size=1,
        maximum_frame_age_ms=5000,
        batch_wait_ms=50,
        socket_path=short_socket_path,
        request_timeout_seconds=2,
        maximum_payload_bytes=1_048_576,
        maximum_images_per_request=2,
        maximum_isolation_attempts=3,
        camera_failure_threshold=100,
        provider_failure_threshold=2,
        provider_failure_minimum_cameras=2,
    )
    detector = AlwaysFailVehicleDetector()
    service = SharedInferenceService(config, detector, RecordingPlateDetector(), TOKEN)
    await service.start()
    camera_a = RemoteVehicleDetector(_client(config, "gate-a"))
    camera_b = RemoteVehicleDetector(_client(config, "gate-b"))

    try:
        for round_index in range(3):
            if not service.running:
                break
            results = await asyncio.gather(
                asyncio.to_thread(camera_a.detect, _image(61 + round_index)),
                asyncio.to_thread(camera_b.detect, _image(71 + round_index)),
                return_exceptions=True,
            )
            assert all(isinstance(result, InferenceError) for result in results)
            await asyncio.sleep(0)

        with pytest.raises(InferenceError, match="circuit breaker"):
            await asyncio.wait_for(service.wait(), timeout=0.5)
        assert {marker for batch in detector.batches for marker in batch} >= {61, 71}
        assert service.stats.provider_failures >= 2
        assert service.stats.circuit_breaker_trips == 1
        assert not service.running
    finally:
        await service.close()


async def test_service_start_cancellation_releases_bound_socket(
    short_socket_path: Path,
    monkeypatch,
) -> None:
    config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=1,
        maximum_batch_size=1,
        socket_path=short_socket_path,
        maximum_images_per_request=1,
        provider_failure_minimum_cameras=1,
    )
    service = SharedInferenceService(
        config,
        RecordingVehicleDetector(),
        RecordingPlateDetector(),
        TOKEN,
    )
    entered = asyncio.Event()
    captured_socket = []

    async def blocked_server(*_args, **kwargs):
        captured_socket.append(kwargs["sock"])
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio, "start_unix_server", blocked_server)
    start = asyncio.create_task(service.start())
    await entered.wait()
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert captured_socket[0].fileno() == -1
    assert not short_socket_path.exists()
    assert service.running is False
