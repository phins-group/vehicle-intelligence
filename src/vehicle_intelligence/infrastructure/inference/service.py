"""Supervisor-owned shared detector service with fair cross-camera batching."""

from __future__ import annotations

import asyncio
import logging
import socket
import threading
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.application.ports import (
    BatchPlateDetector,
    BatchVehicleDetector,
    PlateDetector,
    VehicleDetector,
)
from vehicle_intelligence.config import GPUSchedulerConfig
from vehicle_intelligence.domain import Detection, PlateDetection
from vehicle_intelligence.exceptions import InferenceError, InferenceProtocolError
from vehicle_intelligence.infrastructure.inference.protocol import (
    DetectionResult,
    DetectorKind,
    PingRequest,
    derive_camera_token,
    derive_supervisor_token,
    encode_error_response,
    encode_success_response,
    read_authenticated_request,
    validate_inference_token,
    write_framed_payload,
)
from vehicle_intelligence.infrastructure.inference.socket_path import (
    SocketIdentity,
    prepare_socket_path,
    socket_identity,
    unlink_owned_socket,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SharedInferenceStats:
    requests: int = 0
    rejected: int = 0
    vehicle_batches: int = 0
    plate_batches: int = 0
    vehicle_images: int = 0
    plate_images: int = 0
    maximum_vehicle_batch: int = 0
    maximum_plate_batch: int = 0
    provider_failures: int = 0
    isolation_retries: int = 0
    isolated_image_failures: int = 0
    camera_quarantines: int = 0
    quarantined_requests: int = 0
    circuit_breaker_trips: int = 0


@dataclass(eq=False, slots=True)
class _PendingCall:
    camera_id: str
    images: tuple[NDArray[np.uint8], ...]
    future: asyncio.Future[tuple[tuple[DetectionResult, ...], ...]]
    submitted_monotonic: float
    results: list[tuple[DetectionResult, ...] | None] = field(init=False)
    next_index: int = 0
    completed: int = 0
    finished: bool = False
    dispatched: bool = False

    def __post_init__(self) -> None:
        self.results = [None] * len(self.images)


@dataclass(frozen=True, slots=True)
class _WorkItem:
    call: _PendingCall
    index: int
    image: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _PayloadReservation:
    size: int
    camera_id: str | None
    detector: DetectorKind | None


@dataclass(frozen=True, slots=True)
class _DetectorInvocation:
    results: list[list[DetectionResult]] | None = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        if (self.results is None) == (self.error is None):
            raise ValueError("detector invocation must contain results or an error")


@dataclass(slots=True)
class _IsolationOutcome:
    results: list[list[DetectionResult] | None]
    failed_indices: set[int] = field(default_factory=set)
    confirmed_failure_indices: set[int] = field(default_factory=set)
    provider_failure_cameras: set[str] = field(default_factory=set)
    provider_failures: int = 0
    isolation_retries: int = 0
    exhausted: bool = False


@dataclass(slots=True)
class _ProviderBreakerState:
    consecutive_failures: int = 0
    cameras: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.cameras.clear()


class _FatalDetectorError(InferenceError):
    """A synchronous provider used BaseException control flow and cannot be trusted."""


class _FairInferenceQueue:
    """Bound calls per camera and emit one image per ready camera in round robin."""

    def __init__(self, config: GPUSchedulerConfig) -> None:
        self._maximum_cameras = config.maximum_cameras
        self._maximum_batch_size = config.maximum_batch_size
        self._per_camera_queue_size = config.per_camera_queue_size
        self._batch_wait_seconds = config.batch_wait_ms / 1000
        self._maximum_frame_age_seconds = config.maximum_frame_age_ms / 1000
        self._queues: dict[str, deque[_PendingCall]] = {}
        self._camera_call_counts: dict[str, int] = {}
        self._ready: deque[str] = deque()
        self._ready_set: set[str] = set()
        self._calls: set[_PendingCall] = set()
        self._condition = asyncio.Condition()
        self._closed = False

    async def submit(
        self,
        camera_id: str,
        images: tuple[NDArray[np.uint8], ...],
    ) -> tuple[tuple[DetectionResult, ...], ...]:
        loop = asyncio.get_running_loop()
        call = _PendingCall(camera_id, images, loop.create_future(), loop.time())
        call.future.add_done_callback(self._consume_future_exception)
        async with self._condition:
            if self._closed:
                raise InferenceError("shared inference queue is closed")
            current = self._camera_call_counts.get(camera_id, 0)
            if current == 0 and len(self._camera_call_counts) >= self._maximum_cameras:
                raise InferenceError("shared inference camera capacity is exhausted")
            if current >= self._per_camera_queue_size:
                raise InferenceError("shared inference camera queue is full")
            self._calls.add(call)
            self._camera_call_counts[camera_id] = current + 1
            self._queues.setdefault(camera_id, deque()).append(call)
            self._mark_ready_locked(camera_id)
            self._condition.notify_all()
        try:
            return await asyncio.shield(call.future)
        except asyncio.CancelledError:
            await self._cancel(call)
            raise

    async def next_batch(self) -> tuple[_WorkItem, ...]:
        loop = asyncio.get_running_loop()
        async with self._condition:
            while True:
                self._drop_stale_locked(loop.time())
                if self._closed:
                    return ()
                if not self._ready:
                    await self._condition.wait()
                    continue
                deadline = loop.time() + self._batch_wait_seconds
                while self._pending_images_locked() < self._maximum_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                    except TimeoutError:
                        break
                    if self._closed:
                        return ()
                self._drop_stale_locked(loop.time())
                if self._ready:
                    return self._pop_batch_locked()

    async def complete(
        self,
        items: tuple[_WorkItem, ...],
        results: list[list[DetectionResult]],
    ) -> None:
        async with self._condition:
            for item, detections in zip(items, results, strict=True):
                call = item.call
                if call.finished:
                    continue
                call.results[item.index] = tuple(detections)
                call.completed += 1
                if call.completed == len(call.images):
                    completed = tuple(
                        result if result is not None else () for result in call.results
                    )
                    call.finished = True
                    if not call.future.done():
                        call.future.set_result(completed)
                    self._finish_call_locked(call)
            self._condition.notify_all()

    async def fail(self, items: tuple[_WorkItem, ...], error: InferenceError) -> None:
        async with self._condition:
            for call in {item.call for item in items}:
                if call.finished:
                    continue
                call.finished = True
                if not call.future.done():
                    call.future.set_exception(error)
                self._finish_call_locked(call)
            self._condition.notify_all()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            for call in tuple(self._calls):
                call.finished = True
                if not call.future.done():
                    call.future.set_exception(InferenceError("shared inference service stopped"))
            self._calls.clear()
            self._queues.clear()
            self._camera_call_counts.clear()
            self._ready.clear()
            self._ready_set.clear()
            self._condition.notify_all()

    async def _cancel(self, call: _PendingCall) -> None:
        async with self._condition:
            if call.finished:
                return
            call.finished = True
            if not call.future.done():
                call.future.cancel()
            self._finish_call_locked(call)
            self._condition.notify_all()

    def _pop_batch_locked(self) -> tuple[_WorkItem, ...]:
        batch: list[_WorkItem] = []
        while self._ready and len(batch) < self._maximum_batch_size:
            camera_id = self._ready.popleft()
            self._ready_set.discard(camera_id)
            queue = self._queues.get(camera_id)
            if queue is None:
                continue
            while queue and queue[0].finished:
                queue.popleft()
            if not queue:
                self._queues.pop(camera_id, None)
                continue
            call = queue[0]
            call.dispatched = True
            index = call.next_index
            call.next_index += 1
            batch.append(_WorkItem(call, index, call.images[index]))
            if call.next_index == len(call.images):
                queue.popleft()
            if queue:
                self._mark_ready_locked(camera_id)
            else:
                self._queues.pop(camera_id, None)
        return tuple(batch)

    def _finish_call_locked(self, call: _PendingCall) -> None:
        self._calls.discard(call)
        count = self._camera_call_counts.get(call.camera_id, 0)
        if count <= 1:
            self._camera_call_counts.pop(call.camera_id, None)
        else:
            self._camera_call_counts[call.camera_id] = count - 1
        queue = self._queues.get(call.camera_id)
        if queue is not None:
            self._queues[call.camera_id] = deque(item for item in queue if item is not call)
            if not self._queues[call.camera_id]:
                self._queues.pop(call.camera_id, None)
                self._ready_set.discard(call.camera_id)
                self._ready = deque(item for item in self._ready if item != call.camera_id)

    def _mark_ready_locked(self, camera_id: str) -> None:
        if camera_id not in self._ready_set:
            self._ready.append(camera_id)
            self._ready_set.add(camera_id)

    def _pending_images_locked(self) -> int:
        return sum(
            len(call.images) - call.next_index
            for queue in self._queues.values()
            for call in queue
            if not call.finished
        )

    def _drop_stale_locked(self, now: float) -> None:
        for call in tuple(self._calls):
            if (
                not call.dispatched
                and not call.finished
                and now - call.submitted_monotonic > self._maximum_frame_age_seconds
            ):
                call.finished = True
                if not call.future.done():
                    call.future.set_exception(
                        InferenceError("shared inference request became stale before dispatch")
                    )
                self._finish_call_locked(call)

    @staticmethod
    def _consume_future_exception(
        future: asyncio.Future[tuple[tuple[DetectionResult, ...], ...]],
    ) -> None:
        if not future.cancelled():
            future.exception()


class SharedInferenceService:
    def __init__(
        self,
        config: GPUSchedulerConfig,
        vehicle_detector: VehicleDetector,
        plate_detector: PlateDetector,
        token: str,
    ) -> None:
        try:
            validate_inference_token(token)
        except InferenceProtocolError as exc:
            raise ValueError("shared inference token is invalid") from exc
        self._config = config
        self._vehicle_detector = vehicle_detector
        self._plate_detector = plate_detector
        self._token = token
        self._vehicle_queue = _FairInferenceQueue(config)
        self._plate_queue = _FairInferenceQueue(config)
        self._device_lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None
        self._socket_identity: SocketIdentity | None = None
        self._dispatchers: list[asyncio.Task[None]] = []
        self._client_tasks: set[asyncio.Task[None]] = set()
        self._active_clients = 0
        self._maximum_clients = config.maximum_clients
        self._client_payloads: dict[asyncio.Task[None], _PayloadReservation] = {}
        self._inflight_payload_bytes = 0
        self._camera_payload_bytes: dict[str, int] = {}
        self._camera_admissions: dict[tuple[str, DetectorKind], int] = {}
        self._camera_failure_counts: dict[tuple[str, DetectorKind], int] = {}
        self._camera_quarantine_until: dict[tuple[str, DetectorKind], float] = {}
        self._provider_breakers: dict[DetectorKind, _ProviderBreakerState] = {
            "vehicle": _ProviderBreakerState(),
            "plate": _ProviderBreakerState(),
        }
        self._maximum_camera_payload_bytes = min(
            config.maximum_inflight_payload_bytes,
            config.maximum_payload_bytes * 2,
        )
        self._fatal_event = asyncio.Event()
        self._fatal_error: InferenceError | None = None
        self.stats = SharedInferenceStats()

    @property
    def running(self) -> bool:
        return self._server is not None and self._server.is_serving()

    async def start(self) -> None:
        if self._server is not None:
            return
        path = self._config.socket_path
        prepare_socket_path(path)
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server_socket.bind(str(path))
            self._socket_identity = socket_identity(path)
            server_socket.setblocking(False)
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                sock=server_socket,
            )
            path.chmod(0o600)
        except BaseException as exc:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()
                self._server = None
            server_socket.close()
            self._cleanup_socket()
            if isinstance(exc, (OSError, ValueError)):
                raise InferenceError("cannot start shared inference socket") from exc
            raise
        self._dispatchers = [
            asyncio.create_task(
                self._dispatch("vehicle", self._vehicle_queue),
                name="shared-vehicle-inference",
            ),
            asyncio.create_task(
                self._dispatch("plate", self._plate_queue),
                name="shared-plate-inference",
            ),
        ]
        for dispatcher in self._dispatchers:
            dispatcher.add_done_callback(self._dispatcher_finished)

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()
        await asyncio.gather(self._vehicle_queue.close(), self._plate_queue.close())
        for client_task in tuple(self._client_tasks):
            client_task.cancel()
        if self._client_tasks:
            await asyncio.gather(*tuple(self._client_tasks), return_exceptions=True)
        for dispatcher in self._dispatchers:
            dispatcher.cancel()
        if self._dispatchers:
            await asyncio.gather(*self._dispatchers, return_exceptions=True)
        self._dispatchers.clear()
        self._cleanup_socket()

    async def wait(self) -> None:
        await self._fatal_event.wait()
        raise self._fatal_error or InferenceError("shared inference service became unhealthy")

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
        try:
            if self._active_clients >= self._maximum_clients:
                self.stats.rejected += 1
                await self._write_response(
                    writer,
                    encode_error_response(None, "service_capacity_exhausted"),
                )
                return
            self._active_clients += 1
            try:
                await self._serve_client(reader, writer)
            finally:
                self._active_clients -= 1
        finally:
            writer.close()
            try:
                with suppress(ConnectionError, OSError, TimeoutError):
                    await asyncio.wait_for(
                        writer.wait_closed(),
                        timeout=self._config.request_timeout_seconds,
                    )
            finally:
                if task is not None:
                    self._release_payload(task)
                    self._client_tasks.discard(task)

    async def _serve_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        request_id: str | None = None
        response: bytes
        try:
            request = await asyncio.wait_for(
                read_authenticated_request(
                    reader,
                    self._config.maximum_payload_bytes,
                    self._token,
                    self._config.maximum_images_per_request,
                    self._reserve_payload,
                ),
                timeout=self._config.request_timeout_seconds,
            )
            request_id = request.request_id
            self.stats.requests += 1
            if isinstance(request, PingRequest):
                response = encode_success_response(
                    request,
                    token=derive_supervisor_token(self._token),
                )
            else:
                queue = self._vehicle_queue if request.detector == "vehicle" else self._plate_queue
                try:
                    results = await asyncio.wait_for(
                        queue.submit(request.camera_id, request.images),
                        timeout=self._config.request_timeout_seconds,
                    )
                except TimeoutError as exc:
                    raise InferenceError("shared inference request timed out") from exc
                if (
                    sum(len(detections) for detections in results)
                    > self._config.maximum_payload_bytes // 128
                ):
                    raise InferenceError("shared inference response exceeds safe bounds")
                response = encode_success_response(
                    request,
                    results,
                    token=derive_camera_token(self._token, request.camera_id),
                )
                if len(response) > self._config.maximum_payload_bytes:
                    response = encode_error_response(request_id, "response_too_large")
        except InferenceProtocolError:
            self.stats.rejected += 1
            response = encode_error_response(request_id, "protocol_error")
        except TimeoutError:
            self.stats.rejected += 1
            response = encode_error_response(request_id, "request_timeout")
        except InferenceError:
            self.stats.rejected += 1
            response = encode_error_response(request_id, "request_failed")
        except Exception:
            self.stats.rejected += 1
            logger.exception("shared_inference_client_failed")
            response = encode_error_response(request_id, "internal_error")
        await self._write_response(writer, response)

    async def _write_response(self, writer: asyncio.StreamWriter, response: bytes) -> None:
        with suppress(ConnectionError, InferenceProtocolError, TimeoutError):
            await asyncio.wait_for(
                write_framed_payload(writer, response, self._config.maximum_payload_bytes),
                timeout=self._config.request_timeout_seconds,
            )

    async def _dispatch(
        self,
        detector: DetectorKind,
        queue: _FairInferenceQueue,
    ) -> None:
        while True:
            items = await queue.next_batch()
            if not items:
                return
            try:
                async with self._device_lock:
                    if self._fatal_error is not None:
                        await queue.fail(items, self._fatal_error)
                        return
                    try:
                        outcome = await self._infer_with_isolation(detector, items)
                    except TimeoutError:
                        self._mark_unhealthy(InferenceError("shared detector deadline exceeded"))
                        raise
                    except _FatalDetectorError as exc:
                        # Mark fatal before releasing the device lock so a peer dispatcher
                        # cannot start another call against an untrustworthy backend.
                        self.stats.provider_failures += 1
                        self._mark_unhealthy(exc)
                        raise
                    breaker_error = self._record_fault_outcome(detector, items, outcome)
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                error = self._fatal_error or InferenceError("shared detector deadline exceeded")
                logger.critical(
                    "shared_inference_detector_hung",
                    extra={"detector": detector, "batch_size": len(items)},
                )
                await queue.fail(items, error)
                return
            except _FatalDetectorError:
                error = self._fatal_error or InferenceError(
                    "shared detector provider aborted unexpectedly"
                )
                logger.critical(
                    "shared_inference_provider_aborted",
                    extra={"detector": detector, "batch_size": len(items)},
                    exc_info=True,
                )
                await queue.fail(items, error)
                return
            except Exception:
                error = InferenceError("shared detector returned malformed results")
                logger.exception(
                    "shared_inference_result_invariant_failed",
                    extra={"detector": detector, "batch_size": len(items)},
                )
                self._mark_unhealthy(error)
                await queue.fail(items, error)
                return
            try:
                await self._resolve_outcome(
                    queue,
                    items,
                    outcome,
                    breaker_error or InferenceError("shared detector rejected one or more images"),
                )
            except Exception:
                error = InferenceError("shared detector outcome resolution failed")
                logger.exception(
                    "shared_inference_resolution_failed",
                    extra={"detector": detector, "batch_size": len(items)},
                )
                self._mark_unhealthy(error)
                await queue.fail(items, error)
                return
            if outcome.provider_failures:
                logger.warning(
                    "shared_inference_images_isolated",
                    extra={
                        "detector": detector,
                        "batch_size": len(items),
                        "failed_images": len(outcome.failed_indices),
                        "recovered": not outcome.failed_indices,
                        "provider_failures": outcome.provider_failures,
                        "isolation_retries": outcome.isolation_retries,
                        "attempt_budget_exhausted": outcome.exhausted,
                    },
                )
            if breaker_error is not None:
                return

    async def _infer_with_isolation(
        self,
        detector: DetectorKind,
        items: tuple[_WorkItem, ...],
    ) -> _IsolationOutcome:
        deadline = asyncio.get_running_loop().time() + self._config.request_timeout_seconds
        outcome = _IsolationOutcome([None] * len(items))
        if not self._supports_batch(detector):
            await self._infer_scalar_items(detector, items, deadline, outcome)
            return outcome

        pending: deque[tuple[tuple[int, ...], bool]] = deque([(tuple(range(len(items))), False)])
        attempts = 0
        while pending:
            indices, is_retry = pending.popleft()
            if attempts >= self._config.maximum_isolation_attempts:
                outcome.failed_indices.update(indices)
                outcome.exhausted = True
                continue
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                outcome.failed_indices.update(indices)
                outcome.exhausted = True
                continue
            attempts += 1
            if is_retry:
                outcome.isolation_retries += 1
            invocation = await asyncio.wait_for(
                self._run_detector_thread(
                    detector,
                    [items[index].image for index in indices],
                ),
                timeout=remaining,
            )
            if invocation.error is not None:
                if isinstance(invocation.error, _FatalDetectorError):
                    raise invocation.error
                self._record_invocation_failure(items, indices, outcome)
                if len(indices) == 1:
                    outcome.failed_indices.add(indices[0])
                    outcome.confirmed_failure_indices.add(indices[0])
                    continue
                midpoint = len(indices) // 2
                pending.appendleft((indices[midpoint:], True))
                pending.appendleft((indices[:midpoint], True))
                continue
            results = invocation.results
            self._validate_results(detector, results, len(indices))
            assert results is not None
            self._record_batch(detector, len(indices))
            for index, detections in zip(indices, results, strict=True):
                outcome.results[index] = detections
        outcome.failed_indices.update(
            index for index, result in enumerate(outcome.results) if result is None
        )
        return outcome

    async def _infer_scalar_items(
        self,
        detector: DetectorKind,
        items: tuple[_WorkItem, ...],
        deadline: float,
        outcome: _IsolationOutcome,
    ) -> None:
        for index, item in enumerate(items):
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                outcome.failed_indices.update(range(index, len(items)))
                outcome.exhausted = True
                return
            invocation = await asyncio.wait_for(
                self._run_detector_thread(detector, [item.image]),
                timeout=remaining,
            )
            if invocation.error is not None:
                if isinstance(invocation.error, _FatalDetectorError):
                    raise invocation.error
                self._record_invocation_failure(items, (index,), outcome)
                outcome.failed_indices.add(index)
                outcome.confirmed_failure_indices.add(index)
                continue
            results = invocation.results
            self._validate_results(detector, results, 1)
            assert results is not None
            self._record_batch(detector, 1)
            outcome.results[index] = results[0]

    @staticmethod
    def _record_invocation_failure(
        items: tuple[_WorkItem, ...],
        indices: tuple[int, ...],
        outcome: _IsolationOutcome,
    ) -> None:
        outcome.provider_failures += 1
        outcome.provider_failure_cameras.update(items[index].call.camera_id for index in indices)

    async def _resolve_outcome(
        self,
        queue: _FairInferenceQueue,
        items: tuple[_WorkItem, ...],
        outcome: _IsolationOutcome,
        error: InferenceError,
    ) -> None:
        failed_items = tuple(items[index] for index in sorted(outcome.failed_indices))
        failed_calls = {item.call for item in failed_items}
        if failed_items:
            await queue.fail(failed_items, error)
        successful_items: list[_WorkItem] = []
        successful_results: list[list[DetectionResult]] = []
        for item, result in zip(items, outcome.results, strict=True):
            if result is not None and item.call not in failed_calls:
                successful_items.append(item)
                successful_results.append(result)
        if successful_items:
            await queue.complete(tuple(successful_items), successful_results)

    def _record_fault_outcome(
        self,
        detector: DetectorKind,
        items: tuple[_WorkItem, ...],
        outcome: _IsolationOutcome,
    ) -> InferenceError | None:
        self.stats.provider_failures += outcome.provider_failures
        self.stats.isolation_retries += outcome.isolation_retries
        self.stats.isolated_image_failures += len(outcome.confirmed_failure_indices)
        failed_cameras = {
            items[index].call.camera_id for index in outcome.confirmed_failure_indices
        }
        successful_cameras = {
            item.call.camera_id
            for index, item in enumerate(items)
            if outcome.results[index] is not None
        }
        for camera_id in successful_cameras - failed_cameras:
            self._camera_failure_counts.pop((camera_id, detector), None)
        now = asyncio.get_running_loop().time()
        for camera_id in failed_cameras:
            key = (camera_id, detector)
            failures = self._camera_failure_counts.get(key, 0) + 1
            if failures < self._config.camera_failure_threshold:
                self._camera_failure_counts[key] = failures
                continue
            previously_quarantined = self._camera_quarantine_until.get(key, 0) > now
            self._camera_failure_counts[key] = 0
            self._camera_quarantine_until[key] = now + self._config.camera_quarantine_seconds
            if not previously_quarantined:
                self.stats.camera_quarantines += 1

        breaker = self._provider_breakers[detector]
        if any(result is not None for result in outcome.results):
            breaker.reset()
            return None
        if outcome.provider_failures == 0:
            return None
        breaker.consecutive_failures += 1
        breaker.cameras.update(outcome.provider_failure_cameras)
        if (
            breaker.consecutive_failures < self._config.provider_failure_threshold
            or len(breaker.cameras) < self._config.provider_failure_minimum_cameras
        ):
            return None
        error = InferenceError("shared detector provider circuit breaker opened")
        self.stats.circuit_breaker_trips += 1
        self._mark_unhealthy(error)
        return error

    def _supports_batch(self, detector: DetectorKind) -> bool:
        if detector == "vehicle":
            return isinstance(self._vehicle_detector, BatchVehicleDetector)
        return isinstance(self._plate_detector, BatchPlateDetector)

    async def _run_detector_thread(
        self,
        detector: DetectorKind,
        images: list[NDArray[np.uint8]],
    ) -> _DetectorInvocation:
        loop = asyncio.get_running_loop()
        result: asyncio.Future[_DetectorInvocation] = loop.create_future()

        def run() -> None:
            try:
                detections = self._run_detector(detector, images)
            except asyncio.CancelledError:
                # A synchronous provider cannot cancel this asyncio dispatcher. Treat its
                # control-flow exception as a data-plane failure; only cancellation injected
                # at the await boundary is allowed to stop the dispatcher task.
                self._schedule_thread_completion(
                    loop,
                    self._finish_thread_result,
                    result,
                    _DetectorInvocation(
                        error=InferenceError("shared detector provider cancelled unexpectedly")
                    ),
                )
            except Exception as exc:
                self._schedule_thread_completion(
                    loop,
                    self._finish_thread_result,
                    result,
                    _DetectorInvocation(error=exc),
                )
            except BaseException as exc:
                error = _FatalDetectorError("shared detector provider aborted unexpectedly")
                error.__cause__ = exc
                self._schedule_thread_completion(
                    loop,
                    self._finish_thread_result,
                    result,
                    _DetectorInvocation(error=error),
                )
            else:
                self._schedule_thread_completion(
                    loop,
                    self._finish_thread_result,
                    result,
                    _DetectorInvocation(results=detections),
                )

        threading.Thread(
            target=run,
            name=f"shared-{detector}-inference-call",
            daemon=True,
        ).start()
        return await result

    def _run_detector(
        self,
        detector: DetectorKind,
        images: list[NDArray[np.uint8]],
    ) -> list[list[DetectionResult]]:
        if detector == "vehicle":
            if isinstance(self._vehicle_detector, BatchVehicleDetector):
                return self._vehicle_detector.detect_batch(images)
            return [self._vehicle_detector.detect(image) for image in images]
        if isinstance(self._plate_detector, BatchPlateDetector):
            return self._plate_detector.detect_batch(images)
        return [self._plate_detector.detect(image) for image in images]

    def _record_batch(self, detector: DetectorKind, size: int) -> None:
        if detector == "vehicle":
            self.stats.vehicle_batches += 1
            self.stats.vehicle_images += size
            self.stats.maximum_vehicle_batch = max(self.stats.maximum_vehicle_batch, size)
        else:
            self.stats.plate_batches += 1
            self.stats.plate_images += size
            self.stats.maximum_plate_batch = max(self.stats.maximum_plate_batch, size)

    @staticmethod
    def _validate_results(
        detector: DetectorKind,
        results: object,
        expected_images: int,
    ) -> None:
        if not isinstance(results, list) or len(results) != expected_images:
            raise InferenceError("shared detector result count does not match batch")
        expected_type = Detection if detector == "vehicle" else PlateDetection
        if any(
            not isinstance(detections, list)
            or any(not isinstance(item, expected_type) for item in detections)
            for detections in results
        ):
            raise InferenceError("shared detector result shape is invalid")

    def _reserve_payload(
        self,
        size: int,
        camera_id: str | None,
        detector: DetectorKind | None,
    ) -> None:
        task = asyncio.current_task()
        if task is None or task in self._client_payloads:
            raise InferenceError("shared inference payload ownership is invalid")
        if self._inflight_payload_bytes + size > self._config.maximum_inflight_payload_bytes:
            raise InferenceError("shared inference payload budget is exhausted")
        if camera_id is not None and detector is not None:
            quarantine_key = (camera_id, detector)
            quarantine_until = self._camera_quarantine_until.get(quarantine_key)
            if quarantine_until is not None:
                if quarantine_until > asyncio.get_running_loop().time():
                    self.stats.quarantined_requests += 1
                    raise InferenceError("shared inference camera is temporarily quarantined")
                self._camera_quarantine_until.pop(quarantine_key, None)
                self._camera_failure_counts.pop(quarantine_key, None)
            active_cameras = set(self._camera_payload_bytes)
            if (
                camera_id not in active_cameras
                and len(active_cameras) >= self._config.maximum_cameras
            ):
                raise InferenceError("shared inference camera capacity is exhausted")
            admission_key = (camera_id, detector)
            if self._camera_admissions.get(admission_key, 0) >= self._config.per_camera_queue_size:
                raise InferenceError("shared inference camera queue is full")
            if (
                self._camera_payload_bytes.get(camera_id, 0) + size
                > self._maximum_camera_payload_bytes
            ):
                raise InferenceError("shared inference camera payload budget is exhausted")
            self._camera_admissions[admission_key] = (
                self._camera_admissions.get(admission_key, 0) + 1
            )
            self._camera_payload_bytes[camera_id] = (
                self._camera_payload_bytes.get(camera_id, 0) + size
            )
        self._client_payloads[task] = _PayloadReservation(size, camera_id, detector)
        self._inflight_payload_bytes += size

    def _release_payload(self, task: asyncio.Task[None]) -> None:
        reservation = self._client_payloads.pop(task, None)
        if reservation is None:
            return
        self._inflight_payload_bytes -= reservation.size
        if reservation.camera_id is None or reservation.detector is None:
            return
        admission_key = (reservation.camera_id, reservation.detector)
        admission_count = self._camera_admissions.get(admission_key, 0)
        if admission_count <= 1:
            self._camera_admissions.pop(admission_key, None)
        else:
            self._camera_admissions[admission_key] = admission_count - 1
        camera_bytes = self._camera_payload_bytes.get(reservation.camera_id, 0)
        if camera_bytes <= reservation.size:
            self._camera_payload_bytes.pop(reservation.camera_id, None)
        else:
            self._camera_payload_bytes[reservation.camera_id] = camera_bytes - reservation.size

    def _mark_unhealthy(self, error: InferenceError) -> None:
        if self._fatal_error is not None:
            return
        self._fatal_error = error
        if self._server is not None:
            self._server.close()
        self._fatal_event.set()

    def _dispatcher_finished(self, dispatcher: asyncio.Task[None]) -> None:
        if dispatcher.cancelled() or self._server is None or self._fatal_error is not None:
            return
        error = dispatcher.exception()
        if error is None:
            failure = InferenceError("shared inference dispatcher stopped unexpectedly")
        else:
            logger.error(
                "shared_inference_dispatcher_crashed",
                exc_info=(type(error), error, error.__traceback__),
            )
            failure = InferenceError("shared inference dispatcher crashed")
        self._mark_unhealthy(failure)

    @staticmethod
    def _finish_thread_result(
        future: asyncio.Future[_DetectorInvocation],
        result: _DetectorInvocation,
    ) -> None:
        if not future.done():
            future.set_result(result)

    @staticmethod
    def _schedule_thread_completion(
        loop: asyncio.AbstractEventLoop,
        callback: Callable[..., None],
        *arguments: object,
    ) -> None:
        if loop.is_closed():
            return
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(callback, *arguments)

    def _cleanup_socket(self) -> None:
        identity, self._socket_identity = self._socket_identity, None
        try:
            unlink_owned_socket(self._config.socket_path, identity)
        except InferenceError:
            logger.warning("shared_inference_socket_cleanup_failed", exc_info=True)
