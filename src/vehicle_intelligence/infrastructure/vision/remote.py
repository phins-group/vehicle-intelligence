"""Synchronous detector adapters backed by the local shared inference service."""

from __future__ import annotations

import socket
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.domain import Detection, PlateDetection
from vehicle_intelligence.exceptions import InferenceError, InferenceProtocolError
from vehicle_intelligence.infrastructure.inference.protocol import (
    MAXIMUM_HEADER_BYTES,
    DetectorKind,
    decode_success_response,
    encode_detect_request,
    encode_ping_request,
    receive_framed_payload,
    send_framed_payload,
    validate_inference_token,
)


class UnixInferenceClient:
    def __init__(
        self,
        socket_path: str | Path,
        camera_id: str,
        token: str,
        *,
        timeout_seconds: float,
        maximum_payload_bytes: int,
        maximum_images: int,
    ) -> None:
        path = Path(socket_path)
        if not path.is_absolute() or not camera_id.strip():
            raise ValueError("shared inference endpoint configuration is invalid")
        try:
            validate_inference_token(token)
        except InferenceProtocolError as exc:
            raise ValueError("shared inference endpoint configuration is invalid") from exc
        if timeout_seconds <= 0 or maximum_payload_bytes <= 0 or maximum_images <= 0:
            raise ValueError("shared inference client bounds must be positive")
        self._socket_path = path
        self._camera_id = camera_id.strip()
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._maximum_payload_bytes = maximum_payload_bytes
        self._maximum_images = maximum_images

    def ping(self) -> None:
        request_id = uuid.uuid4().hex
        request = encode_ping_request(
            request_id,
            self._token,
            self._maximum_payload_bytes,
        )
        response = self._exchange(request)
        decode_success_response(response, request_id, None, 0, token=self._token)

    def detect(
        self,
        detector: DetectorKind,
        images: tuple[NDArray[np.uint8], ...],
    ) -> tuple[tuple[Detection | PlateDetection, ...], ...]:
        if not images:
            return ()
        combined: list[tuple[Detection | PlateDetection, ...]] = []
        for chunk in self._chunks(images):
            combined.extend(self._detect_chunk(detector, chunk))
        return tuple(combined)

    def _detect_chunk(
        self,
        detector: DetectorKind,
        images: tuple[NDArray[np.uint8], ...],
    ) -> tuple[tuple[Detection | PlateDetection, ...], ...]:
        request_id = uuid.uuid4().hex
        request = encode_detect_request(
            request_id,
            self._camera_id,
            detector,
            images,
            self._token,
            self._maximum_payload_bytes,
            self._maximum_images,
        )
        response = self._exchange(request)
        return decode_success_response(
            response,
            request_id,
            detector,
            len(images),
            token=self._token,
        )

    def _chunks(
        self,
        images: tuple[NDArray[np.uint8], ...],
    ) -> Sequence[tuple[NDArray[np.uint8], ...]]:
        chunks: list[tuple[NDArray[np.uint8], ...]] = []
        current: list[NDArray[np.uint8]] = []
        raw_bytes = 0
        maximum_raw_bytes = self._maximum_payload_bytes - MAXIMUM_HEADER_BYTES - 4
        for image in images:
            image_bytes = int(image.nbytes)
            if current and (
                len(current) >= self._maximum_images or raw_bytes + image_bytes > maximum_raw_bytes
            ):
                chunks.append(tuple(current))
                current = []
                raw_bytes = 0
            current.append(image)
            raw_bytes += image_bytes
        if current:
            chunks.append(tuple(current))
        return chunks

    def _exchange(self, request: bytes) -> bytes:
        deadline = time.monotonic() + self._timeout_seconds
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(max(0.001, deadline - time.monotonic()))
                client.connect(str(self._socket_path))
                client.settimeout(max(0.001, deadline - time.monotonic()))
                send_framed_payload(client, request, self._maximum_payload_bytes)
                return receive_framed_payload(
                    client,
                    self._maximum_payload_bytes,
                    deadline,
                )
        except (InferenceProtocolError, TimeoutError, OSError) as exc:
            raise InferenceError("shared inference service is unavailable") from exc


class RemoteVehicleDetector:
    def __init__(self, client: UnixInferenceClient) -> None:
        self._client = client

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        return self.detect_batch((image,))[0]

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[Detection]]:
        results = self._client.detect("vehicle", tuple(images))
        return [list(detections) for detections in results]


class RemotePlateDetector:
    def __init__(self, client: UnixInferenceClient) -> None:
        self._client = client

    def detect(self, image: NDArray[np.uint8]) -> list[PlateDetection]:
        return self.detect_batch((image,))[0]

    def detect_batch(self, images: Sequence[NDArray[np.uint8]]) -> list[list[PlateDetection]]:
        results = self._client.detect("plate", tuple(images))
        return [list(detections) for detections in results]
