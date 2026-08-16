"""Length-prefixed, bounded binary protocol for local detector inference."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import socket
import struct
import time
from collections.abc import Callable, MutableMapping
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite, prod
from typing import Literal, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.domain import (
    BoundingBox,
    Detection,
    ModelMetadata,
    PlateDetection,
    Point,
)
from vehicle_intelligence.exceptions import InferenceError, InferenceProtocolError

INFERENCE_SOCKET_ENV = "VEHICLE_INFERENCE_SOCKET"
INFERENCE_TOKEN_ENV = "VEHICLE_INFERENCE_TOKEN"
INFERENCE_TOKEN_FD_ENV = "VEHICLE_INFERENCE_TOKEN_FD"
INFERENCE_CAMERA_ENV = "VEHICLE_INFERENCE_CAMERA_ID"

PROTOCOL_VERSION = 1
MAXIMUM_HEADER_BYTES = 65_536
MAXIMUM_DETECTIONS_PER_IMAGE = 512
_LENGTH = struct.Struct(">I")
_REQUEST_ID = re.compile(r"[0-9a-f]{32}")

type DetectorKind = Literal["vehicle", "plate"]
type DetectionResult = Detection | PlateDetection


@dataclass(frozen=True, slots=True)
class PingRequest:
    request_id: str


@dataclass(frozen=True, slots=True)
class DetectRequest:
    request_id: str
    camera_id: str
    detector: DetectorKind
    images: tuple[NDArray[np.uint8], ...]


type InferenceRequest = PingRequest | DetectRequest


@dataclass(frozen=True, slots=True)
class _DetectHeader:
    request_id: str
    camera_id: str
    detector: DetectorKind
    image_specs: tuple[tuple[tuple[int, ...], int], ...]
    body_sha256: str


def derive_camera_token(master_token: str, camera_id: str) -> str:
    validated_master = _token(master_token)
    validated_camera = _camera_id(camera_id)
    message = f"vehicle-inference-camera-v1\0{validated_camera}".encode()
    return hmac.new(validated_master.encode(), message, hashlib.sha256).hexdigest()


def validate_inference_token(value: object) -> str:
    return _token(value)


def derive_supervisor_token(master_token: str) -> str:
    validated_master = _token(master_token)
    return hmac.new(
        validated_master.encode(),
        b"vehicle-inference-supervisor-v1",
        hashlib.sha256,
    ).hexdigest()


def read_inference_token(environment: MutableMapping[str, str] | None = None) -> str:
    source = os.environ if environment is None else environment
    raw_fd = source.pop(INFERENCE_TOKEN_FD_ENV, None)
    legacy_token = source.pop(INFERENCE_TOKEN_ENV, None)
    if raw_fd is None:
        return _token(legacy_token)
    try:
        descriptor = int(raw_fd)
        if descriptor < 0:
            raise ValueError
    except ValueError as exc:
        raise InferenceProtocolError("shared inference token descriptor is invalid") from exc
    token_bytes = bytearray()
    try:
        while len(token_bytes) <= 256:
            chunk = os.read(descriptor, min(257 - len(token_bytes), 257))
            if not chunk:
                break
            token_bytes.extend(chunk)
    except OSError as exc:
        raise InferenceProtocolError("cannot read shared inference token descriptor") from exc
    finally:
        with suppress(OSError):
            os.close(descriptor)
    if len(token_bytes) > 256:
        raise InferenceProtocolError("shared inference token is invalid")
    try:
        return _token(token_bytes.decode("ascii"))
    except UnicodeDecodeError as exc:
        raise InferenceProtocolError("shared inference token is invalid") from exc


def encode_ping_request(request_id: str, token: str, maximum_payload_bytes: int) -> bytes:
    header: dict[str, object] = {
        "schemaVersion": PROTOCOL_VERSION,
        "type": "ping",
        "requestId": _request_id(request_id),
    }
    header["signature"] = _signature(header, _token(token))
    return _encode_request_header(
        header,
        (),
        maximum_payload_bytes,
    )


def encode_detect_request(
    request_id: str,
    camera_id: str,
    detector: DetectorKind,
    images: tuple[NDArray[np.uint8], ...],
    token: str,
    maximum_payload_bytes: int,
    maximum_images: int,
) -> bytes:
    if detector not in {"vehicle", "plate"}:
        raise InferenceProtocolError("unsupported detector kind")
    if not 1 <= len(images) <= maximum_images:
        raise InferenceProtocolError("inference image count is outside configured bounds")
    if sum(int(image.nbytes) for image in images) > maximum_payload_bytes - _LENGTH.size:
        raise InferenceProtocolError("inference frame exceeds configured payload bounds")
    encoded_images: list[bytes] = []
    specifications: list[dict[str, object]] = []
    for image in images:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise InferenceProtocolError("inference images must be non-empty uint8 HxWx3 arrays")
        if any(dimension <= 0 or dimension > 32_768 for dimension in image.shape):
            raise InferenceProtocolError("inference image dimensions are outside protocol bounds")
        contiguous = np.ascontiguousarray(image)
        encoded = contiguous.tobytes(order="C")
        encoded_images.append(encoded)
        specifications.append(
            {
                "shape": list(contiguous.shape),
                "byteLength": len(encoded),
            }
        )
    body_sha256 = hashlib.sha256()
    for encoded in encoded_images:
        body_sha256.update(encoded)
    header = {
        "schemaVersion": PROTOCOL_VERSION,
        "type": "detect",
        "requestId": _request_id(request_id),
        "cameraId": _camera_id(camera_id),
        "detector": detector,
        "images": specifications,
        "bodySha256": body_sha256.hexdigest(),
    }
    header["signature"] = _signature(header, _token(token))
    return _encode_request_header(
        header,
        tuple(encoded_images),
        maximum_payload_bytes,
    )


def decode_request(
    payload: bytes,
    expected_token: str,
    maximum_images: int,
) -> InferenceRequest:
    try:
        header, raw_offset = _decode_header(payload)
        parsed = _parse_request_header(header, expected_token, maximum_images)
        if isinstance(parsed, PingRequest):
            if raw_offset != len(payload):
                raise InferenceProtocolError("ping request contains trailing bytes")
            return parsed
        return _materialize_detect_request(parsed, payload, raw_offset)
    except InferenceProtocolError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InferenceProtocolError("inference request is malformed") from exc


async def read_authenticated_request(
    reader: asyncio.StreamReader,
    maximum_payload_bytes: int,
    expected_token: str,
    maximum_images: int,
    reserve_payload: Callable[[int, str | None, DetectorKind | None], None] | None = None,
) -> InferenceRequest:
    """Authenticate the bounded header before allocating or reading image bytes."""
    try:
        frame_prefix = await reader.readexactly(_LENGTH.size)
        frame_length = _LENGTH.unpack(frame_prefix)[0]
        _validate_payload_length(frame_length, maximum_payload_bytes)
        header_prefix = await reader.readexactly(_LENGTH.size)
        header_length = _LENGTH.unpack(header_prefix)[0]
        if not 1 <= header_length <= MAXIMUM_HEADER_BYTES:
            raise InferenceProtocolError("inference request header exceeds protocol bounds")
        if _LENGTH.size + header_length > frame_length:
            raise InferenceProtocolError("inference request header is truncated")
        header_bytes = await reader.readexactly(header_length)
        header = json.loads(header_bytes, parse_constant=_reject_json_constant)
        if not isinstance(header, dict):
            raise InferenceProtocolError("inference request header must be an object")
        parsed = _parse_request_header(header, expected_token, maximum_images)
        raw_length = frame_length - _LENGTH.size - header_length
        if isinstance(parsed, PingRequest):
            if raw_length:
                raise InferenceProtocolError("ping request contains trailing bytes")
            if reserve_payload is not None:
                reserve_payload(frame_length, None, None)
            return parsed
        if sum(byte_length for _, byte_length in parsed.image_specs) != raw_length:
            raise InferenceProtocolError("inference image payload length does not match header")
        if reserve_payload is not None:
            reserve_payload(frame_length, parsed.camera_id, parsed.detector)
        raw_images = await reader.readexactly(raw_length)
        return _materialize_detect_request(parsed, raw_images, 0)
    except asyncio.IncompleteReadError as exc:
        raise InferenceProtocolError("inference frame is truncated") from exc
    except InferenceProtocolError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InferenceProtocolError("inference request is malformed") from exc


def encode_success_response(
    request: InferenceRequest,
    results: tuple[tuple[DetectionResult, ...], ...] = (),
    *,
    token: str,
) -> bytes:
    if isinstance(request, PingRequest):
        document: dict[str, object] = {
            "schemaVersion": PROTOCOL_VERSION,
            "requestId": request.request_id,
            "ok": True,
            "type": "pong",
        }
    else:
        if len(results) != len(request.images) or any(
            len(detections) > MAXIMUM_DETECTIONS_PER_IMAGE for detections in results
        ):
            raise InferenceProtocolError("inference response result count exceeds bounds")
        document = {
            "schemaVersion": PROTOCOL_VERSION,
            "requestId": request.request_id,
            "ok": True,
            "type": "result",
            "detector": request.detector,
            "results": [
                [_detection_document(item, request.detector) for item in detections]
                for detections in results
            ],
        }
    document["signature"] = _signature(document, _token(token))
    return json.dumps(
        document,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def encode_error_response(request_id: str | None, code: str) -> bytes:
    safe_request_id = (
        request_id if isinstance(request_id, str) and _REQUEST_ID.fullmatch(request_id) else None
    )
    document = {
        "schemaVersion": PROTOCOL_VERSION,
        "requestId": safe_request_id,
        "ok": False,
        "error": {"code": code[:64]},
    }
    return json.dumps(
        document,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def decode_success_response(
    payload: bytes,
    request_id: str,
    detector: DetectorKind | None,
    expected_images: int,
    *,
    token: str,
) -> tuple[tuple[DetectionResult, ...], ...]:
    try:
        document = json.loads(payload, parse_constant=_reject_json_constant)
        if not isinstance(document, dict) or document.get("schemaVersion") != PROTOCOL_VERSION:
            raise InferenceProtocolError("inference response version is invalid")
        expected_request_id = _request_id(request_id)
        if document.get("ok") is not True:
            if document.get("requestId") not in {expected_request_id, None}:
                raise InferenceProtocolError("inference response request id does not match")
            error = document.get("error")
            code = error.get("code") if isinstance(error, dict) else "unknown"
            raise InferenceError(f"shared inference request failed: {code}")
        if document.get("requestId") != expected_request_id:
            raise InferenceProtocolError("inference response request id does not match")
        _verify_signature(document, _token(token))
        if detector is None:
            if document.get("type") != "pong":
                raise InferenceProtocolError("inference ping response is invalid")
            return ()
        if document.get("type") != "result" or document.get("detector") != detector:
            raise InferenceProtocolError("inference response detector does not match")
        result_documents = document.get("results")
        if not isinstance(result_documents, list) or len(result_documents) != expected_images:
            raise InferenceProtocolError("inference response result count does not match")
        if any(
            not isinstance(detections, list) or len(detections) > MAXIMUM_DETECTIONS_PER_IMAGE
            for detections in result_documents
        ):
            raise InferenceProtocolError("inference response result count exceeds bounds")
        return tuple(
            tuple(_detection_from_document(item, detector) for item in detections)
            if isinstance(detections, list)
            else _raise_invalid_results()
            for detections in result_documents
        )
    except (InferenceError, InferenceProtocolError):
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InferenceProtocolError("inference response is malformed") from exc


def frame_payload(payload: bytes, maximum_payload_bytes: int) -> bytes:
    _validate_payload_length(len(payload), maximum_payload_bytes)
    return _LENGTH.pack(len(payload)) + payload


async def read_framed_payload(
    reader: asyncio.StreamReader,
    maximum_payload_bytes: int,
) -> bytes:
    try:
        prefix = await reader.readexactly(_LENGTH.size)
        length = _LENGTH.unpack(prefix)[0]
        _validate_payload_length(length, maximum_payload_bytes)
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise InferenceProtocolError("inference frame is truncated") from exc


async def write_framed_payload(
    writer: asyncio.StreamWriter,
    payload: bytes,
    maximum_payload_bytes: int,
) -> None:
    writer.write(frame_payload(payload, maximum_payload_bytes))
    await writer.drain()


def send_framed_payload(sock: socket.socket, payload: bytes, maximum_payload_bytes: int) -> None:
    sock.sendall(frame_payload(payload, maximum_payload_bytes))


def receive_framed_payload(
    sock: socket.socket,
    maximum_payload_bytes: int,
    deadline_monotonic: float | None = None,
) -> bytes:
    prefix = _receive_exact(sock, _LENGTH.size, deadline_monotonic)
    length = _LENGTH.unpack(prefix)[0]
    _validate_payload_length(length, maximum_payload_bytes)
    return _receive_exact(sock, length, deadline_monotonic)


def _encode_request_header(
    header: dict[str, object],
    images: tuple[bytes, ...],
    maximum_payload_bytes: int,
) -> bytes:
    header_bytes = json.dumps(
        header,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if not header_bytes or len(header_bytes) > MAXIMUM_HEADER_BYTES:
        raise InferenceProtocolError("inference request header exceeds protocol bounds")
    payload = _LENGTH.pack(len(header_bytes)) + header_bytes + b"".join(images)
    _validate_payload_length(len(payload), maximum_payload_bytes)
    return payload


def _decode_header(payload: bytes) -> tuple[dict[str, object], int]:
    if len(payload) < _LENGTH.size:
        raise InferenceProtocolError("inference request header is truncated")
    header_length = _LENGTH.unpack(payload[: _LENGTH.size])[0]
    if not 1 <= header_length <= MAXIMUM_HEADER_BYTES:
        raise InferenceProtocolError("inference request header exceeds protocol bounds")
    header_end = _LENGTH.size + header_length
    if header_end > len(payload):
        raise InferenceProtocolError("inference request header is truncated")
    header = json.loads(
        payload[_LENGTH.size : header_end],
        parse_constant=_reject_json_constant,
    )
    if not isinstance(header, dict):
        raise InferenceProtocolError("inference request header must be an object")
    return header, header_end


def _parse_request_header(
    header: dict[str, object],
    expected_token: str,
    maximum_images: int,
) -> PingRequest | _DetectHeader:
    if header.get("schemaVersion") != PROTOCOL_VERSION:
        raise InferenceProtocolError("unsupported inference protocol version")
    request_id = _request_id(header.get("requestId"))
    request_type = header.get("type")
    if request_type == "ping":
        _verify_signature(header, derive_supervisor_token(expected_token))
        return PingRequest(request_id)
    if request_type != "detect":
        raise InferenceProtocolError("unsupported inference request type")
    camera_id = _camera_id(header.get("cameraId"))
    expected_camera_token = derive_camera_token(expected_token, camera_id)
    _verify_signature(header, expected_camera_token)
    raw_detector = header.get("detector")
    if raw_detector not in {"vehicle", "plate"}:
        raise InferenceProtocolError("unsupported detector kind")
    detector = cast(DetectorKind, raw_detector)
    image_documents = header.get("images")
    if not isinstance(image_documents, list) or not 1 <= len(image_documents) <= maximum_images:
        raise InferenceProtocolError("inference image count is outside configured bounds")
    image_specs = tuple(_image_specification(item) for item in image_documents)
    body_sha256 = header.get("bodySha256")
    if (
        not isinstance(body_sha256, str)
        or len(body_sha256) != 64
        or any(character not in "0123456789abcdef" for character in body_sha256)
    ):
        raise InferenceProtocolError("inference image payload digest is invalid")
    return _DetectHeader(request_id, camera_id, detector, image_specs, body_sha256)


def _materialize_detect_request(
    header: _DetectHeader,
    payload: bytes,
    raw_offset: int,
) -> DetectRequest:
    raw = payload[raw_offset:]
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), header.body_sha256):
        raise InferenceProtocolError("inference image payload digest does not match")
    images: list[NDArray[np.uint8]] = []
    cursor = raw_offset
    for shape, byte_length in header.image_specs:
        end = cursor + byte_length
        if end > len(payload):
            raise InferenceProtocolError("inference image payload is truncated")
        image = np.frombuffer(payload, dtype=np.uint8, count=byte_length, offset=cursor)
        images.append(image.reshape(shape))
        cursor = end
    if cursor != len(payload):
        raise InferenceProtocolError("inference request contains trailing bytes")
    return DetectRequest(
        header.request_id,
        header.camera_id,
        header.detector,
        tuple(images),
    )


def _image_specification(specification: object) -> tuple[tuple[int, ...], int]:
    if not isinstance(specification, dict):
        raise InferenceProtocolError("inference image specification is invalid")
    raw_shape = specification.get("shape")
    byte_length = specification.get("byteLength")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or any(type(value) is not int or value <= 0 or value > 32_768 for value in raw_shape)
        or type(byte_length) is not int
        or byte_length <= 0
    ):
        raise InferenceProtocolError("inference image specification is invalid")
    shape = tuple(raw_shape)
    if shape[2] != 3:
        raise InferenceProtocolError("inference image channel count is unsupported")
    if prod(shape) != byte_length:
        raise InferenceProtocolError("inference image byte length does not match shape")
    return shape, byte_length


def _detection_document(item: DetectionResult, detector: DetectorKind) -> dict[str, object]:
    model = {
        "name": item.model.name,
        "version": item.model.version,
        "hash": item.model.hash,
    }
    document: dict[str, object] = {
        "bbox": list(item.bbox.as_xyxy()),
        "confidence": item.confidence,
        "model": model,
    }
    if detector == "vehicle":
        if not isinstance(item, Detection):
            raise InferenceProtocolError("vehicle response contains a plate detection")
        document.update({"classId": item.class_id, "className": item.class_name})
    else:
        if not isinstance(item, PlateDetection):
            raise InferenceProtocolError("plate response contains a vehicle detection")
        document["corners"] = (
            [[point.x, point.y] for point in item.corners] if item.corners is not None else None
        )
    return document


def _detection_from_document(document: object, detector: DetectorKind) -> DetectionResult:
    if not isinstance(document, dict):
        raise InferenceProtocolError("inference detection result is invalid")
    raw_bbox = document.get("bbox")
    raw_model = document.get("model")
    if (
        not isinstance(raw_bbox, list)
        or len(raw_bbox) != 4
        or any(type(value) is not int for value in raw_bbox)
        or not isinstance(raw_model, dict)
    ):
        raise InferenceProtocolError("inference detection result is invalid")
    bbox = BoundingBox(*raw_bbox)
    model = ModelMetadata(
        _nonempty_string(raw_model.get("name"), "model name"),
        _nonempty_string(raw_model.get("version"), "model version"),
        _optional_string(raw_model.get("hash")),
    )
    confidence = _finite_number(document.get("confidence"), "confidence")
    if detector == "vehicle":
        class_id = document.get("classId")
        if type(class_id) is not int:
            raise InferenceProtocolError("vehicle class id is invalid")
        return Detection(
            bbox,
            confidence,
            class_id,
            _nonempty_string(document.get("className"), "vehicle class name"),
            model,
        )
    raw_corners = document.get("corners")
    corners = None
    if raw_corners is not None:
        if not isinstance(raw_corners, list) or len(raw_corners) != 4:
            raise InferenceProtocolError("plate corners are invalid")
        points: list[Point] = []
        for raw_point in raw_corners:
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise InferenceProtocolError("plate corners are invalid")
            points.append(
                Point(
                    _finite_number(raw_point[0], "plate corner"),
                    _finite_number(raw_point[1], "plate corner"),
                )
            )
        corners = cast(tuple[Point, Point, Point, Point], tuple(points))
    return PlateDetection(bbox, confidence, model, corners)


def _request_id(value: object) -> str:
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        raise InferenceProtocolError("inference request id is invalid")
    return value


def _camera_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(character in value for character in "/\\\0")
    ):
        raise InferenceProtocolError("inference camera id is invalid")
    return value.strip()


def _token(value: object) -> str:
    if not isinstance(value, str) or len(value) < 32 or len(value) > 256:
        raise InferenceProtocolError("shared inference token is invalid")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise InferenceProtocolError("shared inference token must be ASCII") from exc
    return value


def _signature(document: dict[str, object], token: str) -> str:
    unsigned = {key: value for key, value in document.items() if key != "signature"}
    canonical = json.dumps(
        unsigned,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hmac.new(token.encode(), canonical, hashlib.sha256).hexdigest()


def _verify_signature(document: dict[str, object], token: str) -> None:
    supplied = document.get("signature")
    if (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or not hmac.compare_digest(supplied, _signature(document, token))
    ):
        raise InferenceProtocolError("shared inference authentication failed")


def _finite_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InferenceProtocolError(f"inference {field} is invalid")
    numeric = float(value)
    if not isfinite(numeric):
        raise InferenceProtocolError(f"inference {field} is invalid")
    return numeric


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise InferenceProtocolError(f"inference {field} is invalid")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, "model hash")


def _raise_invalid_results() -> NoReturn:
    raise InferenceProtocolError("inference response results are invalid")


def _validate_payload_length(length: int, maximum_payload_bytes: int) -> None:
    if maximum_payload_bytes <= 0 or not 1 <= length <= maximum_payload_bytes:
        raise InferenceProtocolError("inference frame exceeds configured payload bounds")


def _receive_exact(
    sock: socket.socket,
    length: int,
    deadline_monotonic: float | None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        if deadline_monotonic is not None:
            timeout = deadline_monotonic - time.monotonic()
            if timeout <= 0:
                raise TimeoutError("shared inference response deadline exceeded")
            sock.settimeout(timeout)
        chunk = sock.recv(remaining)
        if not chunk:
            raise InferenceProtocolError("inference frame is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
