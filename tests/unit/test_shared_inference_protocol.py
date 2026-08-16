import asyncio
import json

import numpy as np
import pytest

from vehicle_intelligence.domain import (
    BoundingBox,
    Detection,
    ModelMetadata,
    PlateDetection,
    Point,
)
from vehicle_intelligence.exceptions import InferenceError, InferenceProtocolError
from vehicle_intelligence.infrastructure.inference.protocol import (
    DetectRequest,
    decode_request,
    decode_success_response,
    derive_camera_token,
    encode_detect_request,
    encode_error_response,
    encode_success_response,
    frame_payload,
    read_authenticated_request,
    read_framed_payload,
)

TOKEN = "t" * 32
REQUEST_ID = "a" * 32
CAMERA_TOKEN = derive_camera_token(TOKEN, "gate-01")


def test_detect_request_round_trips_raw_images_without_base64() -> None:
    images = (
        np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        np.full((4, 5, 3), 217, dtype=np.uint8),
    )

    payload = encode_detect_request(
        REQUEST_ID,
        "gate-01",
        "vehicle",
        images,
        CAMERA_TOKEN,
        maximum_payload_bytes=1_048_576,
        maximum_images=4,
    )
    decoded = decode_request(payload, TOKEN, maximum_images=4)

    assert isinstance(decoded, DetectRequest)
    assert decoded.camera_id == "gate-01"
    assert decoded.detector == "vehicle"
    assert len(decoded.images) == 2
    np.testing.assert_array_equal(decoded.images[0], images[0])
    np.testing.assert_array_equal(decoded.images[1], images[1])


def test_protocol_rejects_authentication_and_payload_overflow() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    payload = encode_detect_request(
        REQUEST_ID,
        "gate-01",
        "plate",
        (image,),
        CAMERA_TOKEN,
        maximum_payload_bytes=1_048_576,
        maximum_images=1,
    )

    with pytest.raises(InferenceProtocolError, match="authentication"):
        decode_request(payload, "x" * 32, maximum_images=1)
    with pytest.raises(InferenceProtocolError, match="payload bounds"):
        encode_detect_request(
            REQUEST_ID,
            "gate-01",
            "plate",
            (image,),
            CAMERA_TOKEN,
            maximum_payload_bytes=128,
            maximum_images=1,
        )


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 1), (8, 8, 4)])
def test_protocol_rejects_non_bgr_images_before_batch_admission(shape) -> None:
    with pytest.raises(InferenceProtocolError, match="HxWx3"):
        encode_detect_request(
            REQUEST_ID,
            "gate-01",
            "vehicle",
            (np.zeros(shape, dtype=np.uint8),),
            CAMERA_TOKEN,
            maximum_payload_bytes=1_048_576,
            maximum_images=1,
        )


async def test_frame_reader_rejects_oversized_length_before_reading_body() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((4097).to_bytes(4, "big"))

    with pytest.raises(InferenceProtocolError, match="payload bounds"):
        await read_framed_payload(reader, maximum_payload_bytes=4096)


async def test_bad_signature_is_rejected_before_image_body_is_read() -> None:
    inner = encode_detect_request(
        REQUEST_ID,
        "gate-01",
        "vehicle",
        (np.zeros((128, 128, 3), dtype=np.uint8),),
        derive_camera_token("wrong-" + "x" * 32, "gate-01"),
        maximum_payload_bytes=1_048_576,
        maximum_images=1,
    )
    framed = frame_payload(inner, 1_048_576)
    header_length = int.from_bytes(framed[4:8], "big")
    reader = asyncio.StreamReader()
    reader.feed_data(framed[: 8 + header_length])

    with pytest.raises(InferenceProtocolError, match="authentication"):
        await asyncio.wait_for(
            read_authenticated_request(reader, 1_048_576, TOKEN, 1),
            timeout=0.1,
        )


def test_camera_capability_is_bound_to_camera_and_body_digest() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    wrong_camera = encode_detect_request(
        REQUEST_ID,
        "gate-02",
        "plate",
        (image,),
        CAMERA_TOKEN,
        maximum_payload_bytes=1_048_576,
        maximum_images=1,
    )
    with pytest.raises(InferenceProtocolError, match="authentication"):
        decode_request(wrong_camera, TOKEN, maximum_images=1)

    valid = bytearray(
        encode_detect_request(
            REQUEST_ID,
            "gate-01",
            "plate",
            (image,),
            CAMERA_TOKEN,
            maximum_payload_bytes=1_048_576,
            maximum_images=1,
        )
    )
    valid[-1] ^= 1
    with pytest.raises(InferenceProtocolError, match="digest"):
        decode_request(bytes(valid), TOKEN, maximum_images=1)


def test_vehicle_and_plate_results_round_trip_with_strict_mapping() -> None:
    model = ModelMetadata("detector", "v1", "a" * 64)
    vehicle = Detection(BoundingBox(1, 2, 10, 20), 0.8, 3, "truck", model)
    plate = PlateDetection(
        BoundingBox(2, 3, 8, 9),
        0.9,
        model,
        (Point(2, 3), Point(8, 3), Point(8, 9), Point(2, 9)),
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    vehicle_request = DetectRequest(REQUEST_ID, "gate-01", "vehicle", (image,))
    plate_request = DetectRequest(REQUEST_ID, "gate-01", "plate", (image,))

    vehicle_payload = encode_success_response(
        vehicle_request,
        ((vehicle,),),
        token=CAMERA_TOKEN,
    )
    plate_payload = encode_success_response(
        plate_request,
        ((plate,),),
        token=CAMERA_TOKEN,
    )

    assert decode_success_response(
        vehicle_payload,
        REQUEST_ID,
        "vehicle",
        1,
        token=CAMERA_TOKEN,
    ) == ((vehicle,),)
    assert decode_success_response(
        plate_payload,
        REQUEST_ID,
        "plate",
        1,
        token=CAMERA_TOKEN,
    ) == ((plate,),)


def test_remote_error_response_fails_closed() -> None:
    payload = encode_error_response(REQUEST_ID, "request_failed")

    with pytest.raises(InferenceError, match="request_failed"):
        decode_success_response(payload, REQUEST_ID, "vehicle", 1, token=CAMERA_TOKEN)


def test_success_response_signature_rejects_tampering() -> None:
    model = ModelMetadata("detector", "v1")
    request = DetectRequest(
        REQUEST_ID,
        "gate-01",
        "vehicle",
        (np.zeros((2, 2, 3), dtype=np.uint8),),
    )
    payload = encode_success_response(
        request,
        ((Detection(BoundingBox(0, 0, 2, 2), 0.8, 1, "car", model),),),
        token=CAMERA_TOKEN,
    )
    document = json.loads(payload)
    document["results"][0][0]["className"] = "forged"
    tampered = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()

    with pytest.raises(InferenceProtocolError, match="authentication"):
        decode_success_response(
            tampered,
            REQUEST_ID,
            "vehicle",
            1,
            token=CAMERA_TOKEN,
        )
