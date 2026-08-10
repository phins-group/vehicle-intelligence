import json

import pytest

from vehicle_intelligence.exceptions import EventContractError
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec


def test_event_envelope_codec_round_trip(sample_event) -> None:
    codec = JsonEventEnvelopeCodec()

    payload = codec.encode(sample_event)

    envelope = json.loads(payload)
    assert envelope["type"] == "vehicle.entered"
    assert envelope["schemaVersion"] == 1
    assert envelope["correlationId"] == sample_event.track_id
    assert codec.decode(payload) == sample_event


def test_event_envelope_codec_rejects_incoherent_type(sample_event) -> None:
    codec = JsonEventEnvelopeCodec()
    envelope = json.loads(codec.encode(sample_event))
    envelope["type"] = "vehicle.exited"

    with pytest.raises(EventContractError, match="does not match"):
        codec.decode(json.dumps(envelope))


def test_event_envelope_codec_rejects_unknown_schema(sample_event) -> None:
    codec = JsonEventEnvelopeCodec()
    envelope = json.loads(codec.encode(sample_event))
    envelope["schemaVersion"] = 99

    with pytest.raises(EventContractError, match="unsupported envelope schema"):
        codec.decode(json.dumps(envelope))
