"""JSON codec for the versioned vehicle-event envelope."""

from __future__ import annotations

from datetime import UTC

from pydantic import ValidationError

from vehicle_intelligence.contracts import EventEnvelope
from vehicle_intelligence.domain import EventType, VehicleEvent
from vehicle_intelligence.exceptions import EventContractError
from vehicle_intelligence.infrastructure.serialization import (
    document_to_event,
    event_to_jsonable,
)

EVENT_ENVELOPE_SCHEMA_VERSION = 1
EVENT_TOPICS: dict[EventType, str] = {
    EventType.VEHICLE_DETECTED: "vehicle.detected",
    EventType.VEHICLE_ENTER: "vehicle.entered",
    EventType.VEHICLE_EXIT: "vehicle.exited",
}
TOPIC_EVENTS = {topic: event_type for event_type, topic in EVENT_TOPICS.items()}


class JsonEventEnvelopeCodec:
    def encode(self, event: VehicleEvent) -> str:
        envelope = EventEnvelope(
            id=event.id,
            type=EVENT_TOPICS[event.event_type],
            schemaVersion=EVENT_ENVELOPE_SCHEMA_VERSION,
            occurredAt=event.occurred_at,
            source=f"vision-worker/{event.camera.id}",
            correlationId=event.track_id,
            data=event_to_jsonable(event),
        )
        return envelope.model_dump_json(by_alias=True)

    def decode(self, payload: str) -> VehicleEvent:
        try:
            envelope = EventEnvelope.model_validate_json(payload)
            if envelope.schema_version != EVENT_ENVELOPE_SCHEMA_VERSION:
                raise ValueError(f"unsupported envelope schema version: {envelope.schema_version}")
            expected_event_type = TOPIC_EVENTS.get(envelope.type)
            if expected_event_type is None:
                raise ValueError(f"unsupported vehicle event type: {envelope.type}")
            event = document_to_event(envelope.data)
            self._validate_coherence(envelope, event, expected_event_type)
            return event
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise EventContractError(f"invalid vehicle event envelope: {exc}") from exc

    @staticmethod
    def _validate_coherence(
        envelope: EventEnvelope,
        event: VehicleEvent,
        expected_event_type: EventType,
    ) -> None:
        if envelope.id != event.id:
            raise ValueError("envelope id does not match data event id")
        if event.event_type is not expected_event_type:
            raise ValueError("envelope type does not match data event type")
        if envelope.source != f"vision-worker/{event.camera.id}":
            raise ValueError("envelope source does not match data camera id")
        if envelope.correlation_id is not None and envelope.correlation_id != event.track_id:
            raise ValueError("envelope correlationId does not match data trackId")
        if envelope.occurred_at.astimezone(UTC) != event.occurred_at.astimezone(UTC):
            raise ValueError("envelope occurredAt does not match data occurredAt")
