from __future__ import annotations

from vehicle_intelligence.domain import VehicleFingerprint, VehicleIdentity


def identity_to_jsonable(identity: VehicleIdentity) -> dict[str, object]:
    return {
        "id": identity.id,
        "schemaVersion": identity.schema_version,
        "revision": identity.revision,
        "status": identity.status.value,
        "primaryPlate": identity.primary_plate,
        "plates": [
            {
                "text": plate.text,
                "confidence": plate.confidence,
                "firstSeenAt": plate.first_seen_at.isoformat(),
                "lastSeenAt": plate.last_seen_at.isoformat(),
            }
            for plate in identity.plates
        ],
        "attributes": {
            "type": identity.vehicle_type,
            "color": identity.color,
        },
        "firstSeenAt": identity.first_seen_at.isoformat(),
        "lastSeenAt": identity.last_seen_at.isoformat(),
        "observationCount": identity.observation_count,
        "metadata": identity.metadata,
    }


def fingerprint_to_jsonable(fingerprint: VehicleFingerprint) -> dict[str, object]:
    embedding = fingerprint.embedding
    return {
        "id": fingerprint.id,
        "schemaVersion": fingerprint.schema_version,
        "vehicleId": fingerprint.vehicle_id,
        "sourceEventId": fingerprint.source_event_id,
        "cameraId": fingerprint.camera_id,
        "observedAt": fingerprint.observed_at.isoformat(),
        "plate": (
            {"text": fingerprint.plate, "confidence": fingerprint.plate_confidence}
            if fingerprint.plate is not None
            else None
        ),
        "vehicle": {
            "type": fingerprint.vehicle_type,
            "confidence": fingerprint.vehicle_confidence,
            "color": fingerprint.color,
        },
        "embedding": (
            {
                "id": embedding.id,
                "model": {
                    "name": embedding.model.name,
                    "version": embedding.model.version,
                    "hash": embedding.model.model_hash,
                    "dimension": embedding.model.dimension,
                },
            }
            if embedding is not None
            else None
        ),
    }
