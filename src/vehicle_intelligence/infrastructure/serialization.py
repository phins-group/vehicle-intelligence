"""Mapping between domain events and durable/public document shapes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from vehicle_intelligence.domain import (
    AITrace,
    CameraSnapshot,
    CharacterCorrection,
    Direction,
    EventStatus,
    EventType,
    MediaReferences,
    ModelMetadata,
    PlateEvidence,
    PlateReview,
    VehicleEvent,
    VehicleEvidence,
)


def _model_to_document(model: ModelMetadata | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {"model": model.name, "version": model.version, "hash": model.hash}


def _correction_to_document(correction: CharacterCorrection) -> dict[str, Any]:
    return {
        "position": correction.position,
        "from": correction.from_character,
        "to": correction.to_character,
        "confidence": correction.confidence,
    }


def _plate_prediction_to_document(plate: PlateEvidence) -> dict[str, Any]:
    return {
        "raw": plate.raw,
        "normalized": plate.normalized,
        "confidence": plate.confidence,
        "observationCount": plate.observation_count,
        "partial": plate.partial,
        "corrections": [_correction_to_document(item) for item in plate.corrections],
    }


def _plate_review_to_document(review: PlateReview | None) -> dict[str, Any] | None:
    if review is None:
        return None
    return {
        "normalized": review.normalized,
        "revision": review.revision,
        "reviewedAt": review.reviewed_at.astimezone(UTC),
        "reviewedBy": {
            "id": review.reviewed_by,
            "displayName": review.reviewer_display_name,
        },
        "note": review.note,
    }


def event_to_document(event: VehicleEvent) -> dict[str, Any]:
    return {
        "_id": event.id,
        "schemaVersion": event.schema_version,
        "camera": {
            "id": event.camera.id,
            "name": event.camera.name,
            "zone": event.camera.zone,
        },
        "trackId": event.track_id,
        "vehicleId": event.vehicle_id,
        "eventType": event.event_type.value,
        "direction": event.direction.value,
        "status": event.status.value,
        "plate": (
            _plate_to_document(event.plate, event.schema_version)
            if event.plate is not None
            else None
        ),
        "vehicle": {
            "type": event.vehicle.type,
            "confidence": event.vehicle.confidence,
            "color": event.vehicle.color,
        },
        "media": {
            "snapshotKey": event.media.snapshot_key,
            "vehicleCropKey": event.media.vehicle_crop_key,
            "plateCropKey": event.media.plate_crop_key,
            "clipKey": event.media.clip_key,
        },
        "ai": {
            "vehicleDetector": _model_to_document(event.ai.vehicle_detector),
            "plateDetector": _model_to_document(event.ai.plate_detector),
            "ocr": _model_to_document(event.ai.ocr),
            "configVersion": event.ai.config_version,
        },
        "occurredAt": event.occurred_at.astimezone(UTC),
        "createdAt": event.created_at.astimezone(UTC),
        "metadata": event.metadata,
    }


def event_to_jsonable(event: VehicleEvent) -> dict[str, Any]:
    document = event_to_document(event)
    document["occurredAt"] = document["occurredAt"].isoformat().replace("+00:00", "Z")
    document["createdAt"] = document["createdAt"].isoformat().replace("+00:00", "Z")
    plate_document = document.get("plate")
    if event.plate is not None and isinstance(plate_document, dict):
        plate_document.setdefault("prediction", _plate_prediction_to_document(event.plate))
        plate_document.setdefault("review", _plate_review_to_document(event.plate.review))
        plate_document.setdefault("final", event.plate.final_normalized)
        review = plate_document.get("review")
        if isinstance(review, dict) and isinstance(review.get("reviewedAt"), datetime):
            review["reviewedAt"] = review["reviewedAt"].isoformat().replace("+00:00", "Z")
    return document


def _plate_to_document(plate: PlateEvidence, schema_version: int) -> dict[str, Any]:
    legacy = _plate_prediction_to_document(plate)
    if schema_version < 2:
        return legacy
    return {
        **legacy,
        "prediction": _plate_prediction_to_document(plate),
        "review": _plate_review_to_document(plate.review),
        "final": plate.final_normalized,
    }


def _aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _model_from_document(value: dict[str, Any] | None) -> ModelMetadata | None:
    if not value:
        return None
    return ModelMetadata(
        name=str(value["model"]),
        version=str(value["version"]),
        hash=value.get("hash"),
    )


def document_to_event(document: dict[str, Any]) -> VehicleEvent:
    camera = document["camera"]
    vehicle = document["vehicle"]
    media = document.get("media") or {}
    ai = document.get("ai") or {}
    plate_document = document.get("plate")
    plate = None
    if plate_document:
        prediction = plate_document.get("prediction") or plate_document
        review_document = plate_document.get("review")
        review = None
        if review_document:
            reviewer = review_document.get("reviewedBy") or {}
            review = PlateReview(
                normalized=str(review_document["normalized"]),
                revision=int(review_document["revision"]),
                reviewed_at=_aware(review_document["reviewedAt"]),
                reviewed_by=str(reviewer["id"]),
                reviewer_display_name=str(reviewer.get("displayName") or reviewer["id"]),
                note=review_document.get("note"),
            )
        plate = PlateEvidence(
            raw=str(prediction["raw"]),
            normalized=str(prediction["normalized"]),
            confidence=float(prediction["confidence"]),
            observation_count=int(prediction.get("observationCount", 1)),
            partial=bool(prediction.get("partial", False)),
            corrections=tuple(
                CharacterCorrection(
                    position=int(item["position"]),
                    from_character=str(item["from"]),
                    to_character=str(item["to"]),
                    confidence=float(item["confidence"]),
                )
                for item in prediction.get("corrections", [])
            ),
            review=review,
        )
    return VehicleEvent(
        id=str(document.get("_id") or document["id"]),
        schema_version=int(document["schemaVersion"]),
        camera=CameraSnapshot(
            id=str(camera["id"]),
            name=str(camera["name"]),
            zone=camera.get("zone"),
        ),
        track_id=str(document["trackId"]),
        vehicle_id=document.get("vehicleId"),
        event_type=EventType(document["eventType"]),
        occurred_at=_aware(document["occurredAt"]),
        created_at=_aware(document["createdAt"]),
        direction=Direction(document["direction"]),
        status=EventStatus(document["status"]),
        plate=plate,
        vehicle=VehicleEvidence(
            type=str(vehicle["type"]),
            confidence=float(vehicle["confidence"]),
            color=vehicle.get("color"),
        ),
        media=MediaReferences(
            snapshot_key=media.get("snapshotKey"),
            vehicle_crop_key=media.get("vehicleCropKey"),
            plate_crop_key=media.get("plateCropKey"),
            clip_key=media.get("clipKey"),
        ),
        ai=AITrace(
            vehicle_detector=_model_from_document(ai.get("vehicleDetector")),
            plate_detector=_model_from_document(ai.get("plateDetector")),
            ocr=_model_from_document(ai.get("ocr")),
            config_version=ai.get("configVersion"),
        ),
        metadata=document.get("metadata") or {},
    )
