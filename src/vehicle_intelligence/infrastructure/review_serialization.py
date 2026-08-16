from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from vehicle_intelligence.domain import (
    DatasetSample,
    DatasetSampleReason,
    DatasetSampleStatus,
    DatasetSampleType,
    ModelMetadata,
    OCRDatasetPrediction,
)


def dataset_sample_to_document(sample: DatasetSample) -> dict[str, Any]:
    model = sample.prediction.model
    document = {
        "_id": sample.id,
        "schemaVersion": sample.schema_version,
        "type": sample.sample_type.value,
        "status": sample.status.value,
        "sourceEventId": sample.source_event_id,
        "imageKey": sample.image_key,
        "prediction": {
            "raw": sample.prediction.raw,
            "normalized": sample.prediction.normalized,
            "confidence": sample.prediction.confidence,
            "model": (
                {
                    "name": model.name,
                    "version": model.version,
                    "hash": model.hash,
                }
                if model is not None
                else None
            ),
        },
        "label": sample.label,
        "reason": sample.reason.value,
        "review": {
            "revision": sample.review_revision,
            "reviewedBy": {
                "id": sample.reviewed_by,
                "displayName": sample.reviewer_display_name,
            },
            "reviewedAt": sample.reviewed_at.astimezone(UTC),
        },
        "createdAt": sample.created_at.astimezone(UTC),
    }
    if sample.export_id is not None or sample.export_attempts:
        document["export"] = {
            "id": sample.export_id,
            "attempts": sample.export_attempts,
            "claimedAt": (
                sample.export_claimed_at.astimezone(UTC)
                if sample.export_claimed_at is not None
                else None
            ),
            "exportedAt": (
                sample.exported_at.astimezone(UTC) if sample.exported_at is not None else None
            ),
            "manifestSha256": sample.export_manifest_sha256,
            "errorCode": sample.export_error_code,
        }
    return document


def dataset_sample_to_jsonable(sample: DatasetSample) -> dict[str, Any]:
    document = dataset_sample_to_document(sample)
    document["createdAt"] = document["createdAt"].isoformat().replace("+00:00", "Z")
    review = document["review"]
    review["reviewedAt"] = review["reviewedAt"].isoformat().replace("+00:00", "Z")
    export = document.get("export")
    if isinstance(export, dict):
        for field in ("claimedAt", "exportedAt"):
            value = export.get(field)
            if isinstance(value, datetime):
                export[field] = value.isoformat().replace("+00:00", "Z")
    return document


def document_to_dataset_sample(document: dict[str, Any]) -> DatasetSample:
    prediction = document["prediction"]
    model_document = prediction.get("model")
    review = document["review"]
    reviewer = review["reviewedBy"]
    export = document.get("export") or {}
    return DatasetSample(
        id=str(document.get("_id") or document["id"]),
        schema_version=int(document.get("schemaVersion", 1)),
        sample_type=DatasetSampleType(document["type"]),
        status=DatasetSampleStatus(document["status"]),
        source_event_id=str(document["sourceEventId"]),
        image_key=str(document["imageKey"]),
        prediction=OCRDatasetPrediction(
            raw=str(prediction["raw"]),
            normalized=str(prediction["normalized"]),
            confidence=float(prediction["confidence"]),
            model=(
                ModelMetadata(
                    name=str(model_document["name"]),
                    version=str(model_document["version"]),
                    hash=model_document.get("hash"),
                )
                if model_document
                else None
            ),
        ),
        label=str(document["label"]),
        reason=DatasetSampleReason(document["reason"]),
        review_revision=int(review["revision"]),
        reviewed_by=str(reviewer["id"]),
        reviewer_display_name=str(reviewer.get("displayName") or reviewer["id"]),
        reviewed_at=_aware(review["reviewedAt"]),
        created_at=_aware(document["createdAt"]),
        export_id=export.get("id"),
        export_attempts=int(export.get("attempts", 0)),
        export_claimed_at=(
            _aware(export["claimedAt"]) if export.get("claimedAt") is not None else None
        ),
        exported_at=(
            _aware(export["exportedAt"]) if export.get("exportedAt") is not None else None
        ),
        export_manifest_sha256=export.get("manifestSha256"),
        export_error_code=export.get("errorCode"),
    )


def _aware(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
