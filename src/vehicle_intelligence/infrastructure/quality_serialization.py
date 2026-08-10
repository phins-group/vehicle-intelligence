"""Public JSON mapping for model-quality reports."""

from __future__ import annotations

from typing import Any

from vehicle_intelligence.domain import ModelQualityReport, QualityMetrics


def quality_report_to_jsonable(report: ModelQualityReport) -> dict[str, Any]:
    return {
        "schemaVersion": report.schema_version,
        "window": {
            "from": _timestamp(report.from_time),
            "to": _timestamp(report.to_time),
        },
        "generatedAt": _timestamp(report.generated_at),
        "totals": _metrics(report.totals),
        "models": [
            {
                "model": (
                    {
                        "name": item.model.name,
                        "version": item.model.version,
                        "hash": item.model.hash,
                    }
                    if item.model is not None
                    else None
                ),
                "metrics": _metrics(item.metrics),
            }
            for item in report.models
        ],
        "daily": [
            {"day": item.day, "metrics": _metrics(item.metrics)} for item in report.daily
        ],
        "feedback": {
            "total": report.feedback.total,
            "ready": report.feedback.ready,
            "exporting": report.feedback.exporting,
            "exported": report.feedback.exported,
            "exportFailed": report.feedback.export_failed,
            "corrections": report.feedback.corrections,
            "confirmations": report.feedback.confirmations,
        },
        "truncated": report.truncated,
    }


def _metrics(metrics: QualityMetrics) -> dict[str, int | float | None]:
    return {
        "eventCount": metrics.event_count,
        "readablePlateCount": metrics.readable_plate_count,
        "confirmedCount": metrics.confirmed_count,
        "needsReviewCount": metrics.needs_review_count,
        "noPlateCount": metrics.no_plate_count,
        "unreadableCount": metrics.unreadable_count,
        "reviewedCount": metrics.reviewed_count,
        "correctedCount": metrics.corrected_count,
        "ocrSuccessRate": metrics.ocr_success_rate,
        "unknownPlateRate": metrics.unknown_plate_rate,
        "humanCorrectionRate": metrics.human_correction_rate,
        "averagePlateConfidence": metrics.average_plate_confidence,
    }


def _timestamp(value) -> str:
    return value.isoformat().replace("+00:00", "Z")
