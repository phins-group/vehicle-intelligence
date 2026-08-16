"""Offline OCR feedback evaluation and configurable release gates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from vehicle_intelligence.exceptions import DatasetExportError


@dataclass(frozen=True, slots=True)
class OCRDatasetMetrics:
    sample_count: int
    exact_accuracy: float
    character_accuracy: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class OCRDatasetEvaluation:
    overall: OCRDatasetMetrics
    by_split: dict[str, OCRDatasetMetrics]
    by_model: dict[str, OCRDatasetMetrics]
    caveat: str = (
        "Human-review feedback is selection-biased; validate release gates on a separate "
        "representative holdout before deployment."
    )


def evaluate_ocr_records(records: list[dict[str, Any]], bins: int = 10) -> OCRDatasetEvaluation:
    if not records:
        raise DatasetExportError("cannot evaluate an empty OCR dataset")
    if not 2 <= bins <= 100:
        raise DatasetExportError("calibration bin count must be in [2, 100]")
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        _validate_record(record)
        by_split[str(record["split"])].append(record)
        model = record["prediction"].get("model")
        model_key = (
            f"{model.get('name', 'unknown')}@{model.get('version', 'unknown')}"
            if isinstance(model, dict)
            else "unknown"
        )
        by_model[model_key].append(record)
    return OCRDatasetEvaluation(
        overall=_metrics(records, bins),
        by_split={key: _metrics(value, bins) for key, value in sorted(by_split.items())},
        by_model={key: _metrics(value, bins) for key, value in sorted(by_model.items())},
    )


def evaluation_to_jsonable(evaluation: OCRDatasetEvaluation) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "overall": _metrics_json(evaluation.overall),
        "bySplit": {key: _metrics_json(value) for key, value in evaluation.by_split.items()},
        "byModel": {key: _metrics_json(value) for key, value in evaluation.by_model.items()},
        "caveat": evaluation.caveat,
    }


def release_gates(
    evaluation: OCRDatasetEvaluation,
    *,
    minimum_exact_accuracy: float | None = None,
    minimum_character_accuracy: float | None = None,
    maximum_ece: float | None = None,
) -> list[str]:
    for name, threshold in (
        ("minimum exact accuracy", minimum_exact_accuracy),
        ("minimum character accuracy", minimum_character_accuracy),
        ("maximum expected calibration error", maximum_ece),
    ):
        if threshold is not None and not 0 <= threshold <= 1:
            raise DatasetExportError(f"{name} must be in [0, 1]")
    failures: list[str] = []
    metrics = evaluation.overall
    if minimum_exact_accuracy is not None and metrics.exact_accuracy < minimum_exact_accuracy:
        failures.append("exact_accuracy")
    if (
        minimum_character_accuracy is not None
        and metrics.character_accuracy < minimum_character_accuracy
    ):
        failures.append("character_accuracy")
    if maximum_ece is not None and metrics.expected_calibration_error > maximum_ece:
        failures.append("expected_calibration_error")
    return failures


def _metrics(records: list[dict[str, Any]], bins: int) -> OCRDatasetMetrics:
    exact = 0
    edit_distance = 0
    character_denominator = 0
    calibration: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for record in records:
        prediction = record["prediction"]
        predicted = str(prediction["normalized"])
        label = str(record["label"])
        correct = int(predicted == label)
        exact += correct
        edit_distance += _levenshtein(predicted, label)
        character_denominator += max(len(predicted), len(label), 1)
        confidence = float(prediction["confidence"])
        index = min(bins - 1, int(confidence * bins))
        calibration[index].append((confidence, correct))
    total = len(records)
    ece = 0.0
    for bucket in calibration:
        if not bucket:
            continue
        average_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += len(bucket) / total * abs(accuracy - average_confidence)
    return OCRDatasetMetrics(
        sample_count=total,
        exact_accuracy=exact / total,
        character_accuracy=max(0.0, 1 - edit_distance / character_denominator),
        expected_calibration_error=ece,
    )


def _validate_record(record: dict[str, Any]) -> None:
    prediction = record.get("prediction")
    if (
        not isinstance(prediction, dict)
        or not isinstance(prediction.get("normalized"), str)
        or not isinstance(record.get("label"), str)
        or not isinstance(record.get("split"), str)
    ):
        raise DatasetExportError("OCR evaluation record is invalid")
    confidence = prediction.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise DatasetExportError("OCR evaluation confidence is invalid")


def _metrics_json(metrics: OCRDatasetMetrics) -> dict[str, int | float]:
    return {
        "sampleCount": metrics.sample_count,
        "exactAccuracy": metrics.exact_accuracy,
        "characterAccuracy": metrics.character_accuracy,
        "expectedCalibrationError": metrics.expected_calibration_error,
    }


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
