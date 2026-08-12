"""Provider-neutral detector metrics and explicit production release gates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vehicle_intelligence.exceptions import ModelEvaluationError
from vehicle_intelligence.training.config import DetectorReleaseGates


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    ground_truths: int
    predictions: int
    true_positives: int
    false_positives: int
    precision: float
    recall: float
    ap50: float
    map50_95: float


@dataclass(frozen=True, slots=True)
class SliceMetrics:
    ground_truths: int
    matched: int
    recall: float


@dataclass(frozen=True, slots=True)
class DetectorEvaluation:
    ground_truths: int
    predictions: int
    true_positives: int
    false_positives: int
    precision: float
    recall: float
    map50: float
    map50_95: float
    group_count: int
    matched_group_count: int
    group_recall: float
    full_bbox_coverage: float
    by_class: dict[str, ClassMetrics]
    by_camera: dict[str, SliceMetrics]
    by_lighting: dict[str, SliceMetrics]
    iou_threshold: float
    confidence_threshold: float


@dataclass(frozen=True, slots=True)
class _GroundTruth:
    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _Prediction:
    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True, slots=True)
class _MatchResult:
    true_positive_prediction_ids: frozenset[int]
    matched_ground_truth_ids: frozenset[int]
    prediction_to_ground_truth: dict[int, int]
    false_positives: int


def evaluate_detector_files(
    annotations_path: Path,
    predictions_path: Path,
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
    full_bbox_coverage_threshold: float = 0.95,
) -> DetectorEvaluation:
    ground_truth = _read_json(annotations_path, expected=dict, description="COCO annotations")
    predictions = _read_json(predictions_path, expected=list, description="detector predictions")
    return evaluate_detector(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
        full_bbox_coverage_threshold=full_bbox_coverage_threshold,
    )


def evaluate_detector(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.25,
    full_bbox_coverage_threshold: float = 0.95,
) -> DetectorEvaluation:
    for name, value in (
        ("iou_threshold", iou_threshold),
        ("confidence_threshold", confidence_threshold),
        ("full_bbox_coverage_threshold", full_bbox_coverage_threshold),
    ):
        if not 0 <= value <= 1:
            raise ModelEvaluationError(f"{name} must be in [0, 1]")
    categories = _categories(ground_truth)
    images = _images(ground_truth)
    truths = _ground_truths(ground_truth, images, categories)
    parsed_predictions = _predictions(predictions, images, categories)
    operational_predictions = [
        prediction for prediction in parsed_predictions if prediction.score >= confidence_threshold
    ]
    match = _match(truths, operational_predictions, iou_threshold)
    true_positives = len(match.true_positive_prediction_ids)
    precision = _ratio(true_positives, len(operational_predictions))
    recall = _ratio(len(match.matched_ground_truth_ids), len(truths))

    by_class: dict[str, ClassMetrics] = {}
    ap50_values: list[float] = []
    map_values: list[float] = []
    for category_id, class_name in sorted(categories.items()):
        class_truths = [truth for truth in truths if truth.category_id == category_id]
        class_predictions = [
            prediction
            for prediction in operational_predictions
            if prediction.category_id == category_id
        ]
        class_match = _match(class_truths, class_predictions, iou_threshold)
        class_tp = len(class_match.true_positive_prediction_ids)
        ap50 = _average_precision(class_truths, parsed_predictions, category_id, 0.5)
        class_map = _mean(
            _average_precision(class_truths, parsed_predictions, category_id, threshold)
            for threshold in _iou_thresholds()
        )
        if class_truths:
            ap50_values.append(ap50)
            map_values.append(class_map)
        by_class[class_name] = ClassMetrics(
            ground_truths=len(class_truths),
            predictions=len(class_predictions),
            true_positives=class_tp,
            false_positives=class_match.false_positives,
            precision=_ratio(class_tp, len(class_predictions)),
            recall=_ratio(len(class_match.matched_ground_truth_ids), len(class_truths)),
            ap50=ap50,
            map50_95=class_map,
        )

    truth_by_id = {truth.id: truth for truth in truths}
    prediction_by_id = {prediction.id: prediction for prediction in operational_predictions}
    covered = 0
    for prediction_id, truth_id in match.prediction_to_ground_truth.items():
        prediction = prediction_by_id[prediction_id]
        truth = truth_by_id[truth_id]
        if _ground_truth_coverage(prediction.bbox, truth.bbox) >= full_bbox_coverage_threshold:
            covered += 1

    matched_ids = match.matched_ground_truth_ids
    by_camera = _slice_metrics(
        truths,
        matched_ids,
        images,
        lambda image: str(image.get("camera_id") or "unknown"),
    )
    by_lighting = _slice_metrics(truths, matched_ids, images, _lighting)
    groups = {
        str(images[truth.image_id].get("group_id"))
        for truth in truths
        if images[truth.image_id].get("group_id")
    }
    matched_groups = {
        str(images[truth_by_id[truth_id].image_id].get("group_id"))
        for truth_id in matched_ids
        if images[truth_by_id[truth_id].image_id].get("group_id")
    }
    return DetectorEvaluation(
        ground_truths=len(truths),
        predictions=len(operational_predictions),
        true_positives=true_positives,
        false_positives=match.false_positives,
        precision=precision,
        recall=recall,
        map50=_mean(ap50_values),
        map50_95=_mean(map_values),
        group_count=len(groups),
        matched_group_count=len(matched_groups),
        group_recall=_ratio(len(matched_groups), len(groups)),
        full_bbox_coverage=_ratio(covered, len(truths)),
        by_class=by_class,
        by_camera=by_camera,
        by_lighting=by_lighting,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold,
    )


def detector_release_gate_failures(
    evaluation: DetectorEvaluation,
    gates: DetectorReleaseGates,
) -> list[str]:
    failures: list[str] = []
    checks = (
        ("map50", evaluation.map50, gates.minimum_map50),
        ("map50_95", evaluation.map50_95, gates.minimum_map50_95),
        ("precision", evaluation.precision, gates.minimum_precision),
        ("recall", evaluation.recall, gates.minimum_recall),
        ("group_recall", evaluation.group_recall, gates.minimum_group_recall),
        (
            "full_bbox_coverage",
            evaluation.full_bbox_coverage,
            gates.minimum_full_bbox_coverage,
        ),
    )
    for name, actual, minimum in checks:
        if minimum is not None and actual < minimum:
            failures.append(name)
    if gates.minimum_night_recall is not None:
        night = evaluation.by_lighting.get("NIGHT")
        if night is None or night.recall < gates.minimum_night_recall:
            failures.append("night_recall")
    for class_name, minimum in sorted(gates.minimum_per_class_recall.items()):
        metrics = evaluation.by_class.get(class_name)
        if metrics is None or metrics.recall < minimum:
            failures.append(f"class_recall:{class_name}")
    return failures


def evaluation_to_jsonable(evaluation: DetectorEvaluation) -> dict[str, Any]:
    return {
        "groundTruths": evaluation.ground_truths,
        "predictions": evaluation.predictions,
        "truePositives": evaluation.true_positives,
        "falsePositives": evaluation.false_positives,
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "mAP50": evaluation.map50,
        "mAP50_95": evaluation.map50_95,
        "groupCount": evaluation.group_count,
        "matchedGroupCount": evaluation.matched_group_count,
        "groupRecall": evaluation.group_recall,
        "fullBboxCoverage": evaluation.full_bbox_coverage,
        "iouThreshold": evaluation.iou_threshold,
        "confidenceThreshold": evaluation.confidence_threshold,
        "byClass": {
            name: {
                "groundTruths": value.ground_truths,
                "predictions": value.predictions,
                "truePositives": value.true_positives,
                "falsePositives": value.false_positives,
                "precision": value.precision,
                "recall": value.recall,
                "ap50": value.ap50,
                "mAP50_95": value.map50_95,
            }
            for name, value in sorted(evaluation.by_class.items())
        },
        "byCamera": _slices_to_json(evaluation.by_camera),
        "byLighting": _slices_to_json(evaluation.by_lighting),
    }


def _categories(document: dict[str, Any]) -> dict[int, str]:
    raw = document.get("categories")
    if not isinstance(raw, list) or not raw:
        raise ModelEvaluationError("COCO categories are missing")
    categories: dict[int, str] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise ModelEvaluationError("COCO category is invalid")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or item["id"] in categories:
            raise ModelEvaluationError("COCO category id/name is invalid")
        categories[item["id"]] = name.strip().lower()
    return categories


def _images(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = document.get("images")
    if not isinstance(raw, list):
        raise ModelEvaluationError("COCO images are missing")
    images: dict[int, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise ModelEvaluationError("COCO image is invalid")
        if item["id"] in images:
            raise ModelEvaluationError("COCO image ids must be unique")
        images[item["id"]] = item
    return images


def _ground_truths(
    document: dict[str, Any],
    images: dict[int, dict[str, Any]],
    categories: dict[int, str],
) -> list[_GroundTruth]:
    raw = document.get("annotations")
    if not isinstance(raw, list):
        raise ModelEvaluationError("COCO annotations are missing")
    truths: list[_GroundTruth] = []
    seen_ids: set[int] = set()
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise ModelEvaluationError("COCO ground truth is invalid")
        if item["id"] in seen_ids:
            raise ModelEvaluationError("COCO annotation ids must be unique")
        seen_ids.add(item["id"])
        image_id = item.get("image_id")
        category_id = item.get("category_id")
        if image_id not in images or category_id not in categories:
            raise ModelEvaluationError("COCO ground truth reference is invalid")
        truths.append(_GroundTruth(item["id"], image_id, category_id, _bbox(item.get("bbox"))))
    return truths


def _predictions(
    records: list[dict[str, Any]],
    images: dict[int, dict[str, Any]],
    categories: dict[int, str],
) -> list[_Prediction]:
    predictions: list[_Prediction] = []
    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise ModelEvaluationError("detector prediction must be an object")
        image_id = item.get("image_id")
        category_id = item.get("category_id")
        score = item.get("score")
        if image_id not in images or category_id not in categories:
            raise ModelEvaluationError("detector prediction reference is invalid")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ModelEvaluationError("detector prediction score must be in [0, 1]")
        predictions.append(
            _Prediction(index, image_id, category_id, _bbox(item.get("bbox")), float(score))
        )
    return predictions


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ModelEvaluationError("detection bbox must be [x, y, width, height]")
    if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
        raise ModelEvaluationError("detection bbox values must be finite numbers")
    x, y, width, height = (float(item) for item in value)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ModelEvaluationError("detection bbox must have positive in-image area")
    return x, y, width, height


def _match(
    truths: list[_GroundTruth],
    predictions: list[_Prediction],
    threshold: float,
) -> _MatchResult:
    candidates: dict[tuple[int, int], list[_GroundTruth]] = defaultdict(list)
    for truth in truths:
        candidates[(truth.image_id, truth.category_id)].append(truth)
    matched_truths: set[int] = set()
    matched_predictions: set[int] = set()
    mapping: dict[int, int] = {}
    for prediction in sorted(predictions, key=lambda item: (-item.score, item.id)):
        best: _GroundTruth | None = None
        best_iou = threshold
        for truth in candidates[(prediction.image_id, prediction.category_id)]:
            if truth.id in matched_truths:
                continue
            overlap = _iou(prediction.bbox, truth.bbox)
            if overlap >= best_iou:
                best = truth
                best_iou = overlap
        if best is not None:
            matched_truths.add(best.id)
            matched_predictions.add(prediction.id)
            mapping[prediction.id] = best.id
    return _MatchResult(
        frozenset(matched_predictions),
        frozenset(matched_truths),
        mapping,
        len(predictions) - len(matched_predictions),
    )


def _average_precision(
    truths: list[_GroundTruth],
    all_predictions: list[_Prediction],
    category_id: int,
    threshold: float,
) -> float:
    class_truths = [truth for truth in truths if truth.category_id == category_id]
    if not class_truths:
        return 0.0
    class_predictions = sorted(
        (item for item in all_predictions if item.category_id == category_id),
        key=lambda item: (-item.score, item.id),
    )
    matched: set[int] = set()
    truth_by_image: dict[int, list[_GroundTruth]] = defaultdict(list)
    for truth in class_truths:
        truth_by_image[truth.image_id].append(truth)
    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []
    tp = 0
    fp = 0
    for prediction in class_predictions:
        best: _GroundTruth | None = None
        best_iou = threshold
        for truth in truth_by_image[prediction.image_id]:
            if truth.id in matched:
                continue
            overlap = _iou(prediction.bbox, truth.bbox)
            if overlap >= best_iou:
                best = truth
                best_iou = overlap
        if best is None:
            fp += 1
        else:
            tp += 1
            matched.add(best.id)
        cumulative_tp.append(tp)
        cumulative_fp.append(fp)
    if not cumulative_tp:
        return 0.0
    recalls = [value / len(class_truths) for value in cumulative_tp]
    precisions = [
        true_positive / (true_positive + false_positive)
        for true_positive, false_positive in zip(cumulative_tp, cumulative_fp, strict=True)
    ]
    samples = []
    for recall_target in (index / 100 for index in range(101)):
        candidates = [
            precision
            for recall, precision in zip(recalls, precisions, strict=True)
            if recall >= recall_target
        ]
        samples.append(max(candidates, default=0.0))
    return sum(samples) / len(samples)


def _slice_metrics(
    truths: list[_GroundTruth],
    matched_ids: frozenset[int],
    images: dict[int, dict[str, Any]],
    selector: Callable[[dict[str, Any]], str],
) -> dict[str, SliceMetrics]:
    totals: CounterLike = defaultdict(int)
    matched: CounterLike = defaultdict(int)
    for truth in truths:
        key = selector(images[truth.image_id])
        totals[key] += 1
        if truth.id in matched_ids:
            matched[key] += 1
    return {
        key: SliceMetrics(total, matched[key], _ratio(matched[key], total))
        for key, total in sorted(totals.items())
    }


CounterLike = dict[str, int]


def _lighting(image: dict[str, Any]) -> str:
    attributes = image.get("attributes")
    if not isinstance(attributes, dict):
        return "UNKNOWN"
    value = attributes.get("lighting")
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    if attributes.get("isNight") is True or attributes.get("is_night") is True:
        return "NIGHT"
    return "UNKNOWN"


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    intersection = _intersection(left, right)
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return _ratio(intersection, union)


def _ground_truth_coverage(
    prediction: tuple[float, ...],
    truth: tuple[float, ...],
) -> float:
    return _ratio(_intersection(prediction, truth), truth[2] * truth[3])


def _intersection(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
    right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
    width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    return width * height


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _iou_thresholds() -> tuple[float, ...]:
    return tuple(round(0.50 + index * 0.05, 2) for index in range(10))


def _read_json(path: Path, *, expected: type, description: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelEvaluationError(f"{description} cannot be read") from exc
    if not isinstance(value, expected):
        raise ModelEvaluationError(f"{description} has an invalid root type")
    return value


def _slices_to_json(values: dict[str, SliceMetrics]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "groundTruths": value.ground_truths,
            "matched": value.matched,
            "recall": value.recall,
        }
        for key, value in sorted(values.items())
    }
