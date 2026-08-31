"""Framework-neutral detector post-processing primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def nms_indices(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    class_ids: NDArray[np.int64],
    threshold: float,
    *,
    class_agnostic: bool,
) -> NDArray[np.int64]:
    """Return stable score-ordered indices retained by IoU NMS."""

    order = np.argsort(-scores, kind="stable")
    if order.size < 2:
        return order.astype(np.int64, copy=False)

    widths = np.maximum(0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0, boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    if class_agnostic:
        return _suppress_overlaps(boxes, areas, order, threshold)

    retained = np.zeros(scores.shape[0], dtype=np.bool_)
    ordered_class_ids = class_ids[order]
    for class_id in np.unique(ordered_class_ids):
        class_order = order[ordered_class_ids == class_id]
        retained[_suppress_overlaps(boxes, areas, class_order, threshold)] = True
    return order[retained[order]].astype(np.int64, copy=False)


def _suppress_overlaps(
    boxes: NDArray[np.float32],
    areas: NDArray[np.float32],
    order: NDArray[np.int64],
    threshold: float,
) -> NDArray[np.int64]:
    """Apply NMS to one score-ordered class without recomputing box areas."""

    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection_left = np.maximum(boxes[current, 0], boxes[remaining, 0])
        intersection_top = np.maximum(boxes[current, 1], boxes[remaining, 1])
        intersection_right = np.minimum(boxes[current, 2], boxes[remaining, 2])
        intersection_bottom = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.maximum(0, intersection_right - intersection_left) * np.maximum(
            0, intersection_bottom - intersection_top
        )
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= threshold]
    return np.asarray(keep, dtype=np.int64)
