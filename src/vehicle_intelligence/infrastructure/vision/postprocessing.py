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
        current_area = max(0, boxes[current, 2] - boxes[current, 0]) * max(
            0, boxes[current, 3] - boxes[current, 1]
        )
        remaining_area = np.maximum(0, boxes[remaining, 2] - boxes[remaining, 0]) * np.maximum(
            0, boxes[remaining, 3] - boxes[remaining, 1]
        )
        union = current_area + remaining_area - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        suppress = iou > threshold
        if not class_agnostic:
            suppress &= class_ids[remaining] == class_ids[current]
        order = remaining[~suppress]
    return np.asarray(keep, dtype=np.int64)
