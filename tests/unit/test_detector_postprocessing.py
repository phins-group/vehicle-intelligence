import numpy as np

from vehicle_intelligence.infrastructure.vision.postprocessing import nms_indices


def test_class_aware_nms_suppresses_per_class_and_preserves_global_score_order() -> None:
    boxes = np.asarray(
        [
            [0, 0, 10, 10],
            [1, 1, 9, 9],
            [0, 0, 10, 10],
            [20, 20, 20, 25],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.8, 0.9, 0.85, 0.7], dtype=np.float32)
    class_ids = np.asarray([0, 0, 1, 0], dtype=np.int64)

    retained = nms_indices(boxes, scores, class_ids, 0.5, class_agnostic=False)

    assert retained.tolist() == [1, 2, 3]


def test_class_agnostic_nms_suppresses_overlaps_across_classes() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.asarray([0.8, 0.9], dtype=np.float32)
    class_ids = np.asarray([0, 1], dtype=np.int64)

    retained = nms_indices(boxes, scores, class_ids, 0.5, class_agnostic=True)

    assert retained.tolist() == [1]


def test_nms_keeps_input_order_for_equal_scores() -> None:
    boxes = np.asarray([[0, 0, 5, 5], [10, 10, 15, 15]], dtype=np.float32)
    scores = np.asarray([0.8, 0.8], dtype=np.float32)
    class_ids = np.asarray([0, 0], dtype=np.int64)

    retained = nms_indices(boxes, scores, class_ids, 0.5, class_agnostic=False)

    assert retained.tolist() == [0, 1]
