"""Detector-agnostic ByteTrack adapter."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.config import TrackingConfig
from vehicle_intelligence.domain import BoundingBox, Detection, TrackedDetection
from vehicle_intelligence.exceptions import DependencyUnavailableError, InferenceError


class ByteTrackVehicleTracker:
    def __init__(self, config: TrackingConfig, frame_rate: float) -> None:
        try:
            import supervision as sv
        except ImportError as exc:
            raise DependencyUnavailableError(
                "ByteTrack dependencies are unavailable; install the 'vision' extra"
            ) from exc
        self._sv = sv
        try:
            self._tracker = sv.ByteTrack(
                lost_track_buffer=config.lost_track_buffer,
                frame_rate=frame_rate,
                track_activation_threshold=config.activation_threshold,
                minimum_consecutive_frames=config.minimum_consecutive_frames,
                minimum_matching_threshold=config.minimum_matching_threshold,
            )
        except TypeError as exc:
            raise DependencyUnavailableError(
                "installed Supervision package has an incompatible ByteTrack API"
            ) from exc
        self._emit_first_observation = config.minimum_consecutive_frames == 1

    def update(
        self, detections: Sequence[Detection], image: NDArray[np.uint8]
    ) -> list[TrackedDetection]:
        del image  # ByteTrack is motion/box based and intentionally ignores pixels.
        if detections:
            xyxy = np.asarray([item.bbox.as_xyxy() for item in detections], dtype=np.float32)
            confidence = np.asarray([item.confidence for item in detections], dtype=np.float32)
            class_id = np.asarray([item.class_id for item in detections], dtype=int)
            domain_index = np.arange(len(detections), dtype=int)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_id = np.empty((0,), dtype=int)
            domain_index = np.empty((0,), dtype=int)
        sv_detections = self._sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data={"domain_index": domain_index},
        )
        try:
            tracked = self._tracker.update_with_detections(sv_detections)
        except Exception as exc:
            raise InferenceError("ByteTrack update failed") from exc
        if tracked.tracker_id is None:
            tracked_ids: NDArray[np.int_] = np.empty((0,), dtype=int)
        else:
            tracked_ids = tracked.tracker_id
        indices = tracked.data.get("domain_index")
        output: list[TrackedDetection] = []
        used_detection_indices: set[int] = set()
        for row, tracker_id in enumerate(tracked_ids):
            if tracker_id is None or int(tracker_id) < 0:
                continue
            detection_index = self._resolve_detection_index(
                row,
                indices,
                tracked.xyxy[row],
                detections,
            )
            if detection_index is None:
                continue
            used_detection_indices.add(detection_index)
            observed = self._tracked_detection(
                int(tracker_id),
                tracked.xyxy[row],
                detections[detection_index],
            )
            if observed is not None:
                output.append(observed)
        if self._emit_first_observation:
            output.extend(
                self._unconfirmed_current_observations(
                    detections,
                    used_detection_indices,
                )
            )
        return output

    @staticmethod
    def _resolve_detection_index(
        row: int,
        indices: object,
        box: NDArray[np.float32],
        detections: Sequence[Detection],
    ) -> int | None:
        if isinstance(indices, np.ndarray) and row < len(indices):
            index = int(indices[row])
            if 0 <= index < len(detections):
                return index
        if not detections:
            return None
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return min(
            range(len(detections)),
            key=lambda index: (
                (detections[index].bbox.center.x - center_x) ** 2
                + (detections[index].bbox.center.y - center_y) ** 2
            ),
        )

    def _unconfirmed_current_observations(
        self,
        detections: Sequence[Detection],
        used_detection_indices: set[int],
    ) -> list[TrackedDetection]:
        current_frame = getattr(self._tracker, "frame_id", None)
        candidates = [
            track
            for track in getattr(self._tracker, "tracked_tracks", ())
            if not getattr(track, "is_activated", True)
            and getattr(track, "frame_id", None) == current_frame
            and int(getattr(track, "external_track_id", -1)) >= 0
        ]
        output: list[TrackedDetection] = []
        for track in candidates:
            values = np.asarray(track.tlbr, dtype=np.float32)
            available = [
                index for index in range(len(detections)) if index not in used_detection_indices
            ]
            if not available:
                break
            detection_index = max(
                available,
                key=lambda index: self._iou(detections[index].bbox, values),
            )
            if self._iou(detections[detection_index].bbox, values) < 0.5:
                continue
            observed = self._tracked_detection(
                int(track.external_track_id),
                values,
                detections[detection_index],
            )
            if observed is not None:
                output.append(observed)
                used_detection_indices.add(detection_index)
        return output

    @staticmethod
    def _tracked_detection(
        track_id: int,
        values: NDArray[np.float32],
        original: Detection,
    ) -> TrackedDetection | None:
        left, top = math.floor(float(values[0])), math.floor(float(values[1]))
        right, bottom = math.ceil(float(values[2])), math.ceil(float(values[3]))
        if right <= left or bottom <= top:
            return None
        updated = Detection(
            bbox=BoundingBox(left, top, right, bottom),
            confidence=original.confidence,
            class_id=original.class_id,
            class_name=original.class_name,
            model=original.model,
        )
        return TrackedDetection(track_id=track_id, detection=updated)

    @staticmethod
    def _iou(box: BoundingBox, values: NDArray[np.float32]) -> float:
        left = max(float(box.x1), float(values[0]))
        top = max(float(box.y1), float(values[1]))
        right = min(float(box.x2), float(values[2]))
        bottom = min(float(box.y2), float(values[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        tracked_area = max(0.0, float(values[2] - values[0])) * max(
            0.0,
            float(values[3] - values[1]),
        )
        union = box.area + tracked_area - intersection
        return intersection / union if union > 0 else 0.0

    def reset(self) -> None:
        self._tracker.reset()
