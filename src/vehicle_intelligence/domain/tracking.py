from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.domain.detection import ModelMetadata, PlateDetection, TrackedDetection
from vehicle_intelligence.domain.enums import Direction, TrackStatus
from vehicle_intelligence.domain.geometry import BoundingBox, Point
from vehicle_intelligence.domain.plate import PlateObservation


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    frame_id: int
    timestamp: datetime
    center: Point


@dataclass(slots=True)
class ImageCandidate:
    frame_id: int
    timestamp: datetime
    score: float
    image: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 1:
            raise ValueError("image candidate score must be in [0, 1]")


@dataclass(slots=True)
class VehicleTrack:
    camera_id: str
    session_id: str
    local_track_id: int
    first_seen: datetime
    last_seen: datetime
    max_trajectory_points: int
    max_plate_observations: int
    status: TrackStatus = TrackStatus.ACTIVE
    direction: Direction = Direction.UNKNOWN
    frames_seen: int = 0
    plate_detections_seen: int = 0
    last_ocr_attempt_frame_seen: int | None = None
    latest_bbox: BoundingBox | None = None
    vehicle_type_scores: dict[str, float] = field(default_factory=dict)
    vehicle_confidence_sum: float = 0.0
    vehicle_model: ModelMetadata | None = None
    plate_model: ModelMetadata | None = None
    ocr_model: ModelMetadata | None = None
    best_snapshot: ImageCandidate | None = None
    best_vehicle_crop: ImageCandidate | None = None
    best_plate_crop: ImageCandidate | None = None
    trajectory: deque[TrajectoryPoint] = field(init=False)
    plate_observations: deque[PlateObservation] = field(init=False)

    def __post_init__(self) -> None:
        if self.first_seen.tzinfo is None or self.last_seen.tzinfo is None:
            raise ValueError("track timestamps must be timezone-aware")
        if self.max_trajectory_points < 2 or self.max_plate_observations < 1:
            raise ValueError("track history limits are invalid")
        self.trajectory = deque(maxlen=self.max_trajectory_points)
        self.plate_observations = deque(maxlen=self.max_plate_observations)

    @property
    def logical_id(self) -> str:
        return f"{self.camera_id}:{self.session_id}:{self.local_track_id}"

    @property
    def vehicle_type(self) -> str:
        if not self.vehicle_type_scores:
            return "unknown"
        return max(self.vehicle_type_scores, key=self.vehicle_type_scores.__getitem__)

    @property
    def vehicle_confidence(self) -> float:
        return self.vehicle_confidence_sum / self.frames_seen if self.frames_seen else 0.0

    def update(self, tracked: TrackedDetection, frame_id: int, timestamp: datetime) -> None:
        if self.status is TrackStatus.FINALIZED:
            raise ValueError("cannot update a finalized track")
        detection = tracked.detection
        self.frames_seen += 1
        self.last_seen = timestamp
        self.latest_bbox = detection.bbox
        self.vehicle_confidence_sum += detection.confidence
        self.vehicle_type_scores[detection.class_name] = (
            self.vehicle_type_scores.get(detection.class_name, 0.0) + detection.confidence
        )
        self.vehicle_model = detection.model
        self.trajectory.append(TrajectoryPoint(frame_id, timestamp, detection.bbox.center))

    def update_plate_track(
        self,
        detection: PlateDetection,
        tracked_bbox: BoundingBox,
        frame_id: int,
        timestamp: datetime,
    ) -> None:
        """Update a plate-only track without fabricating vehicle evidence."""

        if self.status is TrackStatus.FINALIZED:
            raise ValueError("cannot update a finalized track")
        self.frames_seen += 1
        self.plate_detections_seen += 1
        self.last_seen = timestamp
        self.latest_bbox = tracked_bbox
        self.plate_model = detection.model
        self.trajectory.append(TrajectoryPoint(frame_id, timestamp, tracked_bbox.center))

    def add_plate_observation(self, observation: PlateObservation) -> None:
        if self.status is TrackStatus.FINALIZED:
            raise ValueError("cannot update a finalized track")
        self.plate_observations.append(observation)
        self.plate_model = observation.plate_model
        self.ocr_model = observation.ocr_model

    def mark_finalized(self) -> None:
        self.status = TrackStatus.FINALIZED
