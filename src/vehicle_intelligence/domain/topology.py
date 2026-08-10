from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CameraTopologyEdge:
    id: str
    from_camera_id: str
    to_camera_id: str
    minimum_travel_seconds: float
    maximum_travel_seconds: float
    typical_travel_seconds: float
    enabled: bool
    created_at: datetime
    updated_at: datetime
    revision: int = 1
    schema_version: int = 1
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifiers = (self.id, self.from_camera_id, self.to_camera_id)
        if any(not value.strip() for value in identifiers):
            raise ValueError("topology identifiers are required")
        if self.from_camera_id == self.to_camera_id:
            raise ValueError("topology edge cannot connect a camera to itself")
        if self.minimum_travel_seconds < 0:
            raise ValueError("minimum travel time cannot be negative")
        if self.maximum_travel_seconds <= self.minimum_travel_seconds:
            raise ValueError("maximum travel time must exceed minimum travel time")
        if not (
            self.minimum_travel_seconds
            <= self.typical_travel_seconds
            <= self.maximum_travel_seconds
        ):
            raise ValueError("typical travel time must be inside the configured window")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("topology timestamps must be timezone-aware")
        if self.updated_at < self.created_at:
            raise ValueError("topology lifecycle is inverted")
        if self.revision < 1 or self.schema_version < 1:
            raise ValueError("topology versions must be positive")


@dataclass(frozen=True, slots=True)
class CrossCameraCandidate:
    fingerprint_id: str
    vehicle_id: str
    camera_id: str
    observed_at: datetime
    topology_edge_id: str
    travel_seconds: float
    time_score: float

    def __post_init__(self) -> None:
        identifiers = (
            self.fingerprint_id,
            self.vehicle_id,
            self.camera_id,
            self.topology_edge_id,
        )
        if any(not value.strip() for value in identifiers):
            raise ValueError("cross-camera candidate identifiers are required")
        if self.observed_at.tzinfo is None or self.travel_seconds < 0:
            raise ValueError("cross-camera candidate timing is invalid")
        if not 0 <= self.time_score <= 1:
            raise ValueError("candidate time score must be in [0, 1]")
