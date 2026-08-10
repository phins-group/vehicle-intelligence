from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vehicle_intelligence.domain.enums import Direction, EventStatus, EventType


@dataclass(frozen=True, slots=True)
class JourneyObservation:
    event_id: str
    camera_id: str
    camera_name: str
    zone: str | None
    occurred_at: datetime
    event_type: EventType
    direction: Direction
    status: EventStatus
    plate: str | None
    vehicle_type: str

    def __post_init__(self) -> None:
        if any(not value.strip() for value in (self.event_id, self.camera_id, self.camera_name)):
            raise ValueError("journey observation identifiers are required")
        if self.occurred_at.tzinfo is None:
            raise ValueError("journey observation timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class JourneySegment:
    from_event_id: str
    to_event_id: str
    from_camera_id: str
    to_camera_id: str
    departed_at: datetime
    arrived_at: datetime
    elapsed_seconds: float
    topology_edge_id: str | None
    expected_minimum_seconds: float | None
    expected_maximum_seconds: float | None
    feasible: bool | None

    def __post_init__(self) -> None:
        identifiers = (
            self.from_event_id,
            self.to_event_id,
            self.from_camera_id,
            self.to_camera_id,
        )
        if any(not value.strip() for value in identifiers):
            raise ValueError("journey segment identifiers are required")
        if (
            self.departed_at.tzinfo is None
            or self.arrived_at.tzinfo is None
            or self.arrived_at < self.departed_at
            or self.elapsed_seconds < 0
        ):
            raise ValueError("journey segment timing is invalid")
        expected = (self.expected_minimum_seconds, self.expected_maximum_seconds)
        if self.topology_edge_id is None and any(value is not None for value in expected):
            raise ValueError("journey expected times require a topology edge")


@dataclass(frozen=True, slots=True)
class VehicleJourney:
    vehicle_id: str
    observations: tuple[JourneyObservation, ...]
    segments: tuple[JourneySegment, ...]
    started_at: datetime | None
    ended_at: datetime | None
    truncated: bool

    def __post_init__(self) -> None:
        if not self.vehicle_id.strip():
            raise ValueError("journey vehicle ID is required")
        if len(self.segments) != max(0, len(self.observations) - 1):
            raise ValueError("journey segments must connect consecutive observations")
        if (self.started_at is None) != (self.ended_at is None):
            raise ValueError("journey time range must be complete")
        if self.started_at is not None and self.ended_at < self.started_at:
            raise ValueError("journey time range is inverted")
