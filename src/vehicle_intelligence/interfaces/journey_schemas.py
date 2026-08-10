"""Logical vehicle timeline and journey response contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from vehicle_intelligence.domain import JourneyObservation, JourneySegment, VehicleJourney


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class JourneyObservationPublic(APIModel):
    event_id: str = Field(alias="eventId")
    camera_id: str = Field(alias="cameraId")
    camera_name: str = Field(alias="cameraName")
    zone: str | None
    occurred_at: datetime = Field(alias="occurredAt")
    event_type: str = Field(alias="eventType")
    direction: str
    status: str
    plate: str | None
    vehicle_type: str = Field(alias="vehicleType")

    @classmethod
    def from_domain(cls, value: JourneyObservation) -> JourneyObservationPublic:
        return cls(
            eventId=value.event_id,
            cameraId=value.camera_id,
            cameraName=value.camera_name,
            zone=value.zone,
            occurredAt=value.occurred_at,
            eventType=value.event_type.value,
            direction=value.direction.value,
            status=value.status.value,
            plate=value.plate,
            vehicleType=value.vehicle_type,
        )


class JourneySegmentPublic(APIModel):
    from_event_id: str = Field(alias="fromEventId")
    to_event_id: str = Field(alias="toEventId")
    from_camera_id: str = Field(alias="fromCameraId")
    to_camera_id: str = Field(alias="toCameraId")
    departed_at: datetime = Field(alias="departedAt")
    arrived_at: datetime = Field(alias="arrivedAt")
    elapsed_seconds: float = Field(alias="elapsedSeconds")
    topology_edge_id: str | None = Field(alias="topologyEdgeId")
    expected_minimum_seconds: float | None = Field(alias="expectedMinimumSeconds")
    expected_maximum_seconds: float | None = Field(alias="expectedMaximumSeconds")
    feasible: bool | None

    @classmethod
    def from_domain(cls, value: JourneySegment) -> JourneySegmentPublic:
        return cls(
            fromEventId=value.from_event_id,
            toEventId=value.to_event_id,
            fromCameraId=value.from_camera_id,
            toCameraId=value.to_camera_id,
            departedAt=value.departed_at,
            arrivedAt=value.arrived_at,
            elapsedSeconds=value.elapsed_seconds,
            topologyEdgeId=value.topology_edge_id,
            expectedMinimumSeconds=value.expected_minimum_seconds,
            expectedMaximumSeconds=value.expected_maximum_seconds,
            feasible=value.feasible,
        )


class VehicleTimelinePublic(APIModel):
    vehicle_id: str = Field(alias="vehicleId")
    items: list[JourneyObservationPublic]


class VehicleJourneyPublic(APIModel):
    vehicle_id: str = Field(alias="vehicleId")
    observations: list[JourneyObservationPublic]
    segments: list[JourneySegmentPublic]
    started_at: datetime | None = Field(alias="startedAt")
    ended_at: datetime | None = Field(alias="endedAt")
    truncated: bool

    @classmethod
    def from_domain(cls, value: VehicleJourney) -> VehicleJourneyPublic:
        return cls(
            vehicleId=value.vehicle_id,
            observations=[
                JourneyObservationPublic.from_domain(item)
                for item in value.observations
            ],
            segments=[JourneySegmentPublic.from_domain(item) for item in value.segments],
            startedAt=value.started_at,
            endedAt=value.ended_at,
            truncated=value.truncated,
        )
