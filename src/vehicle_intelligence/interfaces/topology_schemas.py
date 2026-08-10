"""Strict camera-topology and cross-camera candidate API contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from vehicle_intelligence.application.topology import TopologyCreate, TopologyUpdate
from vehicle_intelligence.domain import CameraTopologyEdge, CrossCameraCandidate


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class TravelTimeInput(APIModel):
    minimum_seconds: float = Field(alias="minimumSeconds", ge=0)
    maximum_seconds: float = Field(alias="maximumSeconds", gt=0)
    typical_seconds: float = Field(alias="typicalSeconds", ge=0)


class TopologyCreateRequest(APIModel):
    id: str = Field(min_length=1, max_length=128)
    from_camera_id: str = Field(alias="fromCameraId", min_length=1, max_length=128)
    to_camera_id: str = Field(alias="toCameraId", min_length=1, max_length=128)
    travel_time: TravelTimeInput = Field(alias="travelTime")
    enabled: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_command(self) -> TopologyCreate:
        return TopologyCreate(
            id=self.id,
            from_camera_id=self.from_camera_id,
            to_camera_id=self.to_camera_id,
            minimum_travel_seconds=self.travel_time.minimum_seconds,
            maximum_travel_seconds=self.travel_time.maximum_seconds,
            typical_travel_seconds=self.travel_time.typical_seconds,
            enabled=self.enabled,
            metadata=self.metadata,
        )


class TopologyUpdateRequest(APIModel):
    revision: int = Field(ge=1)
    from_camera_id: str = Field(alias="fromCameraId", min_length=1, max_length=128)
    to_camera_id: str = Field(alias="toCameraId", min_length=1, max_length=128)
    travel_time: TravelTimeInput = Field(alias="travelTime")
    enabled: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_command(self) -> TopologyUpdate:
        return TopologyUpdate(
            revision=self.revision,
            from_camera_id=self.from_camera_id,
            to_camera_id=self.to_camera_id,
            minimum_travel_seconds=self.travel_time.minimum_seconds,
            maximum_travel_seconds=self.travel_time.maximum_seconds,
            typical_travel_seconds=self.travel_time.typical_seconds,
            enabled=self.enabled,
            metadata=self.metadata,
        )


class TravelTimePublic(APIModel):
    minimum_seconds: float = Field(alias="minimumSeconds")
    maximum_seconds: float = Field(alias="maximumSeconds")
    typical_seconds: float = Field(alias="typicalSeconds")


class TopologyPublic(APIModel):
    id: str
    schema_version: int = Field(alias="schemaVersion")
    revision: int
    from_camera_id: str = Field(alias="fromCameraId")
    to_camera_id: str = Field(alias="toCameraId")
    travel_time: TravelTimePublic = Field(alias="travelTime")
    enabled: bool
    metadata: dict[str, object]
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, edge: CameraTopologyEdge) -> TopologyPublic:
        return cls(
            id=edge.id,
            schemaVersion=edge.schema_version,
            revision=edge.revision,
            fromCameraId=edge.from_camera_id,
            toCameraId=edge.to_camera_id,
            travelTime=TravelTimePublic(
                minimumSeconds=edge.minimum_travel_seconds,
                maximumSeconds=edge.maximum_travel_seconds,
                typicalSeconds=edge.typical_travel_seconds,
            ),
            enabled=edge.enabled,
            metadata=edge.metadata,
            createdAt=edge.created_at,
            updatedAt=edge.updated_at,
        )


class TopologyListPublic(APIModel):
    items: list[TopologyPublic]


class CandidatePublic(APIModel):
    fingerprint_id: str = Field(alias="fingerprintId")
    vehicle_id: str = Field(alias="vehicleId")
    camera_id: str = Field(alias="cameraId")
    observed_at: datetime = Field(alias="observedAt")
    topology_edge_id: str = Field(alias="topologyEdgeId")
    travel_seconds: float = Field(alias="travelSeconds")
    time_score: float = Field(alias="timeScore")

    @classmethod
    def from_domain(cls, candidate: CrossCameraCandidate) -> CandidatePublic:
        return cls(
            fingerprintId=candidate.fingerprint_id,
            vehicleId=candidate.vehicle_id,
            cameraId=candidate.camera_id,
            observedAt=candidate.observed_at,
            topologyEdgeId=candidate.topology_edge_id,
            travelSeconds=candidate.travel_seconds,
            timeScore=candidate.time_score,
        )


class CandidateListPublic(APIModel):
    source_fingerprint_id: str = Field(alias="sourceFingerprintId")
    items: list[CandidatePublic]
