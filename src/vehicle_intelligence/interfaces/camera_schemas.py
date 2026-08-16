"""Credential-safe FastAPI camera request/response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, SecretStr, field_validator

from vehicle_intelligence.application.cameras import (
    CameraBatchResult,
    CameraCreate,
    CameraUpdate,
)
from vehicle_intelligence.application.ports import CameraConnectionTestResult
from vehicle_intelligence.domain import (
    Camera,
    CameraDirection,
    CameraHealth,
    CameraStatus,
    Direction,
    OnvifDiscoveredDevice,
    Point,
    SecretUri,
)


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CameraStreamCreate(APIModel):
    rtsp_url: SecretStr = Field(alias="rtspUrl")
    fps_limit: float = Field(default=6.0, alias="fpsLimit", gt=0)

    @field_validator("rtsp_url")
    @classmethod
    def validate_rtsp_url(cls, value: SecretStr) -> SecretStr:
        SecretUri(value.get_secret_value())
        return value


class CameraStreamUpdate(APIModel):
    fps_limit: float = Field(alias="fpsLimit", gt=0)
    rtsp_url: SecretStr | None = Field(default=None, alias="rtspUrl")

    @field_validator("rtsp_url")
    @classmethod
    def validate_optional_rtsp_url(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None:
            SecretUri(value.get_secret_value())
        return value


class CameraLocationInput(APIModel):
    name: str | None = None
    zone: str | None = None


class CameraVisionInput(APIModel):
    vehicle_confidence: float = Field(default=0.4, alias="vehicleConfidence", ge=0, le=1)
    plate_confidence: float = Field(default=0.45, alias="plateConfidence", ge=0, le=1)


class CameraGeometryInput(APIModel):
    vehicle_roi: list[tuple[FiniteFloat, FiniteFloat]] | None = Field(
        default=None,
        alias="vehicleRoi",
    )
    crossing_line: (
        tuple[tuple[FiniteFloat, FiniteFloat], tuple[FiniteFloat, FiniteFloat]] | None
    ) = Field(default=None, alias="crossingLine")
    crossing_positive_to_negative: Direction = Field(
        default=Direction.ENTER,
        alias="crossingPositiveToNegative",
    )
    finalize_on_crossing: bool = Field(default=False, alias="finalizeOnCrossing")

    @field_validator("vehicle_roi")
    @classmethod
    def validate_roi(
        cls,
        value: list[tuple[FiniteFloat, FiniteFloat]] | None,
    ) -> list[tuple[FiniteFloat, FiniteFloat]] | None:
        if value is not None and len(value) < 3:
            raise ValueError("vehicleRoi requires at least three points")
        return value

    @field_validator("crossing_positive_to_negative")
    @classmethod
    def validate_crossing_direction(cls, value: Direction) -> Direction:
        if value is Direction.UNKNOWN:
            raise ValueError("crossing direction must be ENTER or EXIT")
        return value


class CameraCreateRequest(APIModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    stream: CameraStreamCreate
    location: CameraLocationInput = Field(default_factory=CameraLocationInput)
    direction: CameraDirection = CameraDirection.BOTH
    vision: CameraVisionInput = Field(default_factory=CameraVisionInput)
    geometry: CameraGeometryInput = Field(default_factory=CameraGeometryInput)
    enabled: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_command(self) -> CameraCreate:
        return CameraCreate(
            id=self.id,
            name=self.name,
            rtsp_url=self.stream.rtsp_url.get_secret_value(),
            fps_limit=self.stream.fps_limit,
            direction=self.direction,
            enabled=self.enabled,
            vehicle_confidence=self.vision.vehicle_confidence,
            plate_confidence=self.vision.plate_confidence,
            location=self.location.name,
            zone=self.location.zone,
            roi=(
                tuple(Point(x, y) for x, y in self.geometry.vehicle_roi)
                if self.geometry.vehicle_roi is not None
                else None
            ),
            crossing_line=(
                tuple(Point(x, y) for x, y in self.geometry.crossing_line)
                if self.geometry.crossing_line is not None
                else None
            ),
            crossing_positive_to_negative=self.geometry.crossing_positive_to_negative,
            finalize_on_crossing=self.geometry.finalize_on_crossing,
            metadata=self.metadata,
        )


class CameraUpdateRequest(APIModel):
    revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=256)
    stream: CameraStreamUpdate
    location: CameraLocationInput = Field(default_factory=CameraLocationInput)
    direction: CameraDirection = CameraDirection.BOTH
    vision: CameraVisionInput = Field(default_factory=CameraVisionInput)
    geometry: CameraGeometryInput = Field(default_factory=CameraGeometryInput)
    enabled: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    def to_command(self) -> CameraUpdate:
        return CameraUpdate(
            revision=self.revision,
            name=self.name,
            rtsp_url=(
                self.stream.rtsp_url.get_secret_value()
                if self.stream.rtsp_url is not None
                else None
            ),
            fps_limit=self.stream.fps_limit,
            direction=self.direction,
            enabled=self.enabled,
            vehicle_confidence=self.vision.vehicle_confidence,
            plate_confidence=self.vision.plate_confidence,
            location=self.location.name,
            zone=self.location.zone,
            roi=(
                tuple(Point(x, y) for x, y in self.geometry.vehicle_roi)
                if self.geometry.vehicle_roi is not None
                else None
            ),
            crossing_line=(
                tuple(Point(x, y) for x, y in self.geometry.crossing_line)
                if self.geometry.crossing_line is not None
                else None
            ),
            crossing_positive_to_negative=self.geometry.crossing_positive_to_negative,
            finalize_on_crossing=self.geometry.finalize_on_crossing,
            metadata=self.metadata,
        )


class CameraPublicStream(APIModel):
    fps_limit: float = Field(alias="fpsLimit")
    credentials_configured: bool = Field(alias="credentialsConfigured")


class CameraPublicGeometry(APIModel):
    vehicle_roi: list[tuple[float, float]] | None = Field(alias="vehicleRoi")
    crossing_line: list[tuple[float, float]] | None = Field(alias="crossingLine")
    crossing_positive_to_negative: Direction = Field(alias="crossingPositiveToNegative")
    finalize_on_crossing: bool = Field(alias="finalizeOnCrossing")


class CameraPublic(APIModel):
    id: str
    schema_version: int = Field(alias="schemaVersion")
    revision: int
    name: str
    stream: CameraPublicStream
    location: CameraLocationInput
    direction: CameraDirection
    vision: CameraVisionInput
    geometry: CameraPublicGeometry
    enabled: bool
    metadata: dict[str, object]
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, camera: Camera) -> CameraPublic:
        return cls(
            id=camera.id,
            schemaVersion=camera.schema_version,
            revision=camera.revision,
            name=camera.name,
            stream=CameraPublicStream(
                fpsLimit=camera.fps_limit,
                credentialsConfigured=True,
            ),
            location=CameraLocationInput(name=camera.location, zone=camera.zone),
            direction=camera.direction,
            vision=CameraVisionInput(
                vehicleConfidence=camera.vehicle_confidence,
                plateConfidence=camera.plate_confidence,
            ),
            geometry=CameraPublicGeometry(
                vehicleRoi=(
                    [(point.x, point.y) for point in camera.roi] if camera.roi is not None else None
                ),
                crossingLine=(
                    [(point.x, point.y) for point in camera.crossing_line]
                    if camera.crossing_line is not None
                    else None
                ),
                crossingPositiveToNegative=camera.crossing_positive_to_negative,
                finalizeOnCrossing=camera.finalize_on_crossing,
            ),
            enabled=camera.enabled,
            metadata=camera.metadata,
            createdAt=camera.created_at,
            updatedAt=camera.updated_at,
        )


class CameraListPublic(APIModel):
    items: list[CameraPublic]


class CameraBatchCreateRequest(APIModel):
    items: list[CameraCreateRequest] = Field(min_length=1, max_length=100)


class CameraBatchItemPublic(APIModel):
    camera_id: str = Field(alias="cameraId")
    status: str
    camera: CameraPublic | None = None


class CameraBatchPublic(APIModel):
    items: list[CameraBatchItemPublic]
    created_count: int = Field(alias="createdCount")
    conflict_count: int = Field(alias="conflictCount")
    capacity_reached_count: int = Field(alias="capacityReachedCount")

    @classmethod
    def from_domain(cls, result: CameraBatchResult) -> CameraBatchPublic:
        items = [
            CameraBatchItemPublic(
                cameraId=item.camera_id,
                status=item.status.value,
                camera=(CameraPublic.from_domain(item.camera) if item.camera is not None else None),
            )
            for item in result.items
        ]
        return cls(
            items=items,
            createdCount=sum(item.status == "CREATED" for item in items),
            conflictCount=sum(item.status == "CONFLICT" for item in items),
            capacityReachedCount=sum(item.status == "CAPACITY_REACHED" for item in items),
        )


class OnvifDevicePublic(APIModel):
    endpoint_reference: str = Field(alias="endpointReference")
    service_addresses: list[str] = Field(alias="serviceAddresses")
    types: list[str]
    scopes: list[str]
    remote_address: str | None = Field(alias="remoteAddress")
    name: str | None
    hardware: str | None
    locations: list[str]
    metadata_version: int | None = Field(alias="metadataVersion")
    discovered_at: datetime = Field(alias="discoveredAt")

    @classmethod
    def from_domain(cls, device: OnvifDiscoveredDevice) -> OnvifDevicePublic:
        return cls(
            endpointReference=device.endpoint_reference,
            serviceAddresses=list(device.xaddrs),
            types=list(device.types),
            scopes=list(device.scopes),
            remoteAddress=device.remote_address,
            name=device.name,
            hardware=device.hardware,
            locations=list(device.locations),
            metadataVersion=device.metadata_version,
            discoveredAt=device.discovered_at,
        )


class OnvifDiscoveryPublic(APIModel):
    items: list[OnvifDevicePublic]
    count: int


class CameraHealthPublic(APIModel):
    camera_id: str = Field(alias="cameraId")
    status: CameraStatus
    source_fps: float = Field(alias="sourceFps")
    decode_fps: float = Field(alias="decodeFps")
    queue_size: int = Field(alias="queueSize")
    dropped_frames: int = Field(alias="droppedFrames")
    reconnect_count: int = Field(alias="reconnectCount")
    connection_failures: int = Field(alias="connectionFailures")
    stream_epoch: int = Field(alias="streamEpoch")
    last_frame_at: datetime | None = Field(alias="lastFrameAt")
    updated_at: datetime = Field(alias="updatedAt")
    decoded_frames: int = Field(alias="decodedFrames")
    sampled_frames: int = Field(alias="sampledFrames")
    vehicle_detections: int = Field(alias="vehicleDetections")
    plate_detections: int = Field(alias="plateDetections")
    ocr_requests: int = Field(alias="ocrRequests")
    ocr_success: int = Field(alias="ocrSuccess")
    events_created: int = Field(alias="eventsCreated")
    track_count: int = Field(alias="trackCount")
    inference_fps: float = Field(alias="inferenceFps")
    vehicle_inference_latency_ms: float = Field(alias="vehicleInferenceLatencyMs")
    plate_inference_latency_ms: float = Field(alias="plateInferenceLatencyMs")
    ocr_latency_ms: float = Field(alias="ocrLatencyMs")

    @classmethod
    def from_domain(cls, health: CameraHealth) -> CameraHealthPublic:
        return cls(
            cameraId=health.camera_id,
            status=health.status,
            sourceFps=health.source_fps,
            decodeFps=health.decode_fps,
            queueSize=health.queue_size,
            droppedFrames=health.dropped_frames,
            reconnectCount=health.reconnect_count,
            connectionFailures=health.connection_failures,
            streamEpoch=health.stream_epoch,
            lastFrameAt=health.last_frame_at,
            updatedAt=health.updated_at,
            decodedFrames=health.decoded_frames,
            sampledFrames=health.sampled_frames,
            vehicleDetections=health.vehicle_detections,
            plateDetections=health.plate_detections,
            ocrRequests=health.ocr_requests,
            ocrSuccess=health.ocr_success,
            eventsCreated=health.events_created,
            trackCount=health.track_count,
            inferenceFps=health.inference_fps,
            vehicleInferenceLatencyMs=health.vehicle_inference_latency_ms,
            plateInferenceLatencyMs=health.plate_inference_latency_ms,
            ocrLatencyMs=health.ocr_latency_ms,
        )


class CameraHealthSnapshotItemPublic(APIModel):
    camera: CameraPublic
    health: CameraHealthPublic | None


class CameraHealthSnapshotPublic(APIModel):
    items: list[CameraHealthSnapshotItemPublic]


class CameraConnectionTestPublic(APIModel):
    connected: bool
    latency_ms: float = Field(alias="latencyMs")
    tested_at: datetime = Field(alias="testedAt")
    error_code: str | None = Field(alias="errorCode")

    @classmethod
    def from_domain(cls, result: CameraConnectionTestResult) -> CameraConnectionTestPublic:
        return cls(
            connected=result.connected,
            latencyMs=result.latency_ms,
            testedAt=result.tested_at,
            errorCode=result.error_code,
        )
