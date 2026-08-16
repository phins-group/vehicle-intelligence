"""Authenticated live-preview metadata and exact-frame HTTP surface."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.live_monitor import LiveCameraSnapshot, LiveMonitorService
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import LivePlateOverlay, LiveVehicleOverlay, Principal
from vehicle_intelligence.exceptions import CameraNotFoundError
from vehicle_intelligence.interfaces.security import APISecurity


class _PublicModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class LivePlateOverlayPublic(_PublicModel):
    bbox: tuple[int, int, int, int]
    detection_confidence: float = Field(alias="detectionConfidence")
    quality_score: float | None = Field(alias="qualityScore")
    text: str | None
    ocr_confidence: float | None = Field(alias="ocrConfidence")

    @classmethod
    def from_domain(cls, plate: LivePlateOverlay) -> LivePlateOverlayPublic:
        return cls(
            bbox=plate.bbox.as_xyxy(),
            detectionConfidence=plate.detection_confidence,
            qualityScore=plate.quality_score,
            text=plate.text,
            ocrConfidence=plate.ocr_confidence,
        )


class LiveVehicleOverlayPublic(_PublicModel):
    track_id: str = Field(alias="trackId")
    bbox: tuple[int, int, int, int]
    confidence: float
    vehicle_type: str = Field(alias="vehicleType")
    direction: str
    plate: LivePlateOverlayPublic | None

    @classmethod
    def from_domain(cls, vehicle: LiveVehicleOverlay) -> LiveVehicleOverlayPublic:
        return cls(
            trackId=vehicle.track_id,
            bbox=vehicle.bbox.as_xyxy(),
            confidence=vehicle.confidence,
            vehicleType=vehicle.vehicle_type,
            direction=vehicle.direction.value,
            plate=(
                LivePlateOverlayPublic.from_domain(vehicle.plate)
                if vehicle.plate is not None
                else None
            ),
        )


class LiveFramePublic(_PublicModel):
    sequence: int
    frame_id: int = Field(alias="frameId")
    stream_epoch: int = Field(alias="streamEpoch")
    captured_at: datetime = Field(alias="capturedAt")
    received_at: datetime = Field(alias="receivedAt")
    source_width: int = Field(alias="sourceWidth")
    source_height: int = Field(alias="sourceHeight")
    preview_width: int = Field(alias="previewWidth")
    preview_height: int = Field(alias="previewHeight")
    vehicles: list[LiveVehicleOverlayPublic]
    vehicle_roi: list[tuple[float, float]] | None = Field(alias="vehicleRoi")
    crossing_line: list[tuple[float, float]] | None = Field(alias="crossingLine")
    frame_url: str = Field(alias="frameUrl")


class LiveMonitorStatePublic(_PublicModel):
    camera_id: str = Field(alias="cameraId")
    status: Literal["DISABLED", "WAITING", "LIVE", "STALE", "OFFLINE"]
    source_state: Literal["STARTING", "ONLINE", "OFFLINE", "STOPPED"] = Field(alias="sourceState")
    latest: LiveFramePublic | None


def build_live_monitor_router(
    service: LiveMonitorService | None,
    cameras: CameraService | None,
    security: APISecurity,
) -> APIRouter:
    router = APIRouter(tags=["live-monitor"])
    read_access = security.require(Permission.READ_PLATFORM)

    @router.get(
        "/api/cameras/{camera_id}/live",
        response_model=LiveMonitorStatePublic,
        response_model_by_alias=True,
    )
    async def live_state(
        camera_id: str,
        response: Response,
        _principal: Principal = Depends(read_access),
    ) -> LiveMonitorStatePublic:
        response.headers["Cache-Control"] = "no-store"
        managed = _require_service(service, cameras)
        camera = await _camera(managed[1], camera_id)
        return _state_public(managed[0].snapshot(camera_id, enabled=camera.enabled))

    @router.get("/api/cameras/{camera_id}/live/frame")
    async def live_frame(
        camera_id: str,
        sequence: int = Query(ge=1),
        _principal: Principal = Depends(read_access),
    ) -> Response:
        managed = _require_service(service, cameras)
        await _camera(managed[1], camera_id)
        frame = managed[0].frame(camera_id, sequence)
        if frame is None:
            raise HTTPException(status_code=410, detail="live frame is no longer buffered")
        metadata = frame.packet.metadata
        return Response(
            content=frame.packet.jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Live-Sequence": str(frame.sequence),
                "X-Frame-ID": str(metadata.frame_id),
                "X-Stream-Epoch": str(metadata.stream_epoch),
                "X-Captured-At": metadata.captured_at.isoformat(),
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.get("/api/live-monitor/health")
    async def live_health(
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        if service is None:
            return {"status": "DISABLED", "camerasBuffered": 0}
        stats = service.stats
        return {
            "status": stats.source_state.value,
            "camerasBuffered": stats.cameras_buffered,
            "framesReceived": stats.frames_received,
            "framesEvicted": stats.frames_evicted,
            "reconnectCount": stats.reconnect_count,
            "sourceFailures": stats.source_failures,
            "invalidMessages": stats.invalid_messages,
            "lastFrameAt": stats.last_frame_at,
        }

    return router


def _require_service(
    service: LiveMonitorService | None,
    cameras: CameraService | None,
) -> tuple[LiveMonitorService, CameraService]:
    if service is None:
        raise HTTPException(status_code=503, detail="live monitor is disabled")
    if cameras is None:
        raise HTTPException(status_code=503, detail="camera management is unavailable")
    return service, cameras


async def _camera(cameras: CameraService, camera_id: str):
    try:
        return await cameras.get(camera_id)
    except CameraNotFoundError as exc:
        raise HTTPException(status_code=404, detail="camera not found") from exc


def _state_public(snapshot: LiveCameraSnapshot) -> LiveMonitorStatePublic:
    buffered = snapshot.latest
    latest = None
    if buffered is not None:
        packet = buffered.packet
        metadata = packet.metadata
        latest = LiveFramePublic(
            sequence=buffered.sequence,
            frameId=metadata.frame_id,
            streamEpoch=metadata.stream_epoch,
            capturedAt=metadata.captured_at,
            receivedAt=buffered.received_at,
            sourceWidth=metadata.source_width,
            sourceHeight=metadata.source_height,
            previewWidth=packet.preview_width,
            previewHeight=packet.preview_height,
            vehicles=[LiveVehicleOverlayPublic.from_domain(item) for item in metadata.vehicles],
            vehicleRoi=(
                [(point.x, point.y) for point in metadata.vehicle_roi]
                if metadata.vehicle_roi is not None
                else None
            ),
            crossingLine=(
                [(point.x, point.y) for point in metadata.crossing_line]
                if metadata.crossing_line is not None
                else None
            ),
            frameUrl=(f"/api/cameras/{snapshot.camera_id}/live/frame?sequence={buffered.sequence}"),
        )
    return LiveMonitorStatePublic(
        cameraId=snapshot.camera_id,
        status=snapshot.status.value,
        sourceState=snapshot.source_state.value,
        latest=latest,
    )
