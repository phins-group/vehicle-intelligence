"""Versioned live-preview packet codec for cross-process delivery."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vehicle_intelligence.domain import (
    BoundingBox,
    Direction,
    LiveFrameMetadata,
    LiveFramePacket,
    LivePlateOverlay,
    LiveVehicleOverlay,
    Point,
)
from vehicle_intelligence.exceptions import EventContractError


class _Contract(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class PlateOverlayContract(_Contract):
    bbox: tuple[int, int, int, int]
    detection_confidence: float = Field(alias="detectionConfidence", ge=0, le=1)
    quality_score: float | None = Field(default=None, alias="qualityScore", ge=0, le=1)
    text: str | None = Field(default=None, max_length=32)
    ocr_confidence: float | None = Field(default=None, alias="ocrConfidence", ge=0, le=1)


class VehicleOverlayContract(_Contract):
    track_id: str = Field(alias="trackId", min_length=1, max_length=256)
    bbox: tuple[int, int, int, int]
    confidence: float = Field(ge=0, le=1)
    vehicle_type: str = Field(alias="vehicleType", min_length=1, max_length=64)
    direction: Direction
    plate: PlateOverlayContract | None = None


class LiveFrameContract(_Contract):
    schema_version: int = Field(alias="schemaVersion")
    camera_id: str = Field(alias="cameraId", min_length=1, max_length=128)
    frame_id: int = Field(alias="frameId", ge=0)
    stream_epoch: int = Field(alias="streamEpoch", ge=0)
    captured_at: datetime = Field(alias="capturedAt")
    source_width: int = Field(alias="sourceWidth", gt=0)
    source_height: int = Field(alias="sourceHeight", gt=0)
    preview_width: int = Field(alias="previewWidth", gt=0)
    preview_height: int = Field(alias="previewHeight", gt=0)
    vehicles: tuple[VehicleOverlayContract, ...] = ()
    vehicle_roi: tuple[tuple[float, float], ...] | None = Field(
        default=None,
        alias="vehicleRoi",
    )
    crossing_line: tuple[tuple[float, float], tuple[float, float]] | None = Field(
        default=None,
        alias="crossingLine",
    )
    jpeg_base64: str = Field(alias="jpegBase64", min_length=1)


class JsonLiveFrameCodec:
    def __init__(self, maximum_payload_bytes: int) -> None:
        if maximum_payload_bytes <= 0:
            raise ValueError("live payload limit must be positive")
        self._maximum_payload_bytes = maximum_payload_bytes

    def encode(self, packet: LiveFramePacket) -> str:
        metadata = packet.metadata
        contract = LiveFrameContract(
            schemaVersion=metadata.schema_version,
            cameraId=metadata.camera_id,
            frameId=metadata.frame_id,
            streamEpoch=metadata.stream_epoch,
            capturedAt=metadata.captured_at,
            sourceWidth=metadata.source_width,
            sourceHeight=metadata.source_height,
            previewWidth=packet.preview_width,
            previewHeight=packet.preview_height,
            vehicles=tuple(self._vehicle_contract(item) for item in metadata.vehicles),
            vehicleRoi=(
                tuple((point.x, point.y) for point in metadata.vehicle_roi)
                if metadata.vehicle_roi is not None
                else None
            ),
            crossingLine=(
                tuple((point.x, point.y) for point in metadata.crossing_line)
                if metadata.crossing_line is not None
                else None
            ),
            jpegBase64=base64.b64encode(packet.jpeg).decode("ascii"),
        )
        payload = contract.model_dump_json(by_alias=True)
        if len(payload.encode("utf-8")) > self._maximum_payload_bytes:
            raise EventContractError("live frame payload exceeds configured limit")
        return payload

    def decode(self, payload: str) -> LiveFramePacket:
        if len(payload.encode("utf-8")) > self._maximum_payload_bytes:
            raise EventContractError("live frame payload exceeds configured limit")
        try:
            contract = LiveFrameContract.model_validate_json(payload)
            if contract.schema_version != 1:
                raise ValueError(
                    f"unsupported live frame schema version: {contract.schema_version}"
                )
            jpeg = base64.b64decode(contract.jpeg_base64, validate=True)
            metadata = LiveFrameMetadata(
                camera_id=contract.camera_id,
                frame_id=contract.frame_id,
                stream_epoch=contract.stream_epoch,
                captured_at=contract.captured_at,
                source_width=contract.source_width,
                source_height=contract.source_height,
                vehicles=tuple(self._vehicle_domain(item) for item in contract.vehicles),
                vehicle_roi=(
                    tuple(Point(x, y) for x, y in contract.vehicle_roi)
                    if contract.vehicle_roi is not None
                    else None
                ),
                crossing_line=(
                    tuple(Point(x, y) for x, y in contract.crossing_line)
                    if contract.crossing_line is not None
                    else None
                ),
                schema_version=contract.schema_version,
            )
            return LiveFramePacket(
                metadata=metadata,
                jpeg=jpeg,
                preview_width=contract.preview_width,
                preview_height=contract.preview_height,
            )
        except (binascii.Error, TypeError, ValueError, ValidationError) as exc:
            raise EventContractError(f"invalid live frame payload: {exc}") from exc

    @staticmethod
    def _vehicle_contract(item: LiveVehicleOverlay) -> VehicleOverlayContract:
        plate = item.plate
        return VehicleOverlayContract(
            trackId=item.track_id,
            bbox=item.bbox.as_xyxy(),
            confidence=item.confidence,
            vehicleType=item.vehicle_type,
            direction=item.direction,
            plate=(
                PlateOverlayContract(
                    bbox=plate.bbox.as_xyxy(),
                    detectionConfidence=plate.detection_confidence,
                    qualityScore=plate.quality_score,
                    text=plate.text,
                    ocrConfidence=plate.ocr_confidence,
                )
                if plate is not None
                else None
            ),
        )

    @staticmethod
    def _vehicle_domain(item: VehicleOverlayContract) -> LiveVehicleOverlay:
        plate = item.plate
        return LiveVehicleOverlay(
            track_id=item.track_id,
            bbox=BoundingBox(*item.bbox),
            confidence=item.confidence,
            vehicle_type=item.vehicle_type,
            direction=item.direction,
            plate=(
                LivePlateOverlay(
                    bbox=BoundingBox(*plate.bbox),
                    detection_confidence=plate.detection_confidence,
                    quality_score=plate.quality_score,
                    text=plate.text,
                    ocr_confidence=plate.ocr_confidence,
                )
                if plate is not None
                else None
            ),
        )

