"""Phase 1 video-file pipeline CLI composition root."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from vehicle_intelligence.config import Settings, load_settings
from vehicle_intelligence.exceptions import ConfigurationError, VehicleIntelligenceError
from vehicle_intelligence.infrastructure.vision.opencv import OpenCVVideoSource
from vehicle_intelligence.interfaces.composition import (
    execute_pipeline,
    validate_runtime_settings,
)
from vehicle_intelligence.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 1 vehicle pipeline")
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--camera", dest="camera_id")
    parser.add_argument("--camera-name")
    parser.add_argument("--fps-limit", type=float)
    parser.add_argument("--vehicle-model")
    parser.add_argument("--plate-model")
    parser.add_argument(
        "--plate-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="detect and track plates on the full frame without loading a vehicle model",
    )
    parser.add_argument("--device", help="detector device, for example cpu, cuda, or 0")
    parser.add_argument("--ocr-device", help="PaddleOCR device, for example cpu or gpu:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--storage", choices=("local", "minio"))
    parser.add_argument("--mongo", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--event-backend", choices=("direct", "redis"))
    parser.add_argument(
        "--video-start-time",
        type=parse_aware_datetime,
        help="ISO-8601 timestamp for the first frame (defaults to file modification time)",
    )
    return parser


def parse_aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timestamp must be valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone offset")
    return parsed


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    camera_updates = {
        key: value
        for key, value in {
            "id": args.camera_id,
            "name": args.camera_name,
            "fps_limit": args.fps_limit,
        }.items()
        if value is not None
    }
    vehicle_updates = {
        key: value
        for key, value in {"model_path": args.vehicle_model, "device": args.device}.items()
        if value is not None
    }
    plate_updates = {
        key: value
        for key, value in {"model_path": args.plate_model, "device": args.device}.items()
        if value is not None
    }
    ocr_updates = {"device": args.ocr_device} if args.ocr_device is not None else {}
    storage_updates = {
        key: value
        for key, value in {
            "output_directory": args.output,
            "backend": args.storage,
        }.items()
        if value is not None
    }
    mongo_updates = {"enabled": args.mongo} if args.mongo is not None else {}
    event_backend = getattr(args, "event_backend", None)
    event_bus_updates = {"backend": event_backend} if event_backend is not None else {}
    vision_updates: dict[str, object] = {
        "vehicle_detection": settings.vision.vehicle_detection.model_copy(
            update=vehicle_updates
        ),
        "plate_detection": settings.vision.plate_detection.model_copy(update=plate_updates),
        "ocr": settings.vision.ocr.model_copy(update=ocr_updates),
    }
    plate_only = getattr(args, "plate_only", None)
    if plate_only is not None:
        vision_updates["plate_only"] = plate_only
    vision = settings.vision.model_copy(update=vision_updates)
    candidate = settings.model_copy(
        update={
            "camera": settings.camera.model_copy(update=camera_updates),
            "vision": vision,
            "storage": settings.storage.model_copy(update=storage_updates),
            "mongodb": settings.mongodb.model_copy(update=mongo_updates),
            "event_bus": settings.event_bus.model_copy(update=event_bus_updates),
        }
    )
    try:
        return Settings.model_validate(candidate.model_dump())
    except ValueError as exc:
        raise ConfigurationError(f"invalid command-line override: {exc}") from exc


async def run(args: argparse.Namespace) -> int:
    settings = apply_overrides(load_settings(args.config), args)
    configure_logging(settings.app.log_level)
    validate_runtime_settings(settings)

    source = OpenCVVideoSource(
        args.video,
        settings.camera.id,
        settings.camera.fps_limit,
        args.video_start_time,
    )
    result = await execute_pipeline(settings, source)
    for event in result.events:
        plate = event.plate
        print(f"Track: {event.track_id}")
        if not settings.vision.plate_only:
            print(f"Vehicle: {event.vehicle.type.upper()} {event.vehicle.confidence:.2f}")
        print(f"Plate: {plate.final_normalized if plate else 'UNREADABLE'}")
        print(f"OCR Confidence: {plate.confidence:.2f}" if plate else "OCR Confidence: N/A")
        print(f"Direction: {event.direction.value}")
        print()
    destination = (
        f"published to Redis stream {settings.redis.stream}"
        if settings.event_bus.backend == "redis"
        else f"wrote {settings.storage.output_directory / 'events.jsonl'}"
    )
    print(f"Finalized {result.stats.finalized_tracks} track(s); {destination}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except VehicleIntelligenceError as exc:
        parser.exit(2, f"error: {exc}\n")
