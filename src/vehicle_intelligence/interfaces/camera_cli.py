"""Single-camera RTSP vision-worker CLI."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import threading
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import SecretStr

from vehicle_intelligence.config import load_settings
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import ConfigurationError, VehicleIntelligenceError
from vehicle_intelligence.infrastructure.persistence.camera_mongo import (
    MongoCameraHealthRepository,
)
from vehicle_intelligence.infrastructure.vision.rtsp import OpenCVRTSPSource
from vehicle_intelligence.interfaces.cli import apply_overrides
from vehicle_intelligence.interfaces.composition import (
    execute_pipeline,
    validate_runtime_settings,
)
from vehicle_intelligence.logging_config import configure_logging


class StopRequestSource(Protocol):
    def request_stop(self) -> None: ...


class CameraShutdown:
    """Turn the first process signal into a graceful source stop."""

    def __init__(self, source: StopRequestSource) -> None:
        self._source = source
        self._previous: dict[int, object] = {}
        self.exit_code: int | None = None

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self.handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def handle(self, signum: int, _frame: object) -> None:
        if self.exit_code is not None:
            raise KeyboardInterrupt
        self.exit_code = 128 + signum
        self._source.request_stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one RTSP camera vision worker")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--camera", dest="camera_id", required=True)
    parser.add_argument("--camera-name")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rtsp", help="RTSP URL; prefer --rtsp-env to avoid shell history")
    source.add_argument("--rtsp-env", metavar="ENV_VAR", help="environment variable holding URL")
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
    return parser


def resolve_rtsp_url(args: argparse.Namespace) -> SecretStr:
    value = args.rtsp
    if args.rtsp_env:
        value = os.getenv(args.rtsp_env)
        if not value:
            raise ConfigurationError(
                f"RTSP environment variable is missing or empty: {args.rtsp_env}"
            )
    try:
        parsed = urlsplit(value or "")
    except ValueError as exc:
        raise ConfigurationError("RTSP source URL is malformed") from exc
    if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise ConfigurationError("RTSP source must be an rtsp:// or rtsps:// URL with a host")
    return SecretStr(value or "")


def print_event(event: VehicleEvent) -> None:
    plate = event.plate.final_normalized if event.plate else "UNREADABLE"
    confidence = f"{event.plate.confidence:.2f}" if event.plate else "N/A"
    print(
        f"Event: {event.id} | Track: {event.track_id} | Plate: {plate} | "
        f"OCR: {confidence} | Direction: {event.direction.value}",
        flush=True,
    )


async def run(args: argparse.Namespace) -> int:
    base = load_settings(args.config)
    if args.output is None:
        args.output = base.storage.output_directory / args.camera_id
    if args.camera_name is None:
        args.camera_name = args.camera_id
    settings = apply_overrides(base, args)
    configure_logging(settings.app.log_level)
    validate_runtime_settings(settings)
    source = OpenCVRTSPSource(
        resolve_rtsp_url(args),
        settings.camera.id,
        settings.camera.fps_limit,
        settings.rtsp,
    )
    shutdown = CameraShutdown(source)
    shutdown.install()
    try:
        health_repository = (
            MongoCameraHealthRepository(settings.mongodb) if settings.mongodb.enabled else None
        )
        result = await execute_pipeline(
            settings,
            source,
            retain_events=False,
            event_observer=print_event,
            health_repository=health_repository,
        )
    finally:
        shutdown.restore()
    health = source.health
    print(
        f"Stopped {settings.camera.id}; finalized={result.stats.finalized_tracks}, "
        f"dropped={health.dropped_frames}, reconnects={health.reconnect_count}, "
        f"connection_failures={health.connection_failures}",
        flush=True,
    )
    return shutdown.exit_code or 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except VehicleIntelligenceError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "interrupted\n")
