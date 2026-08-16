"""Dedicated process entry point for shared vehicle and plate inference."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from vehicle_intelligence.config import GPUSchedulerConfig, load_settings
from vehicle_intelligence.exceptions import ConfigurationError, VehicleIntelligenceError
from vehicle_intelligence.infrastructure.inference.protocol import read_inference_token
from vehicle_intelligence.infrastructure.inference.service import SharedInferenceService
from vehicle_intelligence.infrastructure.vision.factory import (
    create_plate_detector,
    create_vehicle_detector,
    validate_detector_provider,
)
from vehicle_intelligence.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local shared detector service")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--socket", type=Path)
    return parser


async def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    if not settings.gpu_scheduler.enabled:
        raise ConfigurationError("shared inference service requires gpu_scheduler.enabled=true")
    scheduler = settings.gpu_scheduler
    if args.socket is not None:
        scheduler = GPUSchedulerConfig.model_validate(
            {**scheduler.model_dump(), "socket_path": args.socket}
        )
    validate_detector_provider(
        settings.vision.vehicle_detection.provider,
        "vehicle detector",
    )
    validate_detector_provider(
        settings.vision.plate_detection.provider,
        "plate detector",
    )
    if not settings.vision.plate_detection.model_path:
        raise ConfigurationError("shared inference service requires a plate detector model")

    service = SharedInferenceService(
        scheduler,
        create_vehicle_detector(settings.vision.vehicle_detection),
        create_plate_detector(settings.vision.plate_detection),
        read_inference_token(),
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
            installed.append(signum)
        except NotImplementedError:
            pass
    service_wait: asyncio.Task[None] | None = None
    signal_wait: asyncio.Task[bool] | None = None
    try:
        await service.start()
        service_wait = asyncio.create_task(service.wait(), name="shared-inference-health")
        signal_wait = asyncio.create_task(stop_event.wait(), name="shared-inference-signal")
        completed, _ = await asyncio.wait(
            (service_wait, signal_wait),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if service_wait in completed:
            await service_wait
    finally:
        pending = tuple(task for task in (service_wait, signal_wait) if task is not None)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await service.close()
        for signum in installed:
            loop.remove_signal_handler(signum)
    return 0


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(asyncio.run(run(parser.parse_args())))
    except VehicleIntelligenceError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "interrupted\n")
