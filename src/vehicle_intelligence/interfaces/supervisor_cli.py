"""Host-native multi-camera process supervisor CLI."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from vehicle_intelligence.application.supervisor import CameraSupervisor
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.exceptions import ConfigurationError, VehicleIntelligenceError
from vehicle_intelligence.infrastructure.persistence.camera_mongo import (
    MongoCameraHealthRepository,
    MongoCameraRepository,
)
from vehicle_intelligence.infrastructure.security.aes_gcm import AesGcmCredentialCipher
from vehicle_intelligence.infrastructure.supervision.subprocess import (
    SubprocessCameraWorkerLauncher,
)
from vehicle_intelligence.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconcile one isolated worker per camera")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument(
        "--once",
        action="store_true",
        help="perform one reconciliation pass and stop all launched workers",
    )
    return parser


async def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    if not settings.mongodb.enabled:
        raise ConfigurationError("camera supervisor requires mongodb.enabled=true")
    cipher = AesGcmCredentialCipher.from_config(settings.security)
    cameras = MongoCameraRepository(settings.mongodb, cipher)
    health = MongoCameraHealthRepository(settings.mongodb)
    launcher = SubprocessCameraWorkerLauncher(
        settings.camera_manager.worker_command,
        args.worker_config or settings.camera_manager.worker_config_path,
        settings.camera_manager.worker_shutdown_seconds,
    )
    supervisor = CameraSupervisor(
        cameras,
        health,
        launcher,
        settings.camera_manager.reconcile_interval_seconds,
        settings.camera_manager.restart_backoff_seconds,
        settings.camera_manager.restart_backoff_max_seconds,
        settings.camera_manager.restart_stability_seconds,
        settings.camera_manager.maximum_active_workers,
        settings.camera_manager.maximum_starts_per_reconcile,
    )

    if args.once:
        try:
            await supervisor.initialize()
            await supervisor.reconcile_once()
        finally:
            await supervisor.stop_all()
            await supervisor.close()
    else:
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, stop_event.set)
                installed.append(signum)
            except NotImplementedError:
                pass
        try:
            await supervisor.run(stop_event)
        finally:
            for signum in installed:
                loop.remove_signal_handler(signum)

    stats = supervisor.stats
    print(
        "Camera supervisor stopped; "
        f"started={stats.workers_started}, stopped={stats.workers_stopped}, "
        f"restarted={stats.workers_restarted}, crashes={stats.worker_crashes}, "
        f"start_failures={stats.worker_start_failures}, "
        f"capacity_deferred={stats.workers_capacity_deferred}, "
        f"peak_active={stats.peak_active_workers}",
        flush=True,
    )
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except VehicleIntelligenceError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "interrupted\n")
