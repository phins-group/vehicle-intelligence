"""Bounded retention and MinIO lifecycle reconciliation worker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from contextlib import suppress
from pathlib import Path

from prometheus_client import start_http_server

from vehicle_intelligence.application.retention import RetentionService, RetentionStats
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.exceptions import ConfigurationError, VehicleIntelligenceError
from vehicle_intelligence.infrastructure.observability.metrics import PrometheusMetrics
from vehicle_intelligence.infrastructure.persistence.retention_mongo import (
    MongoRetentionRepository,
)
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage
from vehicle_intelligence.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coordinate vehicle-event and media retention")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--once", action="store_true", help="run one bounded pass")
    return parser


async def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    if not settings.retention.enabled:
        raise ConfigurationError("retention worker requires retention.enabled=true")
    if not settings.mongodb.enabled:
        raise ConfigurationError("retention worker requires mongodb.enabled=true")
    media = (
        MinioMediaStorage(settings.minio)
        if settings.storage.backend == "minio"
        else LocalMediaStorage(settings.storage.output_directory)
    )
    metrics = PrometheusMetrics()
    service = RetentionService(
        settings.retention,
        MongoRetentionRepository(settings.mongodb),
        media,
        lifecycle=media if isinstance(media, MinioMediaStorage) else None,
        metrics=metrics,
    )
    latest: RetentionStats | None = None
    metrics_server = None
    metrics_thread = None
    try:
        if args.once:
            latest = await service.run_once()
            _log_pass(latest)
        else:
            if settings.observability.prometheus_enabled:
                metrics_server, metrics_thread = start_http_server(
                    settings.observability.retention_metrics_port,
                    registry=metrics.registry,
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
            try:
                while not stop_event.is_set():
                    try:
                        latest = await service.run_once()
                        _log_pass(latest)
                    except VehicleIntelligenceError:
                        logger.exception("retention pass failed; worker will retry")
                    with suppress(TimeoutError):
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=settings.retention.worker_interval_seconds,
                        )
            finally:
                for signum in installed:
                    loop.remove_signal_handler(signum)
    finally:
        try:
            await service.close()
        finally:
            try:
                if isinstance(media, MinioMediaStorage):
                    await media.close()
            finally:
                if metrics_server is not None:
                    await asyncio.to_thread(metrics_server.shutdown)
                    metrics_server.server_close()
                if metrics_thread is not None:
                    await asyncio.to_thread(metrics_thread.join, 5)
    if latest is not None:
        print(
            "Retention pass complete; "
            f"claimed={sum(latest.claimed_by_kind.values())}, "
            f"media_deleted={sum(latest.deleted_by_kind.values())}, "
            f"media_failed={sum(latest.failed_by_kind.values())}, "
            f"events_deleted={latest.events_deleted}, "
            f"lifecycle_changed={latest.lifecycle.changed if latest.lifecycle else False}",
            flush=True,
        )
    return 0


def _log_pass(stats: RetentionStats) -> None:
    logger.info(
        "retention_pass_completed",
        extra={
            "claimed": sum(stats.claimed_by_kind.values()),
            "media_deleted": sum(stats.deleted_by_kind.values()),
            "media_failed": sum(stats.failed_by_kind.values()),
            "events_deleted": stats.events_deleted,
            "duration_seconds": round(stats.duration_seconds, 6),
            "lifecycle_changed": stats.lifecycle.changed if stats.lifecycle else False,
        },
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(run(args)))
    except VehicleIntelligenceError as exc:
        parser.exit(2, f"error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "interrupted\n")
