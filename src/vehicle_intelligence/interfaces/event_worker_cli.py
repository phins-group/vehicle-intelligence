"""Redis Streams to MongoDB vehicle-event worker CLI."""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import socket
import uuid
from pathlib import Path

from vehicle_intelligence.config import load_settings
from vehicle_intelligence.exceptions import ConfigurationError, VehicleIntelligenceError
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.interfaces.event_worker_composition import build_event_worker
from vehicle_intelligence.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persist Redis vehicle events into MongoDB")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--consumer-name")
    parser.add_argument(
        "--once",
        action="store_true",
        help="process at most one reclaimed/new batch, then exit",
    )
    return parser


def default_consumer_name() -> str:
    host = socket.gethostname().split(".", maxsplit=1)[0]
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


async def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    if settings.event_bus.backend != "redis":
        raise ConfigurationError("event worker requires event_bus.backend=redis")
    if not settings.mongodb.enabled:
        raise ConfigurationError("event worker requires mongodb.enabled=true")

    mongo_runtime = MongoRuntime(settings.mongodb)
    worker = build_event_worker(
        mongo_runtime,
        settings.redis,
        settings.identity,
        settings.rule_engine,
        settings.realtime,
        args.consumer_name or default_consumer_name(),
    ).worker

    await mongo_runtime.initialize()
    try:
        if args.once:
            try:
                await worker.initialize()
                await worker.run_once()
            finally:
                await worker.close()
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
                await worker.run(stop_event)
            finally:
                for signum in installed:
                    loop.remove_signal_handler(signum)
    finally:
        await mongo_runtime.close()

    stats = worker.stats
    print(
        "Event worker stopped; "
        f"read={stats.messages_read}, reclaimed={stats.messages_reclaimed}, "
        f"persisted={stats.events_persisted}, duplicates={stats.duplicate_events}, "
        f"invalid={stats.invalid_messages}, persistence_failures={stats.persistence_failures}, "
        f"policy_failures={stats.policy_failures}, matched_rules={stats.matched_rules}, "
        f"actions_succeeded={stats.actions_succeeded}, actions_skipped={stats.actions_skipped}, "
        f"realtime_published={stats.realtime_published}, "
        f"realtime_failures={stats.realtime_failures}",
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
