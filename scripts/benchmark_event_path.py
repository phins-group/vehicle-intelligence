"""Real Redis Streams -> worker -> MongoDB load/soak acceptance benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import resource
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis

from vehicle_intelligence.config import (
    IdentityConfig,
    MongoConfig,
    RealtimeConfig,
    RedisConfig,
    RuleEngineConfig,
)
from vehicle_intelligence.domain import (
    AITrace,
    CameraSnapshot,
    Direction,
    EventStatus,
    EventType,
    MediaReferences,
    ModelMetadata,
    PlateEvidence,
    Rule,
    RuleAction,
    RuleActionType,
    RuleCondition,
    RuleConditionOperator,
    VehicleEvent,
    VehicleEvidence,
)
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.redis_streams import (
    EVENT_PAYLOAD_FIELD,
    RedisStreamEventPublisher,
)
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.interfaces.event_worker_composition import build_event_worker


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    run_id: str
    requested_events: int
    persisted_events: int
    duplicate_deliveries: int
    identity_fingerprints: int
    matched_rules: int
    realtime_published: int
    dead_lettered: int
    pending_messages: int
    elapsed_seconds: float
    events_per_second: float
    p95_worker_batch_ms: float
    rss_growth_mb: float
    error_rate: float
    passed: bool
    failures: tuple[str, ...]


def _event(run_id: str, index: int) -> VehicleEvent:
    timestamp = datetime.now(UTC)
    model = ModelMetadata(name="event-path-benchmark", version="1")
    return VehicleEvent(
        id=f"evt_benchmark_{run_id}_{index:08d}",
        schema_version=1,
        camera=CameraSnapshot(id=f"benchmark-camera-{index % 16:02d}", name="Benchmark"),
        track_id=f"benchmark:{run_id}:{index}",
        event_type=EventType.VEHICLE_DETECTED,
        occurred_at=timestamp,
        created_at=timestamp,
        direction=Direction.UNKNOWN,
        status=EventStatus.CONFIRMED,
        vehicle=VehicleEvidence(type="car", confidence=0.95),
        plate=PlateEvidence(
            raw=f"51H{index % 100000:05d}",
            normalized=f"51H-{index % 100000:03d}.{index % 100:02d}",
            confidence=0.92,
            observation_count=4,
        ),
        media=MediaReferences(),
        ai=AITrace(model, model, model, config_version="benchmark-v1"),
        metadata={"benchmarkRun": run_id},
    )


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


async def _publish(
    publisher: RedisStreamEventPublisher,
    events: list[VehicleEvent],
    concurrency: int,
    rate: float | None,
) -> None:
    for offset in range(0, len(events), concurrency):
        started = time.perf_counter()
        batch = events[offset : offset + concurrency]
        await asyncio.gather(*(publisher.publish(event) for event in batch))
        if rate:
            expected = len(batch) / rate
            remaining = expected - (time.perf_counter() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)


async def run(args: argparse.Namespace) -> BenchmarkResult:
    run_id = uuid.uuid4().hex[:12]
    count = args.events
    if args.soak_seconds:
        count = max(count, math.ceil(args.soak_seconds * args.rate))
    stream = f"vehicle.events.benchmark.{run_id}"
    dead_letter = f"vehicle.events.benchmark.dlq.{run_id}"
    redis_config = RedisConfig(
        url=args.redis_url,
        stream=stream,
        dead_letter_stream=dead_letter,
        consumer_group=f"benchmark-workers-{run_id}",
        max_length=max(1000, count * 2),
        dead_letter_max_length=100,
        batch_size=min(1000, args.batch_size),
        block_ms=10,
        claim_idle_ms=1000,
    )
    benchmark_database = f"{args.mongodb_database[:45]}_{run_id}"
    mongo_config = MongoConfig(
        enabled=True,
        uri=args.mongodb_uri,
        database=benchmark_database,
        server_selection_timeout_ms=5000,
    )
    codec = JsonEventEnvelopeCodec()
    publisher = RedisStreamEventPublisher(redis_config, codec)
    mongo_runtime = MongoRuntime(mongo_config)
    components = build_event_worker(
        mongo_runtime,
        redis_config,
        IdentityConfig(enabled=True),
        RuleEngineConfig(enabled=True, external_actions_enabled=False),
        RealtimeConfig(
            enabled=True,
            redis_channel=f"vehicle.events.benchmark.realtime.{run_id}",
        ),
        f"benchmark-{run_id}",
        codec,
    )
    worker = components.worker
    repository = components.events
    if components.rules is None or components.identities is None:
        raise RuntimeError("benchmark production processors were not composed")
    admin = Redis.from_url(args.redis_url, decode_responses=True)
    events = [_event(run_id, index) for index in range(count)]
    duplicates = events[: math.floor(count * args.duplicate_ratio)]
    failures: list[str] = []
    batch_latencies: list[float] = []
    rss_before = _rss_mb()
    started = time.perf_counter()
    try:
        await mongo_runtime.initialize()
        await publisher.initialize()
        await worker.initialize()
        timestamp = datetime.now(UTC)
        await components.rules.create(
            Rule(
                id=f"rule-benchmark-{run_id}",
                name="Event path benchmark",
                enabled=True,
                priority=1,
                conditions=(RuleCondition("camera.id", RuleConditionOperator.EXISTS, True),),
                actions=(RuleAction("log", RuleActionType.LOG),),
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        await _publish(
            publisher,
            events,
            args.publish_concurrency,
            args.rate if args.soak_seconds else None,
        )
        deadline = time.monotonic() + args.deadline_seconds
        while worker.stats.events_persisted < count and time.monotonic() < deadline:
            batch_started = time.perf_counter()
            processed = await worker.run_once()
            if processed:
                batch_latencies.append((time.perf_counter() - batch_started) * 1000)
        if worker.stats.events_persisted != count:
            failures.append("persisted_count")

        await _publish(publisher, duplicates, args.publish_concurrency, None)
        duplicate_target = len(duplicates)
        while worker.stats.duplicate_events < duplicate_target and time.monotonic() < deadline:
            await worker.run_once()
        if worker.stats.duplicate_events != duplicate_target:
            failures.append("duplicate_recovery")

        await admin.xadd(stream, {EVENT_PAYLOAD_FIELD: "not-a-valid-event"})
        await worker.run_once()
        persisted = await repository._collection.count_documents({"metadata.benchmarkRun": run_id})
        identity_fingerprints = await components.identities._fingerprints.count_documents(
            {"sourceEventId": {"$regex": f"^evt_benchmark_{run_id}_"}}
        )
        pending = await admin.xpending(stream, redis_config.consumer_group)
        dead_lettered = await admin.xlen(dead_letter)
        if persisted != count:
            failures.append("mongo_count")
        if pending["pending"] != 0:
            failures.append("pending_messages")
        if dead_lettered != 1:
            failures.append("dead_letter")
        expected_deliveries = count + duplicate_target
        if identity_fingerprints != count:
            failures.append("identity_path")
        if worker.stats.matched_rules != expected_deliveries:
            failures.append("policy_path")
        if (
            worker.stats.actions_succeeded != count
            or worker.stats.actions_skipped != duplicate_target
        ):
            failures.append("action_idempotency")
        if worker.stats.realtime_published != expected_deliveries:
            failures.append("realtime_path")

        elapsed = time.perf_counter() - started
        throughput = count / elapsed if elapsed else 0
        p95 = _p95(batch_latencies)
        error_count = (
            worker.stats.persistence_failures
            + worker.stats.policy_failures
            + worker.stats.realtime_failures
        )
        error_rate = error_count / max(1, count)
        rss_growth = max(0.0, _rss_mb() - rss_before)
        if throughput < args.minimum_throughput:
            failures.append("minimum_throughput")
        if p95 > args.maximum_p95_batch_ms:
            failures.append("maximum_p95_batch_ms")
        if error_rate > args.maximum_error_rate:
            failures.append("maximum_error_rate")
        if rss_growth > args.maximum_rss_growth_mb:
            failures.append("maximum_rss_growth_mb")
        return BenchmarkResult(
            run_id=run_id,
            requested_events=count,
            persisted_events=persisted,
            duplicate_deliveries=worker.stats.duplicate_events,
            identity_fingerprints=identity_fingerprints,
            matched_rules=worker.stats.matched_rules,
            realtime_published=worker.stats.realtime_published,
            dead_lettered=dead_lettered,
            pending_messages=int(pending["pending"]),
            elapsed_seconds=round(elapsed, 4),
            events_per_second=round(throughput, 2),
            p95_worker_batch_ms=round(p95, 2),
            rss_growth_mb=round(rss_growth, 2),
            error_rate=round(error_rate, 6),
            passed=not failures,
            failures=tuple(dict.fromkeys(failures)),
        )
    finally:
        try:
            await worker.close()
        finally:
            try:
                await publisher.close()
            finally:
                try:
                    await admin.delete(stream, dead_letter)
                    await admin.aclose()
                finally:
                    await mongo_runtime.client.drop_database(benchmark_database)
                    await mongo_runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mongodb-uri",
        default=os.getenv("TEST_MONGODB_URI", "mongodb://localhost:27017/?directConnection=true"),
    )
    parser.add_argument("--mongodb-database", default="vehicle_intelligence_benchmark")
    parser.add_argument(
        "--redis-url",
        default=os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15"),
    )
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--publish-concurrency", type=int, default=100)
    parser.add_argument("--duplicate-ratio", type=float, default=0.1)
    parser.add_argument("--deadline-seconds", type=float, default=60)
    parser.add_argument("--soak-seconds", type=float, default=0)
    parser.add_argument("--rate", type=float, default=50)
    parser.add_argument("--minimum-throughput", type=float, default=25)
    parser.add_argument("--maximum-p95-batch-ms", type=float, default=2000)
    parser.add_argument("--maximum-error-rate", type=float, default=0)
    parser.add_argument("--maximum-rss-growth-mb", type=float, default=256)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (
        args.events < 1
        or not 1 <= args.batch_size <= 1000
        or not 1 <= args.publish_concurrency <= 1000
        or not 0 <= args.duplicate_ratio <= 1
        or args.deadline_seconds <= 0
        or args.soak_seconds < 0
        or args.rate <= 0
    ):
        raise SystemExit("invalid benchmark bounds")
    result = asyncio.run(run(args))
    print(json.dumps(asdict(result), sort_keys=True))
    raise SystemExit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
