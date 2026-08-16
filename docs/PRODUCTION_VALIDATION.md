# Production Failure, Load, and Soak Validation

## Event-path benchmark

`scripts/benchmark_event_path.py` drives versioned events through real Redis
Streams and the same worker composition as the production CLI: durable event
write, identity/fingerprint linking, cached rule evaluation plus idempotent
action, and realtime publication. It then redelivers a configured fraction,
injects one invalid contract, and verifies every production stage in addition to
Mongo cardinality, pending/DLQ counts, throughput, p95 worker-batch time, error
rate, and process RSS growth. Each run uses a unique benchmark database and
stream namespace, both removed in `finally`.

Burst example:

```bash
python scripts/benchmark_event_path.py \
  --events 5000 \
  --batch-size 500 \
  --publish-concurrency 200 \
  --minimum-throughput 100 \
  --maximum-p95-batch-ms 2500
```

Rate-paced soak example:

```bash
python scripts/benchmark_event_path.py \
  --events 1 \
  --soak-seconds 3600 \
  --rate 100 \
  --minimum-throughput 50 \
  --maximum-p95-batch-ms 3000
```

The JSON result is machine-readable and the process exits `2` when any bound is
violated. Thresholds are deployment SLO inputs, not universal hardware claims.
The acceptance record contains the measured local baseline.

## Failure matrix

| Failure | Required behavior | Evidence |
|---|---|---|
| Mongo unavailable during event write | Bounded timeout, message remains pending, stale claim retries | Docker pause/unpause acceptance and unit failure test |
| Ambiguous timed-out Mongo write | Unique event key turns retry into duplicate, never second document | Real outage acceptance |
| Later rule action fails | Earlier successful action remains durable and is skipped on redelivery | Real Redis/Mongo integration test |
| HTTP receiver repeatedly fails | Retryable error, capped attempts, target circuit opens | Unit/integration tests |
| Redis restarts | REST stays available; realtime/live report OFFLINE then reconnect ONLINE | Real container stop/start acceptance |
| Slow realtime client | Drop oldest, emit explicit gap, keep bounded queue | Realtime hub tests |
| Invalid event contract | Bounded DLQ and ACK/delete from source stream | Benchmark and Redis integration tests |

Mongo `connect_timeout_ms` and `socket_timeout_ms` are independently configurable.
Set them above measured healthy p99 latency but below the outage-detection SLO.
Redis commands similarly use bounded connect and command timeouts. Recovery relies
on durable Redis pending entries and Mongo/action unique keys, not in-memory retry
state.

## Required real-service CI gate

The `Quality` workflow has a separate required job named
`Real services (MongoDB replica set, Redis, MinIO)`. It starts the digest-pinned
Compose services, initializes and verifies MongoDB replica-set primary state,
checks Redis and MinIO readiness, then runs every integration file gated by
`TEST_MONGODB_URI`, `TEST_REDIS_URL`, or `TEST_MINIO_ENDPOINT`.

The runner invokes `scripts/run_real_service_tests.py` with `--fail-on-skip`.
Missing connection settings fail before pytest starts, and any selected test that
still skips changes the job result to failure. A unit contract also fails when a
new environment-gated integration file is not added to the required manifest.

Run the same gate against already-started local services with:

```bash
TEST_MONGODB_URI='mongodb://127.0.0.1:27017/?replicaSet=rs0&directConnection=true' \
TEST_REDIS_URL='redis://127.0.0.1:6379/15' \
TEST_MINIO_ENDPOINT='127.0.0.1:9000' \
uv run --locked python scripts/run_real_service_tests.py
```

## Operational acceptance order

1. Run lint and the complete suite against dedicated Mongo/Redis/MinIO namespaces.
2. Run burst load at the expected peak event rate and assert all JSON thresholds.
3. Run rate-paced soak for the desired duration; watch Prometheus RSS/latency and
   confirm pending returns to zero.
4. Exercise one dependency at a time. Always install a recovery trap before
   stopping/pausing a container.
5. Verify `/api/system/health`, `/metrics`, Redis XPENDING, DLQ, Mongo counts, and
   action execution states after recovery.
