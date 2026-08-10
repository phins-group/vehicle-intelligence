# Phase 2 Observability and Retention Acceptance Record

## Decision

The observability and retention milestone passed engineering acceptance on
2026-08-09. It accepts low-cardinality Prometheus signals for API, cameras, and
the retention worker; optional OTLP/HTTP FastAPI traces with structured-log
correlation; enriched persisted camera telemetry; and leased, bounded,
dataset-aware MongoDB/MinIO retention.

It does not accept a production trace backend, alert-rule/SLO policy, canonical
media expiration through a broad object-store lifecycle rule, audit/legal
archive deletion, or distributed transactions across MongoDB and MinIO.

## Accepted flow

```text
HTTP request -> normalized route metric -> API /metrics -> Prometheus
            -> sampled FastAPI span -> OTLP/HTTP -> Collector
            -> trace/span IDs in JSON log context

camera worker -> cumulative counters/latest latency -> camera_health
              -> API dynamic camera metrics -> Prometheus

old event -> atomic media lease -> dataset pin check -> object delete
          -> DELETED/FAILED state -> coordinated event delete
          -> retention-worker :9101/metrics -> Prometheus
```

Prometheus integration follows the official client [ASGI export guidance](https://prometheus.github.io/client_python/exporting/http/asgi/)
and [instrumentation model](https://prometheus.github.io/client_python/instrumenting/).
Tracing follows the official OpenTelemetry Python [exporter documentation](https://opentelemetry.io/docs/languages/python/exporters/),
FastAPI [instrumentation documentation](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html),
and [OTLP exporter specification](https://opentelemetry.io/docs/specs/otel/protocol/exporter/).

## Automated and runtime evidence

- Ruff and Python bytecode compilation passed across source, entry points, and
  tests.
- The complete real-service suite passed **146 tests** against MongoDB 8, Redis
  8, and MinIO with no skips. Coverage includes normalized metric labels,
  in-memory OpenTelemetry span/log correlation, configuration security bounds,
  enriched camera health, real Mongo health persistence, bounded leases,
  concurrent-claim exclusion, stale reclaim, failure restoration, dataset media
  and event pins, real object deletion, event ordering, and lifecycle
  reconciliation.
- A real MinIO integration test exposed two operational edge cases before
  acceptance: a new bucket reports no lifecycle configuration as an error, and
  the running MinIO release rejects a standalone S3 incomplete-multipart action.
  The adapter now treats the first as empty state and manages only the two valid
  `debug/`/`temporary/` expiration rules. Abandoned multipart cleanup is an
  explicit MinIO server setting instead of an untrue bucket-policy guarantee.
- Real retention acceptance deleted three unpinned object kinds while preserving
  a `READY` dataset-pinned snapshot and its source event. After changing the
  sample to `EXPORTED`, the next pass deleted the snapshot and then exactly one
  event. All four scoped MinIO objects were absent afterward.
- A separate real-Mongo lease test proved an active lease cannot be claimed by a
  concurrent worker, a stale lease can be reclaimed, a failed delete restores
  the public media key, and a later successful retry records `DELETED` without
  exposing that key.
- `vehicle-retention-worker --once` completed a real isolated MongoDB/MinIO pass
  and reconciled lifecycle. Its scoped smoke database/bucket were removed after
  verification.
- The rebuilt API exposed normalized metrics such as
  `/api/events/{event_id}`—never the requested event ID. Prometheus returned
  `up=1` for both `vehicle-intelligence-api` and
  `vehicle-intelligence-retention`; the worker's retention histogram had one
  observed pass.
- With 100% acceptance sampling, the OpenTelemetry Collector 0.153.0 received
  one API resource-span group containing four spans. Health/metrics endpoints
  were excluded as configured.
- Compose validation and rebuilt API/retention production images passed. The
  running API, MongoDB, Redis, MinIO, Prometheus, collector, retention worker,
  and Nginx web service were healthy/running at acceptance time.
- Strict application/spec TypeScript typecheck passed. All **30 Vitest tests**
  passed. The Angular production build remained **364.79 kB raw / 99.08 kB
  estimated initial transfer**, and the production dependency audit reported
  zero vulnerabilities.

The suite emits one existing FastAPI/Starlette TestClient deprecation warning.
The host Angular build used unsupported odd Node 23, which disabled only an
optional compiler cache; it did not affect build output.

## Security and operational limits

HTTP metrics use method, route template, and status only. Camera metrics use the
configured bounded camera ID and fixed stage/kind values. Plates, event IDs,
request IDs, actor IDs, arbitrary URLs, errors, and secrets are not labels.
OTLP endpoint validation rejects URL credentials, query tokens, fragments, and
non-HTTP schemes; exporter headers remain secret configuration.

Retention is idempotent but not a distributed transaction. It first hides the
normal event media key, then deletes the object, then records the terminal
state. This makes a crash recoverable and prevents the API from issuing a URL
for an object being removed. Every pass and claim set is bounded. Canonical event
retention cannot be configured shorter than any canonical media window.

`READY` dataset samples currently act as indefinite pins until an explicit
export workflow advances their status. Audit logs, alerts, action executions,
and legal archives have independent policies and are not deleted by this worker.
Production must add alert rules/SLOs, authenticated Prometheus/collector network
boundaries, a durable trace backend, backups, and a business-approved retention
schedule.
