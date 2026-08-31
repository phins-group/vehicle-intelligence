# Observability and Retention

## Runtime boundaries

Observability is split by process rather than hidden behind the API:

```text
camera workers -> latest camera_health -> API /metrics -> Prometheus
API requests   -> normalized HTTP metrics -----------> Prometheus
API requests   -> OTLP/HTTP spans -> Collector ------> trace backend/debug
Collector      -> internal :8888/metrics ------------> Prometheus
event worker   -> worker :9102/metrics --------------> Prometheus
retention pass -> worker :9101/metrics --------------> Prometheus
```

Prometheus uses its normal pull model. The API exposes `/metrics` on its internal
service port and the retention worker exposes a dedicated internal metrics port.
The bundled Nginx configuration does not proxy either endpoint. This follows the
official Prometheus Python client's [ASGI guidance](https://prometheus.github.io/client_python/exporting/http/asgi/)
and Prometheus [installation model](https://prometheus.io/docs/prometheus/latest/installation/).

OpenTelemetry is optional. When enabled, the API creates a bounded-ratio,
parent-based tracer and exports spans through OTLP/HTTP with a finite timeout.
The implementation follows the official Python [exporter documentation](https://opentelemetry.io/docs/languages/python/exporters/),
FastAPI [instrumentation documentation](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html),
and the OTLP [exporter specification](https://opentelemetry.io/docs/specs/otel/protocol/exporter/).

## Metrics contract

HTTP labels are deliberately bounded:

```text
method = GET | POST | PUT | DELETE | ...
route  = FastAPI template, for example /api/events/{event_id}
status = HTTP status code
```

The raw URL, event ID, plate, actor, request ID, RTSP URL, and exception text are
never metric labels. Unmatched paths collapse to `UNMATCHED`.

Implemented API/camera metrics include:

- `http_requests_total` and `http_request_duration_seconds`;
- `camera_online`, source/decode/inference FPS, queue depth, and active tracks;
- decoded/sampled/dropped frames, reconnects, vehicle/plate detections;
- OCR requests/success and vehicle/plate/OCR stage latency;
- process, Python runtime, and garbage-collector metrics.

Camera counters are cumulative worker values stored in the latest
`camera_health` document. The API renders a fresh collector from that latest
state, so an API restart does not reset the visible camera totals. Camera ID is
the only per-camera label and is an explicitly bounded configured resource.

The retention worker exports:

- `retention_objects_deleted_total{kind=...}`;
- `retention_object_failures_total{kind=...}`;
- `retention_events_deleted_total`;
- `retention_run_duration_seconds`.

The event worker exports cumulative read/reclaim/persist/duplicate/DLQ,
persistence/policy failure, action, and realtime publication counters from a
scrape-time snapshot of its in-process stats.

Collection failures increment a bounded `observability_collection_errors_total`
counter and do not make `/metrics` fail.

## Traces and structured logs

`observability.opentelemetry_enabled=false` is the safe local default. Enabling
it requires a credential-free HTTP(S) OTLP endpoint; URL user-info, query tokens,
fragments, and non-HTTP schemes are rejected. Sensitive exporter headers use a
secret configuration value and are not logged.

FastAPI route spans exclude `/metrics`, `/api/system/health`, `/livez`, and
`/readyz`. The configured service name/version are resource attributes. A valid
active span adds lowercase 32-character `trace_id` and 16-character `span_id`
fields to structured JSON logs. Request IDs remain a separate correlation
signal and are added as a span attribute.

The bundled development collector uses the debug exporter only as local
acceptance evidence. `infrastructure/otel/collector.production.yml` instead
exports to a durable OTLP/HTTP backend with indefinite retry, fsync, a
byte-bounded disk-backed sending queue, and start/rebound compaction. Its basic
internal metrics are exposed on port 8888 so Prometheus can alert on collector
availability, queue saturation, enqueue loss, and sustained export failure.
Notification routing remains owned by the deployment Alertmanager/monitoring
service.

## Retention ownership

Canonical media is not expired by a broad MinIO lifecycle rule. The application
worker coordinates MongoDB references, dataset pins, object deletion, and event
deletion:

```text
old event + media key
  -> atomic Mongo lease; public media key moves to retention.media.*
  -> delete object (missing is idempotent success)
  -> mark DELETED, retaining the original key as audit metadata
  -> when every canonical media key is absent and no lease is active
  -> delete event only after its event-retention window
```

The lease states are:

```text
absent/FAILED -> DELETING -> DELETED
                     |
                     +---- storage failure -> FAILED + public key restored
```

Each pass has a configured batch limit. A `DELETING` lease can be reclaimed only
after `claim_stale_seconds`, so overlapping workers do not normally delete the
same object while a crashed worker cannot strand work forever. The worker uses
UTC native dates and refuses a naive clock.

`READY`, `EXPORTING`, and `EXPORT_FAILED` dataset samples pin both their
referenced `imageKey` and their source event. Pins apply to every canonical media
kind, not only plate crops. After an exporter verifies the immutable artifact and
changes the sample to `EXPORTED`, a later retention pass can remove the media and
then the event. Dataset documents and image bytes remain separate.

## MinIO lifecycle scope

The worker idempotently reconciles only two owned lifecycle rules:

```text
debug/     -> debug_images_days
temporary/ -> debug_images_days
```

Rules whose IDs do not start with `vip-managed-` are preserved. No lifecycle
rule is installed on `vehicles/`, because such a rule could bypass a dataset pin.
The MinIO object-management model is documented in its official
[object lifecycle guidance](https://min.io/docs/minio/linux/administration/object-management.html).

Abandoned multipart uploads are a server concern. Compose sets MinIO's bounded
stale-upload expiry/cleanup settings. This avoids pretending an incompatible
bucket rule protects canonical prefixes while keeping application retention
authoritative.

## Configuration

```yaml
observability:
  prometheus_enabled: true
  prometheus_path: /metrics
  retention_metrics_port: 9101
  event_worker_metrics_port: 9102
  opentelemetry_enabled: false
  otlp_traces_endpoint: null
  service_name: vehicle-intelligence-api
  service_version: 0.1.0
  trace_sample_ratio: 0.10
  export_timeout_seconds: 5

retention:
  enabled: false
  worker_interval_seconds: 3600
  batch_size: 100
  claim_stale_seconds: 600
  vehicle_events_days: 365
  snapshots_days: 30
  vehicle_crops_days: 30
  plate_crops_days: 30
  event_clips_days: 14
  debug_images_days: 7
  minio_lifecycle_enabled: true
```

Validation requires canonical event retention to be at least as long as every
canonical media window. This guarantees the event still exists while its media
lease is coordinated.

## Operations

Start the optional profiles:

```bash
docker compose --profile observability --profile maintenance --profile event-driven up -d \
  api event-worker prometheus otel-collector retention-worker
```

Run one bounded host-native pass:

```bash
VIP_RETENTION__ENABLED=true \
VIP_MONGODB__ENABLED=true \
VIP_STORAGE__BACKEND=minio \
vehicle-retention-worker --once
```

Useful acceptance queries:

```promql
up{job="vehicle-intelligence-api"}
up{job="vehicle-intelligence-retention"}
up{job="vehicle-intelligence-otel-collector"}
rate(http_requests_total[5m])
histogram_quantile(0.95, sum by (le, route) (
  rate(http_request_duration_seconds_bucket[5m])
))
increase(retention_object_failures_total[1h])
otelcol_exporter_queue_size / clamp_min(otelcol_exporter_queue_capacity, 1)
```

Bundled alert thresholds are starting points. The durable trace backend,
site-specific thresholds, and notification routing remain deployment policy.
