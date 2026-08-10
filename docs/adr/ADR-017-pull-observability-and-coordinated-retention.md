# ADR-017: Pull Observability and Coordinated Retention

- Status: Accepted
- Date: 2026-08-09

## Context

The platform needs cross-process operational signals without coupling domain
code to a telemetry vendor. Media retention also cannot be implemented as a
blind object-store expiration rule: a `READY` training sample may legitimately
pin an image after the normal evidence window, and MongoDB must not advertise an
object that has already disappeared.

## Decision

1. Prometheus pulls a low-cardinality API endpoint and a separate retention
   worker endpoint. Camera metrics are rendered from persisted latest health.
2. FastAPI tracing is optional OpenTelemetry with OTLP/HTTP export, bounded
   sampling and timeout, and service resource metadata. Structured logs attach
   active trace/span IDs.
3. The retention application owns leased, bounded coordination across MongoDB
   and the `MediaObjectCleaner` port. Canonical events are deleted only after all
   canonical media references are cleared and no dataset sample pins the event.
4. MinIO lifecycle owns only `debug/` and `temporary/`. Rules outside the
   `vip-managed-` namespace are preserved. Canonical `vehicles/` objects are not
   lifecycle-expired.
5. Abandoned multipart cleanup stays a MinIO server setting rather than a fake
   application guarantee.

## Consequences

- Domain/application components remain independent of Prometheus, OpenTelemetry,
  MongoDB, and MinIO SDKs; small telemetry protocols sit at orchestration edges.
- Metric dimensions are predictable and safe for time-series storage.
- Multiple retention workers can overlap safely through atomic leases, and
  crashed leases are recoverable.
- Object deletion and MongoDB mutation are not one distributed transaction.
  Moving the public key into lease state first prevents clients from receiving a
  stale media promise; failures restore the key for retry.
- A pinned prefix cannot be accidentally expired by the managed lifecycle rules.
- Production still has to choose alert rules, trace storage, archival/legal
  event retention, and Prometheus high availability.
