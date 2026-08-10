# Phase 2 Policy Engine Acceptance Record

## Decision

The watchlist, declarative rule, durable action, and alert foundation passed
engineering acceptance on 2026-08-09. This accepts central processing from a
canonical Redis-delivered `VehicleEvent` through policy completion and broker
ACK. Authentication/RBAC and manual-mutation audit logs were subsequently
accepted in [PHASE2_SECURITY_ACCEPTANCE.md](PHASE2_SECURITY_ACCEPTANCE.md). This
record still does not accept the Angular UI, production barrier credentials, or
load/soak readiness. Realtime transport was subsequently accepted in
[PHASE2_REALTIME_ACCEPTANCE.md](PHASE2_REALTIME_ACCEPTANCE.md).

## Accepted flow

```text
VehicleEvent v1 delivery
  -> idempotent MongoDB event persistence
  -> normalized active watchlist lookup at occurredAt
  -> bounded enabled-rule evaluation in priority order
  -> deterministic MongoDB action claim
  -> alert / structured log / guarded HTTP action
  -> durable succeeded, failed, or exhausted action state
  -> Redis ACK
```

## Automated evidence

- Vietnamese plate normalization is applied to watchlist create/update and alert
  search; invalid plates and invalid validity intervals are rejected.
- Watchlist activity respects enabled state and inclusive `validFrom`/
  `validUntil` at event occurrence time.
- Rule validation rejects unknown fields, invalid operator/value shapes,
  URL-embedded credentials, unsupported HTTP methods, and invalid severities.
- Rule matching covers watchlist membership plus camera, direction, event type,
  event status, normalized plate, vehicle type, and color context.
- Temporal duplicate delivery executes one deterministic action and creates one
  alert; a retryable handler failure is reclaimed and then succeeds.
- HTTP adapter tests verify exact hostname allowlisting and stable
  `Idempotency-Key` propagation.
- Alert list filters/cursor pagination and idempotent acknowledge/resolve lifecycle
  are covered through FastAPI integration tests.
- Real MongoDB 8 tests verified all policy indexes, optimistic replacements,
  atomic action claims, and one alert/execution after processing the same event
  twice.
- Real Redis 8 plus MongoDB 8 tests verified duplicate stream deliveries persist
  one event, execute one action, create one alert, ACK both deliveries, and leave
  zero pending entries.
- The self-contained suite passed `73 passed, 7 skipped`; the complete suite with
  MongoDB 8, Redis 8, and MinIO passed `80 passed`.
- Ruff, Python bytecode compilation, Compose validation, editable-package install,
  and both production API/event-worker image builds passed.
- Container smoke tests reported `policyEngine: available`, normalized a posted
  watchlist plate to `51H-123.45`, validated/stored a rule, removed both temporary
  records, and initialized a one-shot event worker with zero failures.

The one emitted warning is an upstream FastAPI/Starlette TestClient deprecation
for `httpx`; it does not represent an application test failure.

## Failure and idempotency evidence

An event is not ACKed when policy processing fails. On redelivery the event insert
may be a duplicate, but policy processing still resumes. MongoDB `_id` uniqueness
and an atomic claim filter ensure only an eligible failed or stale execution is
reclaimed. Completed and terminal/exhausted actions are skipped.

External actions carry the deterministic execution ID to the receiver. The
receiver must honor it because a crash after a successful remote side effect but
before the MongoDB success update can cause a safe retry only when both ends
deduplicate.

## Operational limits

- Policy execution is currently composed only in the Redis event worker; direct
  CLI publishers persist events without applying rules.
- Current rules are read at delivery time. There is no point-in-time policy
  snapshot or historical policy replay command.
- External HTTP actions are disabled by default and have no managed secret,
  signature, mTLS, per-target circuit breaker, or dedicated barrier protocol.
- The hostname allowlist must be paired with production egress controls and
  trusted DNS.
- Authentication/RBAC and actor-oriented audit logs were accepted separately;
  centralized identity lifecycle, action signing, and production TLS/secret
  delivery remain outside this policy milestone.
- Actions execute sequentially per event; horizontal/load/soak limits are not yet
  established.
