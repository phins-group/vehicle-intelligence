# Phase 2 Event Bus Acceptance Record

## Decision

The Redis Streams event-delivery foundation passed engineering acceptance on
2026-08-09. This accepts the versioned vision-worker publisher boundary and the
Redis-to-MongoDB event worker. It does not accept the later rules, actions,
alerts, realtime dashboard delivery, or multi-camera orchestration scope. Rules/actions
were subsequently accepted separately in
[PHASE2_POLICY_ENGINE_ACCEPTANCE.md](PHASE2_POLICY_ENGINE_ACCEPTANCE.md); the
current worker ACK point is therefore after that optional post-processing stage.
Realtime SSE/WebSocket delivery was subsequently accepted in
[PHASE2_REALTIME_ACCEPTANCE.md](PHASE2_REALTIME_ACCEPTANCE.md).

## Accepted flow

```text
VehicleEvent v1
  -> JSON EventEnvelope v1
  -> Redis XADD (bounded stream)
  -> XREADGROUP
  -> contract/coherence validation
  -> idempotent MongoDB insert
  -> transactional XACK + XDEL
```

## Automated evidence

- Envelope encode/decode round-trip and rejection of mismatched event type or
  unsupported schema.
- MongoDB failure leaves a message unacknowledged; a reclaimed retry persists and
  ACKs it.
- Duplicate delivery is acknowledged after MongoDB reports an idempotent no-op.
- Invalid JSON/contract is sent to the dead-letter path without persistence.
- A real Redis 8 container exercised publish, consumer-group read, ACK/delete,
  stale-message `XAUTOCLAIM`, and bounded DLQ insertion.
- A real Redis 8 plus MongoDB 8 run published the same event twice, persisted
  exactly one canonical MongoDB document, reported one duplicate, and ended with
  zero pending entries and an empty processed main stream.
- The complete suite passed `51 passed` with Redis 8, MongoDB 8, and MinIO
  enabled. Ruff, Python bytecode compilation, Compose validation, and the
  production event-worker image build also passed.

The default self-contained suite remains broker/database independent; real-service
tests are opt-in through `TEST_REDIS_URL` and `TEST_MONGODB_URI`.

## Operational limits

- The stream is a bounded outage buffer, not an infinite archive. Capacity must
  exceed the planned maximum outage backlog.
- One consumer group owns canonical persistence today. Deleting after ACK assumes
  no second group needs the same stream entry; future consumers should receive a
  separate stream or set `delete_after_ack=false` with an explicit retention plan.
- DLQ replay tooling, delivery-attempt limits, alerting, Redis cluster/Sentinel,
  and production soak/load testing remain future work.
