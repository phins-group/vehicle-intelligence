# Phase 2 Production Validation Acceptance

Accepted on 2026-08-10.

## Implemented

- Configurable bounded Mongo connect/socket timeouts across API, event, camera,
  policy, review, audit, and retention repositories.
- Real-service resilience tests for partial policy completion, durable action
  retry, Redis pending reclaim, realtime reconnect, duplicate suppression, and
  poison-message DLQ isolation.
- Reusable thresholded Redis-to-Mongo burst/soak benchmark with duplicate load,
  invalid-contract injection, cleanup, throughput/p95/error/RSS checks, and
  machine-readable output/exit status.

## Tested

- Complete lint passed and 155 Python tests passed against real Mongo replica set,
  Redis, and MinIO with no skips. Angular typecheck, 30 tests, and production
  build remained green from the preceding acceptance.
- Burst: 5,000 canonical events, 500 duplicates, one invalid event; 5,000 Mongo
  documents, 0 pending, DLQ 1, 233.24 events/s, p95 batch 2,341.40 ms, error 0%,
  RSS growth 18.69 MB.
- Paced soak: 6,000 events over a 60-second 100 events/s ingress period plus
  drain/duplicate/DLQ verification; 69.84 end-to-end events/s, p95 batch
  2,520.13 ms, error 0%, RSS growth 17.47 MB, pending 0.
- Mongo was paused during an event insert. The worker detected a bounded timeout
  and left one pending entry. After unpause it reclaimed exactly once; the first
  write had an ambiguous network outcome, so the unique index classified the
  retry as duplicate while Mongo contained exactly one event.
- A two-action rule created its alert, failed HTTP with 503, retained the Redis
  message, reclaimed it, skipped the completed alert action, retried HTTP once,
  and completed with exactly one alert.
- Redis was stopped and restarted. API REST health remained reachable; Live
  Monitor and realtime transitioned `OFFLINE` then returned `ONLINE`.

## Known limitations

- The recorded soak is a bounded CI/development baseline, not a multi-hour
  hardware certification. The same tool supports longer environment-specific
  runs.
- Single-node replica-set acceptance validates transactional semantics and
  recovery, not multi-node failover/election latency.
- External receiver load was functionally validated; physical barrier vendor
  hardware still requires a site-specific safety acceptance.

## Next

Phase 2 is accepted. Begin Phase 3 with explicit logical vehicle identity,
fingerprints, and versioned embedding/vector ports.
