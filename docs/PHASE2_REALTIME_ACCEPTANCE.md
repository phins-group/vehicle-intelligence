# Phase 2 Realtime Event Delivery Acceptance Record

## Decision

The authorized SSE/WebSocket realtime delivery foundation passed engineering
acceptance on 2026-08-09. This accepts low-latency fan-out of durably processed
canonical vehicle events, bounded per-client backpressure, short reconnect replay,
explicit gap recovery, and Redis subscriber reconnect behavior. It does not
accept the Angular dashboard, durable realtime replay, or production connection
load/soak limits.

## Accepted flow

```text
Redis Stream EventEnvelope v1
  -> MongoDB persistence
  -> policy/action completion
  -> Redis Pub/Sub EventEnvelope v1
  -> reconnecting API subscriber
  -> bounded process-local replay hub
  -> authorized SSE / WebSocket connection
```

MongoDB remains the source of truth. Pub/Sub is explicitly best effort; its
failure does not block the durable stream ACK. Clients recover an explicit gap
through the paginated event API and deduplicate by canonical event ID.

## Automated evidence

- Unit tests verify bounded queues drop the oldest event, retain newer events,
  and emit a `slow_consumer` gap before remaining deliveries.
- Known `Last-Event-ID` values replay later buffered events; expired/foreign IDs
  emit `replay_unavailable` with the REST recovery endpoint.
- Duplicate event IDs are suppressed within the bounded API replay window.
- A fake broker failure/recovery verifies capped reconnect, source health state,
  invalid-contract isolation, and later successful delivery.
- Event-worker tests prove notification occurs after durable processing and a
  realtime publisher failure still persists and ACKs the canonical event.
- API integration tests verify SSE requires Bearer authentication, WebSocket
  header and browser first-frame authentication, generic `4401` rejection,
  canonical envelope delivery, replay, gap controls, and authorized health.
- Real Redis 8 testing published one canonical envelope through Pub/Sub and
  delivered it independently to two local API subscribers.
- The self-contained suite passed `87 passed, 9 skipped`; the complete suite with
  MongoDB 8, Redis 8, and MinIO passed `96 passed`.
- Ruff, Python bytecode compilation, Compose validation, and production
  API/event-worker image builds passed.
- Docker smoke testing reported realtime `ONLINE`, opened an authenticated SSE
  stream, published a canonical event through the durable Redis Stream, observed
  the same envelope on SSE, and retrieved the persisted event from MongoDB over
  REST. Realtime health reported one received/distributed event with zero drops,
  reconnects, source failures, or invalid messages.
- The temporary smoke event was deleted after verification; processed stream
  entries were ACKed/deleted by the normal worker lifecycle.

The one emitted warning is an upstream FastAPI/Starlette TestClient deprecation
for `httpx`; it does not represent an application test failure.

## Delivery and failure semantics

Realtime notification occurs only after persistence and successful policy
processing. It is sent before the main Redis Stream ACK, so a worker crash in
that narrow interval can emit a duplicate on redelivery. Stable event IDs make
client and API-window deduplication deterministic.

An API subscriber reconnects Redis with capped exponential backoff without
blocking REST startup or event queries. SSE/WebSocket clients have fixed queues;
a slow connection cannot block broker consumption or another client. Overflow
prioritizes recent events and signals mandatory MongoDB-backed reconciliation.

## Operational limits

- This milestone publishes finalized vehicle events; alert and camera-health
  realtime topics are not yet wired to the fan-out publisher.
- Redis Pub/Sub has no outage backlog, and process-local replay does not span API
  replicas or restarts.
- Native browser `EventSource` cannot set the current Bearer header; clients must
  use authenticated fetch streaming until secure session cookies/OIDC exist.
- WebSocket first-frame authentication is timeout-bounded, but connection quotas,
  per-principal limits, ingress rate limits, and distributed presence are not
  implemented.
- There are no server-side camera/event filters, compression negotiation,
  Prometheus metrics, or production connection load/soak results yet.
- The Angular operator console was implemented later under the separate
  [operator dashboard acceptance](PHASE2_OPERATOR_DASHBOARD_ACCEPTANCE.md).
