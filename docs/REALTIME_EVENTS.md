# Realtime Events

## Scope

The Phase 2 realtime path delivers finalized canonical vehicle events to
authorized operator clients over SSE and WebSocket. It is a low-latency dashboard
signal, not a second source of truth. MongoDB event history and the paginated
`GET /api/events` API remain the recovery path after disconnects, process
restarts, or backpressure gaps.

These endpoints never carry frames or image bytes. The optional operator
preview uses a separate channel and API described in
[Live Monitor](LIVE_MONITOR.md), so image loss cannot alter canonical event
delivery or recovery.

## Runtime flow

```text
Redis Stream vehicle event
  -> event worker contract validation
  -> idempotent MongoDB persistence
  -> rules and durable actions
  -> best-effort Redis Pub/Sub notification
  -> Redis ACK lifecycle
  -> one subscriber per API process
  -> bounded local replay hub
  -> bounded queue per SSE/WebSocket client
```

Realtime publication occurs after persistence and policy processing and before
the main stream message is ACKed. A Pub/Sub failure is logged and counted but
does not prevent ACK: the event is already recoverable from MongoDB, while making
ephemeral dashboard availability part of the durable transaction would cause
duplicate policy processing and an unbounded pending backlog.

Every API replica subscribes to the configured Pub/Sub channel and fans out to
its own connected clients. The listener reconnects with capped exponential
backoff. Redis outages do not stop the REST API; health reports the realtime
source as `OFFLINE` until it recovers.

## Canonical payload

Vehicle notifications use the same `EventEnvelope` v1 as the Redis Stream, not a
dashboard-specific arbitrary dictionary:

```json
{
  "id": "evt_123",
  "type": "vehicle.entered",
  "schemaVersion": 1,
  "occurredAt": "2026-08-09T12:00:00Z",
  "source": "vision-worker/gate-01",
  "correlationId": "gate-01:10235",
  "data": {
    "_id": "evt_123",
    "eventType": "VEHICLE_ENTER",
    "camera": {"id": "gate-01", "name": "Main Gate", "zone": "ZONE_A"},
    "plate": {"normalized": "51H-123.45", "confidence": 0.96}
  }
}
```

Control messages are separately identifiable versioned envelopes:

- `system.realtime.ready`: connection established and recovery endpoint;
- `system.realtime.heartbeat`: WebSocket liveness signal;
- `system.realtime.gap`: local replay was unavailable or a slow client lost
  queued events.

Clients must retain the latest **vehicle event** ID, ignore control IDs for
replay, and deduplicate vehicle events by ID.

## SSE

Endpoint:

```http
GET /api/events/stream
Authorization: Bearer <key>
Accept: text/event-stream
Last-Event-ID: evt_123
```

Example:

```bash
curl -N \
  -H "Authorization: Bearer ${VEHICLE_API_KEY}" \
  -H "Last-Event-ID: evt_123" \
  http://localhost:8000/api/events/stream
```

Each vehicle notification has an SSE `id` equal to the canonical event ID and a
JSON `data` field containing the complete envelope. Heartbeats are SSE comments,
so they do not replace the browser's `Last-Event-ID`. Native browser
`EventSource` cannot attach a Bearer header; use an authenticated fetch-stream
client until the platform has a secure same-site session-cookie provider. API
keys are never accepted in query parameters.

## WebSocket

Endpoint:

```text
wss://host/ws/events
```

Non-browser clients may send the normal `Authorization: Bearer ...` handshake
header. Browser clients authenticate as the first frame within the configured
timeout:

```json
{
  "type": "authenticate",
  "token": "<raw-key>",
  "lastEventId": "evt_123"
}
```

The token travels inside the TLS-protected WebSocket payload rather than the URL
or access log. The server does not retain or echo it. Invalid/missing credentials
close with `4401`; insufficient permission uses `4403`; invalid subscription
input uses `4400`; a disabled realtime service uses `1013`.

## Replay and gap recovery

The API retains only `replay_size` recent events in process memory. A reconnect
with a buffered `Last-Event-ID` receives later buffered events before live ones.
If the ID has expired, belongs to another API replica, or the client queue
overflowed, the API emits:

```json
{
  "type": "system.realtime.gap",
  "data": {
    "reason": "replay_unavailable",
    "droppedEvents": 0,
    "lastAvailableEventId": "evt_456",
    "recoveryEndpoint": "/api/events"
  }
}
```

On any gap, the client must reconcile through `GET /api/events` and then resume
live delivery. The local replay buffer intentionally does not pretend to be a
durable cross-replica log.

## Backpressure

Every connection owns a fixed `client_queue_size`. Publishing never awaits a
slow socket. When its queue is full, the oldest queued event is removed, the
newest event is retained, and a gap control is delivered before the remaining
events. This bounds memory by:

```text
connected clients * client_queue_size * average envelope size
```

Ingress connection limits and per-principal quotas still belong at the reverse
proxy/API gateway; the current API only bounds each accepted connection and the
authentication wait.

## Configuration and health

```yaml
realtime:
  enabled: true
  redis_channel: vehicle.events.realtime
  client_queue_size: 50
  replay_size: 500
  heartbeat_seconds: 15
  websocket_auth_timeout_seconds: 5
  broker_poll_seconds: 1
  reconnect_initial_seconds: 0.5
  reconnect_max_seconds: 30
```

`GET /api/realtime/health` requires `READ_PLATFORM` and reports source state,
subscriber count, received/distributed/duplicate events, client drops,
reconnects, invalid messages, and the last event time. Public system health only
reports `DISABLED`, `STARTING`, `ONLINE`, `OFFLINE`, or `STOPPED`.

## Current limits

- Only finalized vehicle events are published today; dedicated `alert.created`
  and `camera.online/offline` realtime contracts are not connected yet.
- Redis Pub/Sub is best effort and has no outage backlog; MongoDB reconciliation
  is mandatory.
- Replay is process-local, so load balancers do not guarantee replay affinity.
- A crash after Pub/Sub publish but before Redis Stream ACK can redeliver the same
  event; clients must deduplicate by canonical ID.
- There is no per-camera server-side subscription filter, connection quota,
  distributed presence, or Prometheus export yet.
- The Angular dashboard and backend OIDC/JWKS Bearer validation are implemented.
  The console still lacks a browser authorization-code/PKCE or cookie-session
  flow; operators currently provide a tab-scoped bearer credential.
