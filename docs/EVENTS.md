# Events

## Envelope

Cross-process events use a versioned envelope:

```json
{
  "id": "evt_...",
  "type": "vehicle.entered",
  "schemaVersion": 1,
  "occurredAt": "2026-08-08T13:30:00Z",
  "source": "vision-worker/sample-camera",
  "correlationId": "sample-camera:12",
  "data": {}
}
```

The envelope ID is the stable canonical event ID and `correlationId` is the
logical camera track ID. Video and RTSP workers publish through a
`VehicleEventPublisher` port. The configured adapter is either a direct local
repository publisher or Redis Streams; arbitrary dictionaries are not accepted.

## Canonical vehicle event v1

Required fields are event ID, schema version, camera snapshot, logical track ID,
event type, direction, status, vehicle evidence, UTC occurrence/creation dates,
media keys, AI model trace, and metadata. `plate` is nullable.

```json
{
  "_id": "evt_...",
  "schemaVersion": 1,
  "camera": {"id": "gate-01", "name": "Main Gate", "zone": "ZONE_A"},
  "trackId": "gate-01:10235",
  "vehicleId": null,
  "eventType": "VEHICLE_ENTER",
  "direction": "ENTER",
  "status": "CONFIRMED",
  "plate": {
    "raw": "51H12345",
    "normalized": "51H-123.45",
    "confidence": 0.96,
    "observationCount": 4,
    "corrections": []
  },
  "vehicle": {"type": "car", "confidence": 0.97, "color": null},
  "media": {
    "snapshotKey": "vehicles/2026/08/08/gate-01/evt_.../snapshot.jpg",
    "vehicleCropKey": "vehicles/2026/08/08/gate-01/evt_.../vehicle.jpg",
    "plateCropKey": "vehicles/2026/08/08/gate-01/evt_.../plate.jpg"
  },
  "ai": {
    "vehicleDetector": {"model": "vehicle-yolo", "version": "v1", "hash": null},
    "plateDetector": {"model": "vietnam-plate-yolo", "version": "v1", "hash": null},
    "ocr": {"model": "PP-OCRv5_mobile_rec", "version": "PP-OCRv5", "hash": null}
  },
  "occurredAt": "2026-08-08T13:30:00Z",
  "createdAt": "2026-08-08T13:30:01Z",
  "metadata": {}
}
```

## Event types

- `vehicle.detected`: direction is not known.
- `vehicle.entered`: trajectory crosses in the configured entry direction.
- `vehicle.exited`: trajectory crosses in the configured exit direction.
- `plate.recognized`: future intermediate evidence event; not a canonical track
  finalization event.

## Reviewed vehicle event v2

Human review is the first semantic event-document evolution. A reviewed v1
event is atomically promoted to v2. The original v1 plate fields remain as a
compatibility snapshot, while `prediction` is the immutable AI evidence,
`review` is the current revisioned operator decision, and `final` is the indexed
effective value:

```json
{
  "schemaVersion": 2,
  "status": "CONFIRMED",
  "plate": {
    "raw": "51H1234S",
    "normalized": "51H-123.4S",
    "confidence": 0.68,
    "observationCount": 3,
    "corrections": [],
    "prediction": {
      "raw": "51H1234S",
      "normalized": "51H-123.4S",
      "confidence": 0.68,
      "observationCount": 3,
      "corrections": []
    },
    "review": {
      "normalized": "51H-123.45",
      "revision": 1,
      "reviewedAt": "2026-08-09T13:00:00Z",
      "reviewedBy": {"id": "operator-01", "displayName": "Gate Operator"},
      "note": "Checked plate crop"
    },
    "final": "51H-123.45"
  }
}
```

`PUT /api/events/{eventId}/plate-review` accepts normalized or common display
forms plus `expectedRevision`. The API normalizes server-side and uses an atomic
revision predicate; stale updates return `409`. Retrying the identical request
is idempotent. `OPERATOR` and `ADMIN` may review; `VIEWER` cannot. Corrections and
confirmations produce distinct audit actions.

When a persisted `plateCropKey` resolves to an existing media object, the same
operation upserts one
deterministically identified `PLATE_OCR` dataset sample for `(eventId,
reviewRevision)`. The sample references the crop key, model trace, immutable
prediction and human label; it never embeds image bytes. Events without a plate
available crop are still reviewable but do not fabricate a training sample.

## Idempotency and evolution

The event ID is stable for retries. The semantic idempotency key is camera ID,
track ID, and event type. Consumers must tolerate redelivery. Unreviewed
historical v1 documents remain valid; the reader derives their effective value
from `plate.normalized`. Review lazily migrates only the affected document to v2,
so no destructive bulk rewrite is required.

## Plate history query

`GET /api/vehicles/search` is currently an exact plate-observation query, not a
global vehicle-identity lookup. The API normalizes `plate`, queries the indexed
canonical value, and returns events newest-first with the same opaque cursor
semantics as the event explorer:

```http
GET /api/vehicles/search?plate=51H12345&limit=50&cursor=...
```

```json
{
  "query": "51H-123.45",
  "items": [],
  "nextCursor": null
}
```

`limit` is bounded to 100. Invalid plates return `422`; invalid cursors return
`400`. Reviewed events query `plate.final` through `ix_plate_final_time`; legacy
v1 events fall back to `plate.normalized` through `ix_plate_time`. Both branches
are indexed and never load the collection into Python.
Fuzzy OCR candidates require a future indexed search representation/adapter and
must not degrade into a full scan. Matching plate observations are not merged
into one physical vehicle without independent identity evidence.

## Identity bootstrap and fingerprints

After canonical event persistence, the identity post-processor deterministically
creates one bootstrap `vehicleId` from the event ID, registers exactly one
`VehicleFingerprint`, and conditionally links `vehicle_events.vehicleId`. The
operation is idempotent under Redis redelivery. Two events carrying the same
plate therefore remain two identities until an explicit ReID decision supplies
independent evidence; plate text alone is never global identity.

`GET /api/vehicles/{vehicleId}` returns the bounded aggregate, while
`GET /api/vehicles/{vehicleId}/fingerprints?limit=...` returns immutable evidence.
Embedding vectors are absent from both event and fingerprint responses; only a
versioned/hashable embedding reference may be exposed.

## Event media access

Event documents expose durable object keys but never object bytes or long-lived
public URLs. An authenticated reader obtains temporary evidence access by event
ID:

```http
GET /api/events/evt_123/media
```

```json
{
  "eventId": "evt_123",
  "expiresAt": "2026-08-09T12:05:00Z",
  "media": {
    "snapshot": {
      "key": "vehicles/2026/08/09/gate-01/evt_123/snapshot.jpg",
      "url": "https://media.example/...signed-query...",
      "contentType": "image/jpeg",
      "status": "AVAILABLE"
    },
    "vehicleCrop": null,
    "plateCrop": {
      "key": "vehicles/2026/08/09/gate-01/evt_123/plate.jpg",
      "url": null,
      "contentType": "image/jpeg",
      "status": "MISSING"
    },
    "clip": null
  }
}
```

The server never signs a key supplied by the caller. It resolves the event,
validates only its persisted media references, and checks each object before
signing. `MISSING` means a referenced object is unexpectedly absent because of
an operational failure. Successful coordinated retention clears the public
event key before deleting the object, so normally retained-away media is `null`
rather than `MISSING`. Unknown events return `404`, an
unconfigured/unavailable storage provider returns `503`, and missing/invalid
credentials return `401`/`403` according to the platform RBAC policy.

The response carries `Cache-Control: no-store, private`. URL lifetime is fixed by
server configuration (five minutes by default, at most one hour) and cannot be
extended by a query parameter. Local Compose signs the browser-visible Nginx
origin and proxies the `/vehicle-media/` bucket path to MinIO while preserving
the signed Host, so the existing `'self'` image/media CSP remains intact.

## Redis Streams delivery

The Redis entry contains one `event` field holding the complete JSON envelope.
Default names are:

```text
stream:              vehicle.events
consumer group:      event-processors
dead-letter stream:  vehicle.events.dlq
```

Delivery is at least once:

```text
XADD by vision worker
  -> XREADGROUP by event worker
  -> validate envelope/data coherence
  -> idempotent MongoDB insert
  -> load active watchlists and enabled rules
  -> claim/execute/record matched actions
  -> XACK + optional XDEL in one Redis transaction
```

If MongoDB is temporarily unavailable, the worker does not ACK. After
`claim_idle_ms`, `XAUTOCLAIM` transfers the stale pending message to a live
consumer. The worker also leaves the message pending when policy/action
processing fails. If Mongo persisted the event before failure, redelivery still
runs policy after the duplicate insert no-op; succeeded actions are skipped and
eligible failed/stale actions resume from their durable claim.

Within one read batch, messages are grouped by camera. Each camera group remains
ordered, while up to `redis.worker_concurrency` different cameras progress in
parallel. Successful IDs are sent through one pipelined `XACK` and optional
`XDEL`; a failure leaves that message and all later messages for the same camera
pending. Stale reclaim runs at `redis.reclaim_interval_ms`, not before every
blocking read. Enabled rules are cached and prevalidated for
`rule_engine.rule_cache_ttl_seconds`; the TTL deliberately bounds configuration
staleness without weakening durable action idempotency.

Each action execution ID is the deterministic hash of event ID, rule ID, and
action ID. Alert creation adds its own unique execution index. External action
attempts carry the same execution ID as `Idempotency-Key`. A remote receiver must
honor that key because there is no atomic transaction spanning its side effect
and the worker's MongoDB success update.

Only after event persistence and policy processing complete does the worker ACK
and optionally delete the Redis entry. A malformed/unsupported envelope is
copied to the bounded DLQ and then ACKed/deleted from the main stream.

`max_length` and `dead_letter_max_length` use approximate stream caps. Operators
must size the main cap above the maximum expected outage backlog; trimming an
unprocessed entry can otherwise remove its payload even if a pending reference
still exists. Redis AOF is enabled in the local Compose service, but production
durability policy remains an operations decision.

Policy rules are evaluated from current MongoDB state when a delivery is
processed. Action IDs preserve replay safety for unchanged rule/action identity;
creating a new rule before an old event is redelivered can intentionally apply
that new rule. Historical point-in-time policy snapshots/replay controls are not
implemented yet.

## Realtime delivery

After durable persistence and policy completion, the event worker publishes the
same envelope to `vehicle.events.realtime` using Redis Pub/Sub. API instances
fan it out through authorized SSE (`/api/events/stream`) and WebSocket
(`/ws/events`) connections. This path is best effort and never replaces the
MongoDB event history. Realtime publish failure therefore does not prevent the
durable Redis Stream ACK.

Clients deduplicate by envelope ID. Short reconnects can request bounded
process-local replay with `Last-Event-ID`; an unavailable replay or slow-client
overflow produces a versioned `system.realtime.gap` control and requires REST
reconciliation. See [Realtime events](REALTIME_EVENTS.md) for authentication,
backpressure, recovery, and control contracts.
