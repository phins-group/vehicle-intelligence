# MongoDB Schema

## Design inputs

Current access patterns are recent-event/alert cursor pagination, exact effective
plate history, human-review queues, dataset-feedback export, camera history,
event lookup, active watchlist lookup, prioritized rule evaluation, durable
action claiming, and idempotent writes. Events are
append-heavy and read-heavy after creation. Camera and rule display data are
intentionally snapshotted into historical records so rendering needs no lookup
and survives later renames.

## Collections

### `vehicle_events`

This is the canonical high-write history collection. Shape is defined in
[EVENTS.md](EVENTS.md). It is self-contained for list/detail rendering; media are
object keys. Plate observations, trajectories, frames, and embeddings are not
embedded. The Phase 1 repository performs `insert_one` and treats duplicate
semantic keys as an idempotent no-op.

Indexes:

```javascript
db.vehicle_events.createIndex(
  {"camera.id": 1, trackId: 1, eventType: 1},
  {unique: true, name: "uq_event_track_type"}
)
db.vehicle_events.createIndex(
  {"plate.normalized": 1, occurredAt: -1},
  {name: "ix_plate_time", partialFilterExpression: {"plate.normalized": {$type: "string"}}}
)
db.vehicle_events.createIndex(
  {"plate.final": 1, occurredAt: -1},
  {name: "ix_plate_final_time", partialFilterExpression: {"plate.final": {$type: "string"}}}
)
db.vehicle_events.createIndex(
  {"camera.id": 1, occurredAt: -1},
  {name: "ix_camera_time"}
)
db.vehicle_events.createIndex(
  {vehicleId: 1, occurredAt: -1},
  {name: "ix_vehicle_time", partialFilterExpression: {vehicleId: {$type: "string"}}}
)
db.vehicle_events.createIndex({eventType: 1, occurredAt: -1}, {name: "ix_type_time"})
db.vehicle_events.createIndex({occurredAt: -1, _id: -1}, {name: "ix_time_cursor"})
db.vehicle_events.createIndex(
  {status: 1, occurredAt: -1, _id: -1},
  {name: "ix_status_time_cursor"}
)
```

The cursor index makes `(occurredAt, _id)` deterministic when timestamps tie.
The status index now owns the demonstrated `NEEDS_REVIEW` queue. Reviewed v2
documents use `plate.final`; historical v1 documents lack that field and use the
legacy normalized index. Human review is the only controlled event amendment:
the update predicate includes `plate.review.revision`, preserves the AI evidence,
and promotes the document to schema v2. Direction remains unindexed until query
telemetry justifies its write cost.

The retention worker adds `ix_retention_oldest` on `(occurredAt, _id)` ascending
for deterministic oldest-first bounded claims. During a media deletion it moves
the public key into a bounded lease subdocument before touching object storage:

```javascript
{
  media: {snapshotKey: null},
  retention: {
    media: {
      snapshot: {
        state: "DELETING", // or FAILED / DELETED
        key: "vehicles/2026/08/09/gate-01/evt_123/snapshot.jpg",
        leaseId: "ret_...",
        errorCode: null,
        updatedAt: ISODate("2026-08-09T13:00:00Z")
      }
    }
  }
}
```

Lease IDs are temporary and disappear on terminal state. The original key stays
in `DELETED` retention metadata as bounded cleanup evidence, not as a client
media promise. Failed deletion restores `media.*Key` for retry.

### `cameras`

This implemented configuration collection is read by the API and supervisor.
The document contains sampling/vision config, ROI/line geometry, direction,
enabled state, timestamps, `schemaVersion`, and an optimistic `revision`.
`stream.rtspUrlEncrypted` is an AES-256-GCM token; neither plaintext nor the
token is exposed by public schemas.

Representative shape:

```javascript
{
  _id: "gate-01",
  schemaVersion: 1,
  revision: 3,
  name: "Main Gate",
  stream: {rtspUrlEncrypted: "v1.primary...", fpsLimit: 6},
  location: {name: "Main Entrance", zone: "ZONE_A"},
  direction: "BOTH",
  vision: {vehicleConfidence: 0.4, plateConfidence: 0.45},
  geometry: {
    vehicleRoi: [[120, 300], [1700, 300], [1900, 1050]],
    crossingLine: [[0, 600], [1920, 600]],
    crossingPositiveToNegative: "ENTER",
    finalizeOnCrossing: false
  },
  enabled: true,
  metadata: {},
  createdAt: ISODate("2026-08-09T00:00:00Z"),
  updatedAt: ISODate("2026-08-09T00:01:00Z")
}
```

Indexes:

```javascript
db.cameras.createIndex(
  {enabled: 1, name: 1, _id: 1},
  {name: "ix_enabled_name"}
)
db.cameras.createIndex({updatedAt: -1}, {name: "ix_updated_at"})
```

Updates replace the bounded document only when `_id` and the expected `revision`
match. A fresh encryption nonce is used even when the RTSP value is retained.
Configured-camera admission also owns one bounded `system_config` document:

```javascript
{_id: "camera-capacity", reservedCount: 12}
```

Create reserves a slot with a conditional `findOneAndUpdate` before inserting;
delete releases it. In production both writes share the request transaction, so
concurrent API replicas cannot exceed the configured limit. The counter is not a
public camera statistic and must not be edited independently of `cameras`.

### `camera_health`

This is operational latest state, not append-only frame telemetry. `_id` is the
camera ID and `replaceOne(..., upsert=true)` keeps exactly one document per
camera. It contains `status`, source/decode FPS, queue depth, cumulative drops,
reconnect/failure counters, stream epoch, `lastFrameAt`, and `updatedAt`. Its
bounded `vision` object contains cumulative decoded/sampled frames,
vehicle/plate detections, OCR requests/success, events created, active-track
count, inference FPS, and latest cumulative-average vehicle/plate/OCR latency.

```javascript
db.camera_health.createIndex(
  {status: 1, updatedAt: -1},
  {name: "ix_health_status_updated"}
)
```

### `watchlists`

One bounded document represents one normalized plate/list association. It owns
optional validity dates, enabled state, metadata, `schemaVersion`, and optimistic
`revision`; it does not embed vehicle event history.

```javascript
{
  _id: "blocked-51h-12345",
  schemaVersion: 1,
  revision: 1,
  plate: "51H-123.45",
  listType: "BLACKLIST",
  enabled: true,
  validFrom: ISODate("2026-08-08T17:00:00Z"),
  validUntil: null,
  metadata: {reason: "security review"},
  createdAt: ISODate("2026-08-09T00:00:00Z"),
  updatedAt: ISODate("2026-08-09T00:00:00Z")
}
```

```javascript
db.watchlists.createIndex(
  {plate: 1, enabled: 1, validFrom: 1, validUntil: 1},
  {name: "ix_watchlist_plate_active"}
)
db.watchlists.createIndex(
  {listType: 1, enabled: 1, updatedAt: -1},
  {name: "ix_watchlist_type_enabled"}
)
```

The first index bounds the event-time lookup by canonical plate before validity
predicates are applied. Multiple types/validity windows for one plate are
allowed; `_id` owns API idempotency rather than a unique plate constraint.

### `rules`

Rules are small, revisioned configuration documents. Conditions and actions are
embedded because they are bounded (`32` and `16` respectively), replaced as one
aggregate, and always read together.

```javascript
{
  _id: "rule-gate-blacklist",
  schemaVersion: 1,
  revision: 2,
  name: "Alert for blacklisted entry",
  enabled: true,
  priority: 100,
  conditions: [
    {field: "watchlist", operator: "CONTAINS", value: "BLACKLIST"},
    {field: "direction", operator: "EQ", value: "ENTER"}
  ],
  actions: [
    {
      id: "critical-alert",
      type: "CREATE_ALERT",
      parameters: {severity: "CRITICAL", message: "Blacklisted entry"}
    }
  ],
  metadata: {},
  createdAt: ISODate("2026-08-09T00:00:00Z"),
  updatedAt: ISODate("2026-08-09T00:02:00Z")
}
```

```javascript
db.rules.createIndex(
  {enabled: 1, priority: -1, name: 1, _id: 1},
  {name: "ix_rules_enabled_priority"}
)
db.rules.createIndex({updatedAt: -1}, {name: "ix_rules_updated"})
```

The worker applies an explicit configured maximum to active rules and fails if
the result would be truncated. Arbitrary executable expressions are never stored.

### `alerts`

Alerts are operator-facing durable records. Camera/rule display values and event
evidence are snapshots; source IDs link back to the canonical event and action
execution. Lifecycle updates replace the bounded document with optimistic
revision matching.

```javascript
{
  _id: "alr_...",
  schemaVersion: 1,
  revision: 1,
  source: {eventId: "evt_...", executionId: "act_...", actionId: "critical-alert"},
  rule: {id: "rule-gate-blacklist", name: "Alert for blacklisted entry"},
  camera: {id: "gate-01", name: "Main Gate", zone: "ZONE_A"},
  eventType: "VEHICLE_ENTER",
  direction: "ENTER",
  severity: "CRITICAL",
  status: "OPEN",
  message: "Blacklisted entry",
  plate: "51H-123.45",
  vehicleType: "car",
  occurredAt: ISODate("2026-08-09T00:00:00Z"),
  createdAt: ISODate("2026-08-09T00:00:01Z"),
  updatedAt: ISODate("2026-08-09T00:00:01Z"),
  acknowledgedAt: null,
  acknowledgedBy: null,
  resolvedAt: null,
  resolvedBy: null,
  metadata: {}
}
```

```javascript
db.alerts.createIndex(
  {"source.executionId": 1},
  {unique: true, name: "uq_alert_execution"}
)
db.alerts.createIndex(
  {status: 1, createdAt: -1, _id: -1},
  {name: "ix_alert_status_cursor"}
)
db.alerts.createIndex(
  {plate: 1, createdAt: -1},
  {name: "ix_alert_plate_time", partialFilterExpression: {plate: {$type: "string"}}}
)
db.alerts.createIndex(
  {"camera.id": 1, createdAt: -1},
  {name: "ix_alert_camera_time"}
)
db.alerts.createIndex(
  {"rule.id": 1, createdAt: -1},
  {name: "ix_alert_rule_time"}
)
db.alerts.createIndex(
  {createdAt: -1, _id: -1},
  {name: "ix_alert_cursor"}
)
```

### `action_executions`

This operational/audit collection is the durable idempotency boundary for rule
effects. `_id` is a deterministic SHA-256-derived value from event ID, rule ID,
and action ID. Atomic insert/reclaim prevents concurrent consumers from owning
the same attempt.

```javascript
{
  _id: "act_...",
  schemaVersion: 1,
  eventId: "evt_...",
  ruleId: "rule-gate-blacklist",
  actionId: "critical-alert",
  actionType: "CREATE_ALERT",
  status: "SUCCEEDED",
  attemptCount: 1,
  errorCode: null,
  createdAt: ISODate("2026-08-09T00:00:01Z"),
  updatedAt: ISODate("2026-08-09T00:00:01Z"),
  completedAt: ISODate("2026-08-09T00:00:01Z")
}
```

```javascript
db.action_executions.createIndex(
  {status: 1, updatedAt: 1},
  {name: "ix_action_status_updated"}
)
db.action_executions.createIndex(
  {eventId: 1, ruleId: 1},
  {name: "ix_action_event_rule"}
)
```

The built-in `_id` index is the uniqueness constraint. Failed attempts retain a
bounded error code rather than response bodies/secrets. Action execution
retention must be at least as long as event redelivery/replay can occur; deleting
it earlier removes the idempotency memory.

### `audit_logs`

This is an append-only actor history for successful security-sensitive API
operations. It is separate from `action_executions`: the latter records automated
rule effects, while `audit_logs` identifies the authenticated principal who
changed a resource. No resource embeds an unbounded audit array.

```javascript
{
  _id: "aud_...",
  schemaVersion: 1,
  actor: {
    id: "admin-01",
    displayName: "Platform Admin",
    role: "ADMIN",
    authenticationMethod: "API_KEY"
  },
  action: "CAMERA_UPDATED",
  resource: {type: "CAMERA", id: "gate-01"},
  requestId: "req_...",
  before: {revision: 1, name: "Main Gate"},
  after: {revision: 2, name: "Updated Gate"},
  metadata: {},
  occurredAt: ISODate("2026-08-09T12:00:00Z")
}
```

```javascript
db.audit_logs.createIndex(
  {occurredAt: -1, _id: -1},
  {name: "ix_audit_cursor"}
)
db.audit_logs.createIndex(
  {"actor.id": 1, occurredAt: -1},
  {name: "ix_audit_actor_time"}
)
db.audit_logs.createIndex(
  {"resource.type": 1, "resource.id": 1, occurredAt: -1},
  {name: "ix_audit_resource_time"}
)
db.audit_logs.createIndex(
  {action: 1, occurredAt: -1},
  {name: "ix_audit_action_time"}
)
```

Application sanitization removes secrets before the repository receives a
document. The API exposes read/cursor operations only—no update/delete endpoint.
Standalone MongoDB does not make the resource write and audit append atomic; a
replica-set transaction or outbox is the later strict-atomicity path.

### `dataset_samples`

One bounded document represents one labeled OCR feedback item. The unique source
event/review revision pair makes correction retries idempotent. Images stay in
object storage and the sample keeps only `imageKey`.

```javascript
{
  _id: "dss_...",
  schemaVersion: 2,
  type: "PLATE_OCR",
  status: "READY",
  sourceEventId: "evt_123",
  imageKey: "vehicles/2026/08/09/gate-01/evt_123/plate.jpg",
  prediction: {
    raw: "51H1234S",
    normalized: "51H-123.4S",
    confidence: 0.68,
    model: {name: "plate-ocr", version: "v2", hash: null}
  },
  label: "51H-123.45",
  reason: "HUMAN_CORRECTION",
  review: {
    revision: 1,
    reviewedBy: {id: "operator-01", displayName: "Gate Operator"},
    reviewedAt: ISODate("2026-08-09T13:00:00Z")
  },
  export: {
    id: "ocr-20260810-v1",
    attempts: 1,
    claimedAt: ISODate("2026-08-10T01:00:00Z"),
    exportedAt: null,
    manifestSha256: null,
    errorCode: null
  },
  createdAt: ISODate("2026-08-09T13:00:00Z")
}
```

```javascript
db.dataset_samples.createIndex(
  {sourceEventId: 1, "review.revision": 1},
  {unique: true, name: "uq_dataset_event_review"}
)
db.dataset_samples.createIndex(
  {status: 1, "export.claimedAt": 1, createdAt: 1},
  {name: "ix_dataset_export_claim"}
)
db.dataset_samples.createIndex(
  {"export.id": 1, status: 1, createdAt: 1},
  {
    name: "ix_dataset_export_resume",
    partialFilterExpression: {"export.id": {$type: "string"}}
  }
)
db.dataset_samples.createIndex(
  {type: 1, status: 1, createdAt: -1, _id: -1},
  {name: "ix_dataset_type_status_cursor"}
)
db.dataset_samples.createIndex(
  {reason: 1, createdAt: -1},
  {name: "ix_dataset_reason_time"}
)
db.dataset_samples.createIndex(
  {createdAt: -1, _id: -1},
  {name: "ix_dataset_cursor"}
)
db.dataset_samples.createIndex(
  {imageKey: 1, status: 1},
  {name: "ix_dataset_image_status"}
)
```

The API exposes a bounded operator/admin cursor listing. `READY`, `EXPORTING`,
`EXPORT_FAILED`, and `EXPORTED` form the leased export state machine; mutation is
owned by the dataset exporter rather than the event or OCR pipeline. The
retention worker uses `ix_dataset_image_status` to make every non-terminal sample
an explicit media/source-event pin. `ix_dataset_export_claim` supports reclaiming
stale work, while `ix_dataset_export_resume` reconciles retries by export ID.

### `vehicle_tracks` (optional debug/operations)

If enabled later, stores one bounded summary per finalized track: first/last seen,
direction, final plate, frame/observation counts, and state. It never stores raw
frames or an unbounded trajectory. Phase 1 keeps track state in memory.

### `vehicles`

Implemented logical identity summary. The event worker first creates one
deterministic bootstrap identity per source event; identical plate strings do not
implicitly merge identities. Later ReID decisions may revise identity ownership.

```javascript
{
  _id: "veh_...",
  schemaVersion: 1,
  revision: 1,
  status: "ACTIVE",
  primaryPlate: "51H-123.45",
  plates: [{
    text: "51H-123.45",
    confidence: 0.96,
    firstSeenAt: ISODate("2026-08-10T01:00:00Z"),
    lastSeenAt: ISODate("2026-08-10T01:00:00Z")
  }],
  attributes: {type: "car", color: "white"},
  firstSeenAt: ISODate("2026-08-10T01:00:00Z"),
  lastSeenAt: ISODate("2026-08-10T01:00:00Z"),
  observationCount: 1,
  metadata: {}
}
```

`plates` is application-bounded to 16 aliases. Event history remains in
`vehicle_events` queried by `vehicleId`; there is deliberately no `events[]`.

```javascript
db.vehicles.createIndex(
  {primaryPlate: 1, lastSeenAt: -1},
  {name: "ix_vehicle_primary_plate_time", partialFilterExpression: {primaryPlate: {$type: "string"}}}
)
db.vehicles.createIndex(
  {"plates.text": 1, lastSeenAt: -1},
  {name: "ix_vehicle_plate_alias_time"}
)
db.vehicles.createIndex({status: 1, lastSeenAt: -1}, {name: "ix_vehicle_status_time"})
```

### `vehicle_fingerprints`

One immutable, bounded evidence document per source event. It references any
visual embedding by ID plus model metadata and never embeds the vector/image.

```javascript
{
  _id: "vfp_...",
  schemaVersion: 1,
  vehicleId: "veh_...",
  sourceEventId: "evt_...",
  cameraId: "gate-01",
  observedAt: ISODate("2026-08-10T01:00:00Z"),
  plate: {text: "51H-123.45", confidence: 0.96},
  vehicle: {type: "car", confidence: 0.97, color: "white"},
  embedding: {
    id: "emb_...",
    model: {name: "vehicle-reid", version: "v1", hash: "sha256...", dimension: 256}
  }
}
```

Indexes are unique `sourceEventId`, `(vehicleId, observedAt desc)`,
`(cameraId, observedAt desc)`, and a partial `embedding.id` index. Fingerprint
registration, identity aggregate update, and event link share the event worker's
Mongo replica-set transaction boundary.

### `vehicle_embeddings`

Separately stores normalized vectors with independent model/query lifecycle:

```javascript
{
  _id: "emb_...",
  schemaVersion: 1,
  model: {name: "vehicle-reid", version: "v1", hash: "sha256...", dimension: 256},
  values: [0.012, -0.044],
  createdAt: ISODate("2026-08-10T01:00:00Z"),
  metadata: {vehicleId: "veh_..."}
}
```

The vector repository intentionally accepts only an explicit candidate ID set
(maximum 5,000, normally much lower), filters by exact model
name/version/dimension, computes cosine similarity over that bounded set, and
returns a bounded top-k. It never loads all embeddings into Python. Indexes are
`(model.name, model.version, createdAt desc)` and the partial
`(metadata.vehicleId, createdAt desc)`.

### `camera_topology`

Revisioned directed graph edges used before any visual similarity query:

```javascript
{
  _id: "gate-a-to-warehouse",
  schemaVersion: 1,
  revision: 1,
  fromCameraId: "gate-a",
  toCameraId: "warehouse",
  travelTime: {minimumSeconds: 60, typicalSeconds: 240, maximumSeconds: 600},
  enabled: true,
  metadata: {},
  createdAt: ISODate("2026-08-10T01:00:00Z"),
  updatedAt: ISODate("2026-08-10T01:00:00Z")
}
```

The `(fromCameraId, toCameraId)` pair is unique. `ix_topology_inbound` supports
candidate generation by `(toCameraId, enabled, fromCameraId)`;
`ix_topology_outbound` supports graph administration. For each inbound edge, the
generator uses `ix_fingerprint_camera_time` with a bounded time range and limit.
It never scans all fingerprints and never assumes the reverse edge exists.

### Additional identity collections

### `identity_reviews`

Immutable result/intent document keyed by the client-stable review ID. It stores
action, source/result identity, expected revisions, reviewer, reason, optional
scored fingerprint pair/score, moved counters, and UTC review time. It does not
embed snapshots or event history.

Indexes `(sourceVehicleId, reviewedAt desc)` and `(reviewer.id, reviewedAt desc)`
support investigation/audit. The unique `_id` supplies idempotency: an identical
retry returns the persisted result, while different intent using the same ID is
rejected. Mongo merge/split updates `vehicles`, `vehicle_fingerprints`,
`vehicle_events`, and `identity_reviews` in one transaction. Operations reject
more than 1,000 fingerprints rather than creating an unbounded transaction.

## Embedding versus references

- Camera display fields and compact vehicle/plate evidence are embedded because
  they are immutable event snapshots read together.
- Media, event clips, raw/debug images, and embeddings are referenced by
  object/vector IDs because they are large and have independent retention/query
  behavior.
- Event history references `vehicleId`; it is never embedded into a vehicle.
- Journey/timeline is derived through `ix_vehicle_time`; no duplicate journey
  collection or unbounded `vehicles.events[]` projection is maintained.

## Retention and TTL

Canonical vehicle events have business retention and are removed by a cleanup
worker that coordinates media and dataset requirements. They do not receive a
blind TTL index. TTL is suitable for future temporary processing state and
short-lived health history with a true `expiresAt` semantic. Media retention is
separately configurable and never leaves a public event key claiming an object
still exists; cleanup records media lifecycle state before deletion.

Alerts likewise follow incident/audit policy rather than a blind TTL.
`action_executions` must outlive every possible broker redelivery or replay of its
source event. No TTL index is created for either collection in this milestone.
Audit logs follow a legal/security archival policy and also receive no blind TTL.
Dataset samples require an explicit export/retention policy. A `READY`,
`EXPORTING`, or `EXPORT_FAILED` sample's referenced media and source event do not
expire; coordinated cleanup becomes eligible only at `EXPORTED`. Managed MinIO
lifecycle applies only to `debug/` and `temporary/`, never canonical
`vehicles/` prefixes.

## Growth risks

- Per-frame observations and trajectories can grow without bound, so only a
  bounded in-memory active-track window and aggregate counts cross finalization.
- Embeddings would multiply event size and index memory, so fingerprints keep
  only an `embedding.id` and versioned model trace while vectors live in
  `vehicle_embeddings`.
- High-cardinality arbitrary metadata must be size-limited at ingestion and must
  not be indexed by default.
- Rules/actions are API-bounded, but metadata and action bodies still need a
  production document-size policy before untrusted authors are admitted.
- Alert volume follows rule match volume; cursor pagination is mandatory and
  retention must preserve incident/audit needs.
- `action_executions` grows once per matched action. Cleanup must coordinate with
  broker replay and event-retention windows or it can re-enable old side effects.
- `audit_logs` grows once per mutation/security action. Archive by explicit policy;
  never embed it back into camera/watchlist/rule documents.
- `dataset_samples` grows once per accepted review revision with an available
  crop. Export/archive by bounded cursor and coordinate referenced media
  retention; never copy image bytes into MongoDB.
- Fuzzy plate search must not scan all events. Phase 1 supports normalized prefix
  candidates bounded by time/count; production fuzzy search will use a derived,
  indexed n-gram/search field or dedicated search adapter after measurement.

## Query patterns

| Query | Index |
|---|---|
| Latest events / cursor page | `ix_time_cursor` |
| Exact final plate history | `ix_plate_final_time` plus legacy `ix_plate_time` |
| Human review queue | `ix_status_time_cursor` |
| Camera timeline | `ix_camera_time` |
| Logical vehicle timeline | `ix_vehicle_time` |
| Event type history | `ix_type_time` |
| Retry/deduplicate track event | `uq_event_track_type` |
| Enabled camera reconciliation | `ix_enabled_name` |
| Camera config freshness | `ix_updated_at` |
| Operational camera state | `ix_health_status_updated` |
| Active list membership by plate/time | `ix_watchlist_plate_active` |
| Watchlist administration by type | `ix_watchlist_type_enabled` |
| Enabled rules in evaluation order | `ix_rules_enabled_priority` |
| Alert cursor/status/plate/camera/rule filters | `ix_alert_*` indexes |
| Claim/deduplicate action | Built-in unique `_id` |
| Inspect stale/failed action executions | `ix_action_status_updated` |
| Trace executions for event/rule | `ix_action_event_rule` |
| Audit chronology / cursor | `ix_audit_cursor` |
| Audit by actor | `ix_audit_actor_time` |
| Audit by resource | `ix_audit_resource_time` |
| Audit by action | `ix_audit_action_time` |
| Dataset sample idempotency | `uq_dataset_event_review` |
| Dataset export queue / cursor | `ix_dataset_type_status_cursor`, `ix_dataset_cursor` |
| Atomic/stale export claim | `ix_dataset_export_claim` |
| Resume export by stable ID | `ix_dataset_export_resume` |
| Dataset reason analysis | `ix_dataset_reason_time` |
| Oldest-first retention claim | `ix_retention_oldest` |
| Dataset media pin | `ix_dataset_image_status` |
| Identity lookup by primary/alias plate | `ix_vehicle_primary_plate_time`, `ix_vehicle_plate_alias_time` |
| Identity fingerprint history | `ix_fingerprint_vehicle_time` |
| Cross-camera fingerprint window | `ix_fingerprint_camera_time` |
| Fingerprint idempotency | `uq_fingerprint_source_event` |
| Bounded embedding candidates | `_id` plus exact model predicate |
| Inbound/outbound topology | `ix_topology_inbound`, `ix_topology_outbound` |
| Unique directed edge | `uq_topology_direction` |
| Identity review history | `ix_identity_review_source_time` |
| Reviewer activity | `ix_identity_review_actor_time` |

All dates use MongoDB BSON datetime in UTC. Repository mapping, not domain code,
owns Mongo field names and collection constants.
