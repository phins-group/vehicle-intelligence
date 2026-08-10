# Rules and Alerts

## Runtime boundary

The policy engine consumes a finalized, canonical `VehicleEvent`; it never runs
inside detection, tracking, plate quality, or OCR components.

```text
Redis vehicle event
  -> idempotent MongoDB event persistence
  -> active watchlists for normalized plate at event time
  -> enabled rules ordered by priority
  -> durable action claim
  -> action handler
  -> durable action result
  -> Redis ACK
```

Policy processing is currently composed by `vehicle-event-worker`. Direct
JSONL/Mongo publishers used by video and RTSP CLIs do not evaluate rules. Use the
Redis event backend when policy actions are required.

## Watchlists

Supported types are `WHITELIST`, `BLACKLIST`, `VIP`, `STAFF`, `CONTRACTOR`, and
`DELIVERY`. Input plates are validated and normalized as Vietnamese plates before
storage. An entry matches an event only when it is enabled and the event's
`occurredAt` lies inside its optional inclusive validity interval.

```http
POST   /api/watchlists
GET    /api/watchlists?listType=BLACKLIST&enabled=true&limit=100
GET    /api/watchlists/{id}
PUT    /api/watchlists/{id}
DELETE /api/watchlists/{id}
```

Example create request:

```json
{
  "id": "blocked-51h-12345",
  "plate": "51H12345",
  "listType": "BLACKLIST",
  "enabled": true,
  "validFrom": "2026-08-09T00:00:00+07:00",
  "validUntil": null,
  "metadata": {"reason": "security review"}
}
```

The response stores `51H-123.45`. A full `PUT` includes the last observed
`revision`; a stale revision returns `409` rather than overwriting a concurrent
change.

## Declarative rules

All conditions in a rule use AND semantics. Enabled rules are evaluated by
descending priority, then stable name/ID order. Evaluation fails explicitly if
the active rule count exceeds `rule_engine.evaluation_max_rules`; it never
silently ignores the tail of the rule set.

Supported fields:

| Field | Event value |
|---|---|
| `watchlist` | Active watchlist-type collection for the plate |
| `camera.id` | Camera snapshot ID |
| `camera.zone` | Camera snapshot zone or null |
| `direction` | `ENTER`, `EXIT`, or `UNKNOWN` |
| `eventType` | Canonical vehicle event type |
| `status` | Event confidence/readability status |
| `plate.normalized` | Canonical plate or null |
| `vehicle.type` | Vehicle class |
| `vehicle.color` | Color or null |

Operators are `EQ`, `NEQ`, `IN`, `NOT_IN`, `CONTAINS`, and `EXISTS`.
`IN`/`NOT_IN` require a non-empty list, `EXISTS` requires a boolean, and
`CONTAINS` is intentionally limited to `watchlist`.

```http
POST   /api/rules
GET    /api/rules?enabledOnly=true&limit=100
GET    /api/rules/{id}
PUT    /api/rules/{id}
DELETE /api/rules/{id}
```

Example:

```json
{
  "id": "rule-gate-blacklist",
  "name": "Alert for blacklisted entry at Gate 01",
  "enabled": true,
  "priority": 100,
  "conditions": [
    {"field": "watchlist", "operator": "CONTAINS", "value": "BLACKLIST"},
    {"field": "camera.id", "operator": "EQ", "value": "gate-01"},
    {"field": "direction", "operator": "EQ", "value": "ENTER"}
  ],
  "actions": [
    {
      "id": "critical-alert",
      "type": "CREATE_ALERT",
      "parameters": {
        "severity": "CRITICAL",
        "message": "Blacklisted vehicle entered Gate 01"
      }
    }
  ],
  "metadata": {}
}
```

Rules are schema-validated at create/update time and again before execution, so
an unsupported field, operator shape, URL, method, or severity cannot be treated
as an arbitrary expression.

The Angular `/rules` route exposes the same contract through a structured
builder; it does not accept raw expressions or arbitrary JSON. `/watchlists`
provides validity-aware CRUD. Both routes are readable by all authenticated
roles, while mutation controls are shown only to `ADMIN`; FastAPI RBAC remains
authoritative. Edits send the current optimistic `revision`, action IDs survive
rule edits, and destructive operations require an explicit named confirmation.

## Actions

Supported types are `CREATE_ALERT`, `LOG`, `OPEN_BARRIER`, `WEBHOOK`,
`HTTP_REQUEST`, and `NOTIFICATION`. The latter four currently share the guarded
HTTP adapter; their separate types preserve business intent and allow dedicated
adapters later.

Every `(event ID, rule ID, action ID)` maps to one deterministic execution ID.
MongoDB atomically inserts or reclaims that record. Status is `RUNNING`,
`SUCCEEDED`, or `FAILED`; retry count, timestamps, and a bounded error code are
retained. A duplicate event skips a succeeded action, retries an eligible failed
action, and reclaims a stale running claim only after
`action_claim_stale_seconds`. Retryable failures stop at `action_max_attempts`.

`CREATE_ALERT` uses the execution ID to derive a deterministic alert ID and also
has a unique index on `source.executionId`. `LOG` emits structured event, camera,
track, rule, action, and execution context. Rule-authored messages must not carry
secrets because logs and alerts are durable/observable records.

## External action safety

External actions are fail-closed by default:

```yaml
rule_engine:
  external_actions_enabled: false
  external_allowed_hosts: []
  action_timeout_seconds: 5
  action_max_attempts: 3
  action_claim_stale_seconds: 60
```

To enable them, list exact lower-case hostnames. Wildcards, ports, paths, and URL
credentials are not allowed in the host list; the rule URL itself must use HTTP
or HTTPS and may not embed credentials. Redirect following is disabled. Allowed
methods are `GET`, `POST`, `PUT`, and `PATCH`.

For non-GET actions the adapter sends:

```json
{
  "eventId": "evt_...",
  "ruleId": "rule-gate-blacklist",
  "actionId": "open-gate",
  "cameraId": "gate-01",
  "plate": "51H-123.45",
  "direction": "ENTER",
  "data": {}
}
```

Headers include `Idempotency-Key`, `X-Vehicle-Event-Id`, and
`X-Vehicle-Rule-Id`. HTTP `408`, `429`, and `5xx` responses are retryable; other
non-2xx responses are terminal. The receiver must persist/deduplicate the
idempotency key. This closes the unavoidable uncertainty window where the remote
side effect succeeds but the worker dies before marking the local execution as
succeeded.

The hostname allowlist is an application guard, not a complete network security
boundary. Production should additionally use egress firewalling, trusted DNS,
TLS, and a dedicated authenticated barrier/webhook adapter.

## Alerts

Alerts snapshot the rule name, camera name/zone, event/direction, normalized
plate, vehicle type, occurrence time, source execution, severity, and message so
list/detail rendering needs no lookup. Status transitions are:

```text
OPEN -> ACKNOWLEDGED -> RESOLVED
OPEN ----------------> RESOLVED
```

Repeated transitions to the same state are idempotent. A resolved alert is
terminal. Transition requests retain actor and UTC timestamp and use optimistic
revision replacement.

```http
GET  /api/alerts?status=OPEN&plate=51H12345&cameraId=gate-01&ruleId=rule-id&limit=50
GET  /api/alerts/{id}
POST /api/alerts/{id}/acknowledge  {}
POST /api/alerts/{id}/resolve      {}
```

Alert listing is newest-first and returns an opaque `nextCursor`.

## Current operational boundary

- API-key authentication, RBAC, and actor mutation audit are implemented; OIDC,
  user/session lifecycle, and strict cross-collection audit transactions are not.
- There is no webhook signature/secret manager integration or per-target circuit
  breaker yet.
- Rule changes affect new/redelivered events; rules are not snapshotted onto an
  event before evaluation and there is no historical policy replay command yet.
- Actions execute sequentially in one event worker process. Horizontal consumers
  are safe through MongoDB claims, but throughput/load/soak limits still need
  measurement.
