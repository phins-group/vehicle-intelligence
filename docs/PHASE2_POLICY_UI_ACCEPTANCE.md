# Phase 2 Policy Console Acceptance Record

## Decision

The Angular watchlist and rule-management milestone passed engineering
acceptance on 2026-08-09. This accepts typed read views for all platform roles,
ADMIN create/update/delete flows, validity-aware watchlists, a structured
allowlisted rule builder, optimistic revision handling, stable action IDs,
destructive confirmations, responsive lazy routes, and same-origin deployment.

It does not accept policy simulation/replay, cursor pagination beyond 200
resources, custom external-action bodies, an audit-log UI, arbitrary metadata
editing, or browser connection/load/soak limits.

## Accepted flows

```text
Angular /watchlists
  -> typed watchlist request
  -> FastAPI plate normalization + revision check + RBAC
  -> MongoDB watchlists + append-only audit

Angular /rules
  -> allowlisted field/operator/action builder
  -> typed rule request with stable action IDs
  -> FastAPI validation + revision check + RBAC
  -> MongoDB rules + append-only audit
```

The browser does not evaluate rules or normalize plates authoritatively. It does
not send executable expressions, raw scripts, or database documents.

## Automated evidence

- Strict application/spec TypeScript typecheck passed.
- Fourteen deterministic Vitest tests passed. The seven new policy cases cover
  watchlist lifecycle/search, timezone-aware datetime conversion, list parsing,
  external URL safety, condition value shapes, and action validation.
- The Angular production build passed at 344.20 kB raw / 95.62 kB estimated
  initial transfer. Rules and watchlists remain lazy chunks at 28.91 kB and
  17.27 kB raw respectively.
- Production dependency audit reported zero vulnerabilities. The complete
  build-only dependency graph still reports seven moderate and four high
  advisories inherited through Angular tooling; Node/npm are absent from the
  Nginx runtime image.
- The complete real MongoDB 8, Redis 8, and MinIO suite passed 96 Python tests.
  Ruff and Python bytecode compilation also passed.
- Compose validation and the updated API/Nginx images built successfully. All
  five requested services became healthy/running, and `/`, `/watchlists`, and
  `/rules` returned the SPA through Nginx.
- A same-origin CRUD smoke normalized `51H12345` to `51H-123.45`, advanced a
  watchlist from revision 1 to 2, rejected a stale revision with `409`, retained
  the rule action ID, listed the new rule, and found its `RULE_CREATED` audit.
  The scoped smoke resources were deleted afterward.
- With authentication temporarily enabled in the Compose API, missing read
  access returned `401`; VIEWER and OPERATOR policy reads returned `200`; both
  roles' writes returned `403`; and ADMIN writes returned `201`. `/api/auth/me`
  resolved all three expected roles. Compose was restored to development
  `auth=disabled`, the Nginx API proxy returned `200`, and both test resources
  returned `404` after cleanup.

One unchanged warning comes from FastAPI/Starlette's current TestClient use of
httpx.

## Security and consistency semantics

Frontend role checks are affordances only; FastAPI remains authoritative. Every
edit submits the last observed resource revision, so a concurrent change fails
instead of being overwritten. Existing metadata is retained even though the
current UI does not expose a generic JSON editor.

The rule builder offers only known fields, operators, actions, methods, and alert
severities. External URLs require HTTP(S) and cannot contain credentials in the
client; the backend repeats those checks and additionally enforces the global
enable flag plus exact hostname allowlist. Action IDs are generated once and
retained through ordinary edits because they contribute to durable execution
identity.

Deletes identify the exact rule or plate and require explicit confirmation.
Successful create/update/delete operations remain covered by actor audit logs.

## Operational limits

- Watchlist/rule list APIs have no cursor and return at most 200 items. UI
  counts and client-side search apply to that loaded set.
- Rule conditions use AND semantics only, matching the current backend. There is
  no nested group or OR editor.
- External action forms intentionally expose only URL and method; custom `body`
  data and secret/signature management need a dedicated safe contract.
- There is no rule dry-run against a historical event, policy import/export,
  policy realtime invalidation, or audit-log navigation from a resource yet.
- The in-app browser runtime exposed no browser instance in this environment.
  Compiler, deterministic tests, container routes, REST semantics, RBAC, and
  cleanup were verified; manual cross-browser visual/keyboard QA remains
  required before an external production release.
