# Phase 2 Security and Audit Acceptance Record

## Decision

The API authentication, role-based authorization, request correlation, and
actor-oriented audit foundation passed engineering acceptance on 2026-08-09.
This accepts protected Phase 2 camera, event, vehicle, watchlist, rule, alert,
and audit APIs with explicit `VIEWER`, `OPERATOR`, and `ADMIN` permissions. It
does not accept centralized user lifecycle, OIDC/JWT, or transactionally atomic
resource-and-audit writes.

## Accepted flow

```text
HTTP request
  -> validated/generated request ID
  -> Bearer credential verification
  -> immutable authenticated principal
  -> explicit route permission
  -> application mutation
  -> secret-redacted before/after audit record
  -> append-only audit repository
  -> correlated HTTP response
```

Authentication remains intentionally disabled in the default local-development
configuration. When enabled, startup validation requires at least one active
`ADMIN`, and only SHA-256 API-key verifiers are loaded from configuration. Raw
keys are presented by clients over Bearer authentication and must be protected
by TLS and a secret manager.

## Automated evidence

- Authentication tests cover valid, invalid, too-short, disabled, duplicate,
  and malformed API-key configurations without disclosing verifier input in
  validation errors.
- Protected API tests verify generic `401` responses with a Bearer challenge,
  `403` for insufficient permission, and the complete role matrix: `VIEWER`
  reads, `OPERATOR` additionally tests cameras and transitions alerts, and
  `ADMIN` performs camera/policy mutations and reads audit logs.
- Alert transitions derive the actor from the authenticated principal; a legacy
  client-supplied actor can only agree with that principal and cannot impersonate
  another user.
- Successful camera, watchlist, rule, and alert mutations record action,
  resource, authenticated actor, request ID, timestamps, and bounded before/after
  snapshots.
- Audit sanitization tests cover RTSP credentials, password/token/API-key fields,
  Bearer values, URL user information, and URL query/fragment removal.
- Audit queries cover actor/action/resource/time filters, stable cursor
  pagination, record lookup, authorization, and explicit invalid-cursor errors.
- Real MongoDB 8 tests verified append/query behavior, BSON mappings, filtering,
  redacted persistence, and the actor/resource/action/cursor indexes.
- The self-contained suite passed `80 passed, 8 skipped`; the complete suite
  with MongoDB 8, Redis 8, and MinIO passed `88 passed`.
- Ruff, Python bytecode compilation, Compose validation, and production
  API/event-worker image builds passed.
- An auth-enabled API container reported `authentication: enabled`; a request
  without credentials returned `401`, while the configured ADMIN identity was
  returned by `/api/auth/me`.
- The container smoke test created and deleted a temporary watchlist and verified
  two MongoDB audit records with the correct ADMIN actor and caller-provided
  request IDs. The temporary resource and both audit records were removed after
  verification.

The one emitted warning is an upstream FastAPI/Starlette TestClient deprecation
for `httpx`; it does not represent an application test failure.

## Failure semantics

Authentication failures intentionally return the same public message for
missing and invalid credentials. Authorization is evaluated before route
business logic. Audit append is synchronous and required for successful
security-sensitive API mutations; an audit repository failure is surfaced as
`503` instead of being silently ignored.

MongoDB resource mutation and audit append are currently separate writes. A
resource write may therefore have committed before an audit failure response.
Deployments requiring strict all-or-nothing semantics need a replica-set
transaction or transactional-outbox design before production acceptance.

## Operational limits

- Static API keys provide a replaceable authentication foundation, not login,
  password reset, key expiry/rotation workflow, OIDC, JWT, or user CRUD.
- API authentication has no built-in rate limiter or persisted failed-login and
  failed-authorization audit stream.
- Audit records are append-only through the application repository, but there is
  no cryptographic chain, WORM export, or external SIEM integration.
- API documentation endpoints remain deployment-controlled rather than governed
  by the current RBAC matrix.
- Production TLS termination, secret-manager delivery, network controls, backup,
  load/soak testing, and operational key rotation remain deployment work.
