# ADR-018: Production Identity, Transactional Audit, and Authenticated Egress

- Status: Accepted
- Date: 2026-08-10

## Context

Static API keys, one camera encryption key, separate resource/audit writes, and
hostname-only webhook controls were an appropriate Phase 2 foundation but not a
complete production security boundary. Operators need centralized identity,
online secret rotation, strict audit atomicity, and authenticated physical/action
egress without embedding credentials in rules.

## Decision

- Keep `Authenticator` as the application port and add OIDC/JWKS verification
  with fixed algorithms, exact issuer/audience, bounded claims, cached keys, and
  fail-closed role mapping. Retain API keys for controlled deployments.
- Replace the single camera key internally with an active/decrypt keyring while
  preserving the `v1.<key-id>...` token contract. Rotate by ciphertext CAS and
  audit the bounded maintenance run.
- Compose all API Mongo repositories from one client. When strict mode is on,
  validate replica-set/mongos capability at startup and bind a transaction to
  every audited mutation request.
- Configure external targets and their credentials on the server. Support Bearer
  or canonical HMAC-SHA256 requests, reject unsafe URLs/redirects, and isolate
  retryable target failures through a CLOSED/OPEN/HALF_OPEN circuit breaker.
  Circuit-open deferrals do not consume durable retry attempts.

## Consequences

- A required audit failure cannot leave a successful resource mutation visible
  in the production profile.
- Key rotation is online and backward compatible, but operations must retain old
  decrypt keys until the report reaches zero pending documents.
- OIDC availability and JWKS freshness become identity dependencies; bounded
  caching reduces calls and failure remains closed.
- Server configuration/secret management, not rule authors, owns egress trust.
- MongoDB production deployment now requires replica-set semantics. Standalone
  mode remains explicit for development only.
