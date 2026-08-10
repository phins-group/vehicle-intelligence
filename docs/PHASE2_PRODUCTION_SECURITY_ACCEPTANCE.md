# Phase 2 Production Security Acceptance

Accepted on 2026-08-10.

## Implemented

- Backward-compatible AES-256-GCM camera keyring, active-key writes, old-key
  reads, dry-run/online CAS rotation CLI, unchanged camera revision, and audit.
- OIDC/JWKS JWT verifier with cached signing keys, fixed `RS256`/`ES256`
  allowlist, issuer/audience/expiry/subject checks, bounded claims, role mapping,
  and fail-closed network/key behavior.
- Shared async Mongo client, replica-set capability check, request-scoped bound
  transactions, and atomic camera/policy/alert/review plus required audit writes.
- Structured external targets with HTTPS/exact-host URL controls, managed Bearer
  or HMAC-SHA256 authentication, stable idempotency, timeout, and retry-aware
  CLOSED/OPEN/HALF_OPEN circuit breaker.

## Tested

- Python lint passed; complete real-service suite passed: 153 tests against the
  MongoDB replica set, Redis, and MinIO with no skips.
- Adversarial OIDC tests rejected wrong issuer, audience, expiry, and HS256
  algorithm substitution; valid RS256 mapped the highest configured role.
- External action tests verified HMAC metadata and proved open-circuit deferrals
  do not consume action attempts.
- Real Mongo transaction acceptance observed zero documents after a forced
  resource/audit rollback and two after commit. Repository-level integration
  also forced duplicate audit failure and proved the camera write invisible.
- Real credential rotation reported `scanned=1`, `rotated=1`, `conflicts=0`,
  changed `v1.old` to `v1.new`, preserved revision `1`, and wrote audit records.
- Compose replica-set API started successfully with transactions enabled; an API
  watchlist mutation produced exactly one matching audit record.
- Angular typecheck, 30 tests, and production build passed (initial bundle
  364.79 kB raw / 99.08 kB estimated transfer).

## References

- [OpenID Connect Core 1.0](https://openid.net/specs/openid-connect-core-1_0-18.html)
- [JWT Best Current Practices](https://www.ietf.org/rfc/rfc8725.html)
- [PyJWT JWKS and decode API](https://pyjwt.readthedocs.io/en/stable/api.html)
- [PyMongo transactions](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/transactions/)

## Known limitations

- TLS termination, identity-provider lifecycle, ingress rate limiting, WAF, and
  external secret-manager injection are deployment responsibilities.
- The circuit breaker is process-local; horizontally scaled action workers each
  maintain independent breaker state. Durable action claims still prevent a
  completed logical action from being executed twice.

## Next

Run production failure-recovery, rule/action, realtime connection, load, and
soak acceptance to close Phase 2.
