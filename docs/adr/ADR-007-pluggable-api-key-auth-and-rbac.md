# ADR-007: Pluggable Bearer API-Key Authentication and RBAC

- Status: Accepted
- Date: 2026-08-09

## Context

Camera, watchlist, rule, and alert APIs had no authenticated actor. Building an
internal password database and token issuer would prematurely own password reset,
MFA, account recovery, and session revocation before an organizational identity
provider is selected. Leaving route logic unauthenticated until the UI phase would
permit security-critical whitelist/rule changes without attribution.

## Decision

Define an application `Authenticator` port and an explicit RBAC permission
matrix. The first provider validates high-entropy Bearer API keys against
configured SHA-256 verifiers using constant-time comparison. Raw keys are never
stored in repository/config files. Enabled authentication requires at least one
active administrator. Development bypass is explicit, visible as a synthetic
principal, and reported by health.

Roles are `VIEWER`, `OPERATOR`, and `ADMIN`. Authorization uses capabilities,
not scattered role-name comparisons. Audit actor identity always comes from the
authenticated principal rather than a request body.

## Consequences

- Current machine/operator clients can be protected without introducing user
  tables or coupling routes to a JWT library.
- A future OIDC/JWT authenticator can enter at the port while the permission
  matrix and audit model remain stable.
- API keys are long-lived credentials without intrinsic expiry/session context;
  TLS and secret-manager handling are mandatory.
- Rotation/revocation is configuration-driven until centralized identity is
  implemented.
- Static API keys are a foundation, not the final human-login design for the
  Angular dashboard.

