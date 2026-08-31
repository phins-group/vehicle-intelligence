# ADR-010: Angular Same-Origin Operator Console

- Status: Accepted
- Date: 2026-08-09
- Updated: 2026-08-26

## Context

Phase 2 already exposes authenticated REST queries/mutations and a best-effort
WebSocket stream. Operators need a reusable console without weakening API-key
handling, inventing a second business-logic layer, or treating realtime delivery
as canonical storage. Browser CORS and WebSocket authentication also become
simpler when the UI and API share one origin.

## Decision

Build a standalone Angular 22 application with lazy feature routes, strict
TypeScript, Angular signals for bounded local state, and RxJS for realtime and
refresh coordination. Nginx serves compiled static assets and proxies /api and
/ws to FastAPI on the same origin.

The REST API remains canonical. WebSocket events are deduplicated by stable event
ID, and any gap triggers REST reconciliation. Production authentication uses
OIDC Authorization Code with PKCE S256. Tokens remain in memory; sessionStorage
contains only the short-lived login transaction. Development API keys remain
tab-scoped, are sent as Bearer headers for REST and in the WebSocket first frame,
and are never accepted in navigation URLs.

Frontend role checks control affordances only; FastAPI RBAC remains authoritative.
No raw frame transport or direct MinIO/MongoDB access is added to the browser.

## Consequences

- Development and production use the same relative API/WebSocket paths.
- Feature chunks are loaded only when routed, keeping the initial application
  bundle bounded.
- A browser refresh initiates a new OIDC authorization flow; no refresh or access
  token is persisted by the console. Development API keys remain tab-scoped.
- Realtime outages degrade to REST data instead of breaking the console.
- The browser uses a public OIDC client with PKCE and no distributed client
  secret. Server-side RBAC remains authoritative after token validation.
- Signed media URLs and the remaining policy/identity screens require later
  milestones.
