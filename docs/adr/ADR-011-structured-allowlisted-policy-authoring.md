# ADR-011: Structured Allowlisted Policy Authoring

- Status: Accepted
- Date: 2026-08-09

## Context

The policy API deliberately accepts only known event fields, comparison
operators, action types, and guarded external HTTP parameters. Operators need to
manage those documents from the Angular console. A raw JSON or expression editor
would make simple validation difficult, invite malformed or ineffective rules,
and visually imply an arbitrary-code capability the rule engine does not own.

Policy updates also carry operational identity: MongoDB uses optimistic
revisions, while every rule action ID participates in the deterministic
`(event ID, rule ID, action ID)` execution claim.

## Decision

Provide dedicated lazy `/watchlists` and `/rules` routes backed by typed REST
contracts. Build conditions and actions from fixed controls that mirror the
backend allowlists. Convert list and datetime input explicitly, validate external
HTTP(S) URLs without credentials, and retain metadata not edited by the current
form.

Preserve resource revisions on every edit and preserve action IDs unless an
administrator explicitly changes them. Hide mutation controls from non-ADMIN
roles, require a named confirmation before delete, and continue treating FastAPI
validation, RBAC, audit, URL/host allowlisting, and idempotency as authoritative.

Do not add a raw JSON, expression, script, or arbitrary parameter editor.

## Consequences

- Common policies can be authored without learning the persistence document.
- Invalid operator/value and external URL shapes are rejected before a request,
  then independently validated again by FastAPI.
- Concurrent browser sessions fail with `409` rather than overwriting changes.
- Stable action IDs preserve prior execution identity across ordinary edits.
- New backend fields/actions require an explicit frontend update, which is an
  intentional safety and compatibility review point.
- Watchlist/rule list views are currently bounded by the API's 200-item limit;
  cursor pagination is future work.
