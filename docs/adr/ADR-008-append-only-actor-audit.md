# ADR-008: Append-Only Actor Audit Records

- Status: Accepted
- Date: 2026-08-09

## Context

Security-sensitive mutations need durable attribution and before/after evidence.
Action executions already audit system rule effects, but they do not identify the
human/API principal who changed a camera, watchlist, rule, or alert state.
Embedding an unbounded history in each resource would violate document-growth
rules and make cross-resource investigation expensive.

## Decision

Append one versioned `audit_logs` document per successful sensitive operation.
The record snapshots actor, action, resource, request correlation, sanitized
before/after values, and UTC occurrence time. Expose only ADMIN read endpoints
with cursor pagination and demonstrated indexes. Provide no update/delete API and
no TTL.

Audit sanitization is mandatory in the application service and recursively
redacts credentials/RTSP/Bearer values before any repository adapter receives the
record.

## Consequences

- Resource documents remain bounded while investigations can query actor,
  resource, action, and time independently.
- Historical names/config state survive later resource changes.
- Audit storage grows with mutation volume and needs an explicit archival policy.
- Standalone MongoDB cannot atomically combine a domain write with the audit
  insert. ADR-018 now selects shared-client replica-set transactions for the
  strict production profile; standalone mode remains development-only.
