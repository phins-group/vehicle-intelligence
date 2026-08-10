# ADR-006: Durable Idempotent Rule Actions

- Status: Accepted
- Date: 2026-08-09

## Context

Redis Streams delivers at least once. A vehicle event may therefore be delivered
again after MongoDB or Redis failure, consumer reclaim, or process termination.
Actions such as alert creation and barrier opening must not be invoked once per
delivery. Holding an in-memory deduplication set would fail after restart and
would not coordinate multiple consumers.

## Decision

Before executing an action, persist an `action_executions` document whose ID is
the deterministic hash of `(event ID, rule ID, action ID)`. MongoDB `_id`
uniqueness is the primary claim. A duplicate claimant may atomically reclaim only
a retryable failed execution below its maximum attempts or a running execution
older than the configured stale interval.

The event worker ACKs the Redis message only after event persistence and policy
processing finish. It evaluates policy even when event persistence reports a
duplicate, allowing an action that failed before ACK to resume. External calls
receive the same execution ID as an `Idempotency-Key` on every attempt.

## Consequences

- Alerts and other local effects are deduplicated across retries and consumers.
- Failure/retry state is inspectable and survives process restarts.
- MongoDB availability is required for action execution.
- External exactly-once effects still require the receiver to honor the
  idempotency key. No distributed transaction can atomically cover an arbitrary
  HTTP service and the local MongoDB success update.
- Terminal and exhausted actions are durable skip decisions, permitting the
  Redis message to be ACKed on redelivery rather than remaining pending forever.
