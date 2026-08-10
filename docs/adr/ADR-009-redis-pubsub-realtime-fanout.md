# ADR-009: Redis Pub/Sub for Realtime Fan-out

- Status: Accepted
- Date: 2026-08-09

## Context

The durable Redis Stream is owned by the event-processing consumer group and is
deleted after persistence, policy processing, and ACK. Letting each API instance
consume that stream would conflict with deletion, pending-entry recovery, and
horizontal fan-out semantics. Retaining a second consumer group indefinitely
would also turn dashboard delivery into a durable backlog with one copy per API
replica, although disconnected browsers cannot consume it.

Dashboard delivery needs low latency and multi-instance broadcast, while the
canonical recovery record already exists in MongoDB.

## Decision

After durable event processing succeeds, the event worker publishes the same
versioned JSON envelope to a dedicated Redis Pub/Sub channel. Every API instance
owns one reconnecting subscriber and a bounded in-memory hub. Each client gets a
bounded queue and short process-local replay buffer. Slow-client overflow and
unavailable replay are explicit gap controls that direct clients to the paginated
MongoDB-backed event API.

Realtime publish failure does not block the durable stream ACK. Client and hub
deduplication use the stable canonical event ID.

## Consequences

- All API replicas receive a notification without competing for one stream item.
- Durable policy execution is isolated from slow or disconnected dashboard
  clients.
- Redis/API outages can lose realtime notifications, so clients must reconcile
  from MongoDB.
- Replay is bounded and process-local rather than a claim of durable delivery.
- A future NATS/Kafka implementation can replace the realtime publisher and
  subscriber ports without changing domain events or API connection handling.
