# ADR-002: Separate vision output from event processing

**Status:** Accepted

Vision produces domain events through a `VehicleEventPublisher` port. It does
not call MongoDB, rules, barriers, or web clients. Phase 1 composes a direct
publisher over local JSONL and optional Mongo repositories. Phase 2 selects a
Redis Streams publisher at the same boundary and runs a separate idempotent
MongoDB consumer.

The Redis path is deliberately at least once. Consumer-group messages are ACKed
only after persistence, stale pending messages are reclaimed, malformed contracts
are dead-lettered, and MongoDB's semantic unique index absorbs redelivery. This
keeps transport semantics out of the domain and leaves the publisher/consumer
ports replaceable by NATS, Kafka, or RabbitMQ adapters later.
