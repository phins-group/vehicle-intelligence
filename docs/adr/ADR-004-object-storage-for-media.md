# ADR-004: Store media in object storage

**Status:** Accepted

Snapshots, crops, and optional clips have a lifecycle and access pattern distinct
from event metadata. They are stored behind `MediaStorage` (local in tests/dev,
MinIO in production); MongoDB retains keys and metadata only. This avoids BSON
growth, base64 overhead, and database backup inflation.

