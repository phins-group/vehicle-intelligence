# ADR-001: Use MongoDB for canonical event documents

**Status:** Accepted

Vehicle events are append-heavy, schema-versioned, naturally document-shaped,
and rendered mostly without joins. MongoDB provides atomic document writes,
compound/partial indexes, and idempotency through a unique semantic key.
Historical events snapshot camera display data intentionally. Large media and
unbounded histories remain outside the document.

