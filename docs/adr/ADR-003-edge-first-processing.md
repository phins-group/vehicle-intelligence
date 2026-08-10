# ADR-003: Process video at the edge

**Status:** Accepted

Decode, sampling, detection, tracking, plate OCR, temporal aggregation, and best
frame selection happen near the camera. Central services receive event metadata
and selected media only. This controls bandwidth, latency, and privacy while
allowing central search/rules. Continuous raw video upload is not a default.

