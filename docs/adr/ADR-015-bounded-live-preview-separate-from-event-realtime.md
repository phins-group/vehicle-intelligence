# ADR-015: Keep bounded live preview separate from event realtime

- Status: Accepted
- Date: 2026-08-09

## Context

Operators need a camera preview with vehicle/plate/track overlays. Returning the
encrypted RTSP source to a browser would expose credentials and usually would
not be browser-playable. Sending frames through `/ws/events` would mix
high-volume ephemeral images with canonical event notifications, compromise
event backpressure semantics, and violate the event contract.

Phase 2 also needs a runnable path without first introducing a production video
gateway, HLS/WebRTC infrastructure, or continuous central recording.

## Decision

Use a dedicated optional live-preview path:

- the edge pipeline emits source-coordinate overlay metadata after processing a
  sampled frame;
- a one-slot reporter drops stale work and background-encodes a capped JPEG;
- a versioned, size-limited packet is published on a separate Redis Pub/Sub
  channel;
- each API process stores a bounded ring of the newest packets per bounded
  camera set;
- authenticated HTTP returns metadata first and the exact JPEG sequence second;
- Angular verifies the echoed sequence and draws functional SVG overlays.

No preview packet is canonical or durable. The default YAML disables preview;
deployments enable it explicitly. The event SSE/WebSocket continues to carry
only versioned vehicle and control envelopes.

## Consequences

- RTSP credentials never reach the browser, and preview failure cannot stop the
  core vision/event path.
- Memory, frame rate, resolution, payload size, camera count, pending encoder
  work, publish duration, and browser polling are bounded.
- Pub/Sub loss and API restart simply cause the UI to wait for the next frame;
  there is deliberately no replay or history.
- State and frame requests must reach the same API process because sequence
  buffers are local. Sticky routing or a shared bounded cache is required before
  horizontally scaling this path.
- JPEG/base64 Pub/Sub is suitable for a low-rate operational preview but not a
  full-frame-rate viewing service. HLS/WebRTC can later implement a replaceable
  stream-gateway port while retaining this overlay contract.

