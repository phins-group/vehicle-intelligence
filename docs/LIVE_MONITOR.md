# Live Monitor

## Scope

The Phase 2 live monitor is an optional low-rate preview for operators. It is
not continuous video storage, an event clip, or a replacement for the RTSP
source. The edge worker publishes only sampled, resized JPEGs with the matching
AI overlay metadata when `live_monitor.enabled=true`.

The normal vehicle-event WebSocket never carries images. Live packets use a
separate best-effort Redis Pub/Sub channel and are never persisted to MongoDB,
MinIO, Redis Streams, or browser storage.

## Runtime flow

```text
sampled inference frame
  -> vehicle/plate/track overlay metadata
  -> one-slot latest-frame reporter
  -> background resize + JPEG encode
  -> versioned, size-capped Redis Pub/Sub packet
  -> API subscriber with capped reconnect
  -> bounded three-frame ring per camera
  -> authenticated state JSON + exact JPEG HTTP request
  -> Angular SVG overlay
```

The edge reporter defaults to 2 FPS and 960 pixels maximum width. Its input
queue holds one pending frame; a newer frame replaces an older pending frame.
Encoding runs outside the pipeline thread, publishing has a bounded timeout,
and failures are logged/counted without stopping detection, tracking, OCR, or
event finalization.

## Contract

The Redis contract is schema v1 and includes:

- camera ID, frame ID, stream epoch, captured timestamp;
- source and preview dimensions;
- vehicle bounding boxes, confidence, type, track ID, and direction;
- optional plate bounding box, detection/quality/OCR confidence, and text;
- optional vehicle ROI and crossing line;
- one bounded base64-encoded JPEG for cross-process transport.

Coordinates remain in source-frame pixels. The browser uses an SVG `viewBox`
matching the source dimensions, so geometry stays aligned when the preview is
responsive. The message codec rejects unknown fields, unsupported schema
versions, invalid base64, invalid boxes/confidences, and oversized payloads.

## API

All endpoints require `READ_PLATFORM`:

```http
GET /api/cameras/{cameraId}/live
GET /api/cameras/{cameraId}/live/frame?sequence=123
GET /api/live-monitor/health
```

The state response contains metadata and a same-origin `frameUrl`, but no image
bytes or RTSP URL. The browser fetches that exact sequence with its normal
Bearer header. The API returns `410` when a sequence has already fallen out of
the bounded ring. Both metadata and JPEG responses are `no-store`; the JPEG
response echoes sequence/frame/epoch headers so the UI can reject a mismatched
pair.

Camera state is `DISABLED`, `WAITING`, `LIVE`, `STALE`, or `OFFLINE`. A preview
can be stale while canonical event history and the camera worker remain
otherwise available.

## Angular behavior

`/live-monitor?camera=gate-01` is a lazy, shareable route. It polls only while
the tab is visible, revokes replaced Blob URLs, and never puts frames or API
keys in persistent browser storage. Operators can toggle vehicle boxes, plate
boxes, track IDs, plate text, direction, confidence, ROI, and crossing line.

The camera page links directly to the selected monitor. All roles may read the
monitor; hiding controls is not an authorization boundary.

## Configuration

```yaml
live_monitor:
  enabled: false
  redis_channel: vehicle.live.frames
  preview_fps: 2.0
  preview_max_width: 960
  jpeg_quality: 72
  maximum_payload_bytes: 750000
  publish_timeout_seconds: 1.0
  frame_buffer_size: 3
  maximum_cameras: 256
  stale_after_seconds: 5.0
```

Preview is disabled by default in the portable YAML configuration. Enable it
for both the API and host-native camera workers and point them at the same Redis
and channel. The Compose API enables its subscriber by default; a host-native
supervisor passes the corresponding environment to its children.

## Limits

- This is a low-rate operational preview, not HLS/WebRTC or a 25 FPS viewing
  service. A future `LiveStreamGateway` can add those transports without moving
  overlay metadata into the event channel.
- Redis Pub/Sub is intentionally lossy. There is no replay after an API restart;
  the UI waits for the next frame.
- Sequences are local to an API process. A load balancer must use sticky routing
  for state/frame pairs or replace the in-process ring with a shared bounded
  frame cache.
- Base64 adds transport overhead. Payload, frame rate, resolution, camera count,
  per-camera frame ring, and publish time are all bounded to keep that tradeoff
  explicit.
- Preview access uses the platform Bearer lifecycle and accepts either API-key or
  OIDC/JWKS JWT authentication. Browser authorization-code/PKCE or cookie-session
  login is not implemented, and ingress connection/rate limits remain a
  deployment responsibility.
