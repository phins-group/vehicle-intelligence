# Camera Pipeline (Phase 2 Camera Management)

The RTSP adapter is implemented without changing the Phase 1 vision or domain
ports. `run_camera.py` runs one explicit camera; the central manager and
`vehicle-camera-supervisor` reconcile any number of configured cameras into one
isolated worker process per camera.

```text
RTSP -> OpenCV/FFmpeg decoder -> FPS sampler -> bounded latest-frame queue -> vision pipeline
```

The decoder runs in an isolated background thread. Queue capacity is configurable
and defaults to three. When it is full the producer drops the oldest frame; when
the consumer wakes it selects the newest frame and discards any remaining stale
ones. This makes memory bounded and prioritizes realtime freshness over processing
every frame.

When no frame arrives, the source emits an internal image-free heartbeat. The
heartbeat never enters inference; it only advances track timeout checks. A track
can therefore finalize while a camera remains offline instead of waiting for a
future reconnect.

## Reconnect and track identity

Open/read timeouts and capped exponential reconnect delays are configuration.
The first successful connection uses stream epoch `0`; every successful recovery
after a disconnect increments that epoch. The pipeline finalizes tracks from the
old epoch, resets ByteTrack, and includes the epoch in the logical track session:

```text
gate-01:rtsp-<session>:42
gate-01:rtsp-<session>-e1:42
```

This prevents a tracker-local ID reused after reconnect from being mistaken for
the previous vehicle. Completed track suppression expires after the configured
tracking timeout, so a long-running worker does not accumulate track IDs forever.
Normal EOF, `Ctrl-C`, and task cancellation finalize active tracks once; unexpected
pipeline failures close resources without publishing partial events.

An unexpected decoder-thread failure is surfaced as a `VideoSourceError` so the
worker exits non-zero for its process supervisor; it is never converted into a
normal empty stream.

## Security

Prefer `--rtsp-env ENV_NAME` for direct workers. Managed credentials are accepted
by the camera API, encrypted with AES-256-GCM before MongoDB storage, and never
returned by API responses. Each encryption uses a fresh 12-byte nonce and binds
the token to its camera ID as authenticated associated data. The process
supervisor passes the decrypted URL only through a child environment variable;
it is absent from command arguments and logs, and the child does not inherit the
encryption key.

## Health

Health is a latest-state document plus aggregated counters: online state, source,
decode FPS, queue depth, drops, last frame, reconnect count, stream epoch, and
update time. The worker throttles upserts to
`camera_manager.health_publish_interval_seconds` and forces a terminal state on
close. It also reports decoded/sampled frames, detection/OCR/event counts,
active tracks, inference FPS, and cumulative-average vehicle/plate/OCR latency.
MongoDB retains one document per camera, not one document per frame; the API
renders that state as low-cardinality Prometheus metrics.

`reconnect_count` means successful recoveries and therefore matches the stream
epoch after the initial connection. `connection_failures` separately counts lost
connections and failed open attempts.

## Optional live preview

When `live_monitor.enabled=true`, the camera worker emits a low-rate operational
preview after tracking. One pending frame is retained, older pending previews are
dropped, and resize/JPEG/publish work runs outside the inference path. Redis
Pub/Sub and API buffers are bounded; no preview is written to MongoDB or object
storage, and failures do not stop track finalization. See
[Live Monitor](LIVE_MONITOR.md) for the contract, API, and deployment limits.

## Desired-state reconciliation

The supervisor periodically reads enabled cameras from MongoDB and maintains one
child process for each configuration revision. Disable/delete stops the child;
an update restarts it with the new revision; an unexpected exit affects only
that camera and is retried after capped exponential, per-camera backoff. Active
workers and new starts per reconciliation cycle are independently bounded. A
worker must remain alive for the configured stability interval before its crash
count resets, so rapid failures cannot continuously restart at the minimum
delay. Supervisor-generated states
cover `CONNECTING`, `OFFLINE`, and `STOPPED`; live source metrics overwrite them
once the worker is running.

Connection tests are I/O work executed outside the FastAPI event loop and are
bounded by a configurable semaphore. Results expose only success, latency, frame
dimensions, or a safe error code—never the tested URL or decoder exception text.

## ONVIF discovery and batch admission

The API can send bounded WS-Discovery probes on the local broadcast domain and
return temporary credential-free ONVIF metadata. It never authenticates to the
device, persists a scan result, or derives an RTSP credential. An operator may
use the metadata to prefill the camera form; an ADMIN must still submit the RTSP
URL through the encrypted camera-create path.

Batch creation has a configured request-size limit and explicit per-item
`CREATED`, `CONFLICT`, or `CAPACITY_REACHED` outcomes. Configured-camera capacity
and active-worker capacity are separate controls. See
[ONVIF and multi-camera ingress](ONVIF_AND_MULTI_CAMERA_INGRESS.md) for network,
security, and cross-replica limitations.

## Local relay

The `rtsp-dev` Compose profile provides MediaMTX on port `8554`. It is optional
and is not started by the normal `docker compose up` command. See the root README
for the FFmpeg publisher and worker commands.

## Current limitations

- The supervisor keeps one process per camera as the isolation-first default.
  Fair latest-frame scheduling and real batch coordination are implemented as an
  explicit shared-worker composition, but the supervisor does not switch to it
  automatically.
- The JSONL repository is a local development fallback and builds an in-memory
  query/idempotency index; use MongoDB for a long-running or high-volume worker.
- Redis Streams plus idempotent Mongo processing provide durable event retry;
  media writes still have no separate local object-storage outage journal.
- ONVIF media-profile provisioning is not implemented; credential-free discovery
  and bounded camera admission are implemented.
- API-key and OIDC/JWKS Bearer authentication, RBAC, redacted audit, replica-set
  transaction boundaries, and online AES-GCM keyring rotation are implemented.
  First-party identity-provider/user lifecycle remains external to the platform.
- ROI/line geometry is accepted as camera configuration, but bounds validation
  and automatic coordinate transforms after a resolution change are not yet
  implemented.
