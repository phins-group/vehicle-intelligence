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

## Durable finalization outbox

With `finalization_outbox.enabled=true`, a completed track is first encoded as
one versioned event envelope plus its JPEG evidence and staged below
`storage.output_directory/finalization-outbox`. The camera-scoped entry is written
mode `0600`, flushed with `fsync`, atomically renamed, and its directory is flushed
before the track is marked finalized. MinIO, Redis, MongoDB, and direct-repository
delivery therefore happen only after a locally durable commit.

The worker wakes delivery immediately without blocking track finalization on an
external call, replays existing entries at startup, and continues replaying at
`replay_interval_seconds`. Each attempt and the final shutdown drain are bounded
by `delivery_timeout_seconds`; a finite file run therefore gets one bounded last
chance to reach healthy JSONL/MinIO/event delivery while still retaining pending
entries for the next startup. It keeps an entry if any media write or event
publish fails. Media keys and event IDs are deterministic, so object and
repository replays are idempotent. A process crash after a Redis `XADD` succeeds
but before the local entry is removed can emit the broker message again; consumers
must retain event-ID idempotency. Replay is chronological by event `occurredAt`
within a camera namespace, with a deterministic event-ID hash tie-break. Deploy
only one active worker for a camera ID; overlapping same-camera processes can race
after reading the same entry and produce the same permitted at-least-once duplicate.

`maximum_entries`, `maximum_bytes`, and `maximum_entry_bytes` are hard bounds per
camera namespace, not node-global disk quotas. When a bound is reached the
current track remains unfinalized and the camera worker exits non-zero instead of
continuing to admit unbounded in-memory tracks; its supervisor/operator retry
boundary provides backoff. The outbox never drops the oldest evidence. Strict
schema, path, permission, size, and SHA-256 checks reject corrupt or replaced
entries rather than publishing or deleting them. Size a node for at least
`camera_count * maximum_bytes` plus normal media/event storage and free-space
headroom, alert on both queue occupancy and filesystem free space, and preserve
the configured output directory across worker restarts.

`maximum_entry_bytes` defaults to 32 MiB and also has a 32 MiB configuration ceiling.
Replay can temporarily hold the JSON/base64 representation, decoded JPEG bytes,
and the MinIO SDK's single-part buffer at the same time. Budget each camera worker
for several times this entry limit per active delivery and confirm peak RSS on the
target edge hardware. MinIO writes of in-memory bytes use one HTTP PUT with no
multipart worker fanout; payloads above the S3 5 GiB single-PUT limit are rejected.

For MinIO plus an enabled outbox, configuration validation reserves the first-use
bucket HEAD, optional bucket creation, three media PUTs, and the configured event
publisher timeout inside `delivery_timeout_seconds`. A required explicit MinIO
region prevents a discovery request. LAN defaults use bounded 5-second connect
and 3-second read timeouts with no SDK retry; urllib3 also applies the connect
timeout while sending the request body, so deployment acceptance must verify the
largest evidence object uploads within five seconds at measured p99 throughput.
Raise both that timeout and the outbox delivery deadline together when a slower
link is accepted. Durable outbox replay owns retry and backoff across delivery
attempts.

These hashes detect accidental corruption and replacement that does not also
forge the manifest; they are not keyed authentication. A process with write
access as the camera worker's Unix UID can recompute them, so isolate that UID
and volume and treat such access as a compromised edge node.

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

## OCR load shaping

OCR remains track-aware so sustained views of one vehicle do not repeatedly run
the recognizer without adding useful evidence. The first eligible plate crop is
recognized immediately. Later crops are recognized at most once every
`vision.ocr.track_frame_interval` observed frames for that track. Plate detection,
quality evaluation, tracking, and the live overlay still run on intervening
frames.

Within one OCR attempt, `variant_early_stop_confidence` stops preprocessing
variants only after a result both normalizes to a valid plate and reaches the
configured confidence. Across attempts, OCR stops for the track after
`consensus_stop_min_observations` identical, complete normalized reads at or
above `consensus_stop_min_confidence`. Partial reads never satisfy this consensus.
Set either stop value to `null` to disable that stop. The pre-optimization
behavior is available with `track_frame_interval: 1`,
`variant_early_stop_confidence: null`, and
`consensus_stop_min_observations: null`.

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

With `gpu_scheduler.enabled=true`, the supervisor first owns one shared inference
daemon and only then spawns camera workers. Each worker receives a camera-bound
capability and connects to the configured private Unix socket; vehicle and plate
crops from concurrent cameras are fairly multiplexed into detector batches. A
daemon crash or detector watchdog timeout stops dependent workers, waits for the
capped restart backoff, restarts the daemon, and only then recreates workers.
When disabled, every worker continues to compose its existing local detectors.

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
is reserved atomically by the repository across API replicas; active-worker
capacity remains a separate supervisor control. See
[ONVIF and multi-camera ingress](ONVIF_AND_MULTI_CAMERA_INGRESS.md) for network,
security, and cross-replica limitations.

## Local relay

The `rtsp-dev` Compose profile provides MediaMTX on port `8554`. It is optional
and is not started by the normal `docker compose up` command. See the root README
for the FFmpeg publisher and worker commands.

## Current limitations

- The supervisor keeps one process per camera for decoding, tracking, OCR, and
  failure isolation. When shared inference is enabled, the vehicle/plate daemon
  is intentionally a single host-local failure domain; its watchdog failure
  stops dependent workers before the supervisor performs a backed-off restart.
- The JSONL repository is a single-writer local development fallback and builds
  an in-memory query/idempotency index. Separate processes can otherwise rewrite
  from stale snapshots; use Redis plus MongoDB for managed, long-running,
  multi-process, or high-volume workers.
- The finalization outbox closes the event/media outage gap only when
  `storage.output_directory` is persistent. Redis delivery remains at-least-once
  across a crash after publish, so downstream event-ID idempotency is required.
- ONVIF media-profile provisioning is not implemented; credential-free discovery
  and bounded camera admission are implemented.
- API-key and OIDC/JWKS Bearer authentication, RBAC, redacted audit, replica-set
  transaction boundaries, and online AES-GCM keyring rotation are implemented.
  First-party identity-provider/user lifecycle remains external to the platform.
- ROI/line geometry is accepted as camera configuration, but bounds validation
  and automatic coordinate transforms after a resolution change are not yet
  implemented.
