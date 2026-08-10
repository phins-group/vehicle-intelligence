# Phase 2 RTSP Foundation Acceptance Record

## Decision

The single-camera RTSP foundation passed an engineering acceptance run on
2026-08-09. One worker consumed a real localhost RTSP stream, survived a publisher
disconnect, incremented its stream epoch after recovery, finalized the old track
once, then finalized the new-epoch track on one `Ctrl-C`.

This accepts the RTSP transport/lifecycle path, not production ANPR accuracy or
the remaining Phase 2 camera-management scope.

## Test setup

- Relay: MediaMTX Docker image `bluenviron/mediamtx:1`, TCP RTSP on port `8554`.
- Publisher: FFmpeg realtime loop of the temporary Phase 1 H.264 acceptance video.
- Test reconnect overrides: 0.2-second initial delay, 1-second cap, and 1-second
  read timeout. Production defaults remain more conservative.
- Vision providers: real Ultralytics vehicle/plate inference, ByteTrack, and
  PaddleOCR on CPU.
- Media/model provenance and accuracy caveats are recorded in
  [PHASE1_ACCEPTANCE.md](PHASE1_ACCEPTANCE.md). Test media and weights are not
  committed to this repository.

## Exercised sequence

```text
publisher starts
  -> camera ONLINE, epoch 0
  -> observations accumulate
publisher stops
  -> camera OFFLINE, capped retries
  -> image-free heartbeat reaches tracking timeout
  -> epoch-0 track finalized and emitted once while still offline
publisher restarts
  -> camera ONLINE, epoch 1
one Ctrl-C
  -> epoch-1 track finalized and emitted once
  -> decoder/resources close
  -> process exits 130
```

## Result

- Exactly two JSONL documents were written.
- Track IDs were distinct: the second included `-e1` in its source session.
- Both temporal aggregates normalized to `11A-015.68`; they contained 53 and 64
  OCR observations respectively.
- Exactly six JPEG objects were written: snapshot, vehicle crop, and plate crop
  for each event.
- Final worker stats reported two finalized tracks and bounded stale-frame drops.
- Graceful shutdown completed from one signal without a second forced interrupt.
- Health distinguished one successful reconnect from failed connection attempts.
- Credential redaction, newest-frame selection, queue-drop accounting, epoch
  changes, event non-retention, and cancellation finalization also have automated
  tests.

## Remaining production acceptance

Run soak tests against each intended camera and network profile. Measure recovery
time, long-disconnect behavior, decode/inference FPS, memory stability, RTSP auth
variants, resolution changes, corrupt packets, day/night accuracy, and shutdown
under blocked device reads. The current worker is one camera per process and does
not yet provide central camera CRUD, ONVIF, persisted health, or a multi-camera
supervisor. Redis delivery is accepted separately in
[PHASE2_EVENT_BUS_ACCEPTANCE.md](PHASE2_EVENT_BUS_ACCEPTANCE.md).
