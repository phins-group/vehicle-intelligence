# ADR-024: Fair latest-frame scheduling at the edge

- Status: accepted
- Date: 2026-08-10

## Context

A shared accelerator can improve utilization, but a FIFO shared queue lets one
high-rate camera increase latency for every other camera and can grow without
bound. Edge binaries also need deterministic model/config identity.

## Decision

Use bounded per-camera latest-frame queues and a ready-camera round robin. Drop
oldest/stale frames explicitly, batch only within a configured latency window,
and expose fairness/drop/latency benchmark gates. Package the optimized edge
worker as non-root and admit model artifacts only through a path-contained,
byte-size and SHA-256 verified manifest with mandatory execution providers.

## Consequences

- Overload sacrifices old frames while preserving current observations and
  fairness.
- A provider with a real batch method can share one inference call; scalar
  providers remain correct and visibly scalar.
- A missing/tampered model or unavailable accelerator stops startup.
- One shared accelerator process has a larger blast radius than isolated camera
  processes, so it is an explicit deployment choice and requires local gates.
