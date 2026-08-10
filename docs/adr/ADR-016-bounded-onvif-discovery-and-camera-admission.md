# ADR-016: Bounded ONVIF Discovery and Camera Admission

## Status

Accepted — 2026-08-09

## Context

Operators need to locate cameras on a local network and onboard several devices
without weakening the existing RTSP credential boundary. A multicast responder
is untrusted network input, while a large import or repeated worker crash can
exhaust API, database, CPU, or GPU capacity.

## Decision

Implement ONVIF WS-Discovery behind an application port and a bounded UDP
adapter. Return credential-free metadata only, keep results ephemeral, and
require explicit privileged camera creation with a separately supplied RTSP
credential. Probe both modern device and legacy network-video types, reject
unsafe XML/addresses, and make every network/result bound configurable.

Separate three controls:

- configured-camera admission bounds central state;
- active-worker capacity bounds simultaneous camera processes;
- per-cycle starts plus per-camera capped backoff bounds process storms.

Batch creation returns per-item outcomes instead of silently dropping conflicts
or capacity rejections. Discovery and creation retain independent RBAC and audit
events.

## Consequences

The implementation is testable without an ONVIF SDK and can be replaced by a
device-management provider without changing domain/API contracts. Discovery is
safe to disable and cannot leak camera credentials because it never receives
them. It intentionally does not implement ONVIF authentication, media-profile
selection, routed subnet scanning, or automatic RTSP provisioning.

Application-level capacity is deterministic in one API writer but is not a
strict cross-replica semaphore. A replica-set transaction/admission counter is
required before relying on it as a global licensing or hard resource limit.

