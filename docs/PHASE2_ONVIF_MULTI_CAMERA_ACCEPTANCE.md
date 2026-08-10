# Phase 2 ONVIF and Multi-Camera Ingress Acceptance Record

## Decision

The ONVIF discovery and multi-camera ingress-hardening milestone passed
engineering acceptance on 2026-08-09. It accepts bounded credential-free
WS-Discovery, explicit batch camera admission, separate configured/active worker
capacities, start-rate limiting, capped per-camera crash backoff, audit/RBAC, and
the Angular onboarding workflow.

It does not accept ONVIF authentication/media-profile provisioning, automatic
RTSP derivation, routed subnet scanning, cross-replica transactional admission,
or shared-GPU scheduling.

## Accepted flow

```text
OPERATOR/ADMIN scan
  -> modern + legacy ONVIF WS-Discovery probes
  -> bounded untrusted-response parser
  -> ephemeral credential-free device metadata
  -> optional ADMIN form prefill
  -> separately supplied RTSP URL
  -> AES-256-GCM camera persistence
  -> configured-camera admission
  -> capacity/start-rate bounded supervisor
```

The design follows ONVIF's WS-Discovery model documented in the official
[specifications index](https://www.onvif.org/profiles/specifications/) and
[Device Feature Discovery specification](https://www.onvif.org/wp-content/uploads/2026/01/ONVIF_Device_Feature_Discovery_Specification_25.06.pdf).

## Automated evidence

- Ruff and Python bytecode compilation passed across source, entry points, and
  tests.
- The complete suite passed **133 tests** against MongoDB 8, Redis 8, and MinIO
  with no skips. Coverage includes safe XML parsing, malformed/entity/credential
  rejection, deterministic deduplication, real UDP loopback exchange, batch
  conflict/capacity outcomes, API RBAC/redaction/audit, configuration bounds,
  supervisor active/start limits, stability reset, and exponential backoff.
- A production-image smoke initially exposed that `uvloop` does not implement
  the low-level UDP coroutine primitive used by the first adapter. The adapter
  was corrected to run deadline-bounded blocking UDP in an isolated thread; unit,
  real-UDP, full-suite, rebuilt-image, and Nginx smoke tests then all passed.
- Strict application/spec TypeScript typecheck passed. All **30 Vitest tests**
  passed, including deterministic ONVIF service-address preference and safe
  suggested camera IDs.
- The Angular production build passed at **364.79 kB raw / 99.08 kB estimated
  initial transfer**. The camera route remains lazy at **19.75 kB raw / 5.42 kB
  estimated transfer**. The production dependency audit reported zero
  vulnerabilities.
- Compose validation, API/web production image builds, and a host-native
  supervisor one-pass reconciliation passed. The empty reconciliation reported
  zero starts/crashes/failures/capacity deferrals.
- Through same-origin Nginx, health reported `cameraManagement=available` and
  `onvifDiscovery=available`; a real bounded multicast scan returned `200` with
  an empty local result rather than failing or hanging. The `/cameras` SPA route
  returned `200` with the expected security headers.
- A two-camera batch returned two explicit `CREATED` outcomes and no RTSP URL,
  ciphertext, or submitted secret. Raw MongoDB inspection confirmed two
  encrypted tokens, no plaintext `rtspUrl` field, and no plaintext marker.
  Discovery and both creates produced append-only audit records.
- Both scoped cameras were deleted through the API. MongoDB confirmed zero
  scoped camera/health records while two create, two delete, and discovery audit
  records remained. Final API/Nginx logs contained only successful requests and
  no application traceback.

The suite emits one existing FastAPI/Starlette TestClient deprecation warning.
The local host build used unsupported odd Node 23, which disabled only an
optional compiler cache; the production container build used pinned Node
24.12.0 and npm 11.10.1.

## Security and operational limits

Discovery never receives camera credentials, never logs response payloads, and
does not persist results. UDP response size, total timeout, retry count,
multicast TTL, and result count are configurable bounds. XML DTD/entities,
malformed responses, non-HTTP(S) service addresses, and URL user-info are
discarded. Discovery requires `TEST_CAMERAS`; batch creation requires
`MANAGE_CAMERAS`.

The configured-camera capacity check is an application-level guard and is not a
strict semaphore across concurrent API replicas. Production deployments that
need a licensing-grade hard limit must use a MongoDB replica-set transaction or
single admission writer. Worker capacity and per-cycle starts remain enforced
by each supervisor process. ONVIF multicast normally remains within the API
container's broadcast domain; segmented networks need an explicitly deployed
edge-local discovery component or controlled relay.

