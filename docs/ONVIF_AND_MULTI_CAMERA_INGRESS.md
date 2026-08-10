# ONVIF Discovery and Multi-Camera Ingress

## Accepted boundary

ONVIF support is a bounded, credential-free discovery aid. It sends WS-Discovery
probes for the current `tds:Device` type and the legacy
`dn:NetworkVideoTransmitter` type, then returns safe device metadata. It does not
authenticate to a camera, negotiate a media profile, derive an RTSP URL, or add
a camera automatically.

This follows ONVIF's discovery model: devices are found through WS-Discovery and
successful probe matches advertise device-service addresses. See the official
[ONVIF specifications index](https://www.onvif.org/profiles/specifications/) and
[Device Feature Discovery specification](https://www.onvif.org/wp-content/uploads/2026/01/ONVIF_Device_Feature_Discovery_Specification_25.06.pdf).

```text
operator scan
  -> bounded UDP multicast probes
  -> size-limited XML responses
  -> safe metadata parsing/deduplication
  -> temporary API/UI result
  -> explicit ADMIN camera creation with separately supplied RTSP credential
```

Discovery results contain endpoint reference, HTTP(S) device-service addresses,
device types, scopes, optional name/hardware/location, sender address, metadata
version, and discovery timestamp. They never contain an RTSP credential or an
encrypted camera token and are not persisted by the scan operation.

## Network and parser limits

`onvif_discovery` configures enablement, IPv4 multicast address/port, optional
interface address, timeout, retry count, multicast TTL, result count, and maximum
response bytes. The defaults target `239.255.255.250:3702`, two probes within a
three-second total deadline, a local-network TTL, at most 128 devices, and one
UDP datagram per response.

The adapter rejects oversized datagrams, malformed XML, DTD/entity declarations,
missing endpoint/service addresses, non-HTTP(S) service addresses, and embedded
URL credentials. Results are deduplicated case-insensitively by endpoint
reference and sorted deterministically. Socket and parsing errors are mapped to
a safe `503`; raw network payloads are not logged.

Multicast normally reaches only the API container's broadcast domain. Routed
VLAN discovery requires an explicitly designed relay or an edge-local API; the
platform does not increase TTL or scan arbitrary subnets automatically.

## Camera admission and worker fairness

`POST /api/cameras/batch` accepts a bounded list and returns an explicit outcome
for each input: `CREATED`, `CONFLICT`, or `CAPACITY_REACHED`. Duplicate IDs in a
request are rejected. It applies the same validation, credential encryption, and
audit semantics as single-camera creation. Public responses expose neither the
submitted RTSP URL nor ciphertext.

The configured-camera limit protects central state. The supervisor separately
caps active processes and the number of starts per reconciliation cycle. Pending
cameras are admitted in deterministic camera-ID order on later cycles. Each
camera maintains its own capped exponential crash backoff, and a worker must
remain stable for the configured interval before its failure count resets. This
prevents one crash loop or a large import from creating an unbounded process
storm while preserving camera-level failure isolation.

Capacity checks are application-level in this milestone. Multiple API replicas
creating cameras concurrently require a transactional admission counter or a
single writer to make the global configured-camera limit strict. MongoDB unique
camera IDs still prevent duplicate records.

## API and RBAC

```text
POST /api/cameras/discover  TEST_CAMERAS (OPERATOR, ADMIN)
POST /api/cameras/batch     MANAGE_CAMERAS (ADMIN)
```

Each successful scan appends a redacted audit record containing only the number
of results. Each created batch item gets the normal camera-create audit evidence
with `BATCH` source metadata. Discovery failure does not create or mutate camera
records.

The Angular camera screen presents scan results as temporary cards. Selecting a
device only prefills a safe camera ID, name, location, and ONVIF provenance; the
operator must provide the RTSP URL through the existing password input before
creation.

