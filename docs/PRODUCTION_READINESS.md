# Production Readiness Gate

`vehicle-production-readiness` converts the static production checklist into a
secret-safe, machine-readable gate. It validates configuration and local model
artifact integrity without opening database, Redis, MinIO, camera, or identity
provider connections.

## Run

```bash
vehicle-production-readiness \
  --config configs/default.yaml \
  --base-directory /opt/vehicle-intelligence \
  --output output/production-readiness.json
```

Equivalent source-tree command:

```bash
python run_production_readiness.py \
  --config configs/default.yaml \
  --base-directory "$PWD" \
  --output output/production-readiness.json
```

Use `--strict-warnings` when the deployment policy requires every warning to be
resolved. Exit status is `0` when the gate is ready, `4` when failures remain (or
strict warnings remain), and `2` for invalid configuration.

The base `configs/default.yaml` is deliberately a development configuration and
must fail this gate when run without overrides. A passing report is not produced
by weakening the checks; production supplies `VIP_*` environment/secret-manager
overrides or passes an independently managed complete configuration file.

## Checked invariants

- explicit production environment and immutable configuration version;
- enabled API-key or OIDC/JWKS authentication;
- valid URL-safe base64 AES-256-GCM camera credential keyring;
- MongoDB enabled with transaction boundaries and replica-set/SRV topology;
- Redis Streams event bus plus Redis auth/TLS posture warnings;
- MinIO canonical storage, non-default credentials, signed-media public endpoint,
  and transport posture;
- coordinated retention, Prometheus, optional tracing, and realtime state;
- production debug artifacts disabled;
- present, non-empty, explicitly versioned vehicle/plate artifacts whose SHA-256
  matches configuration;
- OCR version/hash posture and optional ReID embedding artifact integrity;
- HTTPS-only explicit external action targets when egress actions are enabled.

The report never serializes Mongo/Redis URIs, access keys, secret keys, bearer
tokens, camera encryption material, or credential values.

## What a pass does not prove

This is a static deployment-input gate. Production release still requires:

1. the full MongoDB/Redis/MinIO test suite;
2. `scripts/verify_performance_gates.py` over immutable benchmark evidence;
3. backup and restore rehearsal;
4. target-node CUDA/TensorRT and capacity gates where applicable;
5. representative Vietnamese plate accuracy evaluation;
6. real video, RTSP reconnect, camera geometry, and site network acceptance;
7. live OIDC/JWKS, TLS, DNS, firewall, secret-manager, and alert/action tests.

The current workspace contains vehicle YOLO11n artifacts only. It has no
representative video or licensed/evaluated Vietnamese plate checkpoint, so the
default preflight correctly remains not ready for a live production ANPR rollout.
