# Production deployment profile

`docker-compose.production.yml` is a fail-closed application profile for an
on-premises pilot or a small production installation. It runs immutable API and
web images, the durable event and retention workers, Prometheus, and an
OpenTelemetry Collector with a disk-backed export queue. MongoDB, Redis, MinIO,
the OIDC provider, TLS ingress, the trace backend, and alert notification routing
remain site-managed services.

The profile is not a substitute for MongoDB/MinIO high availability or a backup
product. Do not point it at the development services from `docker-compose.yml`.

## Required boundaries

- Terminate HTTPS before the web port. The example binds only to `127.0.0.1` so
  an approved reverse proxy or load balancer owns certificates and public ingress.
- Use authenticated replica-set/SRV MongoDB, authenticated TLS Redis, and private
  TLS MinIO. Rehearse an isolated restore before admitting camera traffic.
- Register the exact `<console-origin>/login` OIDC redirect URI. The public
  client must require Authorization Code with PKCE S256 and must not have a
  browser-distributed client secret.
- Allow CORS on the IdP token endpoint only for the exact console origin. Set
  `VIP_WEB_OIDC_CONNECT_SRC` to the token endpoint origin so CSP permits that
  exchange; do not use `*`.
- Set `VIP_WEB_REALTIME_CONNECT_SRC` to the exact public `wss://` console origin.
  The production profile does not permit scheme-wide WebSocket destinations.
- The media proxy verifies the upstream MinIO certificate and sends SNI. If the
  site uses a private CA, extend or mount the trusted CA bundle at
  `/etc/ssl/certs/ca-certificates.crt`; do not disable `proxy_ssl_verify`.
- Supply release images as immutable `tag@sha256:digest` references and mount a
  read-only model directory whose hashes pass the production-readiness gate.
- Keep every secret in a mode-0600 file. `VIP_*_FILE` variables contain paths,
  not secret values; startup fails when a file is absent, empty, oversized, or
  configured together with its direct environment equivalent.

## Render and start

Copy `deployment/production.env.example` to a site-owned protected path and
replace every placeholder. The example file intentionally contains no
credentials.

```bash
docker compose \
  --env-file /approved/config/vehicle-intelligence.env \
  -f docker-compose.production.yml \
  config --quiet

docker compose \
  --env-file /approved/config/vehicle-intelligence.env \
  -f docker-compose.production.yml \
  up -d
```

Run the static gate inside the same immutable API image and with the same
environment, secrets, model, and data mounts. Do not waive warnings in the
production change record.

```bash
docker compose \
  --env-file /approved/config/vehicle-intelligence.env \
  -f docker-compose.production.yml \
  run --rm api vehicle-production-readiness \
    --config configs/default.yaml \
    --base-directory /app \
    --strict-warnings \
    --output /app/datasets/evidence/production-readiness.json
```

Then verify `/livez`, `/readyz`, OIDC login/logout, RBAC, signed media, realtime,
backup/restore, and the target-hardware load gates from the pilot runbook.

## Observability and alerts

Prometheus loads `infrastructure/prometheus/rules/vehicle-intelligence.yml`.
Thresholds are conservative starting points and must be replaced by the signed
site SLO after load acceptance. Connect Prometheus to a site-owned Alertmanager
or monitoring service and prove one routed test alert before go-live.

The production collector writes queued spans to the `otel_storage` volume,
fsyncs queue updates, bounds serialized queued trace data at 256 MiB, compacts
it on start and after backlog recovery, and retries indefinitely toward
`VIP_OTEL_TRACES_EXPORT_ENDPOINT`. Database overhead is outside that serialized
payload bound, so put the volume on a monitored partition with a site-owned
quota. Alert before the disk or queue limit is reached and accept the documented
loss boundary; telemetry is not a canonical application datastore.

Container JSON logs use Docker's bounded local driver and already contain
request, trace, and span correlation fields. Forward them with the site's log
agent and alert on camera/offline, API error/latency, retention failure, Redis
pending/DLQ, outbox age, disk, and GPU signals available in that deployment.

## Rollback

Keep the previous image digests and configuration version. Rollback changes only
those immutable references after confirming database/schema compatibility; it
must not delete volumes or restore over the active target. Repeat `/readyz`,
OIDC, signed-media, and event-delivery checks after rollback.
