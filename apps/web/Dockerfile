FROM node:24.15.0-alpine@sha256:d1b3b4da11eefd5941e7f0b9cf17783fc99d9c6fc34884a665f40a06dbdfc94f AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --ignore-scripts
COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.30.4-alpine3.24@sha256:93722936b82ec8a1178d48448e619226680d2de3706a1640800e186cd5fa7fd3

ENV WEB_API_UPSTREAM=http://api:8000 \
    WEB_MEDIA_UPSTREAM=http://minio:9000 \
    WEB_REALTIME_CONNECT_SRC="ws: wss:" \
    WEB_OIDC_CONNECT_SRC=""

COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist/web/browser /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD wget -q -O - http://127.0.0.1:8080/healthz >/dev/null || exit 1
