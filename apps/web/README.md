# Vehicle Intelligence Web

Angular 21 operator console for the Vehicle Intelligence API. The application
uses standalone lazy routes, strict TypeScript, signals for local state, RxJS for
realtime coordination, and the official Lucide Angular icon components.

## Development

Use the Node version pinned in .nvmrc, start FastAPI on port 8000, then run:

    nvm use
    npm ci
    npm start

The dev server listens on http://localhost:4200 and proxies /api and /ws to the
backend. API keys are tab-scoped in sessionStorage and must not be placed in
environment files or source code.

## Verification

    npm run typecheck
    npm test
    npm run build
    npm audit --omit=dev

## Container

The multi-stage Dockerfile compiles the application and serves only its static
output from Nginx. From the repository root:

    docker compose up -d mongodb redis api web

See ../../docs/WEB_DASHBOARD.md for architecture, route capabilities, RBAC,
realtime recovery, and operational limits.
