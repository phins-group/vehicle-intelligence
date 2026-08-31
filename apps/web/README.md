# Vehicle Intelligence Web

Angular 22 operator console for the Vehicle Intelligence API. The application
uses standalone lazy routes, strict TypeScript, signals for local state, RxJS for
realtime coordination, and the official Lucide Angular icon components.

## Development

Use the Node version pinned in .nvmrc, start FastAPI on port 8000, then run:

    nvm use
    npm ci
    npm start

The dev server listens on http://localhost:4200 and proxies /api and /ws to the
backend. Production uses OIDC Authorization Code with PKCE; access tokens remain
in memory. API-key mode is retained for development and stores keys only in
tab-scoped sessionStorage. Never place credentials in environment files or
source code.

## Verification

    npm run lint
    npm run typecheck
    npm test
    npx playwright install chromium
    npm run test:e2e
    npm run test:contracts
    npm run build
    npm audit --package-lock-only --omit=dev --audit-level=low
    npm audit --package-lock-only --audit-level=low

The Playwright gate starts an isolated dev server on `127.0.0.1:4300` and mocks
only the API and identity-provider boundaries. It covers route guards, OIDC
Authorization Code + PKCE, fail-closed callback state validation, in-memory OIDC
tokens, and operator RBAC. On a workstation with Chrome already installed, set
`PLAYWRIGHT_USE_SYSTEM_CHROME=1` to avoid downloading a second browser.

## Container

The multi-stage Dockerfile compiles the application and serves only its static
output from Nginx. From the repository root:

    docker compose up -d mongodb redis api web

See ../../docs/WEB_DASHBOARD.md for architecture, route capabilities, RBAC,
realtime recovery, and operational limits.
