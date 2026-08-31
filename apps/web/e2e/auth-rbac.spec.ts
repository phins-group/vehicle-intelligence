import { expect, test, type Page, type Route } from '@playwright/test';

const API_KEY_SESSION_KEY = 'vehicle-intelligence.api-key';
const OIDC_TRANSACTION_KEY = 'vehicle-intelligence.oidc-transaction';

const health = {
  status: 'ok',
  phase: 'ready',
  authentication: 'enabled',
  cameraManagement: 'available',
  onvifDiscovery: 'available',
  policyEngine: 'available',
  auditLog: 'available',
  mediaAccess: 'available',
  humanReview: 'available',
  datasetReview: 'available',
  datasetRegistry: 'available',
  modelTraining: 'available',
  modelQuality: 'available',
  liveMonitor: 'ONLINE',
  realtime: 'available',
};

const oidcConfiguration = {
  enabled: true,
  provider: 'oidc',
  oidc: {
    issuer: 'https://identity.example',
    authorizationEndpoint: 'https://identity.example/authorize',
    tokenEndpoint: 'https://identity.example/token',
    clientId: 'vehicle-console',
    scopes: ['openid', 'profile'],
    endSessionEndpoint: 'https://identity.example/logout',
    callbackPath: '/login',
  },
};

const apiKeyConfiguration = {
  enabled: true,
  provider: 'api_key',
  oidc: null,
};

const operator = {
  id: 'operator',
  displayName: 'Operations User',
  role: 'OPERATOR',
  authenticationMethod: 'OIDC',
};

interface ApiMockOptions {
  configuration: typeof oidcConfiguration | typeof apiKeyConfiguration;
  acceptedToken?: string;
  authenticationMethod?: 'API_KEY' | 'OIDC';
}

async function fulfillJson(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  });
}

async function installApiMock(page: Page, options: ApiMockOptions): Promise<string[]> {
  const authorizationHeaders: string[] = [];
  await page.routeWebSocket(/\/ws\/events(?:\?|$)/, (webSocket) => {
    webSocket.onMessage(() => undefined);
  });
  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (pathname === '/api/system/health') {
      await fulfillJson(route, health);
      return;
    }
    if (pathname === '/api/auth/config') {
      await fulfillJson(route, options.configuration);
      return;
    }
    if (pathname === '/api/auth/me') {
      const authorization = request.headers()['authorization'] ?? '';
      authorizationHeaders.push(authorization);
      if (options.acceptedToken && authorization !== `Bearer ${options.acceptedToken}`) {
        await fulfillJson(route, { detail: 'Unauthorized' }, 401);
        return;
      }
      await fulfillJson(route, {
        ...operator,
        authenticationMethod: options.authenticationMethod ?? 'OIDC',
      });
      return;
    }
    if (pathname === '/api/camera-health' || pathname === '/api/rules') {
      await fulfillJson(route, { items: [] });
      return;
    }
    await fulfillJson(route, { detail: `Unhandled E2E route: ${pathname}` }, 404);
  });
  return authorizationHeaders;
}

async function saveOidcTransaction(
  page: Page,
  state: string,
  returnUrl: string,
): Promise<string> {
  const redirectUri = new URL('/login', page.url()).toString();
  await page.evaluate(
    ({ key, transaction }) => sessionStorage.setItem(key, JSON.stringify(transaction)),
    {
      key: OIDC_TRANSACTION_KEY,
      transaction: {
        state,
        codeVerifier: 'test-code-verifier-with-sufficient-entropy-0123456789',
        redirectUri,
        returnUrl,
        createdAt: Date.now(),
      },
    },
  );
  return redirectUri;
}

test('starts OIDC with S256 PKCE and no persisted bearer token', async ({ page }) => {
  await installApiMock(page, { configuration: oidcConfiguration });
  await page.route('https://identity.example/authorize**', (route) =>
    route.fulfill({ contentType: 'text/html', body: '<p>Identity provider</p>' }),
  );
  await page.goto('/login?returnUrl=%2Fevents');

  const authorizationRequest = page.waitForRequest((request) =>
    request.url().startsWith('https://identity.example/authorize?'),
  );
  await page.getByRole('button', { name: 'Đăng nhập với OIDC' }).click();
  const authorizationUrl = new URL((await authorizationRequest).url());
  await page.waitForURL('https://identity.example/authorize**');
  await page.goto('/login');

  const transaction = await page.evaluate((key) => {
    const raw = sessionStorage.getItem(key);
    return raw === null ? null : (JSON.parse(raw) as { state: string; codeVerifier: string; returnUrl: string });
  }, OIDC_TRANSACTION_KEY);
  expect(transaction).not.toBeNull();
  expect(authorizationUrl.searchParams.get('response_type')).toBe('code');
  expect(authorizationUrl.searchParams.get('client_id')).toBe('vehicle-console');
  expect(authorizationUrl.searchParams.get('code_challenge_method')).toBe('S256');
  expect(authorizationUrl.searchParams.get('state')).toBe(transaction?.state);
  expect(authorizationUrl.searchParams.has('code_verifier')).toBe(false);
  expect(transaction?.returnUrl).toBe('/events');

  const expectedChallenge = await page.evaluate(async (codeVerifier) => {
    const digest = await crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode(codeVerifier),
    );
    let binary = '';
    for (const byte of new Uint8Array(digest)) binary += String.fromCharCode(byte);
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
  }, transaction?.codeVerifier ?? '');
  expect(authorizationUrl.searchParams.get('code_challenge')).toBe(expectedChallenge);
  expect(await page.evaluate((key) => sessionStorage.getItem(key), API_KEY_SESSION_KEY)).toBeNull();
});

test('rejects an OIDC callback when state does not match', async ({ page }) => {
  await installApiMock(page, { configuration: oidcConfiguration });
  let tokenExchangeCount = 0;
  await page.route('https://identity.example/token', async (route) => {
    tokenExchangeCount += 1;
    await fulfillJson(route, { access_token: 'must-not-be-used', token_type: 'Bearer' });
  });
  await page.goto('/login');
  await saveOidcTransaction(page, 'expected-state', '/events');

  await page.goto('/login?code=authorization-code&state=attacker-state');

  await expect(page.getByRole('alert')).toContainText('không khớp');
  await expect(page).toHaveURL(/\/login$/);
  expect(tokenExchangeCount).toBe(0);
  expect(await page.evaluate((key) => sessionStorage.getItem(key), OIDC_TRANSACTION_KEY)).toBeNull();
});

test('completes OIDC in memory and applies operator RBAC', async ({ page }) => {
  const authorizationHeaders = await installApiMock(page, {
    configuration: oidcConfiguration,
    acceptedToken: 'oidc-access-token',
  });
  let tokenRequestBody = '';
  await page.goto('/login');
  const consoleOrigin = new URL(page.url()).origin;
  const redirectUri = await saveOidcTransaction(page, 'matching-state', '/rules');
  await page.route('https://identity.example/token', async (route) => {
    tokenRequestBody = route.request().postData() ?? '';
    await route.fulfill({
      status: 200,
      headers: {
        'access-control-allow-origin': consoleOrigin,
        'content-type': 'application/json',
      },
      body: JSON.stringify({ access_token: 'oidc-access-token', token_type: 'Bearer' }),
    });
  });

  await page.goto('/login?code=authorization-code&state=matching-state');

  await expect(page.getByRole('heading', { name: 'Luật tự động' })).toBeVisible();
  await expect(page.getByRole('note')).toContainText('Chỉ ADMIN');
  await expect(page.getByRole('button', { name: 'Tạo rule' })).toHaveCount(0);
  expect(authorizationHeaders).toContain('Bearer oidc-access-token');
  const tokenParameters = new URLSearchParams(tokenRequestBody);
  expect(tokenParameters.get('grant_type')).toBe('authorization_code');
  expect(tokenParameters.get('code')).toBe('authorization-code');
  expect(tokenParameters.get('redirect_uri')).toBe(redirectUri);
  expect(tokenParameters.get('code_verifier')).toContain('sufficient-entropy');
  expect(await page.evaluate((key) => sessionStorage.getItem(key), API_KEY_SESSION_KEY)).toBeNull();
  expect(await page.evaluate((key) => sessionStorage.getItem(key), OIDC_TRANSACTION_KEY)).toBeNull();
});

test('guards private routes and limits an API-key operator to non-admin camera actions', async ({
  page,
}) => {
  const authorizationHeaders = await installApiMock(page, {
    configuration: apiKeyConfiguration,
    acceptedToken: 'operator-secret',
    authenticationMethod: 'API_KEY',
  });
  await page.goto('/cameras');

  await expect(page.getByRole('heading', { name: 'Đăng nhập hệ thống' })).toBeVisible();
  expect(new URL(page.url()).searchParams.get('returnUrl')).toBe('/cameras');
  await page.getByRole('textbox', { name: 'API key' }).fill('operator-secret');
  await page.getByRole('button', { name: 'Tiếp tục' }).click();

  await expect(page.getByRole('heading', { name: 'Camera', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Quét ONVIF' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Thêm camera' })).toHaveCount(0);
  await expect(page.getByText('OPERATOR', { exact: true })).toBeVisible();
  expect(authorizationHeaders).toContain('Bearer operator-secret');
  expect(await page.evaluate((key) => sessionStorage.getItem(key), API_KEY_SESSION_KEY)).toBe(
    'operator-secret',
  );
});
