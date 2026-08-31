import '@angular/compiler';

import { Injector, runInInjectionContext } from '@angular/core';
import { Observable, of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  AuthenticationConfiguration,
  OidcTokenResponse,
  Principal,
  SystemHealth,
} from '../models/api.models';
import { ApiClientService } from '../services/api-client.service';
import { AuthService } from './auth.service';

const SESSION_KEY = 'vehicle-intelligence.api-key';
const OIDC_TRANSACTION_KEY = 'vehicle-intelligence.oidc-transaction';

function health(authentication: SystemHealth['authentication']): SystemHealth {
  return {
    status: 'ok',
    phase: 'ready',
    authentication,
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
    realtime: 'available'
  };
}

function principal(role: Principal['role'], authenticationMethod: Principal['authenticationMethod']): Principal {
  return {
    id: role.toLocaleLowerCase(),
    displayName: role,
    role,
    authenticationMethod
  };
}

function authentication(
  provider: AuthenticationConfiguration['provider'],
): AuthenticationConfiguration {
  return {
    enabled: provider !== 'disabled',
    provider,
    oidc:
      provider === 'oidc'
        ? {
            issuer: 'https://identity.example',
            authorizationEndpoint: 'https://identity.example/authorize',
            tokenEndpoint: 'https://identity.example/token',
            clientId: 'vehicle-console',
            scopes: ['openid', 'profile'],
            endSessionEndpoint: 'https://identity.example/logout',
            callbackPath: '/login',
          }
        : null,
  };
}

function createService(api: {
  systemHealth: () => Observable<SystemHealth>;
  authenticationConfiguration: () => Observable<AuthenticationConfiguration>;
  currentPrincipal: () => Observable<Principal>;
  exchangeOidcCode: (
    endpoint: string,
    request: {
      code: string;
      clientId: string;
      codeVerifier: string;
      redirectUri: string;
    },
  ) => Observable<OidcTokenResponse>;
}): AuthService {
  const injector = Injector.create({
    providers: [{ provide: ApiClientService, useValue: api }]
  });
  return runInInjectionContext(injector, () => new AuthService());
}

describe('AuthService', () => {
  const stored = new Map<string, string>();

  beforeEach(() => {
    stored.clear();
    vi.stubGlobal('sessionStorage', {
      getItem: vi.fn((key: string) => stored.get(key) ?? null),
      setItem: vi.fn((key: string, value: string) => stored.set(key, value)),
      removeItem: vi.fn((key: string) => stored.delete(key))
    });
  });

  afterEach(() => vi.unstubAllGlobals());

  it('clears a stale token and resolves the development principal when authentication is disabled', async () => {
    stored.set(SESSION_KEY, 'stale-key');
    const currentPrincipal = vi.fn(() => of(principal('ADMIN', 'DEVELOPMENT')));
    const service = createService({
      systemHealth: vi.fn(() => of(health('disabled'))),
      authenticationConfiguration: vi.fn(() => of(authentication('disabled'))),
      currentPrincipal,
      exchangeOidcCode: vi.fn(),
    });

    await service.initialize();

    expect(currentPrincipal).toHaveBeenCalledOnce();
    expect(stored.has(SESSION_KEY)).toBe(false);
    expect(service.apiKey()).toBe('');
    expect(service.state()).toBe('authenticated');
    expect(service.canManageCameras()).toBe(true);
  });

  it('waits for a key when required, then trims, persists and applies operator permissions', async () => {
    const operator = principal('OPERATOR', 'API_KEY');
    const currentPrincipal = vi.fn(() => of(operator));
    const service = createService({
      systemHealth: vi.fn(() => of(health('enabled'))),
      authenticationConfiguration: vi.fn(() => of(authentication('api_key'))),
      currentPrincipal,
      exchangeOidcCode: vi.fn(),
    });

    await service.initialize();
    expect(service.state()).toBe('anonymous');
    expect(currentPrincipal).not.toHaveBeenCalled();

    await expect(service.login('  operator-secret  ')).resolves.toEqual(operator);
    expect(service.apiKey()).toBe('operator-secret');
    expect(stored.get(SESSION_KEY)).toBe('operator-secret');
    expect(service.canTestCameras()).toBe(true);
    expect(service.canManageCameras()).toBe(false);
  });

  it('starts OIDC Authorization Code + S256 PKCE without persisting a bearer token', async () => {
    const assign = vi.fn();
    vi.stubGlobal('window', {
      location: {
        origin: 'https://console.example',
        href: 'https://console.example/login?returnUrl=%2Fevents',
        search: '?returnUrl=%2Fevents',
        assign,
      },
      history: { replaceState: vi.fn() },
    });
    vi.stubGlobal('crypto', {
      getRandomValues: vi.fn((bytes: Uint8Array) => {
        bytes.fill(7);
        return bytes;
      }),
      subtle: {
        digest: vi.fn(async () => new Uint8Array(32).fill(9).buffer),
      },
    });
    const service = createService({
      systemHealth: vi.fn(() => of(health('enabled'))),
      authenticationConfiguration: vi.fn(() => of(authentication('oidc'))),
      currentPrincipal: vi.fn(),
      exchangeOidcCode: vi.fn(),
    });

    await service.initialize();
    await service.beginOidcLogin('/events');

    const authorizationUrl = new URL(assign.mock.calls[0][0]);
    const transaction = JSON.parse(stored.get(OIDC_TRANSACTION_KEY) ?? '{}');
    expect(authorizationUrl.origin + authorizationUrl.pathname).toBe(
      'https://identity.example/authorize',
    );
    expect(authorizationUrl.searchParams.get('response_type')).toBe('code');
    expect(authorizationUrl.searchParams.get('code_challenge_method')).toBe('S256');
    expect(authorizationUrl.searchParams.get('state')).toBe(transaction.state);
    expect(transaction.returnUrl).toBe('/events');
    expect(stored.has(SESSION_KEY)).toBe(false);
    expect(service.bearerToken()).toBe('');
  });

  it('completes a matching OIDC callback and keeps the access token in memory only', async () => {
    const replaceState = vi.fn();
    vi.stubGlobal('window', {
      location: {
        origin: 'https://console.example',
        href: 'https://console.example/login?code=auth-code&state=expected-state',
        search: '?code=auth-code&state=expected-state',
        assign: vi.fn(),
      },
      history: { replaceState },
    });
    vi.stubGlobal('document', { title: 'Vehicle Intelligence' });
    stored.set(
      OIDC_TRANSACTION_KEY,
      JSON.stringify({
        state: 'expected-state',
        codeVerifier: 'verifier',
        redirectUri: 'https://console.example/login',
        returnUrl: '/events',
        createdAt: Date.now(),
      }),
    );
    const oidcPrincipal = principal('OPERATOR', 'OIDC');
    const exchangeOidcCode = vi.fn(() =>
      of({ access_token: 'header.payload.signature', token_type: 'Bearer' }),
    );
    const service = createService({
      systemHealth: vi.fn(() => of(health('enabled'))),
      authenticationConfiguration: vi.fn(() => of(authentication('oidc'))),
      currentPrincipal: vi.fn(() => of(oidcPrincipal)),
      exchangeOidcCode,
    });

    await service.initialize();

    expect(service.state()).toBe('authenticated');
    expect(service.bearerToken()).toBe('header.payload.signature');
    expect(stored.has(SESSION_KEY)).toBe(false);
    expect(stored.has(OIDC_TRANSACTION_KEY)).toBe(false);
    expect(exchangeOidcCode).toHaveBeenCalledWith('https://identity.example/token', {
      code: 'auth-code',
      clientId: 'vehicle-console',
      codeVerifier: 'verifier',
      redirectUri: 'https://console.example/login',
    });
    expect(replaceState).toHaveBeenCalledWith({}, 'Vehicle Intelligence', '/events');
  });

  it('ignores OAuth-shaped query parameters outside the configured callback path', async () => {
    vi.stubGlobal('window', {
      location: {
        origin: 'https://console.example',
        href: 'https://console.example/dashboard?code=untrusted&state=untrusted',
        search: '?code=untrusted&state=untrusted',
        assign: vi.fn(),
      },
      history: { replaceState: vi.fn() },
    });
    const exchangeOidcCode = vi.fn();
    const service = createService({
      systemHealth: vi.fn(() => of(health('enabled'))),
      authenticationConfiguration: vi.fn(() => of(authentication('oidc'))),
      currentPrincipal: vi.fn(),
      exchangeOidcCode,
    });

    await service.initialize();

    expect(service.state()).toBe('anonymous');
    expect(exchangeOidcCode).not.toHaveBeenCalled();
  });

  it('fails a callback closed when OAuth state does not match', async () => {
    const replaceState = vi.fn();
    vi.stubGlobal('window', {
      location: {
        origin: 'https://console.example',
        href: 'https://console.example/login?code=auth-code&state=attacker-state',
        search: '?code=auth-code&state=attacker-state',
        assign: vi.fn(),
      },
      history: { replaceState },
    });
    vi.stubGlobal('document', { title: 'Vehicle Intelligence' });
    stored.set(
      OIDC_TRANSACTION_KEY,
      JSON.stringify({
        state: 'expected-state',
        codeVerifier: 'verifier',
        redirectUri: 'https://console.example/login',
        returnUrl: '/events',
        createdAt: Date.now(),
      }),
    );
    const exchangeOidcCode = vi.fn();
    const service = createService({
      systemHealth: vi.fn(() => of(health('enabled'))),
      authenticationConfiguration: vi.fn(() => of(authentication('oidc'))),
      currentPrincipal: vi.fn(),
      exchangeOidcCode,
    });

    await service.initialize();

    expect(service.state()).toBe('anonymous');
    expect(service.startupError()).toContain('không khớp');
    expect(exchangeOidcCode).not.toHaveBeenCalled();
    expect(stored.has(OIDC_TRANSACTION_KEY)).toBe(false);
    expect(replaceState).toHaveBeenCalledWith({}, 'Vehicle Intelligence', '/login');
  });
});
