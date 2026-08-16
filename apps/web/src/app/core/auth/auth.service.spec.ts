import '@angular/compiler';

import { Injector, runInInjectionContext } from '@angular/core';
import { Observable, of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Principal, SystemHealth } from '../models/api.models';
import { ApiClientService } from '../services/api-client.service';
import { AuthService } from './auth.service';

const SESSION_KEY = 'vehicle-intelligence.api-key';

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

function createService(api: {
  systemHealth: () => Observable<SystemHealth>;
  currentPrincipal: () => Observable<Principal>;
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
      currentPrincipal
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
      currentPrincipal
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
});
