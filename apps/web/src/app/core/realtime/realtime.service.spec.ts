import '@angular/compiler';

import { Injector, runInInjectionContext } from '@angular/core';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AuthService } from '../auth/auth.service';
import { RealtimeGap, VehicleEvent } from '../models/api.models';
import { RealtimeService } from './realtime.service';

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly sent: string[] = [];
  readyState = FakeWebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  send(value: string): void {
    this.sent.push(value);
  }

  close(): void {
    this.readyState = FakeWebSocket.CLOSED;
  }
}

function envelope(type: string, id: string, data: unknown): string {
  return JSON.stringify({
    id,
    type,
    schemaVersion: 1,
    occurredAt: '2026-08-14T08:00:00Z',
    source: 'api/realtime',
    data
  });
}

function event(id: string): VehicleEvent {
  return {
    _id: id,
    schemaVersion: 1,
    camera: { id: 'gate-01', name: 'Gate 01', zone: 'A' },
    trackId: 'gate-01:1',
    vehicleId: null,
    eventType: 'VEHICLE_ENTER',
    direction: 'ENTER',
    status: 'CONFIRMED',
    plate: null,
    vehicle: { type: 'car', confidence: 0.96, color: null },
    media: { snapshotKey: null, vehicleCropKey: null, plateCropKey: null, clipKey: null },
    ai: {},
    occurredAt: '2026-08-14T08:00:00Z',
    createdAt: '2026-08-14T08:00:00Z',
    metadata: {}
  };
}

function createService(auth: Pick<AuthService, 'bearerToken' | 'invalidate'>): RealtimeService {
  const injector = Injector.create({ providers: [{ provide: AuthService, useValue: auth }] });
  return runInInjectionContext(injector, () => new RealtimeService());
}

describe('RealtimeService', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal('WebSocket', FakeWebSocket);
    vi.stubGlobal('window', {
      location: new URL('http://localhost/live'),
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('authenticates, becomes ready and publishes event plus gap recovery signals', () => {
    const auth = { bearerToken: vi.fn(() => 'secret'), invalidate: vi.fn() };
    const service = createService(auth);
    const events: VehicleEvent[] = [];
    const gaps: RealtimeGap[] = [];
    let recoveries = 0;
    service.events$.subscribe((value) => events.push(value));
    service.gaps$.subscribe((value) => gaps.push(value));
    service.recoveryRequested$.subscribe(() => recoveries += 1);

    service.connect();
    const socket = FakeWebSocket.instances[0];
    expect(service.connectionState()).toBe('connecting');
    expect(socket.url).toBe('ws://localhost/ws/events');

    socket.onopen?.({} as Event);
    expect(JSON.parse(socket.sent[0])).toEqual({ type: 'authenticate', token: 'secret' });

    socket.onmessage?.({ data: envelope('system.realtime.ready', 'ready-1', {}) } as MessageEvent);
    socket.onmessage?.({ data: envelope('vehicle.entered', 'evt-1', event('evt-1')) } as MessageEvent);
    socket.onmessage?.({
      data: envelope('system.realtime.gap', 'gap-1', {
        reason: 'slow_consumer',
        droppedEvents: 2,
        lastAvailableEventId: 'evt-2',
        recoveryEndpoint: '/api/events'
      })
    } as MessageEvent);

    expect(service.connectionState()).toBe('connected');
    expect(events.map((value) => value._id)).toEqual(['evt-1']);
    expect(gaps[0].droppedEvents).toBe(2);
    expect(recoveries).toBe(1);
    service.disconnect();
  });

  it('reconnects with the latest cursor and stops after an authorization close', () => {
    const auth = { bearerToken: vi.fn(() => 'secret'), invalidate: vi.fn() };
    const service = createService(auth);
    service.connect();
    const first = FakeWebSocket.instances[0];
    first.onmessage?.({ data: envelope('vehicle.entered', 'evt-1', event('evt-1')) } as MessageEvent);

    first.onclose?.({ code: 1006 } as CloseEvent);
    expect(service.connectionState()).toBe('reconnecting');
    vi.advanceTimersByTime(1_000);

    const second = FakeWebSocket.instances[1];
    expect(new URL(second.url).searchParams.get('lastEventId')).toBe('evt-1');
    second.onclose?.({ code: 4401 } as CloseEvent);

    expect(auth.invalidate).toHaveBeenCalledOnce();
    expect(service.connectionState()).toBe('unavailable');
    vi.advanceTimersByTime(60_000);
    expect(FakeWebSocket.instances).toHaveLength(2);
    service.disconnect();
  });
});
