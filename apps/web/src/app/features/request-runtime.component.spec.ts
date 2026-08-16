import '@angular/compiler';

import { DestroyRef, Injector, runInInjectionContext } from '@angular/core';
import { Observable, Subject, of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AuthService } from '../core/auth/auth.service';
import {
  Alert,
  AlertPage,
  Camera,
  CameraHealth,
  CameraHealthSnapshot,
  EventFilters,
  EventPage,
  PlateReviewResponse,
  SystemHealth,
  VehicleEvent,
} from '../core/models/api.models';
import { RealtimeService } from '../core/realtime/realtime.service';
import { ApiClientService } from '../core/services/api-client.service';
import { AlertsComponent } from './alerts/alerts.component';
import { EventsComponent } from './events/events.component';
import { OcrReviewComponent } from './ocr-review/ocr-review.component';
import { SystemHealthComponent } from './system-health/system-health.component';

class ManualDestroyRef {
  private readonly callbacks = new Set<() => void>();

  onDestroy(callback: () => void): () => void {
    this.callbacks.add(callback);
    return () => this.callbacks.delete(callback);
  }

  destroy(): void {
    for (const callback of this.callbacks) callback();
    this.callbacks.clear();
  }
}

function camera(id: string): Camera {
  return {
    id,
    schemaVersion: 1,
    revision: 1,
    name: `Camera ${id}`,
    stream: { fpsLimit: 6, credentialsConfigured: true },
    location: { name: 'Gate', zone: 'A' },
    direction: 'BOTH',
    vision: { vehicleConfidence: 0.4, plateConfidence: 0.45 },
    geometry: {
      vehicleRoi: null,
      crossingLine: null,
      crossingPositiveToNegative: 'ENTER',
      finalizeOnCrossing: false,
    },
    enabled: true,
    metadata: {},
    createdAt: '2026-08-16T00:00:00.000Z',
    updatedAt: '2026-08-16T00:00:00.000Z',
  };
}

function cameraHealth(cameraId: string): CameraHealth {
  return {
    cameraId,
    status: 'ONLINE',
    sourceFps: 25,
    decodeFps: 24,
    queueSize: 1,
    droppedFrames: 0,
    reconnectCount: 0,
    connectionFailures: 0,
    streamEpoch: 1,
    lastFrameAt: '2026-08-16T00:00:00.000Z',
    updatedAt: '2026-08-16T00:00:00.000Z',
    decodedFrames: 100,
    sampledFrames: 50,
    vehicleDetections: 20,
    plateDetections: 18,
    ocrRequests: 18,
    ocrSuccess: 17,
    eventsCreated: 16,
    trackCount: 2,
    inferenceFps: 6,
    vehicleInferenceLatencyMs: 4,
    plateInferenceLatencyMs: 3,
    ocrLatencyMs: 8,
  };
}

function systemHealth(status: string): SystemHealth {
  return {
    status,
    phase: '4',
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
    realtime: 'ONLINE',
  };
}

function vehicleEvent(
  id: string,
  status: VehicleEvent['status'] = 'CONFIRMED',
): VehicleEvent {
  const baseTimestamp = '2026-08-16T00:00:00.000Z';
  const offset = Number(id);
  const timestamp = Number.isFinite(offset)
    ? new Date(Date.parse(baseTimestamp) + offset).toISOString()
    : baseTimestamp;
  return {
    _id: `event-${id}`,
    schemaVersion: 1,
    camera: { id: 'gate-01', name: 'Gate 01', zone: 'A' },
    trackId: `gate-01:${id}`,
    vehicleId: null,
    eventType: 'VEHICLE_ENTER',
    direction: 'ENTER',
    status,
    plate: null,
    vehicle: { type: 'car', confidence: 0.95, color: null },
    media: {
      snapshotKey: null,
      vehicleCropKey: null,
      plateCropKey: null,
      clipKey: null,
    },
    ai: {},
    occurredAt: timestamp,
    createdAt: timestamp,
    metadata: {},
  };
}

function alert(id: string): Alert {
  return {
    id,
    schemaVersion: 1,
    revision: 1,
    source: {
      eventId: `event-${id}`,
      executionId: `execution-${id}`,
      actionId: 'notify',
    },
    rule: { id: 'rule-01', name: 'Gate rule' },
    camera: { id: 'gate-01', name: 'Gate 01', zone: 'A' },
    eventType: 'VEHICLE_ENTER',
    direction: 'ENTER',
    severity: 'HIGH',
    status: 'OPEN',
    message: `Alert ${id}`,
    plate: null,
    vehicleType: 'car',
    occurredAt: '2026-08-16T00:00:00.000Z',
    createdAt: '2026-08-16T00:00:00.000Z',
    updatedAt: '2026-08-16T00:00:00.000Z',
    acknowledgedAt: null,
    acknowledgedBy: null,
    resolvedAt: null,
    resolvedBy: null,
    metadata: {},
  };
}

function realtimeSubjects(): {
  events: Subject<VehicleEvent>;
  recovery: Subject<void>;
  service: object;
} {
  const events = new Subject<VehicleEvent>();
  const recovery = new Subject<void>();
  return {
    events,
    recovery,
    service: {
      events$: events.asObservable(),
      recoveryRequested$: recovery.asObservable(),
      connectionState: () => 'connected',
    },
  };
}

function setupAlerts() {
  const realtime = realtimeSubjects();
  const destroyRef = new ManualDestroyRef();
  const api = {
    alerts: vi.fn((_filters: unknown = {}) =>
      of({ items: [], nextCursor: null } as AlertPage),
    ),
    cameras: vi.fn(() => of({ items: [] as Camera[] })),
    acknowledgeAlert: vi.fn(),
    resolveAlert: vi.fn(),
  };
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      { provide: RealtimeService, useValue: realtime.service },
      { provide: AuthService, useValue: { canManageAlerts: () => true } },
      { provide: DestroyRef, useValue: destroyRef },
    ],
  });
  const component = runInInjectionContext(
    injector,
    () => new AlertsComponent(),
  );
  return { api, component, destroyRef, realtime };
}

function setupSystemHealth(snapshot?: CameraHealthSnapshot) {
  const destroyRef = new ManualDestroyRef();
  const defaultSnapshot = snapshot ?? {
    items: [
      { camera: camera('gate-01'), health: cameraHealth('gate-01') },
      { camera: camera('gate-02'), health: null },
    ],
  };
  const api = {
    systemHealth: vi.fn(() => of(systemHealth('ok'))),
    realtimeHealth: vi.fn(() => of({ status: 'ONLINE', subscribers: 1 })),
    liveMonitorHealth: vi.fn(() =>
      of({ status: 'ONLINE', camerasBuffered: 1 }),
    ),
    cameraHealthSnapshot: vi.fn(() => of(defaultSnapshot)),
  };
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      { provide: DestroyRef, useValue: destroyRef },
    ],
  });
  const component = runInInjectionContext(
    injector,
    () => new SystemHealthComponent(),
  );
  return { api, component, destroyRef };
}

function setupEvents() {
  const realtime = realtimeSubjects();
  const destroyRef = new ManualDestroyRef();
  const api = {
    events: vi.fn((_filters: EventFilters = {}): Observable<EventPage> =>
      of({ items: [] as VehicleEvent[], nextCursor: null }),
    ),
    cameras: vi.fn(() => of({ items: [] as Camera[] })),
  };
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      { provide: RealtimeService, useValue: realtime.service },
      { provide: DestroyRef, useValue: destroyRef },
    ],
  });
  const component = runInInjectionContext(
    injector,
    () => new EventsComponent(),
  );
  return { api, component, destroyRef, realtime };
}

function setupOcrReview() {
  const realtime = realtimeSubjects();
  const destroyRef = new ManualDestroyRef();
  const api = {
    events: vi.fn((_filters: EventFilters = {}): Observable<EventPage> =>
      of({ items: [] as VehicleEvent[], nextCursor: null }),
    ),
    reviewPlate: vi.fn((): Observable<PlateReviewResponse> =>
      of({
        event: vehicleEvent('reviewed', 'CONFIRMED'),
        changed: true,
        feedbackReason: 'HUMAN_CORRECTION',
        datasetSampleId: null,
      }),
    ),
    event: vi.fn((): Observable<VehicleEvent> =>
      of(vehicleEvent('latest', 'NEEDS_REVIEW')),
    ),
  };
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      { provide: RealtimeService, useValue: realtime.service },
      { provide: AuthService, useValue: { canReviewPlates: () => true } },
      { provide: DestroyRef, useValue: destroyRef },
    ],
  });
  const component = runInInjectionContext(
    injector,
    () => new OcrReviewComponent(),
  );
  return { api, component, destroyRef, realtime };
}

async function flushMicrotasks(): Promise<void> {
  for (let index = 0; index < 5; index += 1) await Promise.resolve();
}

describe('request and realtime runtime behavior', () => {
  let documentState: { hidden: boolean };

  beforeEach(() => {
    vi.useFakeTimers();
    documentState = { hidden: false };
    vi.stubGlobal('document', documentState);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('coalesces alert realtime invalidations instead of reloading per event', async () => {
    const { api, component, destroyRef, realtime } = setupAlerts();
    await component.load(true);

    for (let index = 0; index < 100; index += 1) {
      realtime.events.next(vehicleEvent(String(index)));
    }
    await vi.advanceTimersByTimeAsync(14_999);
    expect(api.alerts).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(1);
    expect(api.alerts).toHaveBeenCalledTimes(2);
    component.ngOnDestroy();
    destroyRef.destroy();
  });

  it('serializes alert refreshes and ignores the invalidated response', async () => {
    const { api, component, destroyRef } = setupAlerts();
    const first = new Subject<AlertPage>();
    const second = new Subject<AlertPage>();
    api.alerts.mockReset();
    api.alerts.mockReturnValueOnce(first).mockReturnValueOnce(second);

    const activeLoad = component.load(true);
    await component.load(true);
    expect(api.alerts).toHaveBeenCalledOnce();

    first.next({ items: [alert('stale')], nextCursor: null });
    first.complete();
    await activeLoad;
    expect(api.alerts).toHaveBeenCalledTimes(2);
    expect(component.alerts()).toEqual([]);

    second.next({ items: [alert('fresh')], nextCursor: null });
    second.complete();
    await flushMicrotasks();
    expect(component.alerts().map((item) => item.id)).toEqual(['fresh']);
    destroyRef.destroy();
  });

  it('keeps alert pagination on the applied filters until the form is submitted again', async () => {
    const { api, component, destroyRef } = setupAlerts();
    api.alerts.mockReset();
    api.alerts
      .mockReturnValueOnce(
        of({ items: [alert('open')], nextCursor: 'open-cursor' } as AlertPage),
      )
      .mockReturnValueOnce(of({ items: [], nextCursor: null } as AlertPage))
      .mockReturnValueOnce(of({ items: [], nextCursor: null } as AlertPage));

    component.status = 'OPEN';
    await component.applyFilters();
    component.status = 'RESOLVED';
    await component.load(false);

    expect(api.alerts.mock.calls[1]?.[0]).toMatchObject({
      cursor: 'open-cursor',
      status: 'OPEN',
    });

    await component.applyFilters();
    expect(api.alerts.mock.calls[2]?.[0]).toMatchObject({
      cursor: null,
      status: 'RESOLVED',
    });
    destroyRef.destroy();
  });

  it('does not commit an alert response after the component is destroyed', async () => {
    const { api, component, destroyRef } = setupAlerts();
    const response = new Subject<AlertPage>();
    api.alerts.mockReset();
    api.alerts.mockReturnValue(response);
    const loading = component.load(true);

    component.ngOnDestroy();
    destroyRef.destroy();
    response.next({ items: [alert('late')], nextCursor: null });
    response.complete();
    await loading;

    expect(component.alerts()).toEqual([]);
    expect(component.loadState.hasLoaded()).toBe(false);
  });

  it('loads one left-joined camera health snapshot and keeps the null row', async () => {
    const { api, component, destroyRef } = setupSystemHealth();

    await component.load();

    expect(api.cameraHealthSnapshot).toHaveBeenCalledOnce();
    expect(component.cameras().map((item) => item.camera.id)).toEqual([
      'gate-01',
      'gate-02',
    ]);
    expect(component.cameras()[1]?.health).toBeNull();
    destroyRef.destroy();
  });

  it('pauses system health polling while hidden and refreshes on visibility', async () => {
    documentState.hidden = true;
    const { api, component, destroyRef } = setupSystemHealth();

    await vi.advanceTimersByTimeAsync(30_000);
    expect(api.systemHealth).not.toHaveBeenCalled();

    documentState.hidden = false;
    component.handleVisibilityChange();
    await flushMicrotasks();
    expect(api.systemHealth).toHaveBeenCalledOnce();
    destroyRef.destroy();
  });

  it('queues a newer system snapshot and prevents the old one from committing', async () => {
    const { api, component, destroyRef } = setupSystemHealth();
    const first = new Subject<CameraHealthSnapshot>();
    const freshSnapshot = {
      items: [{ camera: camera('gate-fresh'), health: null }],
    };
    api.cameraHealthSnapshot.mockReset();
    api.cameraHealthSnapshot
      .mockReturnValueOnce(first)
      .mockReturnValueOnce(of(freshSnapshot));
    api.systemHealth.mockReset();
    api.systemHealth
      .mockReturnValueOnce(of(systemHealth('stale')))
      .mockReturnValueOnce(of(systemHealth('fresh')));

    const activeLoad = component.load();
    await component.load();
    first.next({ items: [{ camera: camera('gate-stale'), health: null }] });
    first.complete();
    await activeLoad;
    await flushMicrotasks();

    expect(api.cameraHealthSnapshot).toHaveBeenCalledTimes(2);
    expect(component.system()?.status).toBe('fresh');
    expect(component.cameras()[0]?.camera.id).toBe('gate-fresh');
    destroyRef.destroy();
  });

  it('buffers Events realtime bursts and renders only one 100-row window', async () => {
    const { api, component, destroyRef, realtime } = setupEvents();
    for (let index = 0; index < 600; index += 1) {
      realtime.events.next(vehicleEvent(String(index)));
    }

    await vi.advanceTimersByTimeAsync(199);
    expect(component.events()).toEqual([]);
    await vi.advanceTimersByTimeAsync(1);

    expect(component.events()).toHaveLength(500);
    expect(component.eventWindow().items).toHaveLength(100);
    expect(component.eventWindow().total).toBe(500);
    expect(component.bufferLimitReached()).toBe(true);
    expect(component.liveCount()).toBe(600);
    expect(api.events).not.toHaveBeenCalled();

    component.showOlderEvents();
    expect(component.eventWindow().start).toBe(100);
    destroyRef.destroy();
  });

  it('serializes Events filter resets and preserves realtime received during the request', async () => {
    const { api, component, destroyRef, realtime } = setupEvents();
    const first = new Subject<EventPage>();
    const second = new Subject<EventPage>();
    api.events.mockReset();
    api.events.mockReturnValueOnce(first).mockReturnValueOnce(second);
    component.cameraId = 'gate-01';
    const activeLoad = component.applyFilters();

    component.cameraId = 'gate-02';
    await component.applyFilters();
    expect(api.events).toHaveBeenCalledOnce();
    first.next({ items: [vehicleEvent('stale')], nextCursor: null });
    first.complete();
    await activeLoad;
    expect(api.events).toHaveBeenCalledTimes(2);
    expect(api.events.mock.calls[1]?.[0]?.cameraId).toBe('gate-02');

    const liveEvent = vehicleEvent('700');
    liveEvent.camera.id = 'gate-02';
    realtime.events.next(liveEvent);
    await vi.advanceTimersByTimeAsync(200);
    second.next({ items: [vehicleEvent('fresh')], nextCursor: null });
    second.complete();
    await flushMicrotasks();

    expect(component.events().map((event) => event._id)).toEqual([
      'event-700',
      'event-fresh',
    ]);
    destroyRef.destroy();
  });

  it('uses the applied Events filters and cursor while draft controls are edited', async () => {
    const { api, component, destroyRef } = setupEvents();
    api.events.mockReset();
    api.events
      .mockReturnValueOnce(
        of({ items: [vehicleEvent('first')], nextCursor: 'gate-01-cursor' }),
      )
      .mockReturnValueOnce(of({ items: [], nextCursor: null }));

    component.cameraId = 'gate-01';
    await component.applyFilters();
    component.cameraId = 'gate-02';
    await component.load(false);

    expect(api.events.mock.calls[1]?.[0]).toMatchObject({
      cameraId: 'gate-01',
      cursor: 'gate-01-cursor',
    });
    destroyRef.destroy();
  });

  it('buffers OCR realtime bursts, filters status, and bounds the render window', async () => {
    const { api, component, destroyRef, realtime } = setupOcrReview();
    realtime.events.next(vehicleEvent('ignored', 'CONFIRMED'));
    for (let index = 0; index < 600; index += 1) {
      realtime.events.next(vehicleEvent(String(index), 'NEEDS_REVIEW'));
    }

    await vi.advanceTimersByTimeAsync(200);

    expect(component.events()).toHaveLength(500);
    expect(component.eventWindow().items).toHaveLength(100);
    expect(component.eventWindow().total).toBe(500);
    expect(component.bufferLimitReached()).toBe(true);
    expect(api.events).not.toHaveBeenCalled();
    destroyRef.destroy();
  });

  it('serializes OCR recovery loads and ignores the invalidated queue response', async () => {
    const { api, component, destroyRef } = setupOcrReview();
    const first = new Subject<EventPage>();
    const second = new Subject<EventPage>();
    api.events.mockReset();
    api.events.mockReturnValueOnce(first).mockReturnValueOnce(second);

    const activeLoad = component.load(true);
    await component.load(true);
    first.next({
      items: [vehicleEvent('stale', 'NEEDS_REVIEW')],
      nextCursor: null,
    });
    first.complete();
    await activeLoad;
    expect(api.events).toHaveBeenCalledTimes(2);

    second.next({
      items: [vehicleEvent('fresh', 'NEEDS_REVIEW')],
      nextCursor: null,
    });
    second.complete();
    await flushMicrotasks();

    expect(component.events().map((event) => event._id)).toEqual([
      'event-fresh',
    ]);
    destroyRef.destroy();
  });

  it('does not reopen an OCR drawer when a submit response arrives after close', async () => {
    const { api, component, destroyRef } = setupOcrReview();
    const pending = new Subject<PlateReviewResponse>();
    api.reviewPlate.mockReturnValue(pending);
    const original = vehicleEvent('review', 'NEEDS_REVIEW');
    component.events.set([original]);
    component.showReview(original);
    component.plateText = '51H12345';

    const submitting = component.submitReview();
    component.closeReview();
    pending.next({
      event: vehicleEvent('review', 'CONFIRMED'),
      changed: true,
      feedbackReason: 'HUMAN_CORRECTION',
      datasetSampleId: null,
    });
    pending.complete();
    await submitting;

    expect(component.selected()).toBeNull();
    expect(component.events()).toEqual([original]);
    expect(component.success()).toBeNull();
    destroyRef.destroy();
  });

  it('does not resurrect a reviewed event from a delayed queue response', async () => {
    const { api, component, destroyRef } = setupOcrReview();
    const stalePage = new Subject<EventPage>();
    const freshPage = new Subject<EventPage>();
    api.events.mockReset();
    api.events.mockReturnValueOnce(stalePage).mockReturnValueOnce(freshPage);
    const original = vehicleEvent('review', 'NEEDS_REVIEW');
    api.reviewPlate.mockReturnValueOnce(
      of({
        event: vehicleEvent('review', 'CONFIRMED'),
        changed: true,
        feedbackReason: 'HUMAN_CORRECTION',
        datasetSampleId: null,
      }),
    );
    component.events.set([original]);
    component.showReview(original);
    component.plateText = '51H12345';

    const loading = component.load(true);
    await component.submitReview();
    expect(component.events()).toEqual([]);

    stalePage.next({ items: [original], nextCursor: null });
    stalePage.complete();
    await loading;
    expect(api.events).toHaveBeenCalledTimes(2);
    expect(component.events()).toEqual([]);

    freshPage.next({ items: [], nextCursor: null });
    freshPage.complete();
    await flushMicrotasks();
    expect(component.events()).toEqual([]);
    destroyRef.destroy();
  });
});
