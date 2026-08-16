import '@angular/compiler';

import { DestroyRef, Injector, runInInjectionContext } from '@angular/core';
import { Subject, of, throwError } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EventPage, VehicleEvent } from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { DashboardComponent } from './dashboard.component';

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

function vehicleEvent(
  id: string,
  occurredAt: string,
  direction: VehicleEvent['direction'],
): VehicleEvent {
  return {
    _id: id,
    schemaVersion: 1,
    camera: { id: 'gate-01', name: 'Gate 01', zone: 'A' },
    trackId: `gate-01:${id}`,
    vehicleId: null,
    eventType: direction === 'EXIT' ? 'VEHICLE_EXIT' : 'VEHICLE_ENTER',
    direction,
    status: 'CONFIRMED',
    plate: null,
    vehicle: { type: 'car', confidence: 0.96, color: null },
    media: {
      snapshotKey: null,
      vehicleCropKey: null,
      plateCropKey: null,
      clipKey: null,
    },
    ai: {},
    occurredAt,
    createdAt: occurredAt,
    metadata: {},
  };
}

function setup(): {
  component: DashboardComponent;
  destroyRef: ManualDestroyRef;
  events: Subject<VehicleEvent>;
  recovery: Subject<void>;
  api: {
    events: ReturnType<typeof vi.fn>;
    cameraHealthSnapshot: ReturnType<typeof vi.fn>;
    alerts: ReturnType<typeof vi.fn>;
  };
} {
  const initial = vehicleEvent('initial', '2026-08-16T01:00:00.000Z', 'ENTER');
  const api = {
    events: vi.fn(() => of({ items: [initial], nextCursor: null })),
    cameraHealthSnapshot: vi.fn(() => of({ items: [] })),
    alerts: vi.fn(() => of({ items: [], nextCursor: null })),
  };
  const events = new Subject<VehicleEvent>();
  const recovery = new Subject<void>();
  const realtime = {
    events$: events.asObservable(),
    recoveryRequested$: recovery.asObservable(),
  };
  const destroyRef = new ManualDestroyRef();
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      { provide: RealtimeService, useValue: realtime },
      { provide: DestroyRef, useValue: destroyRef },
    ],
  });
  const component = runInInjectionContext(
    injector,
    () => new DashboardComponent(),
  );
  return { component, destroyRef, events, recovery, api };
}

describe('DashboardComponent realtime performance', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-08-16T02:00:00.000Z'));
    vi.stubGlobal('document', { hidden: false });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('batches a realtime burst into the dashboard signals without reloading APIs', async () => {
    const { component, destroyRef, events, api } = setup();
    await component.load();

    for (let index = 0; index < 20; index += 1) {
      events.next(
        vehicleEvent(
          `realtime-${index}`,
          new Date(
            Date.parse('2026-08-16T02:00:00.000Z') + index,
          ).toISOString(),
          index % 2 ? 'EXIT' : 'ENTER',
        ),
      );
    }

    await vi.advanceTimersByTimeAsync(249);
    expect(component.events()).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);

    expect(component.events()).toHaveLength(21);
    expect(component.entries()).toBe(11);
    expect(component.exits()).toBe(10);
    expect(api.events).toHaveBeenCalledOnce();
    expect(api.alerts).toHaveBeenCalledOnce();
    expect(api.cameraHealthSnapshot).toHaveBeenCalledOnce();
    destroyRef.destroy();
  });

  it('keeps background refreshes idle while the page is hidden', async () => {
    const { component, destroyRef, api } = setup();
    await component.load();
    vi.stubGlobal('document', { hidden: true });

    await vi.advanceTimersByTimeAsync(120_000);

    expect(api.events).toHaveBeenCalledOnce();
    expect(api.alerts).toHaveBeenCalledOnce();
    expect(api.cameraHealthSnapshot).toHaveBeenCalledOnce();
    destroyRef.destroy();
  });

  it('does not overwrite a realtime event when an older full-load response arrives later', async () => {
    const { component, destroyRef, events, api } = setup();
    const eventPage = new Subject<EventPage>();
    api.events.mockReturnValueOnce(eventPage.asObservable());
    const loading = component.load();

    events.next(
      vehicleEvent('during-load', '2026-08-16T02:00:01.000Z', 'EXIT'),
    );
    await vi.advanceTimersByTimeAsync(250);
    eventPage.next({
      items: [
        vehicleEvent('server-snapshot', '2026-08-16T02:00:00.000Z', 'ENTER'),
      ],
      nextCursor: null,
    });
    eventPage.complete();
    await loading;

    expect(component.events().map((event) => event._id)).toEqual([
      'during-load',
      'server-snapshot',
    ]);
    destroyRef.destroy();
  });

  it('removes the previous day snapshot after the local day rolls over', async () => {
    const { component, destroyRef } = setup();
    await component.load();
    component.truncated.set(true);

    vi.setSystemTime(new Date('2026-08-17T02:00:00.000Z'));
    await vi.advanceTimersByTimeAsync(30_000);

    expect(component.events()).toEqual([]);
    expect(component.truncated()).toBe(false);
    destroyRef.destroy();
  });

  it('retains the last alert snapshot when a background refresh fails', async () => {
    const { component, destroyRef, api } = setup();
    await component.load();
    const snapshot = [{ id: 'alert-keep' }] as ReturnType<
      typeof component.alerts
    >;
    component.alerts.set(snapshot);
    api.alerts.mockReturnValueOnce(throwError(() => new Error('alert outage')));

    await vi.advanceTimersByTimeAsync(30_000);

    expect(component.alerts()).toBe(snapshot);
    expect(component.alertLoadState.hasLoaded()).toBe(true);
    expect(component.alertLoadState.staleError()).toBe(
      'Không thể làm mới cảnh báo đang mở.',
    );
    destroyRef.destroy();
  });

  it('retains the last camera snapshot when a background refresh fails', async () => {
    const { component, destroyRef, api } = setup();
    await component.load();
    const snapshot = [{ id: 'camera-keep' }] as unknown as ReturnType<
      typeof component.cameras
    >;
    component.cameras.set(snapshot);
    api.cameraHealthSnapshot.mockReturnValueOnce(
      throwError(() => new Error('camera outage')),
    );

    await vi.advanceTimersByTimeAsync(60_000);

    expect(component.cameras()).toBe(snapshot);
    expect(component.cameraLoadState.hasLoaded()).toBe(true);
    expect(component.cameraLoadState.staleError()).toBe(
      'Không thể làm mới trạng thái camera.',
    );
    destroyRef.destroy();
  });
});
