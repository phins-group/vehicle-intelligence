import '@angular/compiler';

import { HttpHeaders, HttpResponse } from '@angular/common/http';
import { Injector, runInInjectionContext } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { Subject, finalize, of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  LiveMonitorFrame,
  LiveMonitorState,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { LiveMonitorComponent } from './live-monitor.component';

function frame(sequence = 1): LiveMonitorFrame {
  return {
    sequence,
    frameId: sequence,
    streamEpoch: 1,
    capturedAt: '2026-08-16T02:00:00.000Z',
    receivedAt: '2026-08-16T02:00:00.010Z',
    sourceWidth: 1920,
    sourceHeight: 1080,
    previewWidth: 960,
    previewHeight: 540,
    vehicles: [],
    vehicleRoi: null,
    crossingLine: null,
    frameUrl: `/api/cameras/gate-01/live/frame?sequence=${sequence}`,
  };
}

function state(
  status: LiveMonitorState['status'],
  latest: LiveMonitorFrame | null,
): LiveMonitorState {
  return {
    cameraId: 'gate-01',
    status,
    sourceState: status === 'OFFLINE' ? 'OFFLINE' : 'ONLINE',
    latest,
  };
}

function createComponent(api: unknown): LiveMonitorComponent {
  const route = {
    snapshot: { queryParamMap: { get: vi.fn(() => null) } },
  };
  const router = { navigate: vi.fn(() => Promise.resolve(true)) };
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      { provide: ActivatedRoute, useValue: route },
      { provide: Router, useValue: router },
    ],
  });
  return runInInjectionContext(injector, () => new LiveMonitorComponent());
}

function frameResponse(sequence: number): HttpResponse<Blob> {
  return new HttpResponse({
    body: new Blob(['preview'], { type: 'image/jpeg' }),
    headers: new HttpHeaders({ 'X-Live-Sequence': String(sequence) }),
    status: 200,
  });
}

describe('LiveMonitorComponent polling lifecycle', () => {
  const createObjectUrl = vi.fn(() => 'blob:preview');
  const revokeObjectUrl = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    createObjectUrl.mockClear();
    revokeObjectUrl.mockClear();
    vi.stubGlobal('window', {
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
    });
    vi.stubGlobal('document', { hidden: false });
    vi.stubGlobal('URL', {
      createObjectURL: createObjectUrl,
      revokeObjectURL: revokeObjectUrl,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('unsubscribes an in-flight frame request and ignores a late response on destroy', () => {
    const frameResponses = new Subject<HttpResponse<Blob>>();
    const cancelFrame = vi.fn();
    const api = {
      liveMonitorState: vi.fn(() => of(state('LIVE', frame(7)))),
      liveMonitorFrame: vi.fn(() => frameResponses.pipe(finalize(cancelFrame))),
    };
    const component = createComponent(api);

    component.selectCamera('gate-01');
    expect(api.liveMonitorFrame).toHaveBeenCalledWith('gate-01', 7);
    component.ngOnDestroy();

    expect(cancelFrame).toHaveBeenCalledOnce();
    frameResponses.next(frameResponse(7));
    frameResponses.complete();
    expect(createObjectUrl).not.toHaveBeenCalled();
    expect(component.imageUrl()).toBeNull();
    vi.advanceTimersByTime(10_000);
    expect(api.liveMonitorState).toHaveBeenCalledOnce();
  });

  it('uses a slower cadence while waiting and stops the timer on destroy', () => {
    const api = {
      liveMonitorState: vi.fn(() => of(state('WAITING', null))),
      liveMonitorFrame: vi.fn(),
    };
    const component = createComponent(api);

    component.selectCamera('gate-01');
    expect(api.liveMonitorState).toHaveBeenCalledOnce();
    vi.advanceTimersByTime(2_499);
    expect(api.liveMonitorState).toHaveBeenCalledOnce();
    vi.advanceTimersByTime(1);
    expect(api.liveMonitorState).toHaveBeenCalledTimes(2);
    expect(api.liveMonitorFrame).not.toHaveBeenCalled();

    component.ngOnDestroy();
    vi.advanceTimersByTime(10_000);
    expect(api.liveMonitorState).toHaveBeenCalledTimes(2);
  });

  it('does not start polling if the camera list resolves after destroy', async () => {
    const cameras = new Subject<{ items: [] }>();
    const api = {
      cameras: vi.fn(() => cameras.asObservable()),
      liveMonitorState: vi.fn(),
      liveMonitorFrame: vi.fn(),
    };
    const component = createComponent(api);
    const loading = component.loadCameras();

    component.ngOnDestroy();
    cameras.next({ items: [] });
    cameras.complete();
    await loading;

    expect(api.liveMonitorState).not.toHaveBeenCalled();
  });

  it('pauses automatic polling, keeps the current state and resumes immediately', () => {
    const api = {
      liveMonitorState: vi.fn(() => of(state('WAITING', null))),
      liveMonitorFrame: vi.fn(),
    };
    const component = createComponent(api);

    component.selectCamera('gate-01');
    expect(api.liveMonitorState).toHaveBeenCalledOnce();
    component.togglePause();
    expect(component.paused()).toBe(true);
    vi.advanceTimersByTime(10_000);
    expect(api.liveMonitorState).toHaveBeenCalledOnce();

    component.togglePause();
    expect(component.paused()).toBe(false);
    expect(api.liveMonitorState).toHaveBeenCalledTimes(2);
  });

  it('bounds the screen-reader detection summary', () => {
    const component = createComponent({});
    const populated = frame();
    populated.vehicles = Array.from({ length: 15 }, (_, index) => ({
      trackId: String(index + 1),
      bbox: [0, 0, 10, 10],
      confidence: 0.9,
      vehicleType: 'car',
      direction: 'ENTER',
      plate: null,
    }));

    expect(component.visibleDetections(populated)).toHaveLength(12);
    expect(component.hiddenDetectionCount(populated)).toBe(3);
  });
});
