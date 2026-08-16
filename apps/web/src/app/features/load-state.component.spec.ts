import '@angular/compiler';

import { Injector, runInInjectionContext } from '@angular/core';
import { Observable, of, throwError } from 'rxjs';
import { describe, expect, it, vi } from 'vitest';

import { AuthService } from '../core/auth/auth.service';
import {
  Camera,
  ModelQualityMetrics,
  ModelQualityReport,
} from '../core/models/api.models';
import { ApiClientService } from '../core/services/api-client.service';
import { CamerasComponent } from './cameras/cameras.component';
import { ModelQualityComponent } from './model-quality/model-quality.component';
import { RulesComponent } from './rules/rules.component';
import { WatchlistsComponent } from './watchlists/watchlists.component';

function camera(id: string): Camera {
  return {
    id,
    schemaVersion: 1,
    revision: 0,
    name: 'Gate camera',
    stream: { fpsLimit: 6, credentialsConfigured: true },
    location: { name: 'Gate', zone: 'A' },
    direction: 'ENTRY',
    vision: { vehicleConfidence: 0.4, plateConfidence: 0.45 },
    geometry: {
      vehicleRoi: null,
      crossingLine: null,
      crossingPositiveToNegative: 'ENTER',
      finalizeOnCrossing: false,
    },
    enabled: true,
    metadata: {},
    createdAt: '2026-08-16T00:00:00Z',
    updatedAt: '2026-08-16T00:00:00Z',
  };
}

function metrics(eventCount: number): ModelQualityMetrics {
  return {
    eventCount,
    readablePlateCount: eventCount,
    confirmedCount: eventCount,
    needsReviewCount: 0,
    noPlateCount: 0,
    unreadableCount: 0,
    reviewedCount: 0,
    correctedCount: 0,
    ocrSuccessRate: eventCount ? 1 : 0,
    unknownPlateRate: 0,
    humanCorrectionRate: 0,
    averagePlateConfidence: eventCount ? 0.9 : null,
  };
}

function qualityReport(eventCount: number): ModelQualityReport {
  return {
    schemaVersion: 1,
    window: { from: '2026-08-01T00:00:00Z', to: '2026-08-16T00:00:00Z' },
    generatedAt: '2026-08-16T00:00:00Z',
    totals: metrics(eventCount),
    models: [],
    daily: [],
    feedback: {
      total: 0,
      ready: 0,
      exporting: 0,
      exported: 0,
      exportFailed: 0,
      corrections: 0,
      confirmations: 0,
    },
    truncated: false,
  };
}

describe('component load states', () => {
  it('keeps camera data when a refresh fails', async () => {
    const savedCamera = camera('gate-01');
    const cameraHealthSnapshot = vi
      .fn()
      .mockReturnValueOnce(
        of({ items: [{ camera: savedCamera, health: null }] }),
      )
      .mockReturnValueOnce(throwError(() => ({ status: 0 })));
    const api = { cameraHealthSnapshot };
    const injector = Injector.create({
      providers: [
        { provide: ApiClientService, useValue: api },
        { provide: AuthService, useValue: {} },
      ],
    });
    const component = runInInjectionContext(
      injector,
      () => new CamerasComponent(),
    );

    await component.load();
    expect(component.cameras().map((item) => item.id)).toEqual(['gate-01']);
    expect(component.loadState.hasLoaded()).toBe(true);
    expect(cameraHealthSnapshot).toHaveBeenCalledTimes(1);

    await component.load();
    expect(component.cameras().map((item) => item.id)).toEqual(['gate-01']);
    expect(component.loadState.staleError()).toBe('Không thể kết nối tới API.');
  });

  it('supports initial error, successful retry and stale report retention', async () => {
    const report = qualityReport(12);
    const modelQuality = vi
      .fn<() => Observable<ModelQualityReport>>()
      .mockReturnValueOnce(throwError(() => ({ status: 0 })))
      .mockReturnValueOnce(of(report))
      .mockReturnValueOnce(throwError(() => ({ status: 0 })));
    const injector = Injector.create({
      providers: [{ provide: ApiClientService, useValue: { modelQuality } }],
    });
    const component = runInInjectionContext(
      injector,
      () => new ModelQualityComponent(),
    );

    await vi.waitFor(() => expect(component.loading()).toBe(false));
    expect(component.report()).toBeNull();
    expect(component.loadState.initialError()).toBe(
      'Không thể kết nối tới API.',
    );

    await component.load();
    expect(component.report()).toBe(report);
    expect(component.loadState.hasLoaded()).toBe(true);

    await component.load();
    expect(component.report()).toBe(report);
    expect(component.loadState.staleError()).toBe('Không thể kết nối tới API.');
  });

  it.each([
    {
      create: (api: unknown, auth: unknown) => {
        const injector = Injector.create({
          providers: [
            { provide: ApiClientService, useValue: api },
            { provide: AuthService, useValue: auth },
          ],
        });
        return runInInjectionContext(injector, () => new RulesComponent());
      },
      method: 'rules',
    },
    {
      create: (api: unknown, auth: unknown) => {
        const injector = Injector.create({
          providers: [
            { provide: ApiClientService, useValue: api },
            { provide: AuthService, useValue: auth },
          ],
        });
        return runInInjectionContext(injector, () => new WatchlistsComponent());
      },
      method: 'watchlists',
    },
  ])(
    'does not turn an initial $method failure into an empty result',
    async ({ create, method }) => {
      const request = vi
        .fn()
        .mockReturnValueOnce(throwError(() => ({ status: 0 })))
        .mockReturnValueOnce(of({ items: [] }))
        .mockReturnValueOnce(throwError(() => ({ status: 0 })));
      const component = create({ [method]: request }, {});

      await component.load();
      expect(component.loadState.hasLoaded()).toBe(false);
      expect(component.loadState.initialError()).toBe(
        'Không thể kết nối tới API.',
      );

      await component.load();
      expect(component.loadState.hasLoaded()).toBe(true);
      expect(component.loadState.initialError()).toBeNull();

      await component.load();
      expect(component.loadState.staleError()).toBe(
        'Không thể kết nối tới API.',
      );
    },
  );
});
