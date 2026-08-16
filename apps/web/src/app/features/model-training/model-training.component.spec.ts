import '@angular/compiler';

import { Injector, runInInjectionContext } from '@angular/core';
import { Subject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AuthService } from '../../core/auth/auth.service';
import {
  ModelTrainingLog,
  ModelTrainingRun,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { ModelTrainingComponent } from './model-training.component';

function trainingRun(
  id: string,
  updatedAt = '2026-08-16T00:00:00Z',
): ModelTrainingRun {
  return {
    id,
    role: 'plate',
    status: 'RUNNING',
    sourceId: 'dataset-v1',
    sourceManifestSha256: 'source-sha',
    exportId: 'export-v1',
    exportManifestSha256: 'export-sha',
    datasetRepoId: 'dataset/repo',
    datasetRevision: 'main',
    datasetCommitSha: 'commit-sha',
    modelRepoId: 'model/repo',
    modelName: 'plate-detector',
    modelVersion: 'v1',
    architecture: 'yolo',
    parameters: {
      epochs: 10,
      batchSize: 4,
      workers: 1,
      snapshotEpoch: 1,
      timeoutSeconds: 3600,
      hardwareFlavor: 'gpu-small',
    },
    requestedBy: 'reviewer',
    confirmations: {
      datasetRights: true,
      computeCost: true,
      restrictedData: false,
    },
    createdAt: '2026-08-16T00:00:00Z',
    updatedAt,
    startedAt: '2026-08-16T00:00:00Z',
    finishedAt: null,
    outputBucket: 'models',
    outputPath: id,
    remoteJobId: `remote-${id}`,
    remoteJobUrl: null,
    remoteMessage: null,
    errorCode: null,
  };
}

function createComponent(api: unknown): ModelTrainingComponent {
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      {
        provide: AuthService,
        useValue: {
          canReviewDatasets: () => true,
          canManageDatasets: () => true,
        },
      },
    ],
  });
  return runInInjectionContext(injector, () => new ModelTrainingComponent());
}

describe('ModelTrainingComponent lifecycle', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('window', {
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
      confirm: () => true,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('does not let logs from a previous run overwrite the current selection', async () => {
    const logResponses = new Map<string, Subject<ModelTrainingLog>>([
      ['run-a', new Subject<ModelTrainingLog>()],
      ['run-b', new Subject<ModelTrainingLog>()],
    ]);
    const api = {
      modelTrainingLogs: vi.fn((runId: string) =>
        logResponses.get(runId)!.asObservable(),
      ),
    };
    const component = createComponent(api);
    component.runs.set([trainingRun('run-a'), trainingRun('run-b')]);

    component.selectRun('run-a');
    component.selectRun('run-b');
    logResponses
      .get('run-b')!
      .next({ runId: 'run-b', lines: ['new-run'], available: true });
    await Promise.resolve();
    expect(component.logLines()).toEqual(['new-run']);

    logResponses
      .get('run-a')!
      .next({ runId: 'run-a', lines: ['stale-run'], available: true });
    await Promise.resolve();
    expect(component.selectedRunId()).toBe('run-b');
    expect(component.logLines()).toEqual(['new-run']);

    component.ngOnDestroy();
  });

  it('ignores an in-flight refresh and does not reschedule after destroy', async () => {
    const refresh = new Subject<ModelTrainingRun>();
    const api = {
      modelTrainingRun: vi.fn(() => refresh.asObservable()),
      modelTrainingLogs: vi.fn(),
    };
    const component = createComponent(api);
    const original = trainingRun('run-a');
    component.runs.set([original]);
    component.selectedRunId.set(original.id);

    const pending = component.refreshSelectedRun();
    component.ngOnDestroy();
    refresh.next(trainingRun('run-a', '2026-08-16T01:00:00Z'));
    refresh.complete();
    await pending;
    vi.advanceTimersByTime(10_000);

    expect(component.runs()).toEqual([original]);
    expect(api.modelTrainingLogs).not.toHaveBeenCalled();
    expect(api.modelTrainingRun).toHaveBeenCalledOnce();
  });
});
