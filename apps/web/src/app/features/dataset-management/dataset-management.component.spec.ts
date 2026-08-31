import '@angular/compiler';

import { DOCUMENT } from '@angular/common';
import { Injector, runInInjectionContext } from '@angular/core';
import { Subject, of } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AuthService } from '../../core/auth/auth.service';
import {
  DatasetHubSyncJob,
  DetectorDatasetSamplePreview,
  DetectorDatasetVersion,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { DatasetManagementComponent } from './dataset-management.component';

interface TestableDatasetManagement {
  loadSampleImages(
    items: DetectorDatasetSamplePreview[],
    sourceId: string,
    generation: number,
  ): Promise<void>;
  startPolling(jobId: string): void;
}

function sample(index: number): DetectorDatasetSamplePreview {
  return {
    sourceId: 'dataset-v1',
    sampleId: `sample-${index}`,
    imageSha256: `hash-${index}`,
    cameraId: 'gate-01',
    groupId: 'group-01',
    capturedAt: '2026-08-16T00:00:00Z',
    split: 'train',
    lighting: 'DAY',
    annotationStatus: 'REVIEWED',
    negative: false,
    image: { width: 640, height: 480 },
    annotations: [],
    imageUrl: `/samples/hash-${index}/image`,
  };
}

function syncJob(
  status: DatasetHubSyncJob['status'],
  sourceId = 'dataset-v1',
): DatasetHubSyncJob {
  return {
    id: 'sync-1',
    sourceId,
    sourceManifestSha256: 'source-sha',
    exportId: 'export-v1',
    repoId: 'dataset/repo',
    requestedRevision: 'main',
    status,
    requestedBy: 'reviewer',
    restrictedTransferConfirmed: false,
    createdAt: '2026-08-16T00:00:00Z',
    updatedAt: '2026-08-16T00:00:00Z',
    exportManifestSha256: null,
    hubCommitSha: null,
    hubUrl: null,
    reusedExport: false,
    errorCode: null,
  };
}

function dataset(sourceId: string): DetectorDatasetVersion {
  return {
    sourceId,
    sourceManifestSha256: `${sourceId}-sha`,
    createdAt: '2026-08-16T00:00:00Z',
    sampleCount: 1,
    annotationCount: 1,
    negativeSampleCount: 0,
    reviewQueueCount: 0,
    releaseEligible: true,
    distributionEligible: true,
    privacyClassification: 'INTERNAL',
    parentSourceId: null,
    export: null,
    latestSync: null,
  };
}

function createComponent(api: unknown): DatasetManagementComponent {
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      { provide: DOCUMENT, useValue: { getElementById: () => null } },
      {
        provide: AuthService,
        useValue: {
          canReviewDatasets: () => true,
          canManageDatasets: () => true,
        },
      },
    ],
  });
  return runInInjectionContext(
    injector,
    () => new DatasetManagementComponent(),
  );
}

describe('DatasetManagementComponent resource lifecycle', () => {
  const createObjectUrl = vi.fn(
    (blob: Blob) => `blob:${blob.size}:${createObjectUrl.mock.calls.length}`,
  );
  const revokeObjectUrl = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    createObjectUrl.mockClear();
    revokeObjectUrl.mockClear();
    vi.stubGlobal('window', {
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
    });
    vi.stubGlobal('URL', {
      createObjectURL: createObjectUrl,
      revokeObjectURL: revokeObjectUrl,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('does not create a Blob URL when an image response arrives after destroy', async () => {
    const image = new Subject<Blob>();
    const api = {
      detectorDatasetSampleImage: vi.fn(() => image.asObservable()),
    };
    const component = createComponent(api);
    component.selectedSourceId.set('dataset-v1');
    const testable = component as unknown as TestableDatasetManagement;

    const pending = testable.loadSampleImages([sample(1)], 'dataset-v1', 0);
    component.ngOnDestroy();
    image.next(new Blob(['late-image']));
    image.complete();
    await pending;

    expect(createObjectUrl).not.toHaveBeenCalled();
    expect(component.sampleImageUrls()).toEqual({});
  });

  it('batches signal publication and bounds the thumbnail Blob URL cache', async () => {
    const api = {
      detectorDatasetSampleImage: vi.fn(() => of(new Blob(['thumbnail']))),
    };
    const component = createComponent(api);
    const items = Array.from({ length: 100 }, (_, index) => sample(index));
    component.selectedSourceId.set('dataset-v1');
    component.selectedSample.set(items[0]);
    const publish = vi.spyOn(component.sampleImageUrls, 'set');

    await (component as unknown as TestableDatasetManagement).loadSampleImages(
      items,
      'dataset-v1',
      0,
    );

    expect(Object.keys(component.sampleImageUrls())).toHaveLength(48);
    expect(component.sampleImageUrls()['hash-0']).toBeDefined();
    expect(revokeObjectUrl).toHaveBeenCalledTimes(52);
    expect(publish).toHaveBeenCalledOnce();
    component.ngOnDestroy();
  });

  it('does not update or reschedule a sync poll after destroy', async () => {
    const response = new Subject<DatasetHubSyncJob>();
    const api = { detectorDatasetSync: vi.fn(() => response.asObservable()) };
    const component = createComponent(api);
    (component as unknown as TestableDatasetManagement).startPolling('sync-1');

    vi.advanceTimersByTime(2000);
    expect(api.detectorDatasetSync).toHaveBeenCalledOnce();
    component.ngOnDestroy();
    response.next(syncJob('UPLOADING'));
    response.complete();
    await Promise.resolve();
    vi.advanceTimersByTime(10_000);

    expect(component.activeJob()).toBeNull();
    expect(api.detectorDatasetSync).toHaveBeenCalledOnce();
  });

  it('invalidates an in-flight sync when selecting another dataset', async () => {
    const response = new Subject<DatasetHubSyncJob>();
    const api = {
      syncDetectorDataset: vi.fn(() => response.asObservable()),
      detectorDatasetSamples: vi.fn(() => of({ items: [], nextCursor: null })),
    };
    const component = createComponent(api);
    component.datasets.set([dataset('dataset-a'), dataset('dataset-b')]);
    component.hub.set({
      enabled: true,
      hubEnabled: true,
      repoId: 'dataset/repo',
      credentialsConfigured: true,
      restrictedPrivateSyncEnabled: false,
    });
    component.selectedSourceId.set('dataset-a');
    component.exportId = 'export-a';
    component.revision = 'main';

    const pending = component.startSync();
    expect(api.syncDetectorDataset).toHaveBeenCalledOnce();
    component.selectDataset('dataset-b');
    response.next(syncJob('QUEUED', 'dataset-a'));
    response.complete();
    await pending;

    expect(component.selectedSourceId()).toBe('dataset-b');
    expect(component.activeJob()).toBeNull();
    expect(component.syncing()).toBe(false);
    component.ngOnDestroy();
  });
});
