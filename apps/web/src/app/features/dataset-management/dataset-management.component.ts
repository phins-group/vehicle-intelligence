import { DOCUMENT, DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  OnDestroy,
  OnInit,
  computed,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  LucideBoxes,
  LucideChevronRight,
  LucideCircleCheck,
  LucideDatabase,
  LucideImage,
  LucideRefreshCw,
  LucideShieldCheck,
  LucideTriangleAlert,
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import {
  DatasetHubSyncJob,
  DatasetRegistryResponse,
  DetectorDatasetSampleKind,
  DetectorDatasetSamplePreview,
  DetectorDatasetVersion,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import {
  datasetReadiness,
  defaultDatasetExportId,
  isDatasetSyncActive,
} from '../../core/utils/dataset-registry-utils';

const MAX_CACHED_SAMPLE_IMAGES = 48;

@Component({
  selector: 'app-dataset-management',
  imports: [
    DatePipe,
    DecimalPipe,
    FormsModule,
    LucideBoxes,
    LucideChevronRight,
    LucideCircleCheck,
    LucideDatabase,
    LucideImage,
    LucideRefreshCw,
    LucideShieldCheck,
    LucideTriangleAlert,
  ],
  templateUrl: './dataset-management.component.html',
})
export class DatasetManagementComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiClientService);
  private readonly document = inject(DOCUMENT);
  readonly auth = inject(AuthService);

  readonly datasets = signal<DetectorDatasetVersion[]>([]);
  readonly hub = signal<DatasetRegistryResponse['hub']>({
    enabled: false,
    hubEnabled: false,
    repoId: null,
    credentialsConfigured: false,
    restrictedPrivateSyncEnabled: false,
  });
  readonly loading = signal(true);
  readonly loadState = new AsyncDataState();
  readonly syncing = signal(false);
  readonly error = signal<string | null>(null);
  readonly success = signal<string | null>(null);
  readonly activeJob = signal<DatasetHubSyncJob | null>(null);
  readonly selectedSourceId = signal('');
  readonly samples = signal<DetectorDatasetSamplePreview[]>([]);
  readonly sampleNextCursor = signal<string | null>(null);
  readonly selectedSample = signal<DetectorDatasetSamplePreview | null>(null);
  readonly sampleImageUrls = signal<Record<string, string>>({});
  readonly failedSampleImages = signal<ReadonlySet<string>>(new Set());
  readonly loadingSamples = signal(false);
  readonly loadingMoreSamples = signal(false);
  readonly sampleError = signal<string | null>(null);
  readonly selected = computed(
    () =>
      this.datasets().find(
        (item) => item.sourceId === this.selectedSourceId(),
      ) ?? null,
  );
  readonly totalSamples = computed(() =>
    this.datasets().reduce((total, item) => total + item.sampleCount, 0),
  );
  readonly readyCount = computed(
    () => this.datasets().filter((item) => item.reviewQueueCount === 0).length,
  );
  readonly syncedCount = computed(
    () =>
      this.datasets().filter((item) => item.latestSync?.status === 'COMPLETED')
        .length,
  );
  canStartSync(): boolean {
    const selected = this.selected();
    const hub = this.hub();
    if (
      !selected ||
      !this.auth.canManageDatasets() ||
      !hub.enabled ||
      !hub.hubEnabled ||
      !hub.credentialsConfigured ||
      selected.reviewQueueCount > 0 ||
      !selected.releaseEligible ||
      isDatasetSyncActive(
        this.activeJob()?.status ?? selected.latestSync?.status,
      ) ||
      !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(this.exportId.trim()) ||
      !/^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$/.test(this.revision.trim())
    ) {
      return false;
    }
    if (!selected.distributionEligible) {
      return hub.restrictedPrivateSyncEnabled && this.confirmRestrictedTransfer;
    }
    return true;
  }

  exportId = '';
  revision = 'main';
  confirmRestrictedTransfer = false;
  sampleKind: DetectorDatasetSampleKind = 'ALL';
  sampleLighting: '' | 'DAY' | 'NIGHT' | 'UNKNOWN' = '';
  private pollTimer: number | null = null;
  private destroyed = false;
  private loadGeneration = 0;
  private syncActionGeneration = 0;
  private pollGeneration = 0;
  private sampleRequestGeneration = 0;
  private readonly sampleObjectUrls = new Map<string, string>();
  private readonly sampleImageRequestKeys = new Set<string>();

  ngOnInit(): void {
    if (!this.auth.canReviewDatasets()) {
      this.loading.set(false);
      return;
    }
    void this.load();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.loadGeneration += 1;
    this.syncActionGeneration += 1;
    this.cancelPolling();
    this.sampleRequestGeneration += 1;
    this.sampleImageRequestKeys.clear();
    this.releaseSampleImages();
  }

  async load(): Promise<void> {
    if (this.destroyed) return;
    const generation = ++this.loadGeneration;
    this.loading.set(true);
    this.error.set(null);
    this.loadState.begin();
    try {
      const response = await firstValueFrom(this.api.detectorDatasets());
      if (this.destroyed || generation !== this.loadGeneration) return;
      this.datasets.set(response.items);
      this.hub.set(response.hub);
      this.loadState.succeed();
      if (
        !response.items.some(
          (item) => item.sourceId === this.selectedSourceId(),
        )
      ) {
        this.selectedSourceId.set(response.items[0]?.sourceId ?? '');
        this.populateSyncForm();
      }
      if (this.selectedSourceId()) await this.loadSamples(true);
      else this.resetSampleBrowser();
    } catch (error) {
      if (!this.destroyed && generation === this.loadGeneration) {
        this.loadState.fail(
          apiErrorMessage(error, 'Không thể tải danh mục dataset.'),
        );
      }
    } finally {
      if (!this.destroyed && generation === this.loadGeneration)
        this.loading.set(false);
    }
  }

  selectDataset(sourceId: string): void {
    this.syncActionGeneration += 1;
    this.syncing.set(false);
    this.cancelPolling();
    this.selectedSourceId.set(sourceId);
    this.activeJob.set(null);
    this.error.set(null);
    this.success.set(null);
    this.populateSyncForm();
    void this.loadSamples(true);
  }

  reloadSamples(): void {
    void this.loadSamples(true);
  }

  scrollToSamples(): void {
    this.document
      .getElementById('dataset-samples')
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  async loadSamples(reset: boolean): Promise<void> {
    if (this.destroyed) return;
    const sourceId = this.selectedSourceId();
    if (!sourceId) {
      this.resetSampleBrowser();
      return;
    }
    if (
      !reset &&
      (this.loadingSamples() ||
        this.loadingMoreSamples() ||
        !this.sampleNextCursor())
    ) {
      return;
    }

    const generation = reset
      ? ++this.sampleRequestGeneration
      : this.sampleRequestGeneration;
    if (reset) {
      this.releaseSampleImages();
      this.samples.set([]);
      this.sampleNextCursor.set(null);
      this.selectedSample.set(null);
      this.loadingSamples.set(true);
    } else {
      this.loadingMoreSamples.set(true);
    }
    this.sampleError.set(null);

    try {
      const page = await firstValueFrom(
        this.api.detectorDatasetSamples(sourceId, {
          limit: 12,
          cursor: reset ? null : this.sampleNextCursor(),
          kind: this.sampleKind,
          lighting: this.sampleLighting,
        }),
      );
      if (
        this.destroyed ||
        generation !== this.sampleRequestGeneration ||
        sourceId !== this.selectedSourceId()
      ) {
        return;
      }
      this.samples.update((current) =>
        reset ? page.items : [...current, ...page.items],
      );
      this.sampleNextCursor.set(page.nextCursor);
      if (reset) this.selectedSample.set(page.items[0] ?? null);
      void this.loadSampleImages(page.items, sourceId, generation);
    } catch (error) {
      if (!this.destroyed && generation === this.sampleRequestGeneration) {
        this.sampleError.set(
          apiErrorMessage(error, 'Không thể tải mẫu của dataset.'),
        );
      }
    } finally {
      if (!this.destroyed && generation === this.sampleRequestGeneration) {
        this.loadingSamples.set(false);
        this.loadingMoreSamples.set(false);
      }
    }
  }

  showSample(sample: DetectorDatasetSamplePreview): void {
    if (this.destroyed) return;
    this.selectedSample.set(sample);
    const objectUrl = this.sampleObjectUrls.get(sample.imageSha256);
    if (objectUrl) {
      this.sampleObjectUrls.delete(sample.imageSha256);
      this.sampleObjectUrls.set(sample.imageSha256, objectUrl);
    } else if (!this.failedSampleImages().has(sample.imageSha256)) {
      void this.loadSampleImages(
        [sample],
        this.selectedSourceId(),
        this.sampleRequestGeneration,
      );
    }
  }

  sampleImageUrl(sample: DetectorDatasetSamplePreview): string | null {
    if (this.failedSampleImages().has(sample.imageSha256)) return null;
    return this.sampleImageUrls()[sample.imageSha256] ?? null;
  }

  markSampleImageFailed(sample: DetectorDatasetSamplePreview): void {
    if (this.destroyed) return;
    this.failedSampleImages.update(
      (current) => new Set([...current, sample.imageSha256]),
    );
  }

  readiness(item: DetectorDatasetVersion): ReturnType<typeof datasetReadiness> {
    return datasetReadiness(item);
  }

  async startSync(): Promise<void> {
    const dataset = this.selected();
    if (this.destroyed || !dataset || !this.canStartSync() || this.syncing())
      return;
    const generation = ++this.syncActionGeneration;
    this.syncing.set(true);
    this.error.set(null);
    this.success.set(null);
    try {
      const job = await firstValueFrom(
        this.api.syncDetectorDataset(dataset.sourceId, {
          exportId: this.exportId.trim(),
          revision: this.revision.trim(),
          confirmRestrictedPrivateTransfer: this.confirmRestrictedTransfer,
        }),
      );
      if (this.destroyed || generation !== this.syncActionGeneration) return;
      this.activeJob.set(job);
      if (isDatasetSyncActive(job.status)) this.startPolling(job.id);
      else this.finishJob(job);
    } catch (error) {
      if (!this.destroyed && generation === this.syncActionGeneration) {
        this.error.set(
          apiErrorMessage(error, 'Không thể bắt đầu đồng bộ Hugging Face.'),
        );
      }
    } finally {
      if (!this.destroyed && generation === this.syncActionGeneration) {
        this.syncing.set(false);
      }
    }
  }

  private populateSyncForm(): void {
    const dataset = this.selected();
    if (!dataset) return;
    this.exportId =
      dataset.export?.exportId ?? defaultDatasetExportId(dataset.sourceId);
    this.revision = dataset.latestSync?.requestedRevision ?? 'main';
    this.confirmRestrictedTransfer = false;
    this.activeJob.set(dataset.latestSync);
  }

  private startPolling(jobId: string): void {
    this.cancelPolling();
    this.poll(jobId, this.pollGeneration);
  }

  private poll(jobId: string, generation: number): void {
    if (this.destroyed || generation !== this.pollGeneration) return;
    if (this.pollTimer !== null) window.clearTimeout(this.pollTimer);
    this.pollTimer = window.setTimeout(async () => {
      this.pollTimer = null;
      if (this.destroyed || generation !== this.pollGeneration) return;
      try {
        const job = await firstValueFrom(this.api.detectorDatasetSync(jobId));
        if (this.destroyed || generation !== this.pollGeneration) return;
        this.activeJob.set(job);
        if (isDatasetSyncActive(job.status)) this.poll(job.id, generation);
        else this.finishJob(job);
      } catch (error) {
        if (!this.destroyed && generation === this.pollGeneration) {
          this.error.set(
            apiErrorMessage(error, 'Không thể cập nhật trạng thái đồng bộ.'),
          );
        }
      }
    }, 2000);
  }

  private cancelPolling(): void {
    this.pollGeneration += 1;
    if (this.pollTimer !== null) window.clearTimeout(this.pollTimer);
    this.pollTimer = null;
  }

  private finishJob(job: DatasetHubSyncJob): void {
    if (this.destroyed) return;
    if (job.status === 'COMPLETED') {
      this.success.set(`Đã đồng bộ ${job.exportId} lên ${job.repoId}.`);
      void this.load();
    } else if (job.status === 'FAILED') {
      this.error.set(`Đồng bộ thất bại (${job.errorCode ?? 'UNKNOWN'}).`);
    }
  }

  private async loadSampleImages(
    items: DetectorDatasetSamplePreview[],
    sourceId: string,
    generation: number,
  ): Promise<void> {
    if (
      this.destroyed ||
      !sourceId ||
      generation !== this.sampleRequestGeneration
    )
      return;
    const pending = items.filter((item) => {
      const requestKey = this.sampleImageRequestKey(
        sourceId,
        generation,
        item.imageSha256,
      );
      if (
        this.sampleObjectUrls.has(item.imageSha256) ||
        this.failedSampleImages().has(item.imageSha256) ||
        this.sampleImageRequestKeys.has(requestKey)
      ) {
        return false;
      }
      this.sampleImageRequestKeys.add(requestKey);
      return true;
    });
    if (!pending.length) return;

    const results = await Promise.all(
      pending.map(async (item) => {
        try {
          const blob = await firstValueFrom(
            this.api.detectorDatasetSampleImage(sourceId, item.imageSha256),
          );
          return { item, blob };
        } catch {
          return { item, blob: null };
        }
      }),
    );
    if (!this.destroyed) {
      for (const item of pending) {
        this.sampleImageRequestKeys.delete(
          this.sampleImageRequestKey(sourceId, generation, item.imageSha256),
        );
      }
    }
    if (
      this.destroyed ||
      generation !== this.sampleRequestGeneration ||
      sourceId !== this.selectedSourceId()
    ) {
      return;
    }

    const failed = new Set(this.failedSampleImages());
    let loadedCount = 0;
    for (const result of results) {
      if (result.blob === null) {
        failed.add(result.item.imageSha256);
        continue;
      }
      try {
        const objectUrl = URL.createObjectURL(result.blob);
        const previous = this.sampleObjectUrls.get(result.item.imageSha256);
        if (previous) URL.revokeObjectURL(previous);
        this.sampleObjectUrls.delete(result.item.imageSha256);
        this.sampleObjectUrls.set(result.item.imageSha256, objectUrl);
        failed.delete(result.item.imageSha256);
        loadedCount += 1;
      } catch {
        failed.add(result.item.imageSha256);
      }
    }
    this.trimSampleImageCache();
    this.sampleImageUrls.set(Object.fromEntries(this.sampleObjectUrls));
    this.failedSampleImages.set(failed);
    if (
      items.length > 0 &&
      loadedCount === 0 &&
      !this.destroyed &&
      generation === this.sampleRequestGeneration
    ) {
      this.sampleError.set(
        'Không thể tải ảnh mẫu có xác thực từ dataset source.',
      );
    }
  }

  private trimSampleImageCache(): void {
    const selectedHash = this.selectedSample()?.imageSha256 ?? null;
    while (this.sampleObjectUrls.size > MAX_CACHED_SAMPLE_IMAGES) {
      let removed = false;
      for (const [imageSha256, objectUrl] of this.sampleObjectUrls) {
        if (imageSha256 === selectedHash && this.sampleObjectUrls.size > 1)
          continue;
        URL.revokeObjectURL(objectUrl);
        this.sampleObjectUrls.delete(imageSha256);
        removed = true;
        break;
      }
      if (!removed) break;
    }
  }

  private sampleImageRequestKey(
    sourceId: string,
    generation: number,
    imageSha256: string,
  ): string {
    return `${generation}:${sourceId}:${imageSha256}`;
  }

  private resetSampleBrowser(): void {
    this.sampleRequestGeneration += 1;
    this.releaseSampleImages();
    this.samples.set([]);
    this.sampleNextCursor.set(null);
    this.selectedSample.set(null);
    this.sampleError.set(null);
    this.loadingSamples.set(false);
    this.loadingMoreSamples.set(false);
  }

  private releaseSampleImages(): void {
    for (const objectUrl of this.sampleObjectUrls.values())
      URL.revokeObjectURL(objectUrl);
    this.sampleObjectUrls.clear();
    this.sampleImageRequestKeys.clear();
    this.sampleImageUrls.set({});
    this.failedSampleImages.set(new Set());
  }
}
