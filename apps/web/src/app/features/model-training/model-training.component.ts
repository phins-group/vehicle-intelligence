import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  LucideBrainCircuit,
  LucideCheck,
  LucideCircleCheck,
  LucideClock3,
  LucideExternalLink,
  LucidePlay,
  LucideRefreshCw,
  LucideScrollText,
  LucideShieldCheck,
  LucideSquare,
  LucideTriangleAlert
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import {
  DatasetRegistryResponse,
  DetectorDatasetVersion,
  ModelTrainingCapabilities,
  ModelTrainingRun
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import {
  isModelTrainingActive,
  modelTrainingBlockerMessage,
  trainingDatasetReadiness,
  trainingStatusLabel
} from '../../core/utils/model-training-utils';

@Component({
  selector: 'app-model-training',
  imports: [
    DatePipe,
    DecimalPipe,
    FormsModule,
    LucideBrainCircuit,
    LucideCheck,
    LucideCircleCheck,
    LucideClock3,
    LucideExternalLink,
    LucidePlay,
    LucideRefreshCw,
    LucideScrollText,
    LucideShieldCheck,
    LucideSquare,
    LucideTriangleAlert
  ],
  templateUrl: './model-training.component.html'
})
export class ModelTrainingComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiClientService);
  readonly auth = inject(AuthService);

  readonly capabilities = signal<ModelTrainingCapabilities | null>(null);
  readonly datasets = signal<DetectorDatasetVersion[]>([]);
  readonly runs = signal<ModelTrainingRun[]>([]);
  readonly selectedSourceId = signal('');
  readonly selectedRunId = signal('');
  readonly logLines = signal<string[]>([]);
  readonly logsAvailable = signal(false);
  readonly loading = signal(true);
  readonly starting = signal(false);
  readonly canceling = signal(false);
  readonly refreshingRun = signal(false);
  readonly error = signal<string | null>(null);
  readonly success = signal<string | null>(null);
  readonly logError = signal<string | null>(null);

  readonly selectedDataset = computed(
    () => this.datasets().find((item) => item.sourceId === this.selectedSourceId()) ?? null
  );
  readonly selectedRun = computed(
    () => this.runs().find((item) => item.id === this.selectedRunId()) ?? null
  );
  readonly activeRunCount = computed(
    () => this.runs().filter((item) => isModelTrainingActive(item.status)).length
  );
  readonly completedRunCount = computed(
    () => this.runs().filter((item) => item.status === 'COMPLETED').length
  );
  readonly selectedReadiness = computed(() =>
    trainingDatasetReadiness(
      this.selectedDataset(),
      this.capabilities()?.defaults.datasetRepoId ?? ''
    )
  );

  modelName = 'phins-vn-plate-detector';
  modelVersion = this.defaultVersion();
  epochs = 100;
  batchSize = 16;
  workers = 4;
  snapshotEpoch = 5;
  confirmDatasetRights = false;
  confirmComputeCost = false;
  confirmRestrictedData = false;

  private initializedForm = false;
  private pollTimer: number | null = null;

  ngOnInit(): void {
    if (!this.auth.canReviewDatasets()) {
      this.loading.set(false);
      return;
    }
    void this.load();
  }

  ngOnDestroy(): void {
    if (this.pollTimer !== null) window.clearTimeout(this.pollTimer);
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const [training, datasets] = await Promise.all([
        firstValueFrom(this.api.modelTrainingOverview()),
        firstValueFrom(this.api.detectorDatasets())
      ]);
      this.capabilities.set(training.capabilities);
      this.datasets.set(datasets.items);
      this.runs.set(training.items);
      this.initializeForm(training.capabilities, datasets);
      this.ensureSelections();
      const selectedRun = this.selectedRun();
      if (selectedRun) {
        await this.loadLogs(selectedRun);
        if (isModelTrainingActive(selectedRun.status)) this.schedulePoll(selectedRun.id);
      }
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải quy trình build model.'));
    } finally {
      this.loading.set(false);
    }
  }

  selectDataset(sourceId: string): void {
    this.selectedSourceId.set(sourceId);
    this.error.set(null);
    this.success.set(null);
    this.confirmRestrictedData = false;
  }

  selectRun(runId: string): void {
    this.selectedRunId.set(runId);
    this.logLines.set([]);
    this.logsAvailable.set(false);
    this.logError.set(null);
    const run = this.selectedRun();
    if (!run) return;
    void this.loadLogs(run);
    if (isModelTrainingActive(run.status)) this.schedulePoll(run.id);
  }

  canStart(): boolean {
    const capabilities = this.capabilities();
    const dataset = this.selectedDataset();
    return Boolean(
      this.auth.canManageDatasets() &&
        capabilities?.submissionsEnabled &&
        this.selectedReadiness().ready &&
        dataset &&
        !this.starting() &&
        this.activeRunCount() === 0 &&
        /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(this.modelName.trim()) &&
        /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(this.modelVersion.trim()) &&
        Number.isInteger(this.epochs) &&
        this.epochs >= 1 &&
        this.epochs <= 10000 &&
        Number.isInteger(this.batchSize) &&
        this.batchSize >= 1 &&
        this.batchSize <= 1024 &&
        Number.isInteger(this.workers) &&
        this.workers >= 0 &&
        this.workers <= 128 &&
        Number.isInteger(this.snapshotEpoch) &&
        this.snapshotEpoch >= 1 &&
        this.snapshotEpoch <= this.epochs &&
        this.confirmDatasetRights &&
        this.confirmComputeCost &&
        (dataset.distributionEligible || this.confirmRestrictedData)
    );
  }

  async startRun(): Promise<void> {
    const dataset = this.selectedDataset();
    if (!dataset || !this.canStart()) return;
    this.starting.set(true);
    this.error.set(null);
    this.success.set(null);
    try {
      const run = await firstValueFrom(
        this.api.startModelTraining({
          sourceId: dataset.sourceId,
          modelName: this.modelName.trim(),
          modelVersion: this.modelVersion.trim(),
          epochs: this.epochs,
          batchSize: this.batchSize,
          workers: this.workers,
          snapshotEpoch: this.snapshotEpoch,
          confirmDatasetRights: this.confirmDatasetRights,
          confirmComputeCost: this.confirmComputeCost,
          confirmRestrictedData: this.confirmRestrictedData
        })
      );
      this.upsertRun(run);
      this.selectedRunId.set(run.id);
      this.success.set('Training run đã được ghi audit và đưa vào hàng đợi an toàn.');
      this.schedulePoll(run.id, 700);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể bắt đầu training run.'));
    } finally {
      this.starting.set(false);
    }
  }

  async refreshSelectedRun(): Promise<void> {
    const run = this.selectedRun();
    if (!run || this.refreshingRun()) return;
    this.refreshingRun.set(true);
    try {
      const refreshed = await firstValueFrom(this.api.modelTrainingRun(run.id));
      this.upsertRun(refreshed);
      await this.loadLogs(refreshed);
      if (isModelTrainingActive(refreshed.status)) this.schedulePoll(refreshed.id);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể cập nhật training run.'));
    } finally {
      this.refreshingRun.set(false);
    }
  }

  async cancelSelectedRun(): Promise<void> {
    const run = this.selectedRun();
    if (!run || !isModelTrainingActive(run.status) || this.canceling()) return;
    if (!window.confirm('Hủy GPU training job này? Chi phí đã phát sinh sẽ không được hoàn lại.')) {
      return;
    }
    this.canceling.set(true);
    this.error.set(null);
    try {
      const canceled = await firstValueFrom(this.api.cancelModelTrainingRun(run.id));
      this.upsertRun(canceled);
      this.success.set('Đã gửi yêu cầu hủy và ghi nhận audit cho training run.');
      if (this.pollTimer !== null) window.clearTimeout(this.pollTimer);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể hủy training run.'));
    } finally {
      this.canceling.set(false);
    }
  }

  isActive(run: ModelTrainingRun): boolean {
    return isModelTrainingActive(run.status);
  }

  statusLabel(run: ModelTrainingRun): string {
    return trainingStatusLabel(run.status);
  }

  blockerMessage(code: string): string {
    return modelTrainingBlockerMessage(code);
  }

  stageReached(run: ModelTrainingRun, stage: number): boolean {
    const current =
      {
        QUEUED: 1,
        SUBMITTING: 1,
        SCHEDULING: 2,
        RUNNING: 3,
        COMPLETED: 4,
        FAILED: 3,
        CANCELED: 2
      }[run.status] ?? 0;
    return current >= stage;
  }

  private initializeForm(
    capabilities: ModelTrainingCapabilities,
    registry: DatasetRegistryResponse
  ): void {
    if (this.initializedForm) return;
    const defaults = capabilities.defaults;
    this.epochs = defaults.epochs;
    this.batchSize = defaults.batchSize;
    this.workers = defaults.workers;
    this.snapshotEpoch = defaults.snapshotEpoch;
    const ready = registry.items.find(
      (item) => trainingDatasetReadiness(item, defaults.datasetRepoId).ready
    );
    this.selectedSourceId.set(ready?.sourceId ?? registry.items[0]?.sourceId ?? '');
    this.initializedForm = true;
  }

  private ensureSelections(): void {
    if (!this.datasets().some((item) => item.sourceId === this.selectedSourceId())) {
      this.selectedSourceId.set(this.datasets()[0]?.sourceId ?? '');
    }
    if (!this.runs().some((item) => item.id === this.selectedRunId())) {
      const preferred = this.runs().find((item) => isModelTrainingActive(item.status));
      this.selectedRunId.set(preferred?.id ?? this.runs()[0]?.id ?? '');
    }
  }

  private schedulePoll(runId: string, delay = 3500): void {
    if (this.pollTimer !== null) window.clearTimeout(this.pollTimer);
    this.pollTimer = window.setTimeout(async () => {
      this.pollTimer = null;
      if (this.selectedRunId() !== runId) return;
      await this.refreshSelectedRun();
    }, delay);
  }

  private async loadLogs(run: ModelTrainingRun): Promise<void> {
    if (!run.remoteJobId) {
      this.logLines.set([]);
      this.logsAvailable.set(false);
      return;
    }
    try {
      const logs = await firstValueFrom(this.api.modelTrainingLogs(run.id, 300));
      this.logLines.set(logs.lines);
      this.logsAvailable.set(logs.available);
      this.logError.set(null);
    } catch (error) {
      this.logError.set(apiErrorMessage(error, 'Log từ remote job chưa khả dụng.'));
    }
  }

  private upsertRun(run: ModelTrainingRun): void {
    this.runs.update((items) => {
      const remaining = items.filter((item) => item.id !== run.id);
      return [run, ...remaining].sort((left, right) =>
        right.createdAt.localeCompare(left.createdAt)
      );
    });
  }

  private defaultVersion(): string {
    const now = new Date();
    const date = [now.getFullYear(), now.getMonth() + 1, now.getDate()]
      .map((value, index) => (index === 0 ? String(value) : String(value).padStart(2, '0')))
      .join('');
    return `v${date}.1`;
  }
}
