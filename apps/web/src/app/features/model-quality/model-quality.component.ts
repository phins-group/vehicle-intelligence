import { DatePipe, PercentPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import {
  LucideActivity,
  LucideCircleAlert,
  LucideGauge,
  LucideRefreshCw,
  LucideScanText
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { ModelQualityReport } from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import { qualityBarWidth } from '../../core/utils/model-quality-utils';

@Component({
  selector: 'app-model-quality',
  imports: [
    DatePipe,
    PercentPipe,
    LucideActivity,
    LucideCircleAlert,
    LucideGauge,
    LucideRefreshCw,
    LucideScanText
  ],
  templateUrl: './model-quality.component.html'
})
export class ModelQualityComponent {
  private readonly api = inject(ApiClientService);
  readonly report = signal<ModelQualityReport | null>(null);
  readonly loading = signal(false);
  readonly loadState = new AsyncDataState();
  readonly windowDays = signal(30);
  readonly dailyMaximum = computed(() =>
    Math.max(0, ...(this.report()?.daily.map((point) => point.metrics.eventCount) ?? []))
  );

  constructor() {
    void this.load();
  }

  async load(): Promise<void> {
    if (this.loading()) return;
    this.loading.set(true);
    this.loadState.begin();
    const to = new Date();
    const from = new Date(to.getTime() - this.windowDays() * 86_400_000);
    try {
      this.report.set(
        await firstValueFrom(this.api.modelQuality(from.toISOString(), to.toISOString()))
      );
      this.loadState.succeed();
    } catch (error) {
      this.loadState.fail(apiErrorMessage(error, 'Không thể tải báo cáo chất lượng model.'));
    } finally {
      this.loading.set(false);
    }
  }

  setWindow(days: number): void {
    this.windowDays.set(days);
    void this.load();
  }

  barWidth(value: number): number {
    return qualityBarWidth(value, this.dailyMaximum());
  }
}
