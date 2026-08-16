import { DatePipe } from '@angular/common';
import {
  Component,
  DestroyRef,
  OnDestroy,
  OnInit,
  inject,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import {
  LucideBell,
  LucideCheck,
  LucideCircleCheck,
  LucideRefreshCw,
  LucideSearch,
  LucideTriangleAlert,
} from '@lucide/angular';
import { catchError, firstValueFrom, of, timer } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import { Alert, AlertStatus, Camera } from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';

const REALTIME_REFRESH_INTERVAL_MS = 15_000;

interface AlertListFilters {
  limit: number;
  cursor: string | null;
  status: AlertStatus | '';
  plate: string;
  cameraId: string;
}

@Component({
  selector: 'app-alerts',
  imports: [
    DatePipe,
    FormsModule,
    LucideBell,
    LucideCheck,
    LucideCircleCheck,
    LucideRefreshCw,
    LucideSearch,
    LucideTriangleAlert,
  ],
  templateUrl: './alerts.component.html',
})
export class AlertsComponent implements OnInit, OnDestroy {
  readonly auth = inject(AuthService);
  private readonly api = inject(ApiClientService);
  private readonly realtime = inject(RealtimeService);
  private readonly destroyRef = inject(DestroyRef);
  readonly alerts = signal<Alert[]>([]);
  readonly cameras = signal<Camera[]>([]);
  readonly nextCursor = signal<string | null>(null);
  readonly loading = signal(true);
  readonly loadingMore = signal(false);
  readonly loadState = new AsyncDataState();
  readonly actionError = signal<string | null>(null);
  readonly notice = signal<string | null>(null);
  readonly busyIds = signal<Set<string>>(new Set());
  status: AlertStatus | '' = '';
  plate = '';
  cameraId = '';
  private realtimeRefreshPending = false;
  private appliedFilters: AlertListFilters = this.emptyFilters();
  private requestedFilters: AlertListFilters = this.emptyFilters();
  private requestGeneration = 0;
  private requestInFlight = false;
  private queuedResetFilters: AlertListFilters | null = null;
  private destroyed = false;

  constructor() {
    this.realtime.events$.pipe(takeUntilDestroyed()).subscribe(() => {
      this.realtimeRefreshPending = true;
    });
    this.realtime.recoveryRequested$
      .pipe(takeUntilDestroyed())
      .subscribe(() => {
        this.realtimeRefreshPending = true;
        void this.load(true);
      });
    timer(REALTIME_REFRESH_INTERVAL_MS, REALTIME_REFRESH_INTERVAL_MS)
      .pipe(takeUntilDestroyed())
      .subscribe(() => {
        if (this.realtimeRefreshPending && !this.pageIsHidden())
          void this.load(true);
      });
    this.destroyRef.onDestroy(() => this.markDestroyed());
  }

  ngOnInit(): void {
    void this.loadCameras();
    void this.load(true);
  }

  ngOnDestroy(): void {
    this.markDestroyed();
  }

  async load(reset: boolean): Promise<void> {
    const filters = reset
      ? { ...this.requestedFilters, cursor: null }
      : { ...this.appliedFilters, cursor: this.nextCursor() };
    await this.loadWithFilters(reset, filters);
  }

  async applyFilters(): Promise<void> {
    this.requestedFilters = this.formFilters(null);
    await this.loadWithFilters(true, this.requestedFilters);
  }

  private async loadWithFilters(
    reset: boolean,
    filters: AlertListFilters,
  ): Promise<void> {
    if (this.destroyed) return;
    if (this.requestInFlight) {
      if (reset) {
        this.queuedResetFilters = { ...filters, cursor: null };
        this.requestGeneration += 1;
      }
      return;
    }
    this.requestInFlight = true;
    const generation = ++this.requestGeneration;
    if (reset) this.realtimeRefreshPending = false;
    if (reset) this.loading.set(true);
    else this.loadingMore.set(true);
    this.actionError.set(null);
    this.loadState.begin();
    try {
      const page = await firstValueFrom(this.api.alerts(filters));
      if (generation !== this.requestGeneration) return;
      if (reset) this.appliedFilters = { ...filters, cursor: null };
      this.alerts.set(reset ? page.items : [...this.alerts(), ...page.items]);
      this.nextCursor.set(page.nextCursor);
      this.loadState.succeed();
    } catch (error) {
      if (generation === this.requestGeneration) {
        this.loadState.fail(apiErrorMessage(error, 'Không thể tải cảnh báo.'));
        if (reset) this.realtimeRefreshPending = true;
      }
    } finally {
      this.requestInFlight = false;
      if (this.destroyed) return;
      this.loading.set(false);
      this.loadingMore.set(false);
      if (this.queuedResetFilters !== null) {
        const queuedFilters = this.queuedResetFilters;
        this.queuedResetFilters = null;
        void this.loadWithFilters(true, queuedFilters);
      }
    }
  }

  clearFilters(): void {
    this.status = '';
    this.plate = '';
    this.cameraId = '';
    void this.applyFilters();
  }

  async acknowledge(alert: Alert): Promise<void> {
    await this.transition(alert, 'acknowledge');
  }

  async resolve(alert: Alert): Promise<void> {
    await this.transition(alert, 'resolve');
  }

  isBusy(alertId: string): boolean {
    return this.busyIds().has(alertId);
  }

  private async transition(
    alert: Alert,
    action: 'acknowledge' | 'resolve',
  ): Promise<void> {
    if (this.destroyed || this.isBusy(alert.id)) return;
    this.setBusy(alert.id, true);
    this.actionError.set(null);
    try {
      const updated = await firstValueFrom(
        action === 'acknowledge'
          ? this.api.acknowledgeAlert(alert.id)
          : this.api.resolveAlert(alert.id),
      );
      if (this.destroyed) return;
      this.alerts.update((items) =>
        items.map((item) => (item.id === updated.id ? updated : item)),
      );
      this.notice.set(
        action === 'acknowledge'
          ? 'Đã tiếp nhận cảnh báo.'
          : 'Đã đánh dấu cảnh báo là resolved.',
      );
    } catch (error) {
      if (!this.destroyed) {
        this.actionError.set(
          apiErrorMessage(error, 'Không thể cập nhật cảnh báo.'),
        );
      }
    } finally {
      if (!this.destroyed) this.setBusy(alert.id, false);
    }
  }

  private setBusy(alertId: string, busy: boolean): void {
    const current = new Set(this.busyIds());
    if (busy) current.add(alertId);
    else current.delete(alertId);
    this.busyIds.set(current);
  }

  private async loadCameras(): Promise<void> {
    const page = await firstValueFrom(
      this.api.cameras().pipe(catchError(() => of({ items: [] as Camera[] }))),
    );
    if (!this.destroyed) this.cameras.set(page.items);
  }

  private pageIsHidden(): boolean {
    return typeof document !== 'undefined' && document.hidden;
  }

  private formFilters(cursor: string | null): AlertListFilters {
    return {
      limit: 50,
      cursor,
      status: this.status,
      plate: this.plate.trim(),
      cameraId: this.cameraId,
    };
  }

  private emptyFilters(): AlertListFilters {
    return { limit: 50, cursor: null, status: '', plate: '', cameraId: '' };
  }

  private markDestroyed(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.requestGeneration += 1;
    this.queuedResetFilters = null;
  }
}
