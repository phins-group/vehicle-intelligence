import { DatePipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import {
  LucideBell,
  LucideCheck,
  LucideCircleCheck,
  LucideRefreshCw,
  LucideSearch,
  LucideTriangleAlert
} from '@lucide/angular';
import { auditTime, catchError, firstValueFrom, of } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import { Alert, AlertStatus, Camera } from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';

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
    LucideTriangleAlert
  ],
  templateUrl: './alerts.component.html'
})
export class AlertsComponent implements OnInit {
  readonly auth = inject(AuthService);
  private readonly api = inject(ApiClientService);
  private readonly realtime = inject(RealtimeService);
  readonly alerts = signal<Alert[]>([]);
  readonly cameras = signal<Camera[]>([]);
  readonly nextCursor = signal<string | null>(null);
  readonly loading = signal(true);
  readonly loadingMore = signal(false);
  readonly error = signal<string | null>(null);
  readonly notice = signal<string | null>(null);
  readonly busyIds = signal<Set<string>>(new Set());
  status: AlertStatus | '' = '';
  plate = '';
  cameraId = '';

  constructor() {
    this.realtime.events$
      .pipe(auditTime(1500), takeUntilDestroyed())
      .subscribe(() => void this.load(true));
    this.realtime.recoveryRequested$
      .pipe(takeUntilDestroyed())
      .subscribe(() => void this.load(true));
  }

  ngOnInit(): void {
    void this.loadCameras();
    void this.load(true);
  }

  async load(reset: boolean): Promise<void> {
    if (reset) this.loading.set(true);
    else this.loadingMore.set(true);
    this.error.set(null);
    try {
      const page = await firstValueFrom(
        this.api.alerts({
          limit: 50,
          cursor: reset ? null : this.nextCursor(),
          status: this.status,
          plate: this.plate.trim(),
          cameraId: this.cameraId
        })
      );
      this.alerts.set(reset ? page.items : [...this.alerts(), ...page.items]);
      this.nextCursor.set(page.nextCursor);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải cảnh báo.'));
    } finally {
      this.loading.set(false);
      this.loadingMore.set(false);
    }
  }

  clearFilters(): void {
    this.status = '';
    this.plate = '';
    this.cameraId = '';
    void this.load(true);
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

  private async transition(alert: Alert, action: 'acknowledge' | 'resolve'): Promise<void> {
    if (this.isBusy(alert.id)) return;
    this.setBusy(alert.id, true);
    this.error.set(null);
    try {
      const updated = await firstValueFrom(
        action === 'acknowledge'
          ? this.api.acknowledgeAlert(alert.id)
          : this.api.resolveAlert(alert.id)
      );
      this.alerts.update((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      this.notice.set(
        action === 'acknowledge' ? 'Đã tiếp nhận cảnh báo.' : 'Đã đánh dấu cảnh báo là resolved.'
      );
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể cập nhật cảnh báo.'));
    } finally {
      this.setBusy(alert.id, false);
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
      this.api.cameras().pipe(catchError(() => of({ items: [] as Camera[] })))
    );
    this.cameras.set(page.items);
  }
}
