import { DatePipe, PercentPipe } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  LucideActivity,
  LucideArrowDownRight,
  LucideArrowUpRight,
  LucideCamera,
  LucideCar,
  LucideCircleAlert,
  LucideRadio,
  LucideRefreshCw
} from '@lucide/angular';
import { auditTime, catchError, firstValueFrom, of } from 'rxjs';

import { Alert, Camera, CameraHealth, VehicleEvent } from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { localDayStartIso } from '../../core/utils/event-utils';
import { finalPlate } from '../../core/utils/plate-review-utils';

interface DashboardCamera extends Camera {
  health: CameraHealth | null;
}

@Component({
  selector: 'app-dashboard',
  imports: [
    DatePipe,
    PercentPipe,
    LucideActivity,
    LucideArrowDownRight,
    LucideArrowUpRight,
    LucideCamera,
    LucideCar,
    LucideCircleAlert,
    LucideRadio,
    LucideRefreshCw
  ],
  templateUrl: './dashboard.component.html'
})
export class DashboardComponent implements OnInit {
  private readonly api = inject(ApiClientService);
  private readonly realtime = inject(RealtimeService);
  readonly loading = signal(true);
  readonly error = signal<string | null>(null);
  readonly events = signal<VehicleEvent[]>([]);
  readonly cameras = signal<DashboardCamera[]>([]);
  readonly alerts = signal<Alert[]>([]);
  readonly truncated = signal(false);

  readonly total = computed(() => this.events().length);
  readonly entries = computed(
    () => this.events().filter((event) => event.direction === 'ENTER').length
  );
  readonly exits = computed(
    () => this.events().filter((event) => event.direction === 'EXIT').length
  );
  readonly unknown = computed(
    () => this.events().filter((event) => event.plate === null).length
  );
  readonly onlineCameras = computed(
    () => this.cameras().filter((camera) => camera.health?.status === 'ONLINE').length
  );
  readonly offlineCameras = computed(
    () => this.cameras().filter((camera) => camera.health && camera.health.status !== 'ONLINE').length
  );
  readonly recentEvents = computed(() => this.events().slice(0, 8));
  readonly hourlyActivity = computed(() => this.buildHourlyActivity(this.events()));

  constructor() {
    this.realtime.events$
      .pipe(auditTime(1000), takeUntilDestroyed())
      .subscribe(() => void this.load());
    this.realtime.recoveryRequested$
      .pipe(takeUntilDestroyed())
      .subscribe(() => void this.load());
  }

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    if (this.loading() && this.events().length > 0) return;
    this.loading.set(true);
    this.error.set(null);
    try {
      const [eventPage, cameraPage, alertPage] = await Promise.all([
        firstValueFrom(this.api.events({ limit: 200, from: localDayStartIso() })),
        firstValueFrom(
          this.api.cameras().pipe(catchError(() => of({ items: [] as Camera[] })))
        ),
        firstValueFrom(
          this.api.alerts({ limit: 200, status: 'OPEN' }).pipe(
            catchError(() => of({ items: [] as Alert[], nextCursor: null }))
          )
        )
      ]);
      const cameras = await Promise.all(
        cameraPage.items.map(async (camera) => ({
          ...camera,
          health: await firstValueFrom(
            this.api.cameraHealth(camera.id).pipe(catchError(() => of(null)))
          )
        }))
      );
      this.events.set(eventPage.items);
      this.truncated.set(Boolean(eventPage.nextCursor));
      this.cameras.set(cameras);
      this.alerts.set(alertPage.items);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải dữ liệu tổng quan.'));
    } finally {
      this.loading.set(false);
    }
  }

  plate(event: VehicleEvent): string {
    return finalPlate(event) ?? 'Không đọc được';
  }

  private buildHourlyActivity(events: VehicleEvent[]): Array<{
    label: string;
    entries: number;
    exits: number;
    total: number;
    width: number;
  }> {
    const now = new Date();
    const hours = Array.from({ length: 7 }, (_, index) => {
      const point = new Date(now);
      point.setMinutes(0, 0, 0);
      point.setHours(point.getHours() - (6 - index));
      return point;
    });
    const rows = hours.map((hour) => {
      const end = new Date(hour);
      end.setHours(end.getHours() + 1);
      const matching = events.filter((event) => {
        const occurred = Date.parse(event.occurredAt);
        return occurred >= hour.getTime() && occurred < end.getTime();
      });
      const entries = matching.filter((event) => event.direction === 'ENTER').length;
      const exits = matching.filter((event) => event.direction === 'EXIT').length;
      return {
        label: hour.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }),
        entries,
        exits,
        total: matching.length,
        width: 0
      };
    });
    const maximum = Math.max(1, ...rows.map((row) => row.total));
    return rows.map((row) => ({ ...row, width: Math.max(3, (row.total / maximum) * 100) }));
  }
}
