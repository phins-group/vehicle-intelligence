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
  LucideRefreshCw,
} from '@lucide/angular';
import { bufferTime, filter, firstValueFrom, interval } from 'rxjs';

import {
  Alert,
  Camera,
  CameraHealth,
  VehicleEvent,
} from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import {
  localDayStartIso,
  mergeVehicleEvents,
} from '../../core/utils/event-utils';
import { finalPlate } from '../../core/utils/plate-review-utils';

interface DashboardCamera extends Camera {
  health: CameraHealth | null;
}

const DASHBOARD_EVENT_LIMIT = 200;
const REALTIME_BATCH_INTERVAL_MS = 250;
const ALERT_REFRESH_INTERVAL_MS = 30_000;
const CAMERA_REFRESH_INTERVAL_MS = 60_000;

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
    LucideRefreshCw,
  ],
  templateUrl: './dashboard.component.html',
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
  readonly alertLoadState = new AsyncDataState();
  readonly cameraLoadState = new AsyncDataState();
  private loadingDashboard = false;
  private loadingAlerts = false;
  private loadingCameras = false;

  readonly total = computed(() => this.events().length);
  readonly entries = computed(
    () => this.events().filter((event) => event.direction === 'ENTER').length,
  );
  readonly exits = computed(
    () => this.events().filter((event) => event.direction === 'EXIT').length,
  );
  readonly unknown = computed(
    () => this.events().filter((event) => event.plate === null).length,
  );
  readonly onlineCameras = computed(
    () =>
      this.cameras().filter((camera) => camera.health?.status === 'ONLINE')
        .length,
  );
  readonly offlineCameras = computed(
    () =>
      this.cameras().filter(
        (camera) => camera.health && camera.health.status !== 'ONLINE',
      ).length,
  );
  readonly recentEvents = computed(() => this.events().slice(0, 8));
  readonly hourlyActivity = computed(() =>
    this.buildHourlyActivity(this.events()),
  );

  constructor() {
    this.realtime.events$
      .pipe(
        bufferTime(REALTIME_BATCH_INTERVAL_MS),
        filter((events) => events.length > 0),
        takeUntilDestroyed(),
      )
      .subscribe((events) => this.applyRealtimeEvents(events));
    this.realtime.recoveryRequested$
      .pipe(takeUntilDestroyed())
      .subscribe(() => void this.load());
    interval(ALERT_REFRESH_INTERVAL_MS)
      .pipe(takeUntilDestroyed())
      .subscribe(() => {
        this.pruneExpiredEvents();
        if (this.pageIsVisible()) void this.refreshAlerts();
      });
    interval(CAMERA_REFRESH_INTERVAL_MS)
      .pipe(takeUntilDestroyed())
      .subscribe(() => {
        if (this.pageIsVisible()) void this.refreshCameras();
      });
  }

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    if (this.loadingDashboard) return;
    this.loadingDashboard = true;
    this.loading.set(true);
    this.error.set(null);
    try {
      const [eventPage] = await Promise.all([
        firstValueFrom(
          this.api.events({
            limit: DASHBOARD_EVENT_LIMIT,
            from: localDayStartIso(),
          }),
        ),
        this.refreshCameras(),
        this.refreshAlerts(),
      ]);
      const dayStart = Date.parse(localDayStartIso());
      this.events.update((current) => {
        const currentDayEvents = current.filter(
          (event) => Date.parse(event.occurredAt) >= dayStart,
        );
        const distinctIds = new Set([
          ...eventPage.items.map((event) => event._id),
          ...currentDayEvents.map((event) => event._id),
        ]);
        this.truncated.set(
          Boolean(eventPage.nextCursor) ||
            distinctIds.size > DASHBOARD_EVENT_LIMIT,
        );
        return mergeVehicleEvents(
          currentDayEvents,
          eventPage.items,
          DASHBOARD_EVENT_LIMIT,
        );
      });
    } catch (error) {
      this.error.set(
        apiErrorMessage(error, 'Không thể tải dữ liệu tổng quan.'),
      );
    } finally {
      this.loadingDashboard = false;
      this.loading.set(false);
    }
  }

  plate(event: VehicleEvent): string {
    return finalPlate(event) ?? 'Không đọc được';
  }

  private applyRealtimeEvents(events: VehicleEvent[]): void {
    const dayStart = Date.parse(localDayStartIso());
    const eligible = events.filter((event) => {
      const occurredAt = Date.parse(event.occurredAt);
      return Number.isFinite(occurredAt) && occurredAt >= dayStart;
    });
    this.events.update((current) => {
      const currentDayEvents = current.filter(
        (event) => Date.parse(event.occurredAt) >= dayStart,
      );
      if (currentDayEvents.length !== current.length) this.truncated.set(false);
      if (!eligible.length) return currentDayEvents;

      const knownIds = new Set(currentDayEvents.map((event) => event._id));
      const newIds = new Set(
        eligible
          .filter((event) => !knownIds.has(event._id))
          .map((event) => event._id),
      );
      if (currentDayEvents.length + newIds.size > DASHBOARD_EVENT_LIMIT) {
        this.truncated.set(true);
      }
      return mergeVehicleEvents(
        currentDayEvents,
        eligible,
        DASHBOARD_EVENT_LIMIT,
      );
    });
  }

  private async refreshAlerts(): Promise<void> {
    if (this.loadingAlerts) return;
    this.loadingAlerts = true;
    this.alertLoadState.begin();
    try {
      const page = await firstValueFrom(
        this.api.alerts({ limit: 200, status: 'OPEN' }),
      );
      this.alerts.set(page.items);
      this.alertLoadState.succeed();
    } catch (error) {
      this.alertLoadState.fail(
        apiErrorMessage(error, 'Không thể làm mới cảnh báo đang mở.'),
      );
    } finally {
      this.loadingAlerts = false;
    }
  }

  private async refreshCameras(): Promise<void> {
    if (this.loadingCameras) return;
    this.loadingCameras = true;
    this.cameraLoadState.begin();
    try {
      const snapshot = await firstValueFrom(this.api.cameraHealthSnapshot());
      this.cameras.set(
        snapshot.items.map(({ camera, health }) => ({ ...camera, health })),
      );
      this.cameraLoadState.succeed();
    } catch (error) {
      this.cameraLoadState.fail(
        apiErrorMessage(error, 'Không thể làm mới trạng thái camera.'),
      );
    } finally {
      this.loadingCameras = false;
    }
  }

  private pruneExpiredEvents(): void {
    const dayStart = Date.parse(localDayStartIso());
    this.events.update((current) => {
      const currentDayEvents = current.filter(
        (event) => Date.parse(event.occurredAt) >= dayStart,
      );
      if (currentDayEvents.length === current.length) return current;
      this.truncated.set(false);
      return currentDayEvents;
    });
  }

  private pageIsVisible(): boolean {
    return typeof document === 'undefined' || !document.hidden;
  }

  private buildHourlyActivity(events: VehicleEvent[]): {
    label: string;
    entries: number;
    exits: number;
    total: number;
    width: number;
  }[] {
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
      const entries = matching.filter(
        (event) => event.direction === 'ENTER',
      ).length;
      const exits = matching.filter(
        (event) => event.direction === 'EXIT',
      ).length;
      return {
        label: hour.toLocaleTimeString('vi-VN', {
          hour: '2-digit',
          minute: '2-digit',
        }),
        entries,
        exits,
        total: matching.length,
        width: 0,
      };
    });
    const maximum = Math.max(1, ...rows.map((row) => row.total));
    return rows.map((row) => ({
      ...row,
      width: Math.max(3, (row.total / maximum) * 100),
    }));
  }
}
