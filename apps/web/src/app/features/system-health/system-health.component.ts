import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import {
  LucideActivity,
  LucideCamera,
  LucideCircleAlert,
  LucideCircleCheck,
  LucideGauge,
  LucideRadio,
  LucideRefreshCw,
  LucideServer
} from '@lucide/angular';
import { catchError, firstValueFrom, of, timer } from 'rxjs';

import {
  Camera,
  CameraHealth,
  LiveMonitorHealth,
  RealtimeHealth,
  SystemHealth
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';

interface CameraHealthRow {
  camera: Camera;
  health: CameraHealth | null;
}

@Component({
  selector: 'app-system-health',
  imports: [
    DatePipe,
    DecimalPipe,
    LucideActivity,
    LucideCamera,
    LucideCircleAlert,
    LucideCircleCheck,
    LucideGauge,
    LucideRadio,
    LucideRefreshCw,
    LucideServer
  ],
  templateUrl: './system-health.component.html'
})
export class SystemHealthComponent {
  private readonly api = inject(ApiClientService);
  readonly system = signal<SystemHealth | null>(null);
  readonly realtime = signal<RealtimeHealth | null>(null);
  readonly liveMonitor = signal<LiveMonitorHealth | null>(null);
  readonly cameras = signal<CameraHealthRow[]>([]);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly updatedAt = signal<Date | null>(null);
  readonly onlineCameras = computed(
    () => this.cameras().filter((item) => item.health?.status === 'ONLINE').length
  );
  readonly totalDropped = computed(() =>
    this.cameras().reduce((sum, item) => sum + (item.health?.droppedFrames ?? 0), 0)
  );
  readonly totalReconnects = computed(() =>
    this.cameras().reduce((sum, item) => sum + (item.health?.reconnectCount ?? 0), 0)
  );

  constructor() {
    timer(0, 10_000)
      .pipe(takeUntilDestroyed())
      .subscribe(() => void this.load());
  }

  async load(): Promise<void> {
    if (this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    try {
      const [system, realtime, liveMonitor, cameraPage] = await Promise.all([
        firstValueFrom(this.api.systemHealth()),
        firstValueFrom(
          this.api.realtimeHealth().pipe(
            catchError(() => of({ status: 'UNAVAILABLE', subscribers: 0 } as RealtimeHealth))
          )
        ),
        firstValueFrom(
          this.api.liveMonitorHealth().pipe(
            catchError(() =>
              of({ status: 'UNAVAILABLE', camerasBuffered: 0 } as LiveMonitorHealth)
            )
          )
        ),
        firstValueFrom(
          this.api.cameras().pipe(catchError(() => of({ items: [] as Camera[] })))
        )
      ]);
      const cameras = await Promise.all(
        cameraPage.items.map(async (camera) => ({
          camera,
          health: await firstValueFrom(
            this.api.cameraHealth(camera.id).pipe(catchError(() => of(null)))
          )
        }))
      );
      this.system.set(system);
      this.realtime.set(realtime);
      this.liveMonitor.set(liveMonitor);
      this.cameras.set(cameras);
      this.updatedAt.set(new Date());
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải health của hệ thống.'));
    } finally {
      this.loading.set(false);
    }
  }
}
