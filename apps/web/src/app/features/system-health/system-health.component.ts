import { DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  DestroyRef,
  HostListener,
  OnDestroy,
  computed,
  inject,
  signal
} from '@angular/core';
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
import { AsyncDataState } from '../../core/utils/async-data-state';

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
export class SystemHealthComponent implements OnDestroy {
  private readonly api = inject(ApiClientService);
  private readonly destroyRef = inject(DestroyRef);
  readonly system = signal<SystemHealth | null>(null);
  readonly realtime = signal<RealtimeHealth | null>(null);
  readonly liveMonitor = signal<LiveMonitorHealth | null>(null);
  readonly cameras = signal<CameraHealthRow[]>([]);
  readonly loading = signal(false);
  readonly loadState = new AsyncDataState();
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
  private requestGeneration = 0;
  private requestInFlight = false;
  private refreshQueued = false;
  private destroyed = false;

  constructor() {
    timer(0, 10_000)
      .pipe(takeUntilDestroyed())
      .subscribe(() => {
        if (!this.pageIsHidden()) void this.load();
      });
    this.destroyRef.onDestroy(() => this.markDestroyed());
  }

  ngOnDestroy(): void {
    this.markDestroyed();
  }

  @HostListener('document:visibilitychange')
  handleVisibilityChange(): void {
    if (!this.pageIsHidden()) void this.load();
  }

  async load(): Promise<void> {
    if (this.destroyed) return;
    if (this.requestInFlight) {
      this.refreshQueued = true;
      this.requestGeneration += 1;
      return;
    }
    this.requestInFlight = true;
    const generation = ++this.requestGeneration;
    this.loading.set(true);
    this.loadState.begin();
    try {
      const [system, realtime, liveMonitor, cameraSnapshot] = await Promise.all([
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
        firstValueFrom(this.api.cameraHealthSnapshot())
      ]);
      if (generation !== this.requestGeneration) return;
      this.system.set(system);
      this.realtime.set(realtime);
      this.liveMonitor.set(liveMonitor);
      this.cameras.set(cameraSnapshot.items);
      this.updatedAt.set(new Date());
      this.loadState.succeed();
    } catch (error) {
      if (generation === this.requestGeneration) {
        this.loadState.fail(apiErrorMessage(error, 'Không thể tải health của hệ thống.'));
      }
    } finally {
      this.requestInFlight = false;
      if (this.destroyed) return;
      this.loading.set(false);
      if (this.refreshQueued) {
        this.refreshQueued = false;
        void this.load();
      }
    }
  }

  private pageIsHidden(): boolean {
    return typeof document !== 'undefined' && document.hidden;
  }

  private markDestroyed(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.requestGeneration += 1;
    this.refreshQueued = false;
  }
}
