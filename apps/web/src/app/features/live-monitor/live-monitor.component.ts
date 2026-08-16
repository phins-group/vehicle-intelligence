import { DatePipe, PercentPipe } from '@angular/common';
import {
  Component,
  OnDestroy,
  OnInit,
  computed,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import {
  LucideCctv,
  LucideCircleAlert,
  LucideMaximize2,
  LucideRefreshCw,
  LucideScanLine,
  LucideSlidersHorizontal,
  LucideWifi,
} from '@lucide/angular';
import {
  EMPTY,
  Subscription,
  finalize,
  firstValueFrom,
  map,
  switchMap,
} from 'rxjs';

import {
  Camera,
  LiveMonitorFrame,
  LiveMonitorState,
  LiveVehicleOverlay,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import {
  DEFAULT_OVERLAY_VISIBILITY,
  OverlayVisibility,
  overlayLabelWidth,
  overlayPoints,
  shouldLoadLiveFrame,
  vehicleOverlayLabel,
} from '../../core/utils/live-monitor-utils';

type OverlayKey = keyof OverlayVisibility;

const LIVE_POLL_INTERVAL_MS = 750;
const IDLE_POLL_INTERVAL_MS = 2_500;
const OFFLINE_POLL_INTERVAL_MS = 5_000;
const DETECTION_SUMMARY_LIMIT = 12;

@Component({
  selector: 'app-live-monitor',
  imports: [
    DatePipe,
    PercentPipe,
    FormsModule,
    LucideCctv,
    LucideCircleAlert,
    LucideMaximize2,
    LucideRefreshCw,
    LucideScanLine,
    LucideSlidersHorizontal,
    LucideWifi,
  ],
  templateUrl: './live-monitor.component.html',
})
export class LiveMonitorComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiClientService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private pollTimer: number | null = null;
  private activePoll: Subscription | null = null;
  private generation = 0;
  private polling = false;
  private destroyed = false;

  readonly cameras = signal<Camera[]>([]);
  readonly selectedCameraId = signal('');
  readonly selectedCamera = computed(
    () =>
      this.cameras().find((camera) => camera.id === this.selectedCameraId()) ??
      null,
  );
  readonly state = signal<LiveMonitorState | null>(null);
  readonly renderedFrame = signal<LiveMonitorFrame | null>(null);
  readonly imageUrl = signal<string | null>(null);
  readonly visibility = signal<OverlayVisibility>({
    ...DEFAULT_OVERLAY_VISIBILITY,
  });
  readonly loading = signal(true);
  readonly loadState = new AsyncDataState();
  readonly refreshing = signal(false);
  readonly paused = signal(false);
  readonly error = signal<string | null>(null);

  ngOnInit(): void {
    if (
      typeof window !== 'undefined' &&
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    ) {
      this.paused.set(true);
    }
    void this.loadCameras();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.generation += 1;
    this.stopPolling();
    this.revokeImage();
  }

  async loadCameras(): Promise<void> {
    this.loading.set(true);
    this.loadState.begin();
    try {
      const page = await firstValueFrom(this.api.cameras(true));
      if (this.destroyed) return;
      this.cameras.set(page.items);
      this.loadState.succeed();
      const requested =
        this.route.snapshot.queryParamMap.get('camera')?.trim() ?? '';
      const selected = page.items.some((camera) => camera.id === requested)
        ? requested
        : (page.items[0]?.id ?? '');
      this.activateCamera(selected);
      if (selected && selected !== requested) {
        void this.router.navigate([], {
          relativeTo: this.route,
          queryParams: { camera: selected },
          queryParamsHandling: 'merge',
          replaceUrl: true,
        });
      }
    } catch (error) {
      if (!this.destroyed) {
        this.loadState.fail(
          apiErrorMessage(
            error,
            'Không thể tải danh sách camera cho live monitor.',
          ),
        );
      }
    } finally {
      if (!this.destroyed) this.loading.set(false);
    }
  }

  selectCamera(cameraId: string): void {
    this.activateCamera(cameraId);
    void this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { camera: cameraId || null },
      queryParamsHandling: 'merge',
    });
  }

  refresh(): void {
    this.clearPollTimer();
    this.poll(this.generation, true);
  }

  togglePause(): void {
    if (this.paused()) {
      this.paused.set(false);
      this.poll(this.generation, true);
      return;
    }
    this.paused.set(true);
    this.stopPolling();
  }

  toggle(key: OverlayKey): void {
    this.visibility.update((current) => ({ ...current, [key]: !current[key] }));
  }

  points(points: [number, number][] | null): string {
    return overlayPoints(points);
  }

  label(vehicle: LiveVehicleOverlay): string {
    return vehicleOverlayLabel(vehicle, this.visibility());
  }

  labelWidth(vehicle: LiveVehicleOverlay): number {
    return overlayLabelWidth(
      this.label(vehicle),
      this.renderedFrame()?.sourceWidth ?? 1920,
    );
  }

  labelY(vehicle: LiveVehicleOverlay): number {
    return Math.max(16, vehicle.bbox[1] - 8);
  }

  averageConfidence(frame: LiveMonitorFrame): number | null {
    if (!frame.vehicles.length) return null;
    return (
      frame.vehicles.reduce((total, vehicle) => total + vehicle.confidence, 0) /
      frame.vehicles.length
    );
  }

  visibleDetections(frame: LiveMonitorFrame): LiveVehicleOverlay[] {
    return frame.vehicles.slice(0, DETECTION_SUMMARY_LIMIT);
  }

  hiddenDetectionCount(frame: LiveMonitorFrame): number {
    return Math.max(0, frame.vehicles.length - DETECTION_SUMMARY_LIMIT);
  }

  private activateCamera(cameraId: string): void {
    if (this.destroyed) return;
    this.generation += 1;
    this.stopPolling();
    this.selectedCameraId.set(cameraId);
    this.state.set(null);
    this.renderedFrame.set(null);
    this.revokeImage();
    this.error.set(null);
    if (!cameraId) return;
    const generation = this.generation;
    this.poll(generation, true);
  }

  private poll(generation: number, manual: boolean): void {
    if (
      this.destroyed ||
      this.polling ||
      generation !== this.generation ||
      !this.selectedCameraId()
    ) {
      return;
    }
    this.polling = true;
    if (manual) this.refreshing.set(true);
    const cameraId = this.selectedCameraId();
    const request = this.api.liveMonitorState(cameraId).pipe(
      switchMap((state) => {
        if (this.destroyed || generation !== this.generation) return EMPTY;
        this.state.set(state);
        const latest = state.latest;
        if (
          !shouldLoadLiveFrame(this.renderedFrame()?.sequence ?? null, latest)
        ) {
          this.error.set(null);
          return EMPTY;
        }
        return this.api.liveMonitorFrame(cameraId, latest!.sequence).pipe(
          map((response) => {
            const responseSequence = Number(
              response.headers.get('X-Live-Sequence'),
            );
            if (responseSequence !== latest!.sequence || !response.body?.size) {
              throw new Error('Live frame sequence không khớp metadata.');
            }
            return { frame: latest!, url: URL.createObjectURL(response.body) };
          }),
        );
      }),
      finalize(() => {
        this.activePoll = null;
        this.polling = false;
        if (generation === this.generation && !this.destroyed) {
          this.refreshing.set(false);
          this.scheduleNextPoll(generation);
        }
      }),
    );
    const subscription = request.subscribe({
      next: ({ frame, url }) => {
        if (this.destroyed || generation !== this.generation) {
          URL.revokeObjectURL(url);
          return;
        }
        this.revokeImage();
        this.imageUrl.set(url);
        this.renderedFrame.set(frame);
        this.error.set(null);
      },
      error: (error: unknown) => {
        if (generation === this.generation && !this.destroyed) {
          this.error.set(
            apiErrorMessage(error, 'Không thể nhận live preview mới nhất.'),
          );
        }
      },
    });
    if (!subscription.closed) this.activePoll = subscription;
  }

  private stopPolling(): void {
    this.clearPollTimer();
    const activePoll = this.activePoll;
    this.activePoll = null;
    activePoll?.unsubscribe();
    this.polling = false;
    this.refreshing.set(false);
  }

  private scheduleNextPoll(generation: number): void {
    if (
      this.destroyed ||
      this.paused() ||
      generation !== this.generation ||
      !this.selectedCameraId()
    ) {
      return;
    }
    this.clearPollTimer();
    this.pollTimer = window.setTimeout(() => {
      this.pollTimer = null;
      if (typeof document !== 'undefined' && document.hidden) {
        this.scheduleNextPoll(generation);
        return;
      }
      this.poll(generation, false);
    }, this.nextPollInterval());
  }

  private nextPollInterval(): number {
    if (this.error()) return OFFLINE_POLL_INTERVAL_MS;
    switch (this.state()?.status) {
      case 'LIVE':
        return LIVE_POLL_INTERVAL_MS;
      case 'WAITING':
      case 'STALE':
        return IDLE_POLL_INTERVAL_MS;
      default:
        return OFFLINE_POLL_INTERVAL_MS;
    }
  }

  private clearPollTimer(): void {
    if (this.pollTimer !== null) {
      window.clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private revokeImage(): void {
    const current = this.imageUrl();
    if (current) URL.revokeObjectURL(current);
    this.imageUrl.set(null);
  }
}
