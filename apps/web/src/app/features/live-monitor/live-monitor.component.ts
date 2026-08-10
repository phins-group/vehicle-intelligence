import { DatePipe, PercentPipe } from '@angular/common';
import { Component, OnDestroy, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';
import {
  LucideCctv,
  LucideCircleAlert,
  LucideMaximize2,
  LucideRefreshCw,
  LucideScanLine,
  LucideSlidersHorizontal,
  LucideWifi
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import {
  Camera,
  LiveMonitorFrame,
  LiveMonitorState,
  LiveVehicleOverlay
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import {
  DEFAULT_OVERLAY_VISIBILITY,
  OverlayVisibility,
  overlayLabelWidth,
  overlayPoints,
  shouldLoadLiveFrame,
  vehicleOverlayLabel
} from '../../core/utils/live-monitor-utils';

type OverlayKey = keyof OverlayVisibility;

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
    LucideWifi
  ],
  templateUrl: './live-monitor.component.html'
})
export class LiveMonitorComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiClientService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private pollTimer: number | null = null;
  private generation = 0;
  private polling = false;

  readonly cameras = signal<Camera[]>([]);
  readonly selectedCameraId = signal('');
  readonly selectedCamera = computed(
    () => this.cameras().find((camera) => camera.id === this.selectedCameraId()) ?? null
  );
  readonly state = signal<LiveMonitorState | null>(null);
  readonly renderedFrame = signal<LiveMonitorFrame | null>(null);
  readonly imageUrl = signal<string | null>(null);
  readonly visibility = signal<OverlayVisibility>({ ...DEFAULT_OVERLAY_VISIBILITY });
  readonly loading = signal(true);
  readonly refreshing = signal(false);
  readonly error = signal<string | null>(null);

  ngOnInit(): void {
    void this.loadCameras();
  }

  ngOnDestroy(): void {
    this.stopPolling();
    this.revokeImage();
  }

  async loadCameras(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const page = await firstValueFrom(this.api.cameras(true));
      this.cameras.set(page.items);
      const requested = this.route.snapshot.queryParamMap.get('camera')?.trim() ?? '';
      const selected = page.items.some((camera) => camera.id === requested)
        ? requested
        : page.items[0]?.id ?? '';
      this.activateCamera(selected);
      if (selected && selected !== requested) {
        void this.router.navigate([], {
          relativeTo: this.route,
          queryParams: { camera: selected },
          queryParamsHandling: 'merge',
          replaceUrl: true
        });
      }
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải danh sách camera cho live monitor.'));
    } finally {
      this.loading.set(false);
    }
  }

  selectCamera(cameraId: string): void {
    this.activateCamera(cameraId);
    void this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { camera: cameraId || null },
      queryParamsHandling: 'merge'
    });
  }

  refresh(): void {
    void this.poll(this.generation, true);
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
    return overlayLabelWidth(this.label(vehicle), this.renderedFrame()?.sourceWidth ?? 1920);
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

  private activateCamera(cameraId: string): void {
    this.stopPolling();
    this.generation += 1;
    this.selectedCameraId.set(cameraId);
    this.state.set(null);
    this.renderedFrame.set(null);
    this.revokeImage();
    this.error.set(null);
    if (!cameraId) return;
    const generation = this.generation;
    void this.poll(generation, true);
    this.pollTimer = window.setInterval(() => {
      if (!document.hidden) void this.poll(generation, false);
    }, 750);
  }

  private async poll(generation: number, manual: boolean): Promise<void> {
    if (this.polling || generation !== this.generation || !this.selectedCameraId()) return;
    this.polling = true;
    if (manual) this.refreshing.set(true);
    try {
      const cameraId = this.selectedCameraId();
      const state = await firstValueFrom(this.api.liveMonitorState(cameraId));
      if (generation !== this.generation) return;
      this.state.set(state);
      const latest = state.latest;
      if (!shouldLoadLiveFrame(this.renderedFrame()?.sequence ?? null, latest)) {
        this.error.set(null);
        return;
      }
      const response = await firstValueFrom(
        this.api.liveMonitorFrame(cameraId, latest!.sequence)
      );
      if (generation !== this.generation) return;
      const responseSequence = Number(response.headers.get('X-Live-Sequence'));
      if (responseSequence !== latest!.sequence || !response.body?.size) {
        throw new Error('Live frame sequence không khớp metadata.');
      }
      const url = URL.createObjectURL(response.body);
      if (generation !== this.generation) {
        URL.revokeObjectURL(url);
        return;
      }
      this.revokeImage();
      this.imageUrl.set(url);
      this.renderedFrame.set(latest);
      this.error.set(null);
    } catch (error) {
      if (generation === this.generation) {
        this.error.set(apiErrorMessage(error, 'Không thể nhận live preview mới nhất.'));
      }
    } finally {
      if (generation === this.generation) this.refreshing.set(false);
      this.polling = false;
    }
  }

  private stopPolling(): void {
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private revokeImage(): void {
    const current = this.imageUrl();
    if (current) URL.revokeObjectURL(current);
    this.imageUrl.set(null);
  }
}
