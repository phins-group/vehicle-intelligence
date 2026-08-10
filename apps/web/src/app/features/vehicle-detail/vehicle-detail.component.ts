import { DatePipe, PercentPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, RouterLink } from '@angular/router';
import {
  LucideArrowDownRight,
  LucideCamera,
  LucideCar,
  LucideChevronRight,
  LucideCircleAlert,
  LucideClock,
  LucideInfo,
  LucideMapPin,
  LucideRefreshCw,
  LucideRoute,
  LucideX
} from '@lucide/angular';
import { distinctUntilChanged, firstValueFrom, map } from 'rxjs';

import {
  JourneyObservation,
  VehicleEvent,
  VehicleIdentity,
  VehicleJourney
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import {
  buildJourneySteps,
  durationLabel,
  summarizeJourney
} from '../../core/utils/vehicle-journey-utils';
import { MediaEvidenceComponent } from '../../shared/media-evidence/media-evidence.component';

@Component({
  selector: 'app-vehicle-detail',
  imports: [
    DatePipe,
    PercentPipe,
    RouterLink,
    MediaEvidenceComponent,
    LucideArrowDownRight,
    LucideCamera,
    LucideCar,
    LucideChevronRight,
    LucideCircleAlert,
    LucideClock,
    LucideInfo,
    LucideMapPin,
    LucideRefreshCw,
    LucideRoute,
    LucideX
  ],
  templateUrl: './vehicle-detail.component.html'
})
export class VehicleDetailComponent {
  private readonly api = inject(ApiClientService);
  private readonly route = inject(ActivatedRoute);
  readonly identity = signal<VehicleIdentity | null>(null);
  readonly journey = signal<VehicleJourney | null>(null);
  readonly loading = signal(true);
  readonly eventLoading = signal(false);
  readonly error = signal<string | null>(null);
  readonly selected = signal<VehicleEvent | null>(null);
  readonly steps = computed(() => {
    const value = this.journey();
    return value ? buildJourneySteps(value) : [];
  });
  readonly summary = computed(() => {
    const value = this.journey();
    return value ? summarizeJourney(value) : null;
  });
  readonly mergedInto = computed(() => {
    const value = this.identity()?.metadata['mergedInto'];
    return typeof value === 'string' ? value : null;
  });
  readonly duration = computed(() => durationLabel(this.summary()?.durationSeconds ?? 0));
  vehicleId = '';
  private requestGeneration = 0;

  constructor() {
    this.route.paramMap
      .pipe(
        map((params) => params.get('vehicleId')?.trim() ?? ''),
        distinctUntilChanged(),
        takeUntilDestroyed()
      )
      .subscribe((vehicleId) => {
        this.vehicleId = vehicleId;
        if (vehicleId) void this.load();
      });
  }

  refresh(): void {
    if (this.vehicleId) void this.load();
  }

  async showEvent(observation: JourneyObservation): Promise<void> {
    if (this.eventLoading()) return;
    this.eventLoading.set(true);
    try {
      this.selected.set(await firstValueFrom(this.api.event(observation.eventId)));
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải chi tiết event.'));
    } finally {
      this.eventLoading.set(false);
    }
  }

  closeEvent(): void {
    this.selected.set(null);
  }

  segmentLabel(seconds: number): string {
    return durationLabel(seconds);
  }

  private async load(): Promise<void> {
    const generation = ++this.requestGeneration;
    this.loading.set(true);
    this.error.set(null);
    this.selected.set(null);
    try {
      const [identity, journey] = await Promise.all([
        firstValueFrom(this.api.vehicleIdentity(this.vehicleId)),
        firstValueFrom(this.api.vehicleJourney(this.vehicleId))
      ]);
      if (generation !== this.requestGeneration) return;
      this.identity.set(identity);
      this.journey.set(journey);
    } catch (error) {
      if (generation === this.requestGeneration) {
        this.identity.set(null);
        this.journey.set(null);
        this.error.set(apiErrorMessage(error, 'Không thể tải logical vehicle.'));
      }
    } finally {
      if (generation === this.requestGeneration) this.loading.set(false);
    }
  }
}
