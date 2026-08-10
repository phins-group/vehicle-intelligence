import { DatePipe, PercentPipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import {
  LucideActivity,
  LucideChevronRight,
  LucideClock,
  LucideRadio,
  LucideRefreshCw,
  LucideSearch,
  LucideX
} from '@lucide/angular';
import { catchError, firstValueFrom, of } from 'rxjs';

import {
  Camera,
  Direction,
  EventFilters,
  EventStatus,
  EventType,
  VehicleEvent
} from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { eventMatchesFilters, mergeVehicleEvents } from '../../core/utils/event-utils';
import { finalPlate } from '../../core/utils/plate-review-utils';
import { MediaEvidenceComponent } from '../../shared/media-evidence/media-evidence.component';

@Component({
  selector: 'app-events',
  imports: [
    DatePipe,
    PercentPipe,
    FormsModule,
    RouterLink,
    MediaEvidenceComponent,
    LucideActivity,
    LucideChevronRight,
    LucideClock,
    LucideRadio,
    LucideRefreshCw,
    LucideSearch,
    LucideX
  ],
  templateUrl: './events.component.html'
})
export class EventsComponent implements OnInit {
  private readonly api = inject(ApiClientService);
  readonly realtime = inject(RealtimeService);
  readonly events = signal<VehicleEvent[]>([]);
  readonly cameras = signal<Camera[]>([]);
  readonly nextCursor = signal<string | null>(null);
  readonly loading = signal(true);
  readonly loadingMore = signal(false);
  readonly error = signal<string | null>(null);
  readonly selected = signal<VehicleEvent | null>(null);
  readonly liveCount = signal(0);
  private readonly appliedFilters = signal<EventFilters>({ limit: 50 });

  plate = '';
  cameraId = '';
  eventType: EventType | '' = '';
  direction: Direction | '' = '';
  status: EventStatus | '' = '';
  fromLocal = '';
  toLocal = '';

  constructor() {
    this.realtime.events$.pipe(takeUntilDestroyed()).subscribe((event) => {
      if (!eventMatchesFilters(event, this.appliedFilters())) return;
      const merged = mergeVehicleEvents(this.events(), [event], 500);
      if (merged.length !== this.events().length || merged[0]?._id !== this.events()[0]?._id) {
        this.events.set(merged);
        this.liveCount.update((count) => count + 1);
      }
    });
    this.realtime.recoveryRequested$
      .pipe(takeUntilDestroyed())
      .subscribe(() => void this.load(true));
  }

  ngOnInit(): void {
    void this.loadCameras();
    void this.load(true);
  }

  async applyFilters(): Promise<void> {
    this.liveCount.set(0);
    await this.load(true);
  }

  async load(reset: boolean): Promise<void> {
    if (reset) this.loading.set(true);
    else this.loadingMore.set(true);
    this.error.set(null);
    try {
      const filters = this.filters(reset ? null : this.nextCursor());
      const page = await firstValueFrom(this.api.events(filters));
      this.appliedFilters.set({ ...filters, cursor: null });
      this.events.set(
        reset ? page.items : mergeVehicleEvents(this.events(), page.items, 1000)
      );
      this.nextCursor.set(page.nextCursor);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải danh sách sự kiện.'));
    } finally {
      this.loading.set(false);
      this.loadingMore.set(false);
    }
  }

  clearFilters(): void {
    this.plate = '';
    this.cameraId = '';
    this.eventType = '';
    this.direction = '';
    this.status = '';
    this.fromLocal = '';
    this.toLocal = '';
    void this.applyFilters();
  }

  showDetails(event: VehicleEvent): void {
    this.selected.set(event);
  }

  closeDetails(): void {
    this.selected.set(null);
  }

  displayPlate(event: VehicleEvent): string {
    return finalPlate(event) ?? 'Không đọc được';
  }

  private async loadCameras(): Promise<void> {
    const page = await firstValueFrom(
      this.api.cameras().pipe(catchError(() => of({ items: [] as Camera[] })))
    );
    this.cameras.set(page.items);
  }

  private filters(cursor: string | null): EventFilters {
    return {
      limit: 50,
      cursor,
      cameraId: this.cameraId,
      plate: this.plate.trim(),
      eventType: this.eventType,
      direction: this.direction,
      status: this.status,
      from: this.toIso(this.fromLocal),
      to: this.toIso(this.toLocal)
    };
  }

  private toIso(value: string): string | undefined {
    if (!value) return undefined;
    const timestamp = new Date(value);
    return Number.isNaN(timestamp.getTime()) ? undefined : timestamp.toISOString();
  }
}
