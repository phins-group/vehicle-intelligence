import { DatePipe, PercentPipe } from '@angular/common';
import {
  Component,
  DestroyRef,
  OnDestroy,
  OnInit,
  computed,
  inject,
  signal,
} from '@angular/core';
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
  LucideX,
} from '@lucide/angular';
import { bufferTime, catchError, filter, firstValueFrom, of } from 'rxjs';

import {
  Camera,
  Direction,
  EventFilters,
  EventStatus,
  EventType,
  VehicleEvent,
} from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import {
  eventMatchesFilters,
  mergeVehicleEvents,
} from '../../core/utils/event-utils';
import { listWindow } from '../../core/utils/list-window-utils';
import { finalPlate } from '../../core/utils/plate-review-utils';
import { AccessibleDialogDirective } from '../../shared/accessibility/accessible-dialog.directive';
import { MediaEvidenceComponent } from '../../shared/media-evidence/media-evidence.component';

const REALTIME_BUFFER_MS = 200;
const MAX_BUFFERED_EVENTS = 500;
const RENDER_WINDOW_SIZE = 100;

@Component({
  selector: 'app-events',
  imports: [
    DatePipe,
    PercentPipe,
    FormsModule,
    RouterLink,
    AccessibleDialogDirective,
    MediaEvidenceComponent,
    LucideActivity,
    LucideChevronRight,
    LucideClock,
    LucideRadio,
    LucideRefreshCw,
    LucideSearch,
    LucideX,
  ],
  templateUrl: './events.component.html',
})
export class EventsComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiClientService);
  private readonly destroyRef = inject(DestroyRef);
  readonly realtime = inject(RealtimeService);
  readonly events = signal<VehicleEvent[]>([]);
  readonly cameras = signal<Camera[]>([]);
  readonly nextCursor = signal<string | null>(null);
  readonly loading = signal(true);
  readonly loadingMore = signal(false);
  readonly loadState = new AsyncDataState();
  readonly selected = signal<VehicleEvent | null>(null);
  readonly liveCount = signal(0);
  readonly renderWindowStart = signal(0);
  readonly bufferLimitReached = signal(false);
  readonly eventWindow = computed(() =>
    listWindow(this.events(), this.renderWindowStart(), RENDER_WINDOW_SIZE),
  );
  private readonly appliedFilters = signal<EventFilters>({ limit: 50 });
  private requestedFilters: EventFilters = { limit: 50 };
  private requestGeneration = 0;
  private requestInFlight = false;
  private queuedResetFilters: EventFilters | null = null;
  private destroyed = false;
  private realtimeRevision = 0;
  private realtimeLog: Array<{ revision: number; event: VehicleEvent }> = [];

  plate = '';
  cameraId = '';
  eventType: EventType | '' = '';
  direction: Direction | '' = '';
  status: EventStatus | '' = '';
  fromLocal = '';
  toLocal = '';

  constructor() {
    this.realtime.events$
      .pipe(
        bufferTime(REALTIME_BUFFER_MS),
        filter((events) => events.length > 0),
        takeUntilDestroyed(),
      )
      .subscribe((events) => this.mergeRealtimeEvents(events));
    this.realtime.recoveryRequested$
      .pipe(takeUntilDestroyed())
      .subscribe(() => void this.load(true));
    this.destroyRef.onDestroy(() => this.markDestroyed());
  }

  ngOnInit(): void {
    void this.loadCameras();
    void this.load(true);
  }

  ngOnDestroy(): void {
    this.markDestroyed();
  }

  async applyFilters(): Promise<void> {
    this.liveCount.set(0);
    this.requestedFilters = this.formFilters(null);
    await this.loadWithFilters(true, this.requestedFilters);
  }

  async load(reset: boolean): Promise<void> {
    const filters = reset
      ? { ...this.requestedFilters, cursor: null }
      : { ...this.appliedFilters(), cursor: this.nextCursor() };
    await this.loadWithFilters(reset, filters);
  }

  private async loadWithFilters(
    reset: boolean,
    filters: EventFilters,
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
    const realtimeRevision = this.realtimeRevision;
    if (reset) this.loading.set(true);
    else this.loadingMore.set(true);
    this.loadState.begin();
    try {
      const page = await firstValueFrom(this.api.events(filters));
      if (generation !== this.requestGeneration) return;
      if (reset) this.appliedFilters.set({ ...filters, cursor: null });
      const realtimeEvents = reset
        ? this.realtimeLog
            .filter(
              (item) =>
                item.revision > realtimeRevision &&
                eventMatchesFilters(item.event, filters),
            )
            .map((item) => item.event)
        : this.events();
      const merged = mergeVehicleEvents(
        realtimeEvents,
        page.items,
        MAX_BUFFERED_EVENTS + 1,
      );
      this.events.set(merged.slice(0, MAX_BUFFERED_EVENTS));
      this.bufferLimitReached.set(
        merged.length > MAX_BUFFERED_EVENTS ||
          (merged.length === MAX_BUFFERED_EVENTS && page.nextCursor !== null),
      );
      if (reset) this.renderWindowStart.set(0);
      this.nextCursor.set(page.nextCursor);
      this.loadState.succeed();
    } catch (error) {
      if (generation === this.requestGeneration) {
        this.loadState.fail(
          apiErrorMessage(error, 'Không thể tải danh sách sự kiện.'),
        );
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

  showNewerEvents(): void {
    this.renderWindowStart.set(
      Math.max(0, this.eventWindow().start - RENDER_WINDOW_SIZE),
    );
  }

  showOlderEvents(): void {
    this.renderWindowStart.set(this.eventWindow().start + RENDER_WINDOW_SIZE);
  }

  displayPlate(event: VehicleEvent): string {
    return finalPlate(event) ?? 'Không đọc được';
  }

  private async loadCameras(): Promise<void> {
    const page = await firstValueFrom(
      this.api.cameras().pipe(catchError(() => of({ items: [] as Camera[] }))),
    );
    if (!this.destroyed) this.cameras.set(page.items);
  }

  private mergeRealtimeEvents(incoming: readonly VehicleEvent[]): void {
    const revision = ++this.realtimeRevision;
    this.realtimeLog = [
      ...this.realtimeLog,
      ...incoming.map((event) => ({ revision, event })),
    ].slice(-MAX_BUFFERED_EVENTS);
    const matching = incoming.filter((event) =>
      eventMatchesFilters(event, this.appliedFilters()),
    );
    if (!matching.length) return;
    const current = this.events();
    const currentIds = new Set(current.map((event) => event._id));
    const newIds = new Set(
      matching
        .filter((event) => !currentIds.has(event._id))
        .map((event) => event._id),
    );
    const merged = mergeVehicleEvents(
      current,
      matching,
      MAX_BUFFERED_EVENTS + 1,
    );
    this.events.set(merged.slice(0, MAX_BUFFERED_EVENTS));
    if (merged.length > MAX_BUFFERED_EVENTS) this.bufferLimitReached.set(true);
    if (newIds.size) this.liveCount.update((count) => count + newIds.size);
  }

  private formFilters(cursor: string | null): EventFilters {
    return {
      limit: 50,
      cursor,
      cameraId: this.cameraId,
      plate: this.plate.trim(),
      eventType: this.eventType,
      direction: this.direction,
      status: this.status,
      from: this.toIso(this.fromLocal),
      to: this.toIso(this.toLocal),
    };
  }

  private toIso(value: string): string | undefined {
    if (!value) return undefined;
    const timestamp = new Date(value);
    return Number.isNaN(timestamp.getTime())
      ? undefined
      : timestamp.toISOString();
  }

  private markDestroyed(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.requestGeneration += 1;
    this.queuedResetFilters = null;
  }
}
