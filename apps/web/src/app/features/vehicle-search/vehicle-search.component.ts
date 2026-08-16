import { DatePipe, PercentPipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
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
  LucideSearch,
  LucideX
} from '@lucide/angular';
import { distinctUntilChanged, firstValueFrom, map } from 'rxjs';

import { VehicleEvent } from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { mergeVehicleEvents } from '../../core/utils/event-utils';
import { listWindow } from '../../core/utils/list-window-utils';
import { finalPlate } from '../../core/utils/plate-review-utils';
import {
  chronologicalVehicleEvents,
  summarizePlateHistory
} from '../../core/utils/vehicle-history-utils';
import { AccessibleDialogDirective } from '../../shared/accessibility/accessible-dialog.directive';
import { MediaEvidenceComponent } from '../../shared/media-evidence/media-evidence.component';

const PAGE_SIZE = 50;
const CLIENT_HISTORY_LIMIT = 500;
const TIMELINE_WINDOW_SIZE = 100;

@Component({
  selector: 'app-vehicle-search',
  imports: [
    DatePipe,
    PercentPipe,
    FormsModule,
    RouterLink,
    AccessibleDialogDirective,
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
    LucideSearch,
    LucideX
  ],
  templateUrl: './vehicle-search.component.html'
})
export class VehicleSearchComponent {
  private readonly api = inject(ApiClientService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  readonly events = signal<VehicleEvent[]>([]);
  readonly resultQuery = signal<string | null>(null);
  readonly nextCursor = signal<string | null>(null);
  readonly loading = signal(false);
  readonly loadingMore = signal(false);
  readonly searched = signal(false);
  readonly error = signal<string | null>(null);
  readonly selected = signal<VehicleEvent | null>(null);
  readonly summary = computed(() => summarizePlateHistory(this.events()));
  readonly timeline = computed(() => chronologicalVehicleEvents(this.events()));
  readonly timelineWindowStart = signal(0);
  readonly timelineWindow = computed(() =>
    listWindow(this.timeline(), this.timelineWindowStart(), TIMELINE_WINDOW_SIZE)
  );
  readonly clientBoundReached = computed(
    () => this.events().length >= CLIENT_HISTORY_LIMIT && this.nextCursor() !== null
  );
  private requestGeneration = 0;
  plate = '';

  constructor() {
    this.route.queryParamMap
      .pipe(
        map((params) => params.get('plate')?.trim() ?? ''),
        distinctUntilChanged(),
        takeUntilDestroyed()
      )
      .subscribe((plate) => {
        this.plate = plate;
        if (plate) void this.load(true, plate);
        else this.resetResults();
      });
  }

  submitSearch(): void {
    const candidate = this.plate.trim();
    if (candidate.length < 4) {
      this.error.set('Nhập biển số có ít nhất 4 ký tự.');
      return;
    }
    const current = this.route.snapshot.queryParamMap.get('plate')?.trim() ?? '';
    if (current === candidate) {
      void this.load(true, candidate);
      return;
    }
    void this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { plate: candidate },
      queryParamsHandling: 'merge'
    });
  }

  clearSearch(): void {
    void this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { plate: null },
      queryParamsHandling: 'merge'
    });
  }

  refresh(): void {
    const query = this.resultQuery() ?? this.plate.trim();
    if (query) void this.load(true, query);
  }

  loadMore(): void {
    const query = this.resultQuery();
    if (!query || !this.nextCursor() || this.clientBoundReached()) return;
    void this.load(false, query);
  }

  showDetails(event: VehicleEvent): void {
    this.selected.set(event);
  }

  closeDetails(): void {
    this.selected.set(null);
  }

  showOlderTimeline(): void {
    const window = this.timelineWindow();
    this.timelineWindowStart.set(Math.max(0, window.start - TIMELINE_WINDOW_SIZE));
  }

  showNewerTimeline(): void {
    const window = this.timelineWindow();
    this.timelineWindowStart.set(window.start + TIMELINE_WINDOW_SIZE);
  }

  displayPlate(event: VehicleEvent): string {
    return finalPlate(event) ?? this.resultQuery() ?? 'Không đọc được';
  }

  private async load(reset: boolean, query: string): Promise<void> {
    const generation = reset ? ++this.requestGeneration : this.requestGeneration;
    if (reset) {
      this.loading.set(true);
      this.loadingMore.set(false);
      this.searched.set(true);
      this.events.set([]);
      this.timelineWindowStart.set(0);
      this.resultQuery.set(null);
      this.nextCursor.set(null);
      this.selected.set(null);
    } else {
      if (this.loadingMore()) return;
      this.loadingMore.set(true);
    }
    this.error.set(null);
    try {
      const remaining = Math.max(0, CLIENT_HISTORY_LIMIT - this.events().length);
      const limit = Math.min(PAGE_SIZE, remaining || PAGE_SIZE);
      const page = await firstValueFrom(
        this.api.searchVehicleHistory(query, limit, reset ? null : this.nextCursor())
      );
      if (generation !== this.requestGeneration) return;
      this.resultQuery.set(page.query);
      this.events.set(
        reset
          ? page.items
          : mergeVehicleEvents(this.events(), page.items, CLIENT_HISTORY_LIMIT)
      );
      if (!reset) this.timelineWindowStart.set(0);
      this.nextCursor.set(page.nextCursor);
    } catch (error) {
      if (generation === this.requestGeneration) {
        this.error.set(apiErrorMessage(error, 'Không thể tra cứu lịch sử biển số.'));
      }
    } finally {
      if (generation === this.requestGeneration) {
        this.loading.set(false);
        this.loadingMore.set(false);
      }
    }
  }

  private resetResults(): void {
    this.requestGeneration += 1;
    this.searched.set(false);
    this.events.set([]);
    this.timelineWindowStart.set(0);
    this.resultQuery.set(null);
    this.nextCursor.set(null);
    this.selected.set(null);
    this.error.set(null);
    this.loading.set(false);
    this.loadingMore.set(false);
  }
}
