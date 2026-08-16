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
import {
  LucideCheck,
  LucideChevronRight,
  LucideClock,
  LucideDatabase,
  LucideRefreshCw,
  LucideScanText,
  LucideX,
} from '@lucide/angular';
import { bufferTime, filter, firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import { VehicleEvent } from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import { mergeVehicleEvents } from '../../core/utils/event-utils';
import { listWindow } from '../../core/utils/list-window-utils';
import {
  finalPlate,
  plateReviewRevision,
} from '../../core/utils/plate-review-utils';
import { AccessibleDialogDirective } from '../../shared/accessibility/accessible-dialog.directive';
import { MediaEvidenceComponent } from '../../shared/media-evidence/media-evidence.component';

const REALTIME_BUFFER_MS = 200;
const MAX_BUFFERED_EVENTS = 500;
const RENDER_WINDOW_SIZE = 100;

@Component({
  selector: 'app-ocr-review',
  imports: [
    DatePipe,
    PercentPipe,
    FormsModule,
    AccessibleDialogDirective,
    MediaEvidenceComponent,
    LucideCheck,
    LucideChevronRight,
    LucideClock,
    LucideDatabase,
    LucideRefreshCw,
    LucideScanText,
    LucideX,
  ],
  templateUrl: './ocr-review.component.html',
})
export class OcrReviewComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiClientService);
  private readonly destroyRef = inject(DestroyRef);
  readonly auth = inject(AuthService);
  private readonly realtime = inject(RealtimeService);
  readonly events = signal<VehicleEvent[]>([]);
  readonly nextCursor = signal<string | null>(null);
  readonly loading = signal(true);
  readonly loadingMore = signal(false);
  readonly submitting = signal(false);
  readonly loadState = new AsyncDataState();
  readonly reviewError = signal<string | null>(null);
  readonly success = signal<string | null>(null);
  readonly selected = signal<VehicleEvent | null>(null);
  readonly renderWindowStart = signal(0);
  readonly bufferLimitReached = signal(false);
  readonly eventWindow = computed(() =>
    listWindow(this.events(), this.renderWindowStart(), RENDER_WINDOW_SIZE),
  );
  private requestGeneration = 0;
  private requestInFlight = false;
  private resetQueued = false;
  private destroyed = false;
  private realtimeRevision = 0;
  private realtimeLog: Array<{ revision: number; event: VehicleEvent }> = [];
  private reviewRequestGeneration = 0;

  plateText = '';
  note = '';

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
    if (!this.auth.canReviewPlates()) {
      this.loading.set(false);
      return;
    }
    void this.load(true);
  }

  ngOnDestroy(): void {
    this.markDestroyed();
  }

  async load(reset: boolean): Promise<void> {
    if (this.destroyed || !this.auth.canReviewPlates()) return;
    if (this.requestInFlight) {
      if (reset) {
        this.resetQueued = true;
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
      const page = await firstValueFrom(
        this.api.events({
          limit: 50,
          cursor: reset ? null : this.nextCursor(),
          status: 'NEEDS_REVIEW',
        }),
      );
      if (generation !== this.requestGeneration) return;
      const realtimeEvents = reset
        ? this.realtimeLog
            .filter((item) => item.revision > realtimeRevision)
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
          apiErrorMessage(error, 'Không thể tải hàng đợi OCR cần duyệt.'),
        );
      }
    } finally {
      this.requestInFlight = false;
      if (this.destroyed) return;
      this.loading.set(false);
      this.loadingMore.set(false);
      if (this.resetQueued) {
        this.resetQueued = false;
        void this.load(true);
      }
    }
  }

  showReview(event: VehicleEvent): void {
    this.reviewRequestGeneration += 1;
    this.submitting.set(false);
    this.selected.set(event);
    this.plateText = finalPlate(event) ?? '';
    this.note = '';
    this.reviewError.set(null);
    this.success.set(null);
  }

  closeReview(): void {
    this.reviewRequestGeneration += 1;
    this.submitting.set(false);
    this.selected.set(null);
    this.reviewError.set(null);
    this.success.set(null);
  }

  showNewerEvents(): void {
    this.renderWindowStart.set(
      Math.max(0, this.eventWindow().start - RENDER_WINDOW_SIZE),
    );
  }

  showOlderEvents(): void {
    this.renderWindowStart.set(this.eventWindow().start + RENDER_WINDOW_SIZE);
  }

  async submitReview(): Promise<void> {
    const event = this.selected();
    const text = this.plateText.trim();
    if (
      this.destroyed ||
      !event ||
      !this.auth.canReviewPlates() ||
      this.submitting()
    ) {
      return;
    }
    if (text.length < 4) {
      this.reviewError.set('Biển số cần ít nhất 4 ký tự.');
      return;
    }
    const generation = ++this.reviewRequestGeneration;
    this.submitting.set(true);
    this.reviewError.set(null);
    this.success.set(null);
    try {
      const outcome = await firstValueFrom(
        this.api.reviewPlate(event._id, {
          text,
          expectedRevision: plateReviewRevision(event),
          note: this.note.trim() || null,
        }),
      );
      if (!this.reviewRequestIsCurrent(generation, event._id)) return;
      this.selected.set(outcome.event);
      this.events.update((items) =>
        items.filter((item) => item._id !== event._id),
      );
      this.invalidateInFlightListAfterReview();
      const feedback = outcome.datasetSampleId
        ? `Dataset sample ${outcome.datasetSampleId} đã sẵn sàng.`
        : 'Plate crop không khả dụng nên không tạo dataset sample.';
      this.success.set(
        `${outcome.feedbackReason === 'HUMAN_CONFIRMATION' ? 'Đã xác nhận' : 'Đã sửa'} ${outcome.event.plate?.final}. ${feedback}`,
      );
    } catch (error) {
      if (!this.reviewRequestIsCurrent(generation, event._id)) return;
      this.reviewError.set(
        apiErrorMessage(error, 'Không thể lưu kết quả duyệt OCR.'),
      );
      if (
        typeof error === 'object' &&
        error !== null &&
        'status' in error &&
        error.status === 409
      ) {
        await this.refreshSelected(event._id, generation);
      }
    } finally {
      if (this.reviewRequestIsCurrent(generation, event._id)) {
        this.submitting.set(false);
      }
    }
  }

  displayPlate(event: VehicleEvent): string {
    return finalPlate(event) ?? 'Không đọc được';
  }

  private mergeRealtimeEvents(incoming: readonly VehicleEvent[]): void {
    const matching = incoming.filter(
      (event) => event.status === 'NEEDS_REVIEW',
    );
    if (!matching.length) return;
    const revision = ++this.realtimeRevision;
    this.realtimeLog = [
      ...this.realtimeLog,
      ...matching.map((event) => ({ revision, event })),
    ].slice(-MAX_BUFFERED_EVENTS);
    const merged = mergeVehicleEvents(
      this.events(),
      matching,
      MAX_BUFFERED_EVENTS + 1,
    );
    this.events.set(merged.slice(0, MAX_BUFFERED_EVENTS));
    if (merged.length > MAX_BUFFERED_EVENTS) this.bufferLimitReached.set(true);
  }

  private async refreshSelected(
    eventId: string,
    generation: number,
  ): Promise<void> {
    try {
      const latest = await firstValueFrom(this.api.event(eventId));
      if (!this.reviewRequestIsCurrent(generation, eventId)) return;
      this.selected.set(latest);
      this.plateText = finalPlate(latest) ?? '';
      if (latest.status !== 'NEEDS_REVIEW') {
        this.events.update((items) =>
          items.filter((item) => item._id !== eventId),
        );
      }
    } catch {
      // Keep the concurrency error visible; the operator can close and refresh the queue.
    }
  }

  private invalidateInFlightListAfterReview(): void {
    if (!this.requestInFlight) return;
    this.resetQueued = true;
    this.requestGeneration += 1;
  }

  private markDestroyed(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.requestGeneration += 1;
    this.reviewRequestGeneration += 1;
    this.resetQueued = false;
  }

  private reviewRequestIsCurrent(generation: number, eventId: string): boolean {
    return (
      !this.destroyed &&
      generation === this.reviewRequestGeneration &&
      this.selected()?._id === eventId
    );
  }
}
