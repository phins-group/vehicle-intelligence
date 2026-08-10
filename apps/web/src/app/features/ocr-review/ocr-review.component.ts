import { DatePipe, PercentPipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import {
  LucideCheck,
  LucideChevronRight,
  LucideClock,
  LucideDatabase,
  LucideRefreshCw,
  LucideScanText,
  LucideX
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import { VehicleEvent } from '../../core/models/api.models';
import { RealtimeService } from '../../core/realtime/realtime.service';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { mergeVehicleEvents } from '../../core/utils/event-utils';
import { finalPlate, plateReviewRevision } from '../../core/utils/plate-review-utils';
import { MediaEvidenceComponent } from '../../shared/media-evidence/media-evidence.component';

@Component({
  selector: 'app-ocr-review',
  imports: [
    DatePipe,
    PercentPipe,
    FormsModule,
    MediaEvidenceComponent,
    LucideCheck,
    LucideChevronRight,
    LucideClock,
    LucideDatabase,
    LucideRefreshCw,
    LucideScanText,
    LucideX
  ],
  templateUrl: './ocr-review.component.html'
})
export class OcrReviewComponent implements OnInit {
  private readonly api = inject(ApiClientService);
  readonly auth = inject(AuthService);
  private readonly realtime = inject(RealtimeService);
  readonly events = signal<VehicleEvent[]>([]);
  readonly nextCursor = signal<string | null>(null);
  readonly loading = signal(true);
  readonly loadingMore = signal(false);
  readonly submitting = signal(false);
  readonly error = signal<string | null>(null);
  readonly reviewError = signal<string | null>(null);
  readonly success = signal<string | null>(null);
  readonly selected = signal<VehicleEvent | null>(null);

  plateText = '';
  note = '';

  constructor() {
    this.realtime.events$.pipe(takeUntilDestroyed()).subscribe((event) => {
      if (event.status !== 'NEEDS_REVIEW') return;
      this.events.set(mergeVehicleEvents(this.events(), [event], 500));
    });
    this.realtime.recoveryRequested$
      .pipe(takeUntilDestroyed())
      .subscribe(() => void this.load(true));
  }

  ngOnInit(): void {
    if (!this.auth.canReviewPlates()) {
      this.loading.set(false);
      return;
    }
    void this.load(true);
  }

  async load(reset: boolean): Promise<void> {
    if (!this.auth.canReviewPlates()) return;
    if (reset) this.loading.set(true);
    else this.loadingMore.set(true);
    this.error.set(null);
    try {
      const page = await firstValueFrom(
        this.api.events({
          limit: 50,
          cursor: reset ? null : this.nextCursor(),
          status: 'NEEDS_REVIEW'
        })
      );
      this.events.set(
        reset ? page.items : mergeVehicleEvents(this.events(), page.items, 500)
      );
      this.nextCursor.set(page.nextCursor);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải hàng đợi OCR cần duyệt.'));
    } finally {
      this.loading.set(false);
      this.loadingMore.set(false);
    }
  }

  showReview(event: VehicleEvent): void {
    this.selected.set(event);
    this.plateText = finalPlate(event) ?? '';
    this.note = '';
    this.reviewError.set(null);
    this.success.set(null);
  }

  closeReview(): void {
    this.selected.set(null);
    this.reviewError.set(null);
    this.success.set(null);
  }

  async submitReview(): Promise<void> {
    const event = this.selected();
    const text = this.plateText.trim();
    if (!event || !this.auth.canReviewPlates() || this.submitting()) return;
    if (text.length < 4) {
      this.reviewError.set('Biển số cần ít nhất 4 ký tự.');
      return;
    }
    this.submitting.set(true);
    this.reviewError.set(null);
    this.success.set(null);
    try {
      const outcome = await firstValueFrom(
        this.api.reviewPlate(event._id, {
          text,
          expectedRevision: plateReviewRevision(event),
          note: this.note.trim() || null
        })
      );
      this.selected.set(outcome.event);
      this.events.update((items) => items.filter((item) => item._id !== event._id));
      const feedback = outcome.datasetSampleId
        ? `Dataset sample ${outcome.datasetSampleId} đã sẵn sàng.`
        : 'Plate crop không khả dụng nên không tạo dataset sample.';
      this.success.set(
        `${outcome.feedbackReason === 'HUMAN_CONFIRMATION' ? 'Đã xác nhận' : 'Đã sửa'} ${outcome.event.plate?.final}. ${feedback}`
      );
    } catch (error) {
      this.reviewError.set(apiErrorMessage(error, 'Không thể lưu kết quả duyệt OCR.'));
      if (typeof error === 'object' && error !== null && 'status' in error && error.status === 409) {
        await this.refreshSelected(event._id);
      }
    } finally {
      this.submitting.set(false);
    }
  }

  displayPlate(event: VehicleEvent): string {
    return finalPlate(event) ?? 'Không đọc được';
  }

  private async refreshSelected(eventId: string): Promise<void> {
    try {
      const latest = await firstValueFrom(this.api.event(eventId));
      this.selected.set(latest);
      this.plateText = finalPlate(latest) ?? '';
      if (latest.status !== 'NEEDS_REVIEW') {
        this.events.update((items) => items.filter((item) => item._id !== eventId));
      }
    } catch {
      // Keep the concurrency error visible; the operator can close and refresh the queue.
    }
  }
}
