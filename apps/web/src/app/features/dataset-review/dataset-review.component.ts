import { DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  OnDestroy,
  OnInit,
  computed,
  inject,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  LucideBoxSelect,
  LucideCheck,
  LucideChevronRight,
  LucideDatabase,
  LucideImage,
  LucideLayers,
  LucideMousePointer2,
  LucideRefreshCw,
  LucideRotateCcw,
  LucideSave,
  LucideTrash2,
  LucideX,
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import {
  DetectorPromotionJob,
  DetectorReviewAction,
  DetectorReviewBox,
  DetectorReviewDecision,
  DetectorReviewItem,
  DetectorReviewSource,
  DetectorReviewStatus,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import {
  BoxNudgeKey,
  CanvasPoint,
  boxFromPoints,
  boxesMatchSuggestions,
  clampBox,
  defaultReviewBox,
  detectorReviewReason,
  editableBoxes,
  nudgeReviewBox,
  pointerToImage,
} from '../../core/utils/dataset-review-utils';

type BoxField = 'x' | 'y' | 'width' | 'height';

function isBoxNudgeKey(key: string): key is BoxNudgeKey {
  return ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(key);
}

@Component({
  selector: 'app-dataset-review',
  imports: [
    DatePipe,
    DecimalPipe,
    FormsModule,
    LucideBoxSelect,
    LucideCheck,
    LucideChevronRight,
    LucideDatabase,
    LucideImage,
    LucideLayers,
    LucideMousePointer2,
    LucideRefreshCw,
    LucideRotateCcw,
    LucideSave,
    LucideTrash2,
    LucideX,
  ],
  templateUrl: './dataset-review.component.html',
})
export class DatasetReviewComponent implements OnInit, OnDestroy {
  private readonly api = inject(ApiClientService);
  readonly auth = inject(AuthService);

  readonly sources = signal<DetectorReviewSource[]>([]);
  readonly items = signal<DetectorReviewItem[]>([]);
  readonly nextCursor = signal<string | null>(null);
  readonly selected = signal<DetectorReviewItem | null>(null);
  readonly history = signal<DetectorReviewDecision[]>([]);
  readonly previewUrl = signal<string | null>(null);
  readonly boxes = signal<DetectorReviewBox[]>([]);
  readonly selectedBox = signal<number | null>(null);
  readonly draftBox = signal<DetectorReviewBox | null>(null);
  readonly loading = signal(true);
  readonly loadState = new AsyncDataState();
  readonly loadingItems = signal(false);
  readonly loadingMore = signal(false);
  readonly loadingDetail = signal(false);
  readonly submitting = signal(false);
  readonly error = signal<string | null>(null);
  readonly reviewError = signal<string | null>(null);
  readonly success = signal<string | null>(null);
  readonly bboxStatus = signal('');
  readonly promotionJob = signal<DetectorPromotionJob | null>(null);
  readonly promotionStarting = signal(false);
  readonly promoting = computed(
    () =>
      this.promotionStarting() ||
      ['QUEUED', 'RUNNING'].includes(this.promotionJob()?.status ?? ''),
  );
  readonly selectedSource = computed(
    () =>
      this.sources().find((source) => source.sourceId === this.sourceId) ??
      null,
  );
  readonly canApprove = computed(() => {
    const item = this.selected();
    return (
      item !== null && boxesMatchSuggestions(this.boxes(), item.suggestions)
    );
  });

  sourceId = '';
  statusFilter: DetectorReviewStatus | '' = 'PENDING_REVIEW';
  reasonFilter = '';
  note = '';
  targetSourceId = '';

  private dragStart: CanvasPoint | null = null;
  private previewObjectUrl: string | null = null;
  private promotionTimer: number | null = null;
  private destroyed = false;
  private sourceRequestGeneration = 0;
  private itemListGeneration = 0;
  private detailRequestGeneration = 0;
  private submissionGeneration = 0;
  private promotionGeneration = 0;

  ngOnInit(): void {
    if (!this.auth.canReviewDatasets()) {
      this.loading.set(false);
      return;
    }
    void this.loadSources();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.sourceRequestGeneration += 1;
    this.itemListGeneration += 1;
    this.detailRequestGeneration += 1;
    this.submissionGeneration += 1;
    this.promotionGeneration += 1;
    this.releasePreview();
    if (this.promotionTimer !== null) window.clearTimeout(this.promotionTimer);
    this.promotionTimer = null;
  }

  async loadSources(): Promise<void> {
    if (this.destroyed) return;
    const generation = ++this.sourceRequestGeneration;
    this.loading.set(true);
    this.error.set(null);
    this.loadState.begin();
    try {
      const result = await firstValueFrom(this.api.detectorReviewSources());
      if (this.destroyed || generation !== this.sourceRequestGeneration) return;
      this.sources.set(result.items);
      this.loadState.succeed();
      if (!result.items.some((source) => source.sourceId === this.sourceId)) {
        this.sourceId = result.items[0]?.sourceId ?? '';
      }
      this.refreshTargetSourceId();
      if (this.sourceId) await this.loadItems(true);
      else this.items.set([]);
    } catch (error) {
      if (!this.destroyed && generation === this.sourceRequestGeneration) {
        this.loadState.fail(
          apiErrorMessage(error, 'Không thể tải detector review sources.'),
        );
      }
    } finally {
      if (!this.destroyed && generation === this.sourceRequestGeneration) {
        this.loading.set(false);
      }
    }
  }

  async sourceChanged(): Promise<void> {
    this.closeItem();
    this.clearPromotionState();
    this.reasonFilter = '';
    this.refreshTargetSourceId();
    await this.loadItems(true);
  }

  async applyFilters(): Promise<void> {
    this.closeItem();
    await this.loadItems(true);
  }

  async loadItems(reset: boolean): Promise<void> {
    if (
      this.destroyed ||
      !this.sourceId ||
      (!reset && (this.loadingItems() || this.loadingMore()))
    ) {
      return;
    }
    const sourceId = this.sourceId;
    const generation = reset
      ? ++this.itemListGeneration
      : this.itemListGeneration;
    if (reset) {
      this.items.set([]);
      this.nextCursor.set(null);
    }
    reset ? this.loadingItems.set(true) : this.loadingMore.set(true);
    this.error.set(null);
    try {
      const page = await firstValueFrom(
        this.api.detectorReviewItems({
          sourceId,
          limit: 50,
          cursor: reset ? null : this.nextCursor(),
          status: this.statusFilter,
          reason: this.reasonFilter,
        }),
      );
      if (
        this.destroyed ||
        generation !== this.itemListGeneration ||
        sourceId !== this.sourceId
      ) {
        return;
      }
      this.items.update((current) =>
        reset ? page.items : [...current, ...page.items],
      );
      this.nextCursor.set(page.nextCursor);
    } catch (error) {
      if (!this.destroyed && generation === this.itemListGeneration) {
        this.error.set(
          apiErrorMessage(error, 'Không thể tải hàng đợi detector dataset.'),
        );
      }
    } finally {
      if (!this.destroyed && generation === this.itemListGeneration) {
        this.loadingItems.set(false);
        this.loadingMore.set(false);
      }
    }
  }

  async openItem(item: DetectorReviewItem): Promise<void> {
    if (this.destroyed || item.sourceId !== this.sourceId) return;
    const generation = ++this.detailRequestGeneration;
    this.loadingDetail.set(true);
    this.reviewError.set(null);
    this.success.set(null);
    this.releasePreview();
    try {
      const [detail, image, history] = await Promise.all([
        firstValueFrom(
          this.api.detectorReviewItem(item.sourceId, item.reviewId),
        ),
        firstValueFrom(
          this.api.detectorReviewImage(item.sourceId, item.reviewId),
        ),
        firstValueFrom(
          this.api.detectorReviewHistory(item.sourceId, item.reviewId),
        ),
      ]);
      if (this.destroyed || generation !== this.detailRequestGeneration) return;
      this.selected.set(detail);
      this.boxes.set(editableBoxes(detail));
      this.history.set(history.items);
      this.selectedBox.set(null);
      this.bboxStatus.set('');
      this.note = detail.decision?.note ?? '';
      this.previewObjectUrl = URL.createObjectURL(image);
      this.previewUrl.set(this.previewObjectUrl);
    } catch (error) {
      if (!this.destroyed && generation === this.detailRequestGeneration) {
        this.reviewError.set(
          apiErrorMessage(error, 'Không thể tải ảnh và nhãn cần duyệt.'),
        );
      }
    } finally {
      if (!this.destroyed && generation === this.detailRequestGeneration) {
        this.loadingDetail.set(false);
      }
    }
  }

  closeItem(): void {
    this.detailRequestGeneration += 1;
    this.submissionGeneration += 1;
    this.loadingDetail.set(false);
    this.submitting.set(false);
    this.selected.set(null);
    this.history.set([]);
    this.boxes.set([]);
    this.draftBox.set(null);
    this.selectedBox.set(null);
    this.bboxStatus.set('');
    this.reviewError.set(null);
    this.success.set(null);
    this.note = '';
    this.releasePreview();
  }

  resetBoxes(): void {
    const item = this.selected();
    if (!item) return;
    this.boxes.set(editableBoxes(item));
    this.selectedBox.set(null);
    this.bboxStatus.set('Đã khôi phục bounding box theo dữ liệu ban đầu.');
  }

  addKeyboardBox(): void {
    const image = this.selected()?.image;
    if (!image) return;
    const box = defaultReviewBox(image);
    this.boxes.update((items) => [...items, box]);
    const index = this.boxes().length - 1;
    this.selectedBox.set(index);
    this.bboxStatus.set(`Đã thêm bbox ${index + 1} ở giữa ảnh.`);
  }

  selectBox(index: number): void {
    if (index < 0 || index >= this.boxes().length) return;
    this.selectedBox.set(index);
  }

  boxAccessibleLabel(box: DetectorReviewBox, index: number): string {
    return `BBox ${index + 1}: x ${box.x}, y ${box.y}, rộng ${box.width}, cao ${box.height}`;
  }

  boxKeydown(index: number, event: KeyboardEvent): void {
    if (event.defaultPrevented || event.isComposing) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.selectBox(index);
      return;
    }
    if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault();
      const fallback = (
        event.currentTarget as Element | null
      )?.ownerDocument.getElementById('add-keyboard-bbox');
      this.removeBox(index);
      fallback?.focus();
      return;
    }
    if (!isBoxNudgeKey(event.key)) return;
    const image = this.selected()?.image;
    if (!image) return;
    event.preventDefault();
    const step = event.shiftKey ? 10 : 1;
    this.boxes.update((items) =>
      items.map((box, boxIndex) =>
        boxIndex === index
          ? nudgeReviewBox(box, image, event.key as BoxNudgeKey, step)
          : box,
      ),
    );
    this.selectedBox.set(index);
  }

  canvasPointerDown(event: PointerEvent): void {
    const image = this.selected()?.image;
    if (!image || event.button !== 0) return;
    const canvas = event.currentTarget as SVGSVGElement;
    canvas.setPointerCapture(event.pointerId);
    this.dragStart = pointerToImage(
      event.clientX,
      event.clientY,
      canvas.getBoundingClientRect(),
      image,
    );
    this.draftBox.set(null);
    this.selectedBox.set(null);
    event.preventDefault();
  }

  canvasPointerMove(event: PointerEvent): void {
    const image = this.selected()?.image;
    if (!image || !this.dragStart) return;
    const canvas = event.currentTarget as SVGSVGElement;
    const point = pointerToImage(
      event.clientX,
      event.clientY,
      canvas.getBoundingClientRect(),
      image,
    );
    this.draftBox.set(boxFromPoints(this.dragStart, point, image));
  }

  canvasPointerUp(event: PointerEvent): void {
    const image = this.selected()?.image;
    if (!image || !this.dragStart) return;
    const canvas = event.currentTarget as SVGSVGElement;
    const point = pointerToImage(
      event.clientX,
      event.clientY,
      canvas.getBoundingClientRect(),
      image,
    );
    const box = boxFromPoints(this.dragStart, point, image);
    this.dragStart = null;
    this.draftBox.set(null);
    if (box) {
      this.boxes.update((items) => [...items, box]);
      this.selectedBox.set(this.boxes().length - 1);
    }
  }

  selectExistingBox(index: number, event: PointerEvent): void {
    event.stopPropagation();
    this.selectedBox.set(index);
  }

  updateBox(index: number, field: BoxField, value: number | string): void {
    const image = this.selected()?.image;
    if (!image) return;
    const parsed = typeof value === 'number' ? value : Number(value);
    this.boxes.update((items) =>
      items.map((box, boxIndex) =>
        boxIndex === index ? clampBox({ ...box, [field]: parsed }, image) : box,
      ),
    );
  }

  removeBox(index: number): void {
    this.boxes.update((items) =>
      items.filter((_, boxIndex) => boxIndex !== index),
    );
    this.selectedBox.set(null);
    this.bboxStatus.set(`Đã xóa bbox ${index + 1}.`);
  }

  async submit(action: DetectorReviewAction): Promise<void> {
    const item = this.selected();
    if (this.destroyed || !item || this.submitting()) return;
    if (action === 'APPROVE' && !this.canApprove()) {
      this.reviewError.set(
        'Approve chỉ dùng khi bbox còn nguyên như model đề xuất.',
      );
      return;
    }
    if (action === 'CORRECT' && !this.boxes().length) {
      this.reviewError.set(
        'Hãy vẽ ít nhất một bbox hoặc chọn “Không có biển số”.',
      );
      return;
    }
    if (action === 'REJECT' && !this.note.trim()) {
      this.reviewError.set('Cần ghi rõ lý do loại ảnh.');
      return;
    }
    const generation = ++this.submissionGeneration;
    this.submitting.set(true);
    this.reviewError.set(null);
    this.success.set(null);
    try {
      const reviewed = await firstValueFrom(
        this.api.reviewDetectorSample(item.sourceId, item.reviewId, {
          action,
          expectedRevision: item.revision,
          annotations: action === 'CORRECT' ? this.boxes() : [],
          note: this.note.trim() || null,
        }),
      );
      if (this.destroyed || generation !== this.submissionGeneration) return;
      this.selected.set(reviewed);
      this.boxes.set(editableBoxes(reviewed));
      this.success.set(
        `Đã lưu ${reviewed.status} · revision ${reviewed.revision}.`,
      );
      await this.refreshAfterReview(reviewed.reviewId, generation);
    } catch (error) {
      if (this.destroyed || generation !== this.submissionGeneration) return;
      this.reviewError.set(
        apiErrorMessage(error, 'Không thể lưu quyết định detector review.'),
      );
      if (
        typeof error === 'object' &&
        error !== null &&
        'status' in error &&
        error.status === 409
      ) {
        await this.reloadSelected(item);
      }
    } finally {
      if (!this.destroyed && generation === this.submissionGeneration) {
        this.submitting.set(false);
      }
    }
  }

  async startPromotion(): Promise<void> {
    if (
      this.destroyed ||
      !this.auth.canManageDatasets() ||
      !this.sourceId ||
      this.promoting()
    ) {
      return;
    }
    if (!this.selectedSource()?.promotionEligible) {
      this.error.set(
        'Source này chỉ dành cho kiểm duyệt; cần xác minh quyền dữ liệu trước khi tạo source production.',
      );
      return;
    }
    if ((this.selectedSource()?.reviewedCount ?? 0) === 0) {
      this.error.set(
        'Cần hoàn tất ít nhất một quyết định review trước khi promote.',
      );
      return;
    }
    const target = this.targetSourceId.trim();
    if (!target) {
      this.error.set('Target source ID là bắt buộc.');
      return;
    }
    const generation = ++this.promotionGeneration;
    this.error.set(null);
    this.promotionStarting.set(true);
    try {
      const job = await firstValueFrom(
        this.api.promoteDetectorSource(this.sourceId, target),
      );
      if (this.destroyed || generation !== this.promotionGeneration) return;
      this.promotionJob.set(job);
      this.schedulePromotionPoll(job.id, generation);
    } catch (error) {
      if (!this.destroyed && generation === this.promotionGeneration) {
        this.error.set(
          apiErrorMessage(error, 'Không thể khởi tạo promotion job.'),
        );
      }
    } finally {
      if (!this.destroyed && generation === this.promotionGeneration) {
        this.promotionStarting.set(false);
      }
    }
  }

  reasonLabel(reason: string): string {
    return detectorReviewReason(reason);
  }

  reasonOptions(): string[] {
    return Object.keys(this.selectedSource()?.reasonCounts ?? {}).sort();
  }

  count(status: DetectorReviewStatus): number {
    return this.selectedSource()?.statusCounts[status] ?? 0;
  }

  confidence(item: DetectorReviewItem): number | null {
    const value = item.suggestions[0]?.attributes['confidence'];
    return typeof value === 'number' ? value : null;
  }

  private async refreshAfterReview(
    reviewId: string,
    submissionGeneration: number,
  ): Promise<void> {
    const isCurrent = (): boolean =>
      !this.destroyed && submissionGeneration === this.submissionGeneration;
    if (!isCurrent()) return;
    await this.loadItems(true);
    if (!isCurrent()) return;
    await this.loadSourcesSummaryOnly();
    if (!isCurrent()) return;
    if (this.statusFilter === 'PENDING_REVIEW') {
      const next = this.items().find((item) => item.reviewId !== reviewId);
      if (next) await this.openItem(next);
    } else {
      const latest = this.items().find((item) => item.reviewId === reviewId);
      if (latest) await this.openItem(latest);
    }
  }

  private async loadSourcesSummaryOnly(): Promise<void> {
    try {
      const result = await firstValueFrom(this.api.detectorReviewSources());
      if (!this.destroyed) this.sources.set(result.items);
    } catch {
      // The saved revision remains visible; a later refresh can recover summary counts.
    }
  }

  private async reloadSelected(item: DetectorReviewItem): Promise<void> {
    try {
      await this.openItem(item);
    } catch {
      // Keep the conflict message visible when refresh also fails.
    }
  }

  private refreshTargetSourceId(): void {
    if (!this.sourceId) {
      this.targetSourceId = '';
      return;
    }
    const versionMatch = this.sourceId.match(/^(.*?)-v(\d+)$/);
    this.targetSourceId = versionMatch
      ? `${versionMatch[1]}-v${Number(versionMatch[2]) + 1}`
      : `${this.sourceId}-reviewed-v2`;
  }

  private schedulePromotionPoll(jobId: string, generation: number): void {
    if (this.destroyed || generation !== this.promotionGeneration) return;
    if (this.promotionTimer !== null) window.clearTimeout(this.promotionTimer);
    this.promotionTimer = window.setTimeout(async () => {
      this.promotionTimer = null;
      if (this.destroyed || generation !== this.promotionGeneration) return;
      try {
        const job = await firstValueFrom(this.api.detectorPromotion(jobId));
        if (this.destroyed || generation !== this.promotionGeneration) return;
        this.promotionJob.set(job);
        if (job.status === 'QUEUED' || job.status === 'RUNNING') {
          this.schedulePromotionPoll(jobId, generation);
        } else if (job.status === 'COMPLETED') {
          await this.loadSourcesSummaryOnly();
        }
      } catch (error) {
        if (!this.destroyed && generation === this.promotionGeneration) {
          this.error.set(
            apiErrorMessage(error, 'Không thể cập nhật trạng thái promotion.'),
          );
        }
      }
    }, 1500);
  }

  private clearPromotionState(): void {
    this.promotionGeneration += 1;
    if (this.promotionTimer !== null) window.clearTimeout(this.promotionTimer);
    this.promotionTimer = null;
    this.promotionJob.set(null);
    this.promotionStarting.set(false);
  }

  private releasePreview(): void {
    if (this.previewObjectUrl !== null)
      URL.revokeObjectURL(this.previewObjectUrl);
    this.previewObjectUrl = null;
    this.previewUrl.set(null);
  }
}
