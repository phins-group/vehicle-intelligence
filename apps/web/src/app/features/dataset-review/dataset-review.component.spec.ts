import '@angular/compiler';

import { Injector, runInInjectionContext } from '@angular/core';
import { of, Subject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AuthService } from '../../core/auth/auth.service';
import {
  DetectorPromotionJob,
  DetectorReviewDecision,
  DetectorReviewItem,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { DatasetReviewComponent } from './dataset-review.component';

interface ReviewResponse {
  detail: Subject<DetectorReviewItem>;
  image: Subject<Blob>;
  history: Subject<{ items: DetectorReviewDecision[] }>;
}

interface TestableDatasetReview {
  schedulePromotionPoll(jobId: string, generation: number): void;
}

function reviewItem(id: string): DetectorReviewItem {
  return {
    sourceId: 'review-v1',
    reviewId: id,
    sourceImageSha256: `source-${id}`,
    sourceFilenameSha256: `filename-${id}`,
    reason: 'LOW_CONFIDENCE',
    status: 'PENDING_REVIEW',
    revision: 1,
    suggestions: [],
    decision: null,
    imageUrl: `/review/${id}/image`,
    image: { width: 640, height: 480 },
  };
}

function promotionJob(
  status: DetectorPromotionJob['status'],
): DetectorPromotionJob {
  return {
    id: 'promotion-1',
    sourceId: 'review-v1',
    targetSourceId: 'review-v2',
    status,
    createdAt: '2026-08-16T00:00:00Z',
    updatedAt: '2026-08-16T00:00:00Z',
    requestedBy: 'reviewer',
    reviewedSampleCount: 1,
    pendingSampleCount: 0,
    decisionSnapshotSha256: 'snapshot-sha',
    outputDirectory: null,
    manifestSha256: null,
    errorCode: null,
  };
}

function createComponent(api: unknown): DatasetReviewComponent {
  const injector = Injector.create({
    providers: [
      { provide: ApiClientService, useValue: api },
      {
        provide: AuthService,
        useValue: {
          canReviewDatasets: () => true,
          canManageDatasets: () => true,
        },
      },
    ],
  });
  return runInInjectionContext(injector, () => new DatasetReviewComponent());
}

describe('DatasetReviewComponent lifecycle', () => {
  const createObjectUrl = vi.fn(
    (blob: Blob) => `blob:${blob.size}:${createObjectUrl.mock.calls.length}`,
  );
  const revokeObjectUrl = vi.fn();

  beforeEach(() => {
    vi.useFakeTimers();
    createObjectUrl.mockClear();
    revokeObjectUrl.mockClear();
    vi.stubGlobal('window', {
      setTimeout: globalThis.setTimeout,
      clearTimeout: globalThis.clearTimeout,
    });
    vi.stubGlobal('URL', {
      createObjectURL: createObjectUrl,
      revokeObjectURL: revokeObjectUrl,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('opens the source with the largest pending queue by default', async () => {
    const source = (sourceId: string, pendingCount: number) => ({
      sourceId,
      sourceManifestSha256: `${sourceId}-manifest`,
      sourceType: 'FIRST_PARTY_DETECTOR_SOURCE',
      collectionMethod: 'FIRST_PARTY_USER_COLLECTED',
      rightsStatus: 'REVIEW_REQUIRED',
      promotionEligible: false,
      releaseEligible: false,
      distributionEligible: false,
      queueCount: pendingCount,
      statusCounts: { PENDING_REVIEW: pendingCount },
      reasonCounts: {},
      reviewedCount: 0,
      pendingCount,
    });
    const api = {
      detectorReviewSources: vi.fn(() =>
        of({
          items: [
            source('production-v3', 0),
            source('traffic-review-v1', 626),
            source('warehouse-review-v1', 2_900),
          ],
        }),
      ),
      detectorReviewItems: vi.fn(() =>
        of({ items: [], nextCursor: null }),
      ),
    };
    const component = createComponent(api);

    await component.loadSources();

    expect(component.sourceId).toBe('warehouse-review-v1');
    expect(api.detectorReviewItems).toHaveBeenCalledWith({
      sourceId: 'warehouse-review-v1',
      limit: 50,
      cursor: null,
      status: 'PENDING_REVIEW',
      reason: '',
    });
    component.ngOnDestroy();
  });

  it('keeps the newest item when an older detail response arrives last', async () => {
    const responses = new Map<string, ReviewResponse>();
    for (const id of ['item-a', 'item-b']) {
      responses.set(id, {
        detail: new Subject<DetectorReviewItem>(),
        image: new Subject<Blob>(),
        history: new Subject<{ items: DetectorReviewDecision[] }>(),
      });
    }
    const api = {
      detectorReviewItem: vi.fn(
        (_sourceId: string, id: string) => responses.get(id)!.detail,
      ),
      detectorReviewImage: vi.fn(
        (_sourceId: string, id: string) => responses.get(id)!.image,
      ),
      detectorReviewHistory: vi.fn(
        (_sourceId: string, id: string) => responses.get(id)!.history,
      ),
    };
    const component = createComponent(api);
    component.sourceId = 'review-v1';
    const itemA = reviewItem('item-a');
    const itemB = reviewItem('item-b');

    const pendingA = component.openItem(itemA);
    const pendingB = component.openItem(itemB);
    responses.get('item-b')!.detail.next(itemB);
    responses.get('item-b')!.image.next(new Blob(['item-b']));
    responses.get('item-b')!.history.next({ items: [] });
    await pendingB;
    expect(component.selected()?.reviewId).toBe('item-b');
    expect(component.previewUrl()).toContain('blob:');

    responses.get('item-a')!.detail.next(itemA);
    responses.get('item-a')!.image.next(new Blob(['item-a']));
    responses.get('item-a')!.history.next({ items: [] });
    await pendingA;
    expect(component.selected()?.reviewId).toBe('item-b');
    expect(createObjectUrl).toHaveBeenCalledOnce();
    component.ngOnDestroy();
  });

  it('does not create a late preview after close or destroy', async () => {
    const detail = new Subject<DetectorReviewItem>();
    const image = new Subject<Blob>();
    const history = new Subject<{ items: DetectorReviewDecision[] }>();
    const api = {
      detectorReviewItem: vi.fn(() => detail.asObservable()),
      detectorReviewImage: vi.fn(() => image.asObservable()),
      detectorReviewHistory: vi.fn(() => history.asObservable()),
    };
    const component = createComponent(api);
    component.sourceId = 'review-v1';
    const item = reviewItem('item-a');

    const pending = component.openItem(item);
    component.closeItem();
    detail.next(item);
    image.next(new Blob(['late-image']));
    history.next({ items: [] });
    await pending;
    expect(component.selected()).toBeNull();
    expect(createObjectUrl).not.toHaveBeenCalled();

    component.ngOnDestroy();
  });

  it('does not update or reschedule a promotion poll after destroy', async () => {
    const response = new Subject<DetectorPromotionJob>();
    const api = { detectorPromotion: vi.fn(() => response.asObservable()) };
    const component = createComponent(api);
    (component as unknown as TestableDatasetReview).schedulePromotionPoll(
      'promotion-1',
      0,
    );

    vi.advanceTimersByTime(1500);
    expect(api.detectorPromotion).toHaveBeenCalledOnce();
    component.ngOnDestroy();
    response.next(promotionJob('RUNNING'));
    response.complete();
    await Promise.resolve();
    vi.advanceTimersByTime(10_000);

    expect(component.promotionJob()).toBeNull();
    expect(api.detectorPromotion).toHaveBeenCalledOnce();
  });

  it('does not reopen another item when the dialog closes during post-submit refresh', async () => {
    const reviewResponse = new Subject<DetectorReviewItem>();
    const page = new Subject<{
      items: DetectorReviewItem[];
      nextCursor: null;
    }>();
    const reviewed = reviewItem('item-a');
    const api = {
      reviewDetectorSample: vi.fn(() => reviewResponse.asObservable()),
      detectorReviewItems: vi.fn(() => page.asObservable()),
      detectorReviewSources: vi.fn(),
      detectorReviewItem: vi.fn(),
      detectorReviewImage: vi.fn(),
      detectorReviewHistory: vi.fn(),
    };
    const component = createComponent(api);
    component.sourceId = 'review-v1';
    component.selected.set(reviewed);
    component.note = 'reject duplicate frame';

    const pending = component.submit('REJECT');
    reviewResponse.next(reviewed);
    reviewResponse.complete();
    await Promise.resolve();
    expect(api.detectorReviewItems).toHaveBeenCalledOnce();
    component.closeItem();
    page.next({ items: [reviewItem('item-b')], nextCursor: null });
    page.complete();
    await pending;

    expect(component.selected()).toBeNull();
    expect(api.detectorReviewSources).not.toHaveBeenCalled();
    expect(api.detectorReviewItem).not.toHaveBeenCalled();
    component.ngOnDestroy();
  });

  it('rejects an item left over from the previous review source', async () => {
    const api = {
      detectorReviewItem: vi.fn(),
      detectorReviewImage: vi.fn(),
      detectorReviewHistory: vi.fn(),
    };
    const component = createComponent(api);
    component.sourceId = 'review-v2';

    await component.openItem(reviewItem('old-item'));

    expect(api.detectorReviewItem).not.toHaveBeenCalled();
    expect(component.selected()).toBeNull();
    component.ngOnDestroy();
  });

  it('supports creating, selecting, nudging and deleting a bbox from the keyboard', () => {
    const component = createComponent({});
    component.selected.set(reviewItem('keyboard-item'));
    component.addKeyboardBox();

    expect(component.boxes()).toEqual([
      { x: 224, y: 204, width: 192, height: 72 },
    ]);
    expect(component.selectedBox()).toBe(0);
    expect(component.boxAccessibleLabel(component.boxes()[0]!, 0)).toContain(
      'BBox 1',
    );

    const preventNudge = vi.fn();
    component.boxKeydown(0, {
      key: 'ArrowRight',
      shiftKey: true,
      isComposing: false,
      defaultPrevented: false,
      preventDefault: preventNudge,
    } as unknown as KeyboardEvent);
    expect(preventNudge).toHaveBeenCalledOnce();
    expect(component.boxes()[0]?.x).toBe(234);

    const focusFallback = vi.fn();
    component.boxKeydown(0, {
      key: 'Delete',
      shiftKey: false,
      isComposing: false,
      defaultPrevented: false,
      preventDefault: vi.fn(),
      currentTarget: {
        ownerDocument: {
          getElementById: () => ({ focus: focusFallback }),
        },
      },
    } as unknown as KeyboardEvent);
    expect(component.boxes()).toEqual([]);
    expect(focusFallback).toHaveBeenCalledOnce();
    expect(component.bboxStatus()).toBe('Đã xóa bbox 1.');
    component.ngOnDestroy();
  });
});
