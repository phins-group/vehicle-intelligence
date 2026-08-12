import { describe, expect, it } from 'vitest';

import {
  boxFromPoints,
  boxesMatchSuggestions,
  clampBox,
  detectorReviewReason,
  editableBoxes,
  pointerToImage
} from './dataset-review-utils';

describe('dataset review geometry', () => {
  it('projects browser coordinates to source image coordinates', () => {
    expect(
      pointerToImage(150, 100, { left: 50, top: 50, width: 200, height: 100 }, { width: 1000, height: 500 })
    ).toEqual({ x: 500, y: 250 });
  });

  it('normalizes drag direction and clamps a box to image bounds', () => {
    expect(
      boxFromPoints({ x: 90, y: 80 }, { x: -10, y: 10 }, { width: 80, height: 60 })
    ).toEqual({ x: 0, y: 10, width: 80, height: 50 });
    expect(clampBox({ x: 75, y: 55, width: 30, height: 20 }, { width: 80, height: 60 })).toEqual({
      x: 75,
      y: 55,
      width: 5,
      height: 5
    });
  });

  it('keeps human decisions ahead of model suggestions', () => {
    const item = {
      suggestions: [{ className: 'license_plate' as const, bbox: { x: 1, y: 2, width: 3, height: 4 }, attributes: {} }],
      decision: {
        annotations: [{ className: 'license_plate' as const, bbox: { x: 5, y: 6, width: 7, height: 8 }, attributes: {} }]
      }
    } as Parameters<typeof editableBoxes>[0];
    expect(editableBoxes(item)).toEqual([{ x: 5, y: 6, width: 7, height: 8 }]);
  });

  it('only enables approve when the boxes still match model suggestions', () => {
    const suggestion = [{ className: 'license_plate' as const, bbox: { x: 1, y: 2, width: 3, height: 4 }, attributes: {} }];
    expect(boxesMatchSuggestions([{ x: 1, y: 2, width: 3, height: 4 }], suggestion)).toBe(true);
    expect(boxesMatchSuggestions([{ x: 1, y: 2, width: 4, height: 4 }], suggestion)).toBe(false);
  });

  it('labels video extraction suggestions as human-review work', () => {
    expect(detectorReviewReason('VIDEO_MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW')).toBe(
      'Video mới — model đề xuất, cần duyệt'
    );
  });
});
