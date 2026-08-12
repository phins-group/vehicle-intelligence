import {
  DetectorReviewAnnotation,
  DetectorReviewBox,
  DetectorReviewItem
} from '../models/api.models';

export interface ImageDimensions {
  width: number;
  height: number;
}

export interface CanvasRectangle {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface CanvasPoint {
  x: number;
  y: number;
}

export function editableBoxes(item: DetectorReviewItem): DetectorReviewBox[] {
  const annotations = item.decision?.annotations ?? item.suggestions;
  return annotations.map((annotation) => ({ ...annotation.bbox }));
}

export function pointerToImage(
  clientX: number,
  clientY: number,
  canvas: CanvasRectangle,
  image: ImageDimensions
): CanvasPoint {
  const horizontal = canvas.width > 0 ? (clientX - canvas.left) / canvas.width : 0;
  const vertical = canvas.height > 0 ? (clientY - canvas.top) / canvas.height : 0;
  return {
    x: clamp(horizontal * image.width, 0, image.width),
    y: clamp(vertical * image.height, 0, image.height)
  };
}

export function boxFromPoints(
  start: CanvasPoint,
  end: CanvasPoint,
  image: ImageDimensions,
  minimumSize = 2
): DetectorReviewBox | null {
  const x1 = clamp(Math.min(start.x, end.x), 0, image.width);
  const y1 = clamp(Math.min(start.y, end.y), 0, image.height);
  const x2 = clamp(Math.max(start.x, end.x), 0, image.width);
  const y2 = clamp(Math.max(start.y, end.y), 0, image.height);
  if (x2 - x1 < minimumSize || y2 - y1 < minimumSize) return null;
  return {
    x: round(x1),
    y: round(y1),
    width: round(x2 - x1),
    height: round(y2 - y1)
  };
}

export function clampBox(box: DetectorReviewBox, image: ImageDimensions): DetectorReviewBox {
  const x = clamp(finite(box.x), 0, Math.max(0, image.width - 1));
  const y = clamp(finite(box.y), 0, Math.max(0, image.height - 1));
  const width = clamp(finite(box.width), 1, image.width - x);
  const height = clamp(finite(box.height), 1, image.height - y);
  return { x: round(x), y: round(y), width: round(width), height: round(height) };
}

export function boxesMatchSuggestions(
  boxes: DetectorReviewBox[],
  suggestions: DetectorReviewAnnotation[],
  tolerance = 0.01
): boolean {
  if (!suggestions.length || boxes.length !== suggestions.length) return false;
  return boxes.every((box, index) => {
    const expected = suggestions[index].bbox;
    return (['x', 'y', 'width', 'height'] as const).every(
      (field) => Math.abs(box[field] - expected[field]) <= tolerance
    );
  });
}

export function detectorReviewReason(reason: string): string {
  return (
    {
      MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW: 'Model đề xuất — cần xác nhận',
      AUTO_LABEL_CONFLICT_REQUIRES_HUMAN_REVIEW: 'Nhãn model xung đột',
      MISSING_VERIFIED_ANNOTATION: 'Chưa có nhãn được xác minh'
    }[reason] ?? reason
  );
}

function finite(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}
