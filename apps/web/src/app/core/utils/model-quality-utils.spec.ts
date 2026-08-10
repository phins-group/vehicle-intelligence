import { describe, expect, it } from 'vitest';

import { qualityBarWidth } from './model-quality-utils';

describe('qualityBarWidth', () => {
  it('normalizes and clamps daily event bars', () => {
    expect(qualityBarWidth(25, 100)).toBe(25);
    expect(qualityBarWidth(120, 100)).toBe(100);
    expect(qualityBarWidth(-1, 100)).toBe(0);
    expect(qualityBarWidth(1, 0)).toBe(0);
  });
});
