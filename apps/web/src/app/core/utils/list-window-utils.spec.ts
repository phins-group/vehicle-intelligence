import { describe, expect, it } from 'vitest';

import { listWindow } from './list-window-utils';

describe('listWindow', () => {
  it('keeps the rendered collection bounded without dropping the total', () => {
    const source = Array.from({ length: 1000 }, (_, index) => index);

    expect(listWindow(source, 900, 100)).toEqual({
      items: source.slice(900),
      start: 900,
      end: 1000,
      total: 1000,
      hasPrevious: true,
      hasNext: false
    });
  });

  it('clamps a stale start to the final stable page after a collection shrinks', () => {
    const source = Array.from({ length: 75 }, (_, index) => index);
    const result = listWindow(source, 900, 50);

    expect(result.items).toEqual(source.slice(50));
    expect(result.start).toBe(50);
    expect(result.end).toBe(75);
    expect(result.hasPrevious).toBe(true);
    expect(result.hasNext).toBe(false);
  });

  it('normalizes invalid bounds and handles an empty collection', () => {
    expect(listWindow([], Number.NaN, 0)).toEqual({
      items: [],
      start: 0,
      end: 0,
      total: 0,
      hasPrevious: false,
      hasNext: false
    });
  });
});
