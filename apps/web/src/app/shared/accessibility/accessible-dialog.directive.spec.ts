import '@angular/compiler';

import { describe, expect, it } from 'vitest';

import { dialogTabTarget } from './accessible-dialog.directive';

function focusable(): HTMLElement {
  return {} as HTMLElement;
}

describe('dialogTabTarget', () => {
  it('wraps forward focus from the last element to the first', () => {
    const first = focusable();
    const last = focusable();

    expect(dialogTabTarget([first, last], last, false)).toBe(first);
  });

  it('wraps backward focus from the first element to the last', () => {
    const first = focusable();
    const last = focusable();

    expect(dialogTabTarget([first, last], first, true)).toBe(last);
  });

  it('recovers focus that is outside the dialog in the requested direction', () => {
    const first = focusable();
    const last = focusable();

    expect(dialogTabTarget([first, last], null, false)).toBe(first);
    expect(dialogTabTarget([first, last], null, true)).toBe(last);
  });

  it('allows native Tab movement between elements inside the dialog', () => {
    const first = focusable();
    const middle = focusable();
    const last = focusable();

    expect(dialogTabTarget([first, middle, last], middle, false)).toBeNull();
    expect(dialogTabTarget([first, middle, last], middle, true)).toBeNull();
  });
});
