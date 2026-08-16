import { describe, expect, it } from 'vitest';

import { AsyncDataState } from './async-data-state';

describe('AsyncDataState', () => {
  it('distinguishes an initial failure from a stale refresh failure', () => {
    const state = new AsyncDataState();

    state.fail('initial failure');
    expect(state.hasLoaded()).toBe(false);
    expect(state.initialError()).toBe('initial failure');
    expect(state.staleError()).toBeNull();

    state.begin();
    state.succeed();
    state.fail('refresh failure');

    expect(state.hasLoaded()).toBe(true);
    expect(state.initialError()).toBeNull();
    expect(state.staleError()).toBe('refresh failure');
  });

  it('clears an earlier error when a retry begins and succeeds', () => {
    const state = new AsyncDataState();
    state.fail('offline');

    state.begin();
    expect(state.error()).toBeNull();

    state.succeed();
    expect(state.hasLoaded()).toBe(true);
    expect(state.error()).toBeNull();
  });
});
