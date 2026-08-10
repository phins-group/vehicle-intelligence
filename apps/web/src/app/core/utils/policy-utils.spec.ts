import { describe, expect, it } from 'vitest';

import { RuleAction, RuleCondition, WatchlistEntry } from '../models/api.models';
import {
  datetimeLocalToIso,
  isValidExternalActionUrl,
  parseListInput,
  ruleActionIsValid,
  ruleConditionIsValid,
  watchlistLifecycle,
  watchlistMatchesSearch
} from './policy-utils';

function entry(overrides: Partial<WatchlistEntry> = {}): WatchlistEntry {
  return {
    id: 'blocked-51h-12345',
    schemaVersion: 1,
    revision: 1,
    plate: '51H-123.45',
    listType: 'BLACKLIST',
    enabled: true,
    validFrom: null,
    validUntil: null,
    metadata: {},
    createdAt: '2026-08-09T00:00:00Z',
    updatedAt: '2026-08-09T00:00:00Z',
    ...overrides
  };
}

describe('policy utilities', () => {
  const now = new Date('2026-08-09T05:00:00Z');

  it('derives watchlist lifecycle from enable and validity windows', () => {
    expect(watchlistLifecycle(entry(), now)).toBe('ACTIVE');
    expect(watchlistLifecycle(entry({ enabled: false }), now)).toBe('DISABLED');
    expect(
      watchlistLifecycle(entry({ validFrom: '2026-08-09T06:00:00Z' }), now)
    ).toBe('SCHEDULED');
    expect(
      watchlistLifecycle(entry({ validUntil: '2026-08-09T04:59:59Z' }), now)
    ).toBe('EXPIRED');
  });

  it('matches canonical plates without punctuation and also matches IDs', () => {
    expect(watchlistMatchesSearch(entry(), '51h12345')).toBe(true);
    expect(watchlistMatchesSearch(entry(), 'blocked-51h')).toBe(true);
    expect(watchlistMatchesSearch(entry(), '59A')).toBe(false);
  });

  it('converts a local datetime to an aware ISO timestamp and rejects invalid input', () => {
    expect(datetimeLocalToIso('')).toBeNull();
    expect(datetimeLocalToIso('invalid')).toBeNull();
    expect(datetimeLocalToIso('2026-08-09T12:00')).toMatch(/^2026-08-09T\d{2}:00:00\.000Z$/);
  });

  it('turns comma, semicolon, and newline input into a deduplicated list', () => {
    expect(parseListInput('ENTER, EXIT; ENTER\nUNKNOWN')).toEqual([
      'ENTER',
      'EXIT',
      'UNKNOWN'
    ]);
  });

  it('only accepts safe external action URLs', () => {
    expect(isValidExternalActionUrl('https://barrier.internal/open')).toBe(true);
    expect(isValidExternalActionUrl('https://user:secret@barrier.internal/open')).toBe(false);
    expect(isValidExternalActionUrl('file:///tmp/open')).toBe(false);
  });

  it('validates structured condition value shapes', () => {
    const condition = (value: Partial<RuleCondition>): RuleCondition => ({
      field: 'direction',
      operator: 'EQ',
      value: 'ENTER',
      ...value
    });
    expect(ruleConditionIsValid(condition({}))).toBe(true);
    expect(ruleConditionIsValid(condition({ operator: 'IN', value: ['ENTER', 'EXIT'] }))).toBe(true);
    expect(ruleConditionIsValid(condition({ operator: 'IN', value: [] }))).toBe(false);
    expect(
      ruleConditionIsValid(condition({ field: 'camera.id', operator: 'CONTAINS' }))
    ).toBe(false);
    expect(ruleConditionIsValid(condition({ operator: 'EXISTS', value: true }))).toBe(true);
  });

  it('validates alert and external actions without accepting arbitrary types', () => {
    const action = (value: Partial<RuleAction>): RuleAction => ({
      id: 'action-1',
      type: 'LOG',
      parameters: {},
      ...value
    });
    expect(ruleActionIsValid(action({}))).toBe(true);
    expect(
      ruleActionIsValid(
        action({ type: 'CREATE_ALERT', parameters: { severity: 'CRITICAL', message: 'Match' } })
      )
    ).toBe(true);
    expect(
      ruleActionIsValid(
        action({ type: 'CREATE_ALERT', parameters: { severity: 'SEVERE' } })
      )
    ).toBe(false);
    expect(
      ruleActionIsValid(
        action({
          type: 'OPEN_BARRIER',
          parameters: { url: 'https://barrier.internal/open', method: 'POST' }
        })
      )
    ).toBe(true);
  });
});
