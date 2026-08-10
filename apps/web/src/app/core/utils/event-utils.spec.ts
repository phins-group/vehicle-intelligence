import { describe, expect, it } from 'vitest';

import { VehicleEvent } from '../models/api.models';
import { apiErrorMessage } from './api-error';
import {
  eventMatchesFilters,
  localDayStartIso,
  mergeVehicleEvents,
  parseRealtimeMessage
} from './event-utils';

function event(id: string, occurredAt: string, plate = '51H-123.45'): VehicleEvent {
  return {
    _id: id,
    schemaVersion: 1,
    camera: { id: 'gate-01', name: 'Gate 01', zone: 'A' },
    trackId: 'gate-01:1',
    vehicleId: null,
    eventType: 'VEHICLE_ENTER',
    direction: 'ENTER',
    status: 'CONFIRMED',
    plate: { raw: '51H12345', normalized: plate, confidence: 0.95, observationCount: 4, corrections: [] },
    vehicle: { type: 'car', confidence: 0.96, color: 'white' },
    media: { snapshotKey: null, vehicleCropKey: null, plateCropKey: null, clipKey: null },
    ai: {},
    occurredAt,
    createdAt: occurredAt,
    metadata: {}
  };
}

describe('event utilities', () => {
  it('deduplicates and orders realtime events newest first', () => {
    const old = event('evt_old', '2026-08-08T10:00:00Z');
    const recent = event('evt_recent', '2026-08-08T11:00:00Z');
    expect(mergeVehicleEvents([old], [old, recent]).map((item) => item._id)).toEqual([
      'evt_recent',
      'evt_old'
    ]);
  });

  it('enforces the configured in-memory bound', () => {
    const items = Array.from({ length: 5 }, (_, index) =>
      event('evt_' + index, '2026-08-08T1' + index + ':00:00Z')
    );
    expect(mergeVehicleEvents([], items, 2)).toHaveLength(2);
  });

  it('parses a canonical vehicle envelope and rejects malformed JSON', () => {
    const payload = JSON.stringify({
      id: 'evt_recent',
      type: 'vehicle.entered',
      schemaVersion: 1,
      occurredAt: '2026-08-08T11:00:00Z',
      source: 'vision-worker',
      data: event('evt_recent', '2026-08-08T11:00:00Z')
    });
    expect(parseRealtimeMessage(payload).kind).toBe('event');
    expect(parseRealtimeMessage('{broken').kind).toBe('invalid');
  });

  it('parses an explicit realtime gap control', () => {
    const payload = JSON.stringify({
      id: 'ctl_gap',
      type: 'system.realtime.gap',
      schemaVersion: 1,
      occurredAt: '2026-08-08T11:00:00Z',
      source: 'api/realtime',
      data: {
        reason: 'slow_consumer',
        droppedEvents: 2,
        lastAvailableEventId: 'evt_recent',
        recoveryEndpoint: '/api/events'
      }
    });
    expect(parseRealtimeMessage(payload).kind).toBe('gap');
  });

  it('matches normalized plate input without punctuation', () => {
    expect(eventMatchesFilters(event('evt', '2026-08-08T11:00:00Z'), { plate: '51H12345' })).toBe(true);
    expect(eventMatchesFilters(event('evt', '2026-08-08T11:00:00Z'), { cameraId: 'gate-02' })).toBe(false);
  });

  it('creates local midnight with an ISO timezone-aware value', () => {
    expect(localDayStartIso(new Date(2026, 7, 9, 18, 30))).toBe(
      new Date(2026, 7, 9).toISOString()
    );
  });

  it('maps safe API errors without exposing arbitrary response values', () => {
    expect(apiErrorMessage({ status: 0 }, 'fallback')).toBe('Không thể kết nối tới API.');
    expect(apiErrorMessage({ status: 403 }, 'fallback')).toContain('không có quyền');
    expect(apiErrorMessage({ error: { unrelated: 'secret-value' } }, 'fallback')).toBe('fallback');
  });
});
