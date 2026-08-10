import { describe, expect, it } from 'vitest';

import { VehicleEvent } from '../models/api.models';
import { chronologicalVehicleEvents, summarizePlateHistory } from './vehicle-history-utils';

function event(
  id: string,
  occurredAt: string,
  overrides: Partial<VehicleEvent> = {}
): VehicleEvent {
  return {
    _id: id,
    schemaVersion: 1,
    camera: { id: 'gate-01', name: 'Gate 01', zone: 'A' },
    trackId: 'gate-01:' + id,
    vehicleId: null,
    eventType: 'VEHICLE_ENTER',
    direction: 'ENTER',
    status: 'CONFIRMED',
    plate: {
      raw: '51H12345',
      normalized: '51H-123.45',
      confidence: 0.9,
      observationCount: 4,
      corrections: []
    },
    vehicle: { type: 'car', confidence: 0.95, color: 'white' },
    media: { snapshotKey: null, vehicleCropKey: null, plateCropKey: null, clipKey: null },
    ai: {},
    occurredAt,
    createdAt: occurredAt,
    metadata: {},
    ...overrides
  };
}

describe('vehicle history utilities', () => {
  it('orders observations from oldest to newest with a stable ID tie-breaker', () => {
    const items = [
      event('evt-c', '2026-08-09T10:01:00Z'),
      event('evt-b', '2026-08-09T10:00:00Z'),
      event('evt-a', '2026-08-09T10:00:00Z')
    ];
    expect(chronologicalVehicleEvents(items).map((item) => item._id)).toEqual([
      'evt-a',
      'evt-b',
      'evt-c'
    ]);
  });

  it('summarizes only loaded plate observations without inventing identity', () => {
    const summary = summarizePlateHistory([
      event('evt-1', '2026-08-09T10:00:00Z'),
      event('evt-2', '2026-08-09T10:05:00Z', {
        camera: { id: 'gate-02', name: 'Gate 02', zone: 'B' },
        vehicleId: 'veh-01',
        direction: 'EXIT',
        eventType: 'VEHICLE_EXIT',
        plate: {
          raw: '51H12345',
          normalized: '51H-123.45',
          confidence: 0.8,
          observationCount: 3,
          corrections: []
        }
      })
    ]);
    expect(summary).toMatchObject({
      loadedCount: 2,
      cameraCount: 2,
      entries: 1,
      exits: 1,
      logicalVehicleIdCount: 1,
      unresolvedIdentityCount: 1,
      oldestLoadedAt: '2026-08-09T10:00:00Z',
      latestLoadedAt: '2026-08-09T10:05:00Z'
    });
    expect(summary.averagePlateConfidence).toBeCloseTo(0.85);
  });

  it('uses deterministic lexical ordering when observation modes tie', () => {
    const summary = summarizePlateHistory([
      event('evt-1', '2026-08-09T10:00:00Z', { vehicle: { type: 'truck', confidence: 0.9, color: 'white' } }),
      event('evt-2', '2026-08-09T10:05:00Z', { vehicle: { type: 'car', confidence: 0.9, color: 'black' } })
    ]);
    expect(summary.commonVehicleType).toBe('car');
    expect(summary.commonColor).toBe('black');
  });

  it('returns explicit empty summary values', () => {
    expect(summarizePlateHistory([])).toEqual({
      loadedCount: 0,
      cameraCount: 0,
      entries: 0,
      exits: 0,
      unknownDirections: 0,
      latestLoadedAt: null,
      oldestLoadedAt: null,
      commonVehicleType: null,
      commonColor: null,
      averagePlateConfidence: null,
      logicalVehicleIdCount: 0,
      unresolvedIdentityCount: 0
    });
  });
});
