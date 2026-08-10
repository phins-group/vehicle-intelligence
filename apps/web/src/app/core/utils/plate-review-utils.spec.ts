import { describe, expect, it } from 'vitest';

import { VehicleEvent } from '../models/api.models';
import { finalPlate, isHumanReviewed, plateReviewRevision } from './plate-review-utils';

function event(): VehicleEvent {
  return {
    _id: 'evt-review',
    schemaVersion: 2,
    camera: { id: 'gate-01', name: 'Gate 01', zone: null },
    trackId: 'gate-01:1',
    vehicleId: null,
    eventType: 'VEHICLE_ENTER',
    direction: 'ENTER',
    status: 'CONFIRMED',
    plate: {
      raw: '51H1234S',
      normalized: '51H-123.4S',
      confidence: 0.68,
      observationCount: 3,
      corrections: [],
      prediction: {
        raw: '51H1234S',
        normalized: '51H-123.4S',
        confidence: 0.68,
        observationCount: 3,
        corrections: []
      },
      review: {
        normalized: '51H-123.45',
        revision: 1,
        reviewedAt: '2026-08-09T03:00:00Z',
        reviewedBy: { id: 'operator-01', displayName: 'Gate Operator' },
        note: null
      },
      final: '51H-123.45'
    },
    vehicle: { type: 'car', confidence: 0.97, color: null },
    media: {
      snapshotKey: null,
      vehicleCropKey: null,
      plateCropKey: 'vehicles/plate.jpg',
      clipKey: null
    },
    ai: {},
    occurredAt: '2026-08-09T02:59:00Z',
    createdAt: '2026-08-09T02:59:01Z',
    metadata: {}
  };
}

describe('plate review helpers', () => {
  it('uses the human-reviewed final plate without losing the prediction', () => {
    const reviewed = event();

    expect(finalPlate(reviewed)).toBe('51H-123.45');
    expect(reviewed.plate?.prediction?.normalized).toBe('51H-123.4S');
    expect(plateReviewRevision(reviewed)).toBe(1);
    expect(isHumanReviewed(reviewed)).toBe(true);
  });

  it('falls back to the AI normalized value for legacy-shaped events', () => {
    const legacy = event();
    if (legacy.plate) {
      legacy.plate.final = '';
      legacy.plate.review = null;
    }

    expect(finalPlate(legacy)).toBe('51H-123.4S');
    expect(plateReviewRevision(legacy)).toBe(0);
  });
});
