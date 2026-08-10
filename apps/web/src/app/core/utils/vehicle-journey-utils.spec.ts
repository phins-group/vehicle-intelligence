import { describe, expect, it } from 'vitest';

import { VehicleJourney } from '../models/api.models';
import {
  buildJourneySteps,
  durationLabel,
  summarizeJourney
} from './vehicle-journey-utils';

const journey: VehicleJourney = {
  vehicleId: 'veh-1',
  startedAt: '2026-08-10T08:00:00Z',
  endedAt: '2026-08-10T08:05:00Z',
  truncated: false,
  observations: [
    {
      eventId: 'evt-a',
      cameraId: 'a',
      cameraName: 'Gate A',
      zone: null,
      occurredAt: '2026-08-10T08:00:00Z',
      eventType: 'VEHICLE_ENTER',
      direction: 'ENTER',
      status: 'CONFIRMED',
      plate: '51H-123.45',
      vehicleType: 'car'
    },
    {
      eventId: 'evt-b',
      cameraId: 'b',
      cameraName: 'Warehouse',
      zone: null,
      occurredAt: '2026-08-10T08:05:00Z',
      eventType: 'VEHICLE_DETECTED',
      direction: 'UNKNOWN',
      status: 'CONFIRMED',
      plate: '51H-123.45',
      vehicleType: 'car'
    }
  ],
  segments: [
    {
      fromEventId: 'evt-a',
      toEventId: 'evt-b',
      fromCameraId: 'a',
      toCameraId: 'b',
      departedAt: '2026-08-10T08:00:00Z',
      arrivedAt: '2026-08-10T08:05:00Z',
      elapsedSeconds: 300,
      topologyEdgeId: 'a-b',
      expectedMinimumSeconds: 60,
      expectedMaximumSeconds: 600,
      feasible: true
    }
  ]
};

describe('vehicle journey utilities', () => {
  it('aligns each segment with the arrival observation', () => {
    const steps = buildJourneySteps(journey);
    expect(steps[0].segmentBefore).toBeNull();
    expect(steps[1].segmentBefore?.topologyEdgeId).toBe('a-b');
  });

  it('summarizes loaded journey without inventing topology', () => {
    expect(summarizeJourney(journey)).toEqual({
      observationCount: 2,
      cameraCount: 2,
      durationSeconds: 300,
      infeasibleSegments: 0,
      unknownTopologySegments: 0
    });
    expect(durationLabel(300)).toBe('5m 0s');
  });
});
