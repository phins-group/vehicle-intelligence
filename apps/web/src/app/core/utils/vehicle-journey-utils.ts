import {
  JourneyObservation,
  JourneySegment,
  VehicleJourney
} from '../models/api.models';

export interface JourneyStep {
  observation: JourneyObservation;
  segmentBefore: JourneySegment | null;
}

export interface JourneySummary {
  observationCount: number;
  cameraCount: number;
  durationSeconds: number;
  infeasibleSegments: number;
  unknownTopologySegments: number;
}

export function buildJourneySteps(journey: VehicleJourney): JourneyStep[] {
  return journey.observations.map((observation, index) => ({
    observation,
    segmentBefore: index > 0 ? (journey.segments[index - 1] ?? null) : null
  }));
}

export function summarizeJourney(journey: VehicleJourney): JourneySummary {
  const started = journey.startedAt ? Date.parse(journey.startedAt) : Number.NaN;
  const ended = journey.endedAt ? Date.parse(journey.endedAt) : Number.NaN;
  return {
    observationCount: journey.observations.length,
    cameraCount: new Set(journey.observations.map((item) => item.cameraId)).size,
    durationSeconds:
      Number.isFinite(started) && Number.isFinite(ended)
        ? Math.max(0, (ended - started) / 1000)
        : 0,
    infeasibleSegments: journey.segments.filter((item) => item.feasible === false).length,
    unknownTopologySegments: journey.segments.filter((item) => item.feasible === null).length
  };
}

export function durationLabel(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const remaining = whole % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${remaining}s`;
  return `${remaining}s`;
}
