import { VehicleEvent } from '../models/api.models';

export interface PlateHistorySummary {
  loadedCount: number;
  cameraCount: number;
  entries: number;
  exits: number;
  unknownDirections: number;
  latestLoadedAt: string | null;
  oldestLoadedAt: string | null;
  commonVehicleType: string | null;
  commonColor: string | null;
  averagePlateConfidence: number | null;
  logicalVehicleIdCount: number;
  unresolvedIdentityCount: number;
}

export function summarizePlateHistory(events: readonly VehicleEvent[]): PlateHistorySummary {
  const chronological = chronologicalVehicleEvents(events);
  const confidences = events
    .map((event) => event.plate?.confidence)
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  const vehicleIds = new Set(
    events.map((event) => event.vehicleId).filter((value): value is string => Boolean(value))
  );
  return {
    loadedCount: events.length,
    cameraCount: new Set(events.map((event) => event.camera.id)).size,
    entries: events.filter((event) => event.direction === 'ENTER').length,
    exits: events.filter((event) => event.direction === 'EXIT').length,
    unknownDirections: events.filter((event) => event.direction === 'UNKNOWN').length,
    latestLoadedAt: chronological.at(-1)?.occurredAt ?? null,
    oldestLoadedAt: chronological[0]?.occurredAt ?? null,
    commonVehicleType: mostCommon(events.map((event) => event.vehicle.type)),
    commonColor: mostCommon(
      events.map((event) => event.vehicle.color).filter((value): value is string => Boolean(value))
    ),
    averagePlateConfidence: confidences.length
      ? confidences.reduce((total, value) => total + value, 0) / confidences.length
      : null,
    logicalVehicleIdCount: vehicleIds.size,
    unresolvedIdentityCount: events.filter((event) => event.vehicleId === null).length
  };
}

export function chronologicalVehicleEvents(events: readonly VehicleEvent[]): VehicleEvent[] {
  return [...events].sort(
    (left, right) =>
      Date.parse(left.occurredAt) - Date.parse(right.occurredAt) ||
      left._id.localeCompare(right._id)
  );
}

function mostCommon(values: readonly string[]): string | null {
  if (!values.length) return null;
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()]
    .sort(([leftValue, leftCount], [rightValue, rightCount]) =>
      rightCount - leftCount || leftValue.localeCompare(rightValue)
    )[0]?.[0] ?? null;
}
