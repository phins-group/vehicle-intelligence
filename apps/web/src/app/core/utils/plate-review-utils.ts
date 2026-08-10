import { VehicleEvent } from '../models/api.models';

export function finalPlate(event: VehicleEvent): string | null {
  if (!event.plate) return null;
  return event.plate.final || event.plate.normalized;
}

export function plateReviewRevision(event: VehicleEvent): number {
  return event.plate?.review?.revision ?? 0;
}

export function isHumanReviewed(event: VehicleEvent): boolean {
  return plateReviewRevision(event) > 0;
}
