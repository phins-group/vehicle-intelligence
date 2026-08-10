import { LiveMonitorFrame, LiveVehicleOverlay } from '../models/api.models';

export interface OverlayVisibility {
  vehicleBoxes: boolean;
  plateBoxes: boolean;
  trackIds: boolean;
  plateText: boolean;
  confidence: boolean;
  direction: boolean;
  roi: boolean;
  crossingLine: boolean;
}

export const DEFAULT_OVERLAY_VISIBILITY: OverlayVisibility = {
  vehicleBoxes: true,
  plateBoxes: true,
  trackIds: true,
  plateText: true,
  confidence: true,
  direction: true,
  roi: true,
  crossingLine: true
};

export function shouldLoadLiveFrame(
  renderedSequence: number | null,
  frame: LiveMonitorFrame | null
): boolean {
  return frame !== null && frame.sequence !== renderedSequence;
}

export function overlayPoints(points: [number, number][] | null): string {
  return (points ?? []).map(([x, y]) => `${x},${y}`).join(' ');
}

export function vehicleOverlayLabel(
  vehicle: LiveVehicleOverlay,
  visibility: OverlayVisibility
): string {
  const parts: string[] = [];
  if (visibility.trackIds) parts.push(vehicle.trackId);
  if (visibility.plateText && vehicle.plate?.text) parts.push(vehicle.plate.text);
  if (visibility.direction && vehicle.direction !== 'UNKNOWN') parts.push(vehicle.direction);
  if (visibility.confidence) parts.push(`${Math.round(vehicle.confidence * 100)}%`);
  return parts.join(' · ');
}

export function overlayLabelWidth(label: string, sourceWidth: number): number {
  return Math.min(Math.max(96, label.length * 8 + 16), Math.max(96, sourceWidth));
}

