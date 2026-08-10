import { EventFilters, RealtimeEnvelope, RealtimeGap, VehicleEvent } from '../models/api.models';
import { finalPlate } from './plate-review-utils';

const REALTIME_EVENT_TYPES = new Set([
  'vehicle.detected',
  'vehicle.entered',
  'vehicle.exited'
]);

export type ParsedRealtimeMessage =
  | { kind: 'event'; envelope: RealtimeEnvelope<VehicleEvent> }
  | { kind: 'gap'; envelope: RealtimeEnvelope<RealtimeGap> }
  | { kind: 'control'; envelope: RealtimeEnvelope<unknown> }
  | { kind: 'invalid' };

export function parseRealtimeMessage(raw: string): ParsedRealtimeMessage {
  try {
    const value = JSON.parse(raw) as Partial<RealtimeEnvelope<unknown>>;
    if (
      typeof value.id !== 'string' ||
      typeof value.type !== 'string' ||
      typeof value.data !== 'object' ||
      value.data === null
    ) {
      return { kind: 'invalid' };
    }
    const envelope = value as RealtimeEnvelope<unknown>;
    if (REALTIME_EVENT_TYPES.has(envelope.type) && isVehicleEvent(envelope.data)) {
      return {
        kind: 'event',
        envelope: envelope as RealtimeEnvelope<VehicleEvent>
      };
    }
    if (envelope.type === 'system.realtime.gap') {
      return {
        kind: 'gap',
        envelope: envelope as RealtimeEnvelope<RealtimeGap>
      };
    }
    return { kind: 'control', envelope };
  } catch {
    return { kind: 'invalid' };
  }
}

export function mergeVehicleEvents(
  current: readonly VehicleEvent[],
  incoming: readonly VehicleEvent[],
  maximum = 200
): VehicleEvent[] {
  const byId = new Map<string, VehicleEvent>();
  for (const event of [...incoming, ...current]) {
    if (!byId.has(event._id)) {
      byId.set(event._id, event);
    }
  }
  return [...byId.values()]
    .sort((left, right) => Date.parse(right.occurredAt) - Date.parse(left.occurredAt))
    .slice(0, Math.max(0, maximum));
}

export function eventMatchesFilters(event: VehicleEvent, filters: EventFilters): boolean {
  const plate = filters.plate?.trim().toLocaleUpperCase();
  if (filters.cameraId && event.camera.id !== filters.cameraId) return false;
  if (filters.eventType && event.eventType !== filters.eventType) return false;
  if (filters.direction && event.direction !== filters.direction) return false;
  if (filters.status && event.status !== filters.status) return false;
  if (plate && !finalPlate(event)?.replace(/[-. ]/g, '').includes(plate.replace(/[-. ]/g, ''))) {
    return false;
  }
  if (filters.from && Date.parse(event.occurredAt) < Date.parse(filters.from)) return false;
  if (filters.to && Date.parse(event.occurredAt) > Date.parse(filters.to)) return false;
  return true;
}

export function localDayStartIso(now = new Date()): string {
  return new Date(now.getFullYear(), now.getMonth(), now.getDate()).toISOString();
}

function isVehicleEvent(value: unknown): value is VehicleEvent {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Partial<VehicleEvent>;
  return (
    typeof candidate._id === 'string' &&
    typeof candidate.occurredAt === 'string' &&
    typeof candidate.camera === 'object' &&
    candidate.camera !== null
  );
}
