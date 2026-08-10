import { EventMediaAccess, SignedMediaAsset } from '../models/api.models';

export type MediaSlot = 'snapshot' | 'vehicleCrop' | 'plateCrop' | 'clip';

export interface DisplayMediaAsset {
  slot: MediaSlot;
  label: string;
  asset: SignedMediaAsset;
}

const MEDIA_SLOTS: ReadonlyArray<{ slot: MediaSlot; label: string }> = [
  { slot: 'snapshot', label: 'Toàn cảnh sự kiện' },
  { slot: 'vehicleCrop', label: 'Phương tiện' },
  { slot: 'plateCrop', label: 'Biển số' },
  { slot: 'clip', label: 'Event clip' }
];

export function displayMediaAssets(access: EventMediaAccess | null): DisplayMediaAsset[] {
  if (!access) return [];
  return MEDIA_SLOTS.flatMap(({ slot, label }) => {
    const asset = access.media[slot];
    return asset ? [{ slot, label, asset }] : [];
  });
}

export function mediaRefreshDelay(
  expiresAt: string,
  nowMilliseconds: number,
  safetyWindowMilliseconds = 30_000
): number | null {
  const expires = Date.parse(expiresAt);
  if (!Number.isFinite(expires)) return null;
  return Math.max(1_000, expires - nowMilliseconds - safetyWindowMilliseconds);
}
