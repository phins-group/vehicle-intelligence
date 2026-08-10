import { describe, expect, it } from 'vitest';

import { EventMediaAccess } from '../models/api.models';
import { displayMediaAssets, mediaRefreshDelay } from './media-access-utils';

const access: EventMediaAccess = {
  eventId: 'evt-media',
  expiresAt: '2026-08-09T12:05:00Z',
  media: {
    snapshot: {
      key: 'vehicles/test/snapshot.jpg',
      url: 'https://media.example/snapshot?signature=one',
      contentType: 'image/jpeg',
      status: 'AVAILABLE'
    },
    vehicleCrop: null,
    plateCrop: {
      key: 'vehicles/test/plate.jpg',
      url: null,
      contentType: 'image/jpeg',
      status: 'MISSING'
    },
    clip: {
      key: 'vehicles/test/event.mp4',
      url: 'https://media.example/clip?signature=two',
      contentType: 'video/mp4',
      status: 'AVAILABLE'
    }
  }
};

describe('media access utilities', () => {
  it('keeps canonical slot order and excludes absent references', () => {
    expect(displayMediaAssets(access).map((item) => item.slot)).toEqual([
      'snapshot',
      'plateCrop',
      'clip'
    ]);
  });

  it('retains missing objects so the UI can show durable evidence gaps', () => {
    const plate = displayMediaAssets(access)[1];
    expect(plate.label).toBe('Biển số');
    expect(plate.asset.status).toBe('MISSING');
  });

  it('refreshes thirty seconds before expiry', () => {
    expect(mediaRefreshDelay(access.expiresAt, Date.parse('2026-08-09T12:00:00Z'))).toBe(
      270_000
    );
  });

  it('uses a bounded retry for expired URLs and ignores invalid timestamps', () => {
    expect(mediaRefreshDelay(access.expiresAt, Date.parse('2026-08-09T12:06:00Z'))).toBe(1_000);
    expect(mediaRefreshDelay('invalid', 0)).toBeNull();
  });
});
