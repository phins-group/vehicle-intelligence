import { describe, expect, it } from 'vitest';

import { LiveMonitorFrame, LiveVehicleOverlay } from '../models/api.models';
import {
  DEFAULT_OVERLAY_VISIBILITY,
  overlayLabelWidth,
  overlayPoints,
  shouldLoadLiveFrame,
  vehicleOverlayLabel
} from './live-monitor-utils';

const vehicle: LiveVehicleOverlay = {
  trackId: 'gate-01:session:12',
  bbox: [10, 20, 110, 120],
  confidence: 0.956,
  vehicleType: 'car',
  direction: 'ENTER',
  plate: {
    bbox: [40, 80, 90, 100],
    detectionConfidence: 0.9,
    qualityScore: 0.8,
    text: '51H-123.45',
    ocrConfidence: 0.92
  }
};

const frame = { sequence: 7 } as LiveMonitorFrame;

describe('live monitor overlay utilities', () => {
  it('loads only a new synchronized frame sequence', () => {
    expect(shouldLoadLiveFrame(null, frame)).toBe(true);
    expect(shouldLoadLiveFrame(6, frame)).toBe(true);
    expect(shouldLoadLiveFrame(7, frame)).toBe(false);
    expect(shouldLoadLiveFrame(7, null)).toBe(false);
  });

  it('builds deterministic geometry and labels from enabled overlays', () => {
    expect(overlayPoints([[1, 2], [3, 4]])).toBe('1,2 3,4');
    expect(vehicleOverlayLabel(vehicle, DEFAULT_OVERLAY_VISIBILITY)).toBe(
      'gate-01:session:12 · 51H-123.45 · ENTER · 96%'
    );
  });

  it('bounds label backgrounds to the source width', () => {
    expect(overlayLabelWidth('A', 1920)).toBe(96);
    expect(overlayLabelWidth('x'.repeat(300), 640)).toBe(640);
  });
});

