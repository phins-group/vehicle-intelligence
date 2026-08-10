import { describe, expect, it } from 'vitest';

import { OnvifDiscoveredDevice } from '../models/api.models';
import { preferredOnvifAddress, suggestedCameraId } from './onvif-utils';

function device(overrides: Partial<OnvifDiscoveredDevice> = {}): OnvifDiscoveredDevice {
  return {
    endpointReference: 'urn:uuid:camera-01',
    serviceAddresses: ['http://192.0.2.10/onvif', 'https://192.0.2.10/onvif'],
    types: ['tds:Device'],
    scopes: [],
    remoteAddress: '192.0.2.10',
    name: 'Cổng Chính số 1',
    hardware: 'IPC-42',
    locations: ['Factory/Gate-A'],
    metadataVersion: 1,
    discoveredAt: '2026-08-09T00:00:00Z',
    ...overrides
  };
}

describe('ONVIF camera suggestions', () => {
  it('creates a path-safe deterministic camera id', () => {
    expect(suggestedCameraId(device())).toBe('cong-chinh-so-1');
  });

  it('falls back when device metadata is absent', () => {
    expect(
      suggestedCameraId(device({ name: null, hardware: null, remoteAddress: null }))
    ).toBe('onvif-camera');
  });

  it('prefers an HTTPS device service address', () => {
    expect(preferredOnvifAddress(device())).toBe('https://192.0.2.10/onvif');
  });
});
