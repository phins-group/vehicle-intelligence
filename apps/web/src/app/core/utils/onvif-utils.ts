import { OnvifDiscoveredDevice } from '../models/api.models';

export function suggestedCameraId(device: OnvifDiscoveredDevice): string {
  const source = device.name || device.hardware || device.remoteAddress || 'onvif-camera';
  const normalized = source
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 96);
  return normalized || 'onvif-camera';
}

export function preferredOnvifAddress(device: OnvifDiscoveredDevice): string | null {
  return (
    device.serviceAddresses.find((address) => address.toLowerCase().startsWith('https://')) ||
    device.serviceAddresses[0] ||
    null
  );
}
