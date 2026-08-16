import { DecimalPipe } from '@angular/common';
import {
  Component,
  ElementRef,
  Injector,
  OnInit,
  ViewChild,
  afterNextRender,
  inject,
  signal,
} from '@angular/core';
import { FormsModule, NgForm } from '@angular/forms';
import { RouterLink } from '@angular/router';
import {
  LucideCamera,
  LucideCircleAlert,
  LucideCircleCheck,
  LucideCctv,
  LucideFlaskConical,
  LucideImport,
  LucideLockKeyhole,
  LucideMapPin,
  LucideNetwork,
  LucidePlus,
  LucidePower,
  LucideRadar,
  LucideRefreshCw,
  LucideWifi,
  LucideWifiOff,
  LucideX,
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import {
  Camera,
  CameraConnectionTest,
  CameraCreateRequest,
  CameraHealth,
  OnvifDiscoveredDevice,
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import { AsyncDataState } from '../../core/utils/async-data-state';
import {
  preferredOnvifAddress,
  suggestedCameraId,
} from '../../core/utils/onvif-utils';
import { AccessibleDialogDirective } from '../../shared/accessibility/accessible-dialog.directive';

interface ManagedCamera extends Camera {
  health: CameraHealth | null;
}

interface CameraDraft {
  id: string;
  name: string;
  rtspUrl: string;
  fpsLimit: number;
  location: string;
  zone: string;
  direction: 'ENTRY' | 'EXIT' | 'BOTH';
  vehicleConfidence: number;
  plateConfidence: number;
  enabled: boolean;
}

@Component({
  selector: 'app-cameras',
  imports: [
    DecimalPipe,
    FormsModule,
    RouterLink,
    AccessibleDialogDirective,
    LucideCamera,
    LucideCircleAlert,
    LucideCircleCheck,
    LucideCctv,
    LucideFlaskConical,
    LucideImport,
    LucideLockKeyhole,
    LucideMapPin,
    LucideNetwork,
    LucidePlus,
    LucidePower,
    LucideRadar,
    LucideRefreshCw,
    LucideWifi,
    LucideWifiOff,
    LucideX,
  ],
  templateUrl: './cameras.component.html',
})
export class CamerasComponent implements OnInit {
  readonly auth = inject(AuthService);
  private readonly api = inject(ApiClientService);
  private readonly injector = inject(Injector);
  readonly cameras = signal<ManagedCamera[]>([]);
  readonly loading = signal(true);
  readonly loadState = new AsyncDataState();
  readonly actionError = signal<string | null>(null);
  readonly createError = signal<string | null>(null);
  readonly discoveryError = signal<string | null>(null);
  readonly notice = signal<string | null>(null);
  readonly createOpen = signal(false);
  readonly saving = signal(false);
  readonly discoveryOpen = signal(false);
  readonly discovering = signal(false);
  readonly discoveryRan = signal(false);
  readonly discoveredDevices = signal<OnvifDiscoveredDevice[]>([]);
  readonly selectedOnvif = signal<OnvifDiscoveredDevice | null>(null);
  readonly busyIds = signal<Set<string>>(new Set());
  readonly testResults = signal<Record<string, CameraConnectionTest>>({});
  draft: CameraDraft = this.emptyDraft();

  @ViewChild('cameraFormElement')
  private cameraFormElement?: ElementRef<HTMLFormElement>;

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.actionError.set(null);
    this.loadState.begin();
    try {
      const snapshot = await firstValueFrom(this.api.cameraHealthSnapshot());
      this.cameras.set(
        snapshot.items.map(({ camera, health }) => ({ ...camera, health })),
      );
      this.loadState.succeed();
    } catch (error) {
      this.loadState.fail(apiErrorMessage(error, 'Không thể tải camera.'));
    } finally {
      this.loading.set(false);
    }
  }

  openCreate(): void {
    this.draft = this.emptyDraft();
    this.selectedOnvif.set(null);
    this.actionError.set(null);
    this.createError.set(null);
    this.createOpen.set(true);
  }

  closeCreate(): void {
    this.draft.rtspUrl = '';
    this.selectedOnvif.set(null);
    this.createError.set(null);
    this.createOpen.set(false);
  }

  async discoverOnvif(): Promise<void> {
    if (this.discovering()) return;
    this.discoveryOpen.set(true);
    this.discovering.set(true);
    this.discoveryRan.set(false);
    this.actionError.set(null);
    this.discoveryError.set(null);
    try {
      const result = await firstValueFrom(this.api.discoverOnvifCameras());
      this.discoveredDevices.set(result.items);
      this.discoveryRan.set(true);
      this.notice.set(
        result.count
          ? `Đã tìm thấy ${result.count} thiết bị ONVIF.`
          : 'Không tìm thấy thiết bị ONVIF trong thời gian quét.',
      );
    } catch (error) {
      this.discoveryError.set(
        apiErrorMessage(error, 'Không thể quét thiết bị ONVIF.'),
      );
    } finally {
      this.discovering.set(false);
    }
  }

  closeDiscovery(): void {
    this.discoveryError.set(null);
    this.discoveryOpen.set(false);
  }

  useDiscoveredDevice(device: OnvifDiscoveredDevice): void {
    const address = preferredOnvifAddress(device);
    this.draft = {
      ...this.emptyDraft(),
      id: suggestedCameraId(device),
      name: device.name || device.hardware || 'ONVIF Camera',
      location: device.locations[0] || '',
    };
    this.selectedOnvif.set({
      ...device,
      serviceAddresses: address ? [address] : device.serviceAddresses,
    });
    this.actionError.set(null);
    this.createError.set(null);
    this.createOpen.set(true);
  }

  async createCamera(form: NgForm): Promise<void> {
    if (this.saving()) return;
    if (form.invalid) {
      form.control.markAllAsTouched();
      this.createError.set(
        'Hãy sửa các trường được đánh dấu trước khi tạo camera.',
      );
      afterNextRender(
        () =>
          this.cameraFormElement?.nativeElement
            .querySelector<HTMLElement>(':invalid, [aria-invalid="true"]')
            ?.focus(),
        { injector: this.injector },
      );
      return;
    }
    this.saving.set(true);
    this.createError.set(null);
    const request: CameraCreateRequest = {
      id: this.draft.id.trim(),
      name: this.draft.name.trim(),
      stream: { rtspUrl: this.draft.rtspUrl, fpsLimit: this.draft.fpsLimit },
      location: {
        name: this.draft.location.trim() || null,
        zone: this.draft.zone.trim() || null,
      },
      direction: this.draft.direction,
      vision: {
        vehicleConfidence: this.draft.vehicleConfidence,
        plateConfidence: this.draft.plateConfidence,
      },
      enabled: this.draft.enabled,
      metadata: this.discoveryMetadata(),
    };
    try {
      await firstValueFrom(this.api.createCamera(request));
      this.notice.set('Đã tạo camera ' + request.name + '.');
      this.closeCreate();
      await this.load();
    } catch (error) {
      this.createError.set(apiErrorMessage(error, 'Không thể tạo camera.'));
    } finally {
      this.draft.rtspUrl = '';
      this.saving.set(false);
    }
  }

  async toggle(camera: ManagedCamera): Promise<void> {
    if (this.isBusy(camera.id)) return;
    this.setBusy(camera.id, true);
    this.actionError.set(null);
    try {
      await firstValueFrom(
        this.api.setCameraEnabled(camera.id, !camera.enabled),
      );
      this.notice.set(
        (camera.enabled ? 'Đã tắt ' : 'Đã bật ') + camera.name + '.',
      );
      await this.load();
    } catch (error) {
      this.actionError.set(
        apiErrorMessage(error, 'Không thể thay đổi trạng thái camera.'),
      );
    } finally {
      this.setBusy(camera.id, false);
    }
  }

  async test(camera: ManagedCamera): Promise<void> {
    if (this.isBusy(camera.id)) return;
    this.setBusy(camera.id, true);
    this.actionError.set(null);
    try {
      const result = await firstValueFrom(this.api.testCamera(camera.id));
      this.testResults.update((items) => ({ ...items, [camera.id]: result }));
    } catch (error) {
      this.actionError.set(
        apiErrorMessage(error, 'Không thể test kết nối camera.'),
      );
    } finally {
      this.setBusy(camera.id, false);
    }
  }

  isBusy(cameraId: string): boolean {
    return this.busyIds().has(cameraId);
  }

  private setBusy(cameraId: string, busy: boolean): void {
    const current = new Set(this.busyIds());
    if (busy) current.add(cameraId);
    else current.delete(cameraId);
    this.busyIds.set(current);
  }

  private emptyDraft(): CameraDraft {
    return {
      id: '',
      name: '',
      rtspUrl: '',
      fpsLimit: 6,
      location: '',
      zone: '',
      direction: 'BOTH',
      vehicleConfidence: 0.4,
      plateConfidence: 0.45,
      enabled: true,
    };
  }

  private discoveryMetadata(): Record<string, unknown> {
    const device = this.selectedOnvif();
    if (!device) return {};
    return {
      source: 'ONVIF_DISCOVERY',
      onvif: {
        endpointReference: device.endpointReference,
        serviceAddress: preferredOnvifAddress(device),
        hardware: device.hardware,
      },
    };
  }
}
