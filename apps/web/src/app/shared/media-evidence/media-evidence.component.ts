import { DatePipe } from '@angular/common';
import {
  Component,
  Input,
  OnChanges,
  OnDestroy,
  SimpleChanges,
  computed,
  inject,
  signal
} from '@angular/core';
import {
  LucideArrowUpRight,
  LucideClock,
  LucideImage,
  LucideRefreshCw,
  LucideShieldCheck,
  LucideTriangleAlert
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { EventMediaAccess, VehicleEvent } from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import {
  DisplayMediaAsset,
  MediaSlot,
  displayMediaAssets,
  mediaRefreshDelay
} from '../../core/utils/media-access-utils';

@Component({
  selector: 'app-media-evidence',
  imports: [
    DatePipe,
    LucideArrowUpRight,
    LucideClock,
    LucideImage,
    LucideRefreshCw,
    LucideShieldCheck,
    LucideTriangleAlert
  ],
  templateUrl: './media-evidence.component.html'
})
export class MediaEvidenceComponent implements OnChanges, OnDestroy {
  private readonly api = inject(ApiClientService);
  @Input({ required: true }) event!: VehicleEvent;

  readonly access = signal<EventMediaAccess | null>(null);
  readonly loading = signal(false);
  readonly error = signal<string | null>(null);
  readonly failedSlots = signal<ReadonlySet<MediaSlot>>(new Set());
  readonly assets = computed(() => displayMediaAssets(this.access()));
  readonly imageAssets = computed(() =>
    this.assets().filter((item) => item.slot !== 'clip')
  );
  readonly clipAsset = computed(() =>
    this.assets().find((item) => item.slot === 'clip') ?? null
  );
  private requestGeneration = 0;
  private refreshTimer: ReturnType<typeof setTimeout> | null = null;

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['event']) void this.load();
  }

  ngOnDestroy(): void {
    this.requestGeneration += 1;
    this.clearRefreshTimer();
  }

  hasReferences(): boolean {
    const media = this.event?.media;
    return Boolean(
      media?.snapshotKey || media?.vehicleCropKey || media?.plateCropKey || media?.clipKey
    );
  }

  reload(): void {
    void this.load();
  }

  usableUrl(item: DisplayMediaAsset): string | null {
    if (this.failedSlots().has(item.slot) || item.asset.status !== 'AVAILABLE') return null;
    return item.asset.url;
  }

  markFailed(slot: MediaSlot): void {
    this.failedSlots.update((current) => new Set([...current, slot]));
  }

  unavailableMessage(item: DisplayMediaAsset): string {
    return item.asset.status === 'MISSING'
      ? 'Object không còn trong storage.'
      : 'URL đã hết hạn hoặc media không thể tải.';
  }

  private async load(): Promise<void> {
    const generation = ++this.requestGeneration;
    const eventId = this.event?._id;
    this.clearRefreshTimer();
    this.access.set(null);
    this.failedSlots.set(new Set());
    this.error.set(null);
    if (!eventId || !this.hasReferences()) {
      this.loading.set(false);
      return;
    }

    this.loading.set(true);
    try {
      const access = await firstValueFrom(this.api.eventMedia(eventId));
      if (generation !== this.requestGeneration || access.eventId !== eventId) return;
      this.access.set(access);
      this.scheduleRefresh(access.expiresAt, generation);
    } catch (error) {
      if (generation === this.requestGeneration) {
        this.error.set(apiErrorMessage(error, 'Không thể cấp quyền xem media evidence.'));
      }
    } finally {
      if (generation === this.requestGeneration) this.loading.set(false);
    }
  }

  private scheduleRefresh(expiresAt: string, generation: number): void {
    const delay = mediaRefreshDelay(expiresAt, Date.now());
    if (delay === null) return;
    this.refreshTimer = setTimeout(() => {
      if (generation === this.requestGeneration) void this.load();
    }, delay);
  }

  private clearRefreshTimer(): void {
    if (this.refreshTimer !== null) clearTimeout(this.refreshTimer);
    this.refreshTimer = null;
  }
}
