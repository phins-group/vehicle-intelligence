import { DatePipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  LucideCalendarClock,
  LucideListChecks,
  LucidePencil,
  LucidePlus,
  LucideRefreshCw,
  LucideSearch,
  LucideShieldAlert,
  LucideShieldCheck,
  LucideTrash2,
  LucideX
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import {
  WatchlistEntry,
  WatchlistType,
  WatchlistWriteRequest
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import {
  WatchlistLifecycle,
  datetimeLocalToIso,
  isoToDatetimeLocal,
  watchlistLifecycle,
  watchlistMatchesSearch
} from '../../core/utils/policy-utils';

interface WatchlistDraft {
  id: string;
  plate: string;
  listType: WatchlistType;
  enabled: boolean;
  validFrom: string;
  validUntil: string;
  metadata: Record<string, unknown>;
  revision: number | null;
}

const LIST_TYPES: readonly WatchlistType[] = [
  'WHITELIST',
  'BLACKLIST',
  'VIP',
  'STAFF',
  'CONTRACTOR',
  'DELIVERY'
];

@Component({
  selector: 'app-watchlists',
  imports: [
    DatePipe,
    FormsModule,
    LucideCalendarClock,
    LucideListChecks,
    LucidePencil,
    LucidePlus,
    LucideRefreshCw,
    LucideSearch,
    LucideShieldAlert,
    LucideShieldCheck,
    LucideTrash2,
    LucideX
  ],
  templateUrl: './watchlists.component.html'
})
export class WatchlistsComponent implements OnInit {
  readonly auth = inject(AuthService);
  private readonly api = inject(ApiClientService);
  readonly listTypes = LIST_TYPES;
  readonly entries = signal<WatchlistEntry[]>([]);
  readonly loading = signal(true);
  readonly saving = signal(false);
  readonly deleting = signal(false);
  readonly editorOpen = signal(false);
  readonly editingId = signal<string | null>(null);
  readonly pendingDelete = signal<WatchlistEntry | null>(null);
  readonly error = signal<string | null>(null);
  readonly notice = signal<string | null>(null);
  search = '';
  listType: WatchlistType | '' = '';
  enabledFilter: 'ALL' | 'ENABLED' | 'DISABLED' = 'ALL';
  draft: WatchlistDraft = this.emptyDraft();

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const page = await firstValueFrom(this.api.watchlists({ limit: 200 }));
      this.entries.set(page.items);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải danh sách xe.'));
    } finally {
      this.loading.set(false);
    }
  }

  filteredEntries(): WatchlistEntry[] {
    return this.entries().filter((entry) => {
      if (this.listType && entry.listType !== this.listType) return false;
      if (this.enabledFilter === 'ENABLED' && !entry.enabled) return false;
      if (this.enabledFilter === 'DISABLED' && entry.enabled) return false;
      return watchlistMatchesSearch(entry, this.search);
    });
  }

  countType(type: WatchlistType): number {
    return this.entries().filter((entry) => entry.listType === type).length;
  }

  countLifecycle(lifecycle: WatchlistLifecycle): number {
    return this.entries().filter((entry) => watchlistLifecycle(entry) === lifecycle).length;
  }

  lifecycle(entry: WatchlistEntry): WatchlistLifecycle {
    return watchlistLifecycle(entry);
  }

  clearFilters(): void {
    this.search = '';
    this.listType = '';
    this.enabledFilter = 'ALL';
  }

  openCreate(): void {
    this.draft = this.emptyDraft();
    this.editingId.set(null);
    this.error.set(null);
    this.editorOpen.set(true);
  }

  openEdit(entry: WatchlistEntry): void {
    this.draft = {
      id: entry.id,
      plate: entry.plate,
      listType: entry.listType,
      enabled: entry.enabled,
      validFrom: isoToDatetimeLocal(entry.validFrom),
      validUntil: isoToDatetimeLocal(entry.validUntil),
      metadata: { ...entry.metadata },
      revision: entry.revision
    };
    this.editingId.set(entry.id);
    this.error.set(null);
    this.editorOpen.set(true);
  }

  closeEditor(): void {
    if (this.saving()) return;
    this.editorOpen.set(false);
    this.editingId.set(null);
    this.draft = this.emptyDraft();
  }

  draftValidation(): string | null {
    const plate = this.draft.plate.trim();
    if (plate.length < 4) return 'Biển số phải có ít nhất 4 ký tự.';
    const validFrom = datetimeLocalToIso(this.draft.validFrom);
    const validUntil = datetimeLocalToIso(this.draft.validUntil);
    if (this.draft.validFrom && !validFrom) return 'Thời điểm bắt đầu không hợp lệ.';
    if (this.draft.validUntil && !validUntil) return 'Thời điểm kết thúc không hợp lệ.';
    if (validFrom && validUntil && Date.parse(validUntil) <= Date.parse(validFrom)) {
      return 'Thời điểm kết thúc phải sau thời điểm bắt đầu.';
    }
    return null;
  }

  async save(): Promise<void> {
    if (this.saving()) return;
    const validation = this.draftValidation();
    if (validation) {
      this.error.set(validation);
      return;
    }
    this.saving.set(true);
    this.error.set(null);
    const request: WatchlistWriteRequest = {
      plate: this.draft.plate.trim(),
      listType: this.draft.listType,
      enabled: this.draft.enabled,
      validFrom: datetimeLocalToIso(this.draft.validFrom),
      validUntil: datetimeLocalToIso(this.draft.validUntil),
      metadata: { ...this.draft.metadata }
    };
    const editingId = this.editingId();
    try {
      let saved: WatchlistEntry;
      if (editingId !== null && this.draft.revision !== null) {
        saved = await firstValueFrom(
          this.api.updateWatchlist(editingId, { ...request, revision: this.draft.revision })
        );
        this.entries.update((items) => items.map((item) => (item.id === saved.id ? saved : item)));
        this.notice.set('Đã cập nhật ' + saved.plate + ' với revision ' + saved.revision + '.');
      } else {
        const requestedId = this.draft.id.trim();
        saved = await firstValueFrom(
          this.api.createWatchlist(requestedId ? { ...request, id: requestedId } : request)
        );
        this.entries.update((items) => [saved, ...items]);
        this.notice.set('Đã thêm ' + saved.plate + ' vào ' + saved.listType + '.');
      }
      this.closeEditorAfterSave();
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể lưu watchlist entry.'));
    } finally {
      this.saving.set(false);
    }
  }

  requestDelete(entry: WatchlistEntry): void {
    this.pendingDelete.set(entry);
  }

  cancelDelete(): void {
    if (!this.deleting()) this.pendingDelete.set(null);
  }

  async confirmDelete(): Promise<void> {
    const entry = this.pendingDelete();
    if (!entry || this.deleting()) return;
    this.deleting.set(true);
    this.error.set(null);
    try {
      await firstValueFrom(this.api.deleteWatchlist(entry.id));
      this.entries.update((items) => items.filter((item) => item.id !== entry.id));
      this.notice.set('Đã xóa ' + entry.plate + ' khỏi ' + entry.listType + '.');
      this.pendingDelete.set(null);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể xóa watchlist entry.'));
    } finally {
      this.deleting.set(false);
    }
  }

  private closeEditorAfterSave(): void {
    this.editorOpen.set(false);
    this.editingId.set(null);
    this.draft = this.emptyDraft();
  }

  private emptyDraft(): WatchlistDraft {
    return {
      id: '',
      plate: '',
      listType: 'WHITELIST',
      enabled: true,
      validFrom: '',
      validUntil: '',
      metadata: {},
      revision: null
    };
  }
}
