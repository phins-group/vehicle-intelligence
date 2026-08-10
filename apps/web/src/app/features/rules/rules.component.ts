import { DatePipe } from '@angular/common';
import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
  LucideBell,
  LucideBraces,
  LucideGitBranch,
  LucideGlobe2,
  LucidePencil,
  LucidePlus,
  LucideRefreshCw,
  LucideSearch,
  LucideShieldCheck,
  LucideTrash2,
  LucideX,
  LucideZap
} from '@lucide/angular';
import { firstValueFrom } from 'rxjs';

import { AuthService } from '../../core/auth/auth.service';
import {
  AlertSeverity,
  Rule,
  RuleAction,
  RuleActionType,
  RuleCondition,
  RuleConditionField,
  RuleConditionOperator,
  RuleWriteRequest,
  WatchlistType
} from '../../core/models/api.models';
import { ApiClientService } from '../../core/services/api-client.service';
import { apiErrorMessage } from '../../core/utils/api-error';
import {
  RULE_FIELDS,
  isExternalAction,
  parseListInput,
  ruleActionIsValid,
  ruleConditionIsValid
} from '../../core/utils/policy-utils';

interface ConditionDraft {
  field: RuleConditionField;
  operator: RuleConditionOperator;
  value: string;
}

interface ActionDraft {
  uiId: string;
  id: string;
  type: RuleActionType;
  severity: AlertSeverity;
  message: string;
  url: string;
  method: 'GET' | 'POST' | 'PUT' | 'PATCH';
}

interface RuleDraft {
  id: string;
  name: string;
  enabled: boolean;
  priority: number;
  conditions: ConditionDraft[];
  actions: ActionDraft[];
  metadata: Record<string, unknown>;
  revision: number | null;
}

const ACTION_TYPES: readonly RuleActionType[] = [
  'CREATE_ALERT',
  'LOG',
  'OPEN_BARRIER',
  'WEBHOOK',
  'HTTP_REQUEST',
  'NOTIFICATION'
];
const WATCHLIST_TYPES: readonly WatchlistType[] = [
  'WHITELIST',
  'BLACKLIST',
  'VIP',
  'STAFF',
  'CONTRACTOR',
  'DELIVERY'
];
const ALERT_SEVERITIES: readonly AlertSeverity[] = [
  'INFO',
  'LOW',
  'MEDIUM',
  'HIGH',
  'CRITICAL'
];
const FIELD_LABELS: Record<RuleConditionField, string> = {
  watchlist: 'Nhóm watchlist',
  'camera.id': 'Camera ID',
  'camera.zone': 'Camera zone',
  direction: 'Hướng di chuyển',
  eventType: 'Loại sự kiện',
  status: 'Trạng thái sự kiện',
  'plate.normalized': 'Biển số chuẩn hóa',
  'vehicle.type': 'Loại phương tiện',
  'vehicle.color': 'Màu phương tiện'
};

@Component({
  selector: 'app-rules',
  imports: [
    DatePipe,
    FormsModule,
    LucideBell,
    LucideBraces,
    LucideGitBranch,
    LucideGlobe2,
    LucidePencil,
    LucidePlus,
    LucideRefreshCw,
    LucideSearch,
    LucideShieldCheck,
    LucideTrash2,
    LucideX,
    LucideZap
  ],
  templateUrl: './rules.component.html'
})
export class RulesComponent implements OnInit {
  readonly auth = inject(AuthService);
  private readonly api = inject(ApiClientService);
  readonly fields = RULE_FIELDS;
  readonly actionTypes = ACTION_TYPES;
  readonly watchlistTypes = WATCHLIST_TYPES;
  readonly alertSeverities = ALERT_SEVERITIES;
  readonly rules = signal<Rule[]>([]);
  readonly loading = signal(true);
  readonly saving = signal(false);
  readonly deleting = signal(false);
  readonly editorOpen = signal(false);
  readonly editingId = signal<string | null>(null);
  readonly pendingDelete = signal<Rule | null>(null);
  readonly error = signal<string | null>(null);
  readonly notice = signal<string | null>(null);
  search = '';
  enabledFilter: 'ALL' | 'ENABLED' | 'DISABLED' = 'ALL';
  private actionSequence = 0;
  draft: RuleDraft = this.emptyDraft();

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    this.loading.set(true);
    this.error.set(null);
    try {
      const page = await firstValueFrom(this.api.rules(false, 200));
      this.rules.set(this.sortRules(page.items));
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể tải rules.'));
    } finally {
      this.loading.set(false);
    }
  }

  filteredRules(): Rule[] {
    const query = this.search.trim().toLocaleUpperCase();
    return this.rules().filter((rule) => {
      if (this.enabledFilter === 'ENABLED' && !rule.enabled) return false;
      if (this.enabledFilter === 'DISABLED' && rule.enabled) return false;
      return !query || rule.name.toLocaleUpperCase().includes(query) || rule.id.toLocaleUpperCase().includes(query);
    });
  }

  enabledCount(): number {
    return this.rules().filter((rule) => rule.enabled).length;
  }

  alertRuleCount(): number {
    return this.rules().filter((rule) => rule.actions.some((action) => action.type === 'CREATE_ALERT')).length;
  }

  externalRuleCount(): number {
    return this.rules().filter((rule) => rule.actions.some((action) => isExternalAction(action.type))).length;
  }

  clearFilters(): void {
    this.search = '';
    this.enabledFilter = 'ALL';
  }

  openCreate(): void {
    this.draft = this.emptyDraft();
    this.editingId.set(null);
    this.error.set(null);
    this.editorOpen.set(true);
  }

  openEdit(rule: Rule): void {
    this.draft = {
      id: rule.id,
      name: rule.name,
      enabled: rule.enabled,
      priority: rule.priority,
      conditions: rule.conditions.map((condition) => this.conditionToDraft(condition)),
      actions: rule.actions.map((action) => this.actionToDraft(action)),
      metadata: { ...rule.metadata },
      revision: rule.revision
    };
    this.editingId.set(rule.id);
    this.error.set(null);
    this.editorOpen.set(true);
  }

  closeEditor(): void {
    if (this.saving()) return;
    this.editorOpen.set(false);
    this.editingId.set(null);
    this.draft = this.emptyDraft();
  }

  addCondition(): void {
    if (this.draft.conditions.length >= 32) return;
    this.draft.conditions.push(this.newCondition());
  }

  removeCondition(index: number): void {
    if (this.draft.conditions.length > 1) this.draft.conditions.splice(index, 1);
  }

  conditionFieldChanged(index: number): void {
    const condition = this.draft.conditions[index];
    if (!condition) return;
    if (condition.field === 'watchlist') {
      condition.operator = 'CONTAINS';
      condition.value = 'WHITELIST';
    } else {
      if (condition.operator === 'CONTAINS') condition.operator = 'EQ';
      condition.value = this.defaultConditionValue(condition.field, condition.operator);
    }
  }

  conditionOperatorChanged(index: number): void {
    const condition = this.draft.conditions[index];
    if (!condition) return;
    condition.value = this.defaultConditionValue(condition.field, condition.operator);
  }

  operatorsFor(condition: ConditionDraft): readonly RuleConditionOperator[] {
    return condition.field === 'watchlist'
      ? ['CONTAINS', 'EXISTS']
      : ['EQ', 'NEQ', 'IN', 'NOT_IN', 'EXISTS'];
  }

  conditionChoices(condition: ConditionDraft): readonly string[] {
    if (condition.operator === 'IN' || condition.operator === 'NOT_IN' || condition.operator === 'EXISTS') return [];
    switch (condition.field) {
      case 'watchlist': return WATCHLIST_TYPES;
      case 'direction': return ['ENTER', 'EXIT', 'UNKNOWN'];
      case 'eventType': return ['VEHICLE_ENTER', 'VEHICLE_EXIT', 'VEHICLE_DETECTED'];
      case 'status': return ['CONFIRMED', 'LOW_CONFIDENCE', 'NEEDS_REVIEW', 'NO_PLATE', 'UNREADABLE'];
      case 'vehicle.type': return ['car', 'motorcycle', 'bus', 'truck'];
      default: return [];
    }
  }

  conditionLabel(field: RuleConditionField): string {
    return FIELD_LABELS[field];
  }

  conditionPlaceholder(condition: ConditionDraft): string {
    if (condition.operator === 'IN' || condition.operator === 'NOT_IN') return 'Giá trị 1, Giá trị 2';
    if (condition.field === 'camera.id') return 'gate-01';
    if (condition.field === 'camera.zone') return 'ZONE_A';
    if (condition.field === 'plate.normalized') return '51H-123.45';
    if (condition.field === 'vehicle.color') return 'white';
    return 'Giá trị so sánh';
  }

  addAction(): void {
    if (this.draft.actions.length >= 16) return;
    this.draft.actions.push(this.newAction());
  }

  removeAction(index: number): void {
    if (this.draft.actions.length > 1) this.draft.actions.splice(index, 1);
  }

  actionTypeChanged(index: number): void {
    const action = this.draft.actions[index];
    if (!action) return;
    action.severity = 'HIGH';
    action.message = '';
    action.url = '';
    action.method = 'POST';
  }

  externalAction(type: RuleActionType): boolean {
    return isExternalAction(type);
  }

  actionSummary(action: RuleAction): string {
    if (action.type === 'CREATE_ALERT') return String(action.parameters['severity'] ?? 'HIGH');
    if (isExternalAction(action.type)) {
      const method = String(action.parameters['method'] ?? 'POST');
      const url = String(action.parameters['url'] ?? 'URL chưa cấu hình');
      return method + ' · ' + url;
    }
    return String(action.parameters['message'] ?? 'structured log');
  }

  conditionSummary(condition: RuleCondition): string {
    const value = Array.isArray(condition.value) ? condition.value.join(', ') : String(condition.value);
    return this.conditionLabel(condition.field) + ' ' + condition.operator + ' ' + value;
  }

  draftValidation(): string | null {
    if (!this.draft.name.trim()) return 'Tên rule là bắt buộc.';
    if (!Number.isInteger(this.draft.priority) || this.draft.priority < -10000 || this.draft.priority > 10000) {
      return 'Priority phải là số nguyên từ -10000 đến 10000.';
    }
    if (!this.draft.conditions.length || this.draft.conditions.length > 32) return 'Rule cần từ 1 đến 32 conditions.';
    if (!this.draft.actions.length || this.draft.actions.length > 16) return 'Rule cần từ 1 đến 16 actions.';
    const actionIds = this.draft.actions.map((action) => action.id.trim());
    if (new Set(actionIds).size !== actionIds.length) return 'Action ID phải duy nhất trong một rule.';
    const invalidCondition = this.draft.conditions.findIndex((condition) => !ruleConditionIsValid(this.conditionFromDraft(condition)));
    if (invalidCondition >= 0) return 'Condition ' + (invalidCondition + 1) + ' chưa có giá trị hợp lệ.';
    const invalidAction = this.draft.actions.findIndex((action) => !ruleActionIsValid(this.actionFromDraft(action)));
    if (invalidAction >= 0) return 'Action ' + (invalidAction + 1) + ' chưa có cấu hình hợp lệ.';
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
    const request: RuleWriteRequest = {
      name: this.draft.name.trim(),
      enabled: this.draft.enabled,
      priority: this.draft.priority,
      conditions: this.draft.conditions.map((condition) => this.conditionFromDraft(condition)),
      actions: this.draft.actions.map((action) => this.actionFromDraft(action)),
      metadata: { ...this.draft.metadata }
    };
    const editingId = this.editingId();
    try {
      let saved: Rule;
      if (editingId !== null && this.draft.revision !== null) {
        saved = await firstValueFrom(this.api.updateRule(editingId, { ...request, revision: this.draft.revision }));
        this.rules.update((items) => this.sortRules(items.map((item) => (item.id === saved.id ? saved : item))));
        this.notice.set('Đã cập nhật rule ' + saved.name + ' tại revision ' + saved.revision + '.');
      } else {
        const requestedId = this.draft.id.trim();
        saved = await firstValueFrom(this.api.createRule(requestedId ? { ...request, id: requestedId } : request));
        this.rules.update((items) => this.sortRules([saved, ...items]));
        this.notice.set('Đã tạo rule ' + saved.name + '.');
      }
      this.closeEditorAfterSave();
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể lưu rule.'));
    } finally {
      this.saving.set(false);
    }
  }

  requestDelete(rule: Rule): void {
    this.pendingDelete.set(rule);
  }

  cancelDelete(): void {
    if (!this.deleting()) this.pendingDelete.set(null);
  }

  async confirmDelete(): Promise<void> {
    const rule = this.pendingDelete();
    if (!rule || this.deleting()) return;
    this.deleting.set(true);
    this.error.set(null);
    try {
      await firstValueFrom(this.api.deleteRule(rule.id));
      this.rules.update((items) => items.filter((item) => item.id !== rule.id));
      this.notice.set('Đã xóa rule ' + rule.name + '.');
      this.pendingDelete.set(null);
    } catch (error) {
      this.error.set(apiErrorMessage(error, 'Không thể xóa rule.'));
    } finally {
      this.deleting.set(false);
    }
  }

  private conditionFromDraft(draft: ConditionDraft): RuleCondition {
    let value: unknown = draft.value.trim();
    if (draft.operator === 'EXISTS') value = draft.value === 'true';
    if (draft.operator === 'IN' || draft.operator === 'NOT_IN') value = parseListInput(draft.value);
    return { field: draft.field, operator: draft.operator, value };
  }

  private actionFromDraft(draft: ActionDraft): RuleAction {
    let parameters: Record<string, unknown> = {};
    if (isExternalAction(draft.type)) {
      parameters = { url: draft.url.trim(), method: draft.method };
    } else if (draft.type === 'CREATE_ALERT') {
      parameters = { severity: draft.severity };
      if (draft.message.trim()) parameters['message'] = draft.message.trim();
    } else if (draft.message.trim()) {
      parameters = { message: draft.message.trim() };
    }
    return { id: draft.id, type: draft.type, parameters };
  }

  private conditionToDraft(condition: RuleCondition): ConditionDraft {
    const value = Array.isArray(condition.value) ? condition.value.join(', ') : String(condition.value);
    return { field: condition.field, operator: condition.operator, value };
  }

  private actionToDraft(action: RuleAction): ActionDraft {
    const severityValue = String(action.parameters['severity'] ?? 'HIGH') as AlertSeverity;
    const methodValue = String(action.parameters['method'] ?? 'POST').toUpperCase() as ActionDraft['method'];
    return {
      uiId: 'row_' + this.nextIdentifierSuffix(),
      id: action.id,
      type: action.type,
      severity: ALERT_SEVERITIES.includes(severityValue) ? severityValue : 'HIGH',
      message: typeof action.parameters['message'] === 'string' ? action.parameters['message'] : '',
      url: typeof action.parameters['url'] === 'string' ? action.parameters['url'] : '',
      method: ['GET', 'POST', 'PUT', 'PATCH'].includes(methodValue) ? methodValue : 'POST'
    };
  }

  private defaultConditionValue(field: RuleConditionField, operator: RuleConditionOperator): string {
    if (operator === 'EXISTS') return 'true';
    if (operator === 'IN' || operator === 'NOT_IN') return '';
    switch (field) {
      case 'watchlist': return 'WHITELIST';
      case 'direction': return 'ENTER';
      case 'eventType': return 'VEHICLE_ENTER';
      case 'status': return 'CONFIRMED';
      case 'vehicle.type': return 'car';
      default: return '';
    }
  }

  private newCondition(): ConditionDraft {
    return { field: 'watchlist', operator: 'CONTAINS', value: 'WHITELIST' };
  }

  private newAction(): ActionDraft {
    const randomId = this.nextIdentifierSuffix();
    return {
      uiId: 'row_' + randomId,
      id: 'action_' + randomId,
      type: 'CREATE_ALERT',
      severity: 'HIGH',
      message: '',
      url: '',
      method: 'POST'
    };
  }

  private nextIdentifierSuffix(): string {
    this.actionSequence += 1;
    return globalThis.crypto?.randomUUID?.().replaceAll('-', '') ?? (Date.now().toString(36) + this.actionSequence);
  }

  private emptyDraft(): RuleDraft {
    return {
      id: '',
      name: '',
      enabled: true,
      priority: 0,
      conditions: [this.newCondition()],
      actions: [this.newAction()],
      metadata: {},
      revision: null
    };
  }

  private sortRules(items: Rule[]): Rule[] {
    return [...items].sort((left, right) => right.priority - left.priority || left.name.localeCompare(right.name) || left.id.localeCompare(right.id));
  }

  private closeEditorAfterSave(): void {
    this.editorOpen.set(false);
    this.editingId.set(null);
    this.draft = this.emptyDraft();
  }
}
