import {
  AlertSeverity,
  RuleAction,
  RuleActionType,
  RuleCondition,
  RuleConditionField,
  RuleConditionOperator,
  WatchlistEntry
} from '../models/api.models';

export type WatchlistLifecycle = 'ACTIVE' | 'SCHEDULED' | 'EXPIRED' | 'DISABLED';

export const RULE_FIELDS: readonly RuleConditionField[] = [
  'watchlist',
  'camera.id',
  'camera.zone',
  'direction',
  'eventType',
  'status',
  'plate.normalized',
  'vehicle.type',
  'vehicle.color'
];

export const RULE_OPERATORS: readonly RuleConditionOperator[] = [
  'EQ',
  'NEQ',
  'IN',
  'NOT_IN',
  'CONTAINS',
  'EXISTS'
];

export const EXTERNAL_ACTION_TYPES: readonly RuleActionType[] = [
  'OPEN_BARRIER',
  'WEBHOOK',
  'HTTP_REQUEST',
  'NOTIFICATION'
];

const ALERT_SEVERITIES: readonly AlertSeverity[] = [
  'INFO',
  'LOW',
  'MEDIUM',
  'HIGH',
  'CRITICAL'
];

const EXTERNAL_METHODS = ['GET', 'POST', 'PUT', 'PATCH'] as const;

export function watchlistLifecycle(
  entry: Pick<WatchlistEntry, 'enabled' | 'validFrom' | 'validUntil'>,
  now = new Date()
): WatchlistLifecycle {
  if (!entry.enabled) return 'DISABLED';
  const current = now.getTime();
  const starts = entry.validFrom ? Date.parse(entry.validFrom) : Number.NaN;
  const ends = entry.validUntil ? Date.parse(entry.validUntil) : Number.NaN;
  if (Number.isFinite(starts) && starts > current) return 'SCHEDULED';
  if (Number.isFinite(ends) && ends < current) return 'EXPIRED';
  return 'ACTIVE';
}

export function watchlistMatchesSearch(
  entry: Pick<WatchlistEntry, 'id' | 'plate'>,
  search: string
): boolean {
  const query = normalizePolicySearch(search);
  if (!query) return true;
  return (
    normalizePolicySearch(entry.plate).includes(query) ||
    entry.id.toLocaleUpperCase().includes(search.trim().toLocaleUpperCase())
  );
}

export function datetimeLocalToIso(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const date = new Date(trimmed);
  return Number.isNaN(date.getTime()) ? null : date.toISOString();
}

export function isoToDatetimeLocal(value: string | null): string {
  if (!value) return '';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const pad = (part: number): string => String(part).padStart(2, '0');
  return (
    date.getFullYear() +
    '-' +
    pad(date.getMonth() + 1) +
    '-' +
    pad(date.getDate()) +
    'T' +
    pad(date.getHours()) +
    ':' +
    pad(date.getMinutes())
  );
}

export function parseListInput(value: string): string[] {
  return [...new Set(value.split(/[,;\n]/).map((item) => item.trim()).filter(Boolean))];
}

export function isExternalAction(type: RuleActionType): boolean {
  return EXTERNAL_ACTION_TYPES.includes(type);
}

export function isValidExternalActionUrl(value: unknown): boolean {
  if (typeof value !== 'string' || !value.trim()) return false;
  try {
    const parsed = new URL(value);
    return (
      (parsed.protocol === 'http:' || parsed.protocol === 'https:') &&
      Boolean(parsed.hostname) &&
      !parsed.username &&
      !parsed.password
    );
  } catch {
    return false;
  }
}

export function ruleConditionIsValid(condition: RuleCondition): boolean {
  if (!RULE_FIELDS.includes(condition.field) || !RULE_OPERATORS.includes(condition.operator)) {
    return false;
  }
  if (condition.operator === 'CONTAINS') {
    return condition.field === 'watchlist' && nonEmptyString(condition.value);
  }
  if (condition.operator === 'EXISTS') return typeof condition.value === 'boolean';
  if (condition.operator === 'IN' || condition.operator === 'NOT_IN') {
    return Array.isArray(condition.value) && condition.value.some(nonEmptyString);
  }
  return condition.value !== null && condition.value !== undefined && String(condition.value).trim() !== '';
}

export function ruleActionIsValid(action: RuleAction): boolean {
  if (!action.id.trim()) return false;
  if (isExternalAction(action.type)) {
    const method = String(action.parameters['method'] ?? 'POST').toUpperCase();
    return (
      isValidExternalActionUrl(action.parameters['url']) &&
      (EXTERNAL_METHODS as readonly string[]).includes(method)
    );
  }
  if (action.type === 'CREATE_ALERT') {
    const message = action.parameters['message'];
    const severity = action.parameters['severity'];
    return (
      (message === undefined || nonEmptyString(message)) &&
      (severity === undefined || ALERT_SEVERITIES.includes(severity as AlertSeverity))
    );
  }
  return action.type === 'LOG';
}

function normalizePolicySearch(value: string): string {
  return value.toLocaleUpperCase().replace(/[-.\s]/g, '');
}

function nonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && Boolean(value.trim());
}
