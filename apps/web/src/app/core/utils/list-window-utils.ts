export interface ListWindow<T> {
  items: T[];
  start: number;
  end: number;
  total: number;
  hasPrevious: boolean;
  hasNext: boolean;
}

export function listWindow<T>(
  items: readonly T[],
  requestedStart: number,
  requestedSize: number
): ListWindow<T> {
  const size = Number.isFinite(requestedSize)
    ? Math.max(1, Math.floor(requestedSize))
    : 1;
  const maximumStart = items.length > 0 ? Math.floor((items.length - 1) / size) * size : 0;
  const candidateStart = Number.isFinite(requestedStart)
    ? Math.max(0, Math.floor(requestedStart / size) * size)
    : 0;
  const start = Math.min(candidateStart, maximumStart);
  const end = Math.min(items.length, start + size);

  return {
    items: items.slice(start, end),
    start,
    end,
    total: items.length,
    hasPrevious: start > 0,
    hasNext: end < items.length
  };
}
