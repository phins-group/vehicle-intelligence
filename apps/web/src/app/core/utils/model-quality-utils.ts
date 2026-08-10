export function qualityBarWidth(value: number, maximum: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(maximum) || maximum <= 0) return 0;
  return Math.max(0, Math.min(100, (value / maximum) * 100));
}
