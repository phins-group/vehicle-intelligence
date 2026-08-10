export function apiErrorMessage(error: unknown, fallback: string): string {
  if (typeof error !== 'object' || error === null) return fallback;
  const candidate = error as { error?: { detail?: unknown }; status?: number };
  const detail = candidate.error?.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (candidate.status === 0) return 'Không thể kết nối tới API.';
  if (candidate.status === 401) return 'Phiên đăng nhập không hợp lệ.';
  if (candidate.status === 403) return 'Tài khoản không có quyền thực hiện thao tác này.';
  return fallback;
}
