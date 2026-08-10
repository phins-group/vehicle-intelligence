import { HttpInterceptorFn } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { catchError, firstValueFrom, throwError } from 'rxjs';

import { Principal, SystemHealth, UserRole } from '../models/api.models';
import { ApiClientService } from '../services/api-client.service';

const SESSION_KEY = 'vehicle-intelligence.api-key';

export type AuthState = 'checking' | 'anonymous' | 'authenticated';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly api = inject(ApiClientService);
  private readonly token = signal('');
  private readonly principalState = signal<Principal | null>(null);
  private readonly healthState = signal<SystemHealth | null>(null);
  private readonly authState = signal<AuthState>('checking');
  private readonly startupErrorState = signal<string | null>(null);

  readonly principal = this.principalState.asReadonly();
  readonly systemHealth = this.healthState.asReadonly();
  readonly state = this.authState.asReadonly();
  readonly startupError = this.startupErrorState.asReadonly();
  readonly isAuthenticated = computed(() => this.authState() === 'authenticated');
  readonly authenticationRequired = computed(
    () => this.healthState()?.authentication !== 'disabled'
  );
  readonly canManageCameras = computed(() => this.principal()?.role === 'ADMIN');
  readonly canManagePolicies = computed(() => this.principal()?.role === 'ADMIN');
  readonly canTestCameras = computed(() => this.hasAnyRole('ADMIN', 'OPERATOR'));
  readonly canManageAlerts = computed(() => this.hasAnyRole('ADMIN', 'OPERATOR'));
  readonly canReviewPlates = computed(() => this.hasAnyRole('ADMIN', 'OPERATOR'));

  async initialize(): Promise<void> {
    this.authState.set('checking');
    try {
      const health = await firstValueFrom(this.api.systemHealth());
      this.healthState.set(health);
      this.startupErrorState.set(null);
      if (health.authentication === 'disabled') {
        this.removeSavedToken();
        this.token.set('');
        await this.resolvePrincipal();
        return;
      }
      const saved = this.readSavedToken();
      if (health.authentication === 'enabled' && !saved) {
        this.authState.set('anonymous');
        return;
      }
      this.token.set(saved);
      await this.resolvePrincipal();
    } catch {
      this.clearCredentials();
      this.startupErrorState.set('Không thể xác minh trạng thái API. Hãy kiểm tra backend và thử lại.');
    }
  }

  async login(apiKey: string): Promise<Principal> {
    const candidate = apiKey.trim();
    if (!candidate) throw new Error('API key là bắt buộc.');
    this.token.set(candidate);
    try {
      const principal = await firstValueFrom(this.api.currentPrincipal());
      this.saveToken(candidate);
      this.principalState.set(principal);
      this.authState.set('authenticated');
      this.startupErrorState.set(null);
      return principal;
    } catch (error) {
      this.clearCredentials();
      throw error;
    }
  }

  async continueDevelopment(): Promise<Principal> {
    this.token.set('');
    return this.resolvePrincipal();
  }

  logout(): void {
    this.clearCredentials();
  }

  invalidate(): void {
    this.clearCredentials();
  }

  apiKey(): string {
    return this.token();
  }

  hasAnyRole(...roles: UserRole[]): boolean {
    const role = this.principalState()?.role;
    return role !== undefined && roles.includes(role);
  }

  private async resolvePrincipal(): Promise<Principal> {
    try {
      const principal = await firstValueFrom(this.api.currentPrincipal());
      this.principalState.set(principal);
      this.authState.set('authenticated');
      return principal;
    } catch (error) {
      this.clearCredentials();
      throw error;
    }
  }

  private clearCredentials(): void {
    this.token.set('');
    this.principalState.set(null);
    this.authState.set('anonymous');
    this.removeSavedToken();
  }

  private readSavedToken(): string {
    try {
      return sessionStorage.getItem(SESSION_KEY) ?? '';
    } catch {
      return '';
    }
  }

  private saveToken(token: string): void {
    try {
      sessionStorage.setItem(SESSION_KEY, token);
    } catch {
      // The in-memory token still supports this tab when storage is unavailable.
    }
  }

  private removeSavedToken(): void {
    try {
      sessionStorage.removeItem(SESSION_KEY);
    } catch {
      // Storage can be disabled by browser privacy policy.
    }
  }
}

export const authInterceptor: HttpInterceptorFn = (request, next) => {
  const auth = inject(AuthService);
  const token = auth.apiKey();
  if (!request.url.startsWith('/api') || !token) return next(request);
  return next(
    request.clone({
      setHeaders: { Authorization: 'Bearer ' + token }
    })
  ).pipe(
    catchError((error: unknown) => {
      if (typeof error === 'object' && error !== null && 'status' in error && error.status === 401) {
        auth.invalidate();
      }
      return throwError(() => error);
    })
  );
};
