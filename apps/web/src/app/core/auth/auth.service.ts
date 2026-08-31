import { HttpInterceptorFn } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { catchError, firstValueFrom, throwError } from 'rxjs';

import {
  AuthenticationConfiguration,
  OidcConsoleConfiguration,
  OidcTokenResponse,
  Principal,
  SystemHealth,
  UserRole,
} from '../models/api.models';
import { ApiClientService } from '../services/api-client.service';

const API_KEY_SESSION_KEY = 'vehicle-intelligence.api-key';
const OIDC_TRANSACTION_KEY = 'vehicle-intelligence.oidc-transaction';
const OIDC_TRANSACTION_MAX_AGE_MS = 10 * 60 * 1000;

interface OidcTransaction {
  state: string;
  codeVerifier: string;
  redirectUri: string;
  returnUrl: string;
  createdAt: number;
}

class OidcFlowError extends Error {}

export type AuthState = 'checking' | 'anonymous' | 'authenticated';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly api = inject(ApiClientService);
  private readonly token = signal('');
  private readonly principalState = signal<Principal | null>(null);
  private readonly healthState = signal<SystemHealth | null>(null);
  private readonly configurationState = signal<AuthenticationConfiguration | null>(null);
  private readonly authState = signal<AuthState>('checking');
  private readonly startupErrorState = signal<string | null>(null);

  readonly principal = this.principalState.asReadonly();
  readonly systemHealth = this.healthState.asReadonly();
  readonly configuration = this.configurationState.asReadonly();
  readonly state = this.authState.asReadonly();
  readonly startupError = this.startupErrorState.asReadonly();
  readonly isAuthenticated = computed(() => this.authState() === 'authenticated');
  readonly authenticationRequired = computed(
    () => this.configurationState()?.enabled ?? this.healthState()?.authentication !== 'disabled',
  );
  readonly provider = computed(() => this.configurationState()?.provider ?? null);
  readonly oidcLoginAvailable = computed(
    () => this.configurationState()?.provider === 'oidc' && this.configurationState()?.oidc !== null,
  );
  readonly canManageCameras = computed(() => this.principal()?.role === 'ADMIN');
  readonly canManagePolicies = computed(() => this.principal()?.role === 'ADMIN');
  readonly canTestCameras = computed(() => this.hasAnyRole('ADMIN', 'OPERATOR'));
  readonly canManageAlerts = computed(() => this.hasAnyRole('ADMIN', 'OPERATOR'));
  readonly canReviewPlates = computed(() => this.hasAnyRole('ADMIN', 'OPERATOR'));
  readonly canReviewDatasets = computed(() => this.hasAnyRole('ADMIN', 'OPERATOR'));
  readonly canManageDatasets = computed(() => this.principal()?.role === 'ADMIN');

  async initialize(): Promise<void> {
    this.authState.set('checking');
    try {
      const [health, configuration] = await Promise.all([
        firstValueFrom(this.api.systemHealth()),
        firstValueFrom(this.api.authenticationConfiguration()),
      ]);
      this.healthState.set(health);
      this.configurationState.set(configuration);
      this.startupErrorState.set(null);

      if (!configuration.enabled) {
        this.removeSavedApiKey();
        this.token.set('');
        await this.resolvePrincipal();
        return;
      }
      if (configuration.provider === 'oidc') {
        this.removeSavedApiKey();
        if (configuration.oidc === null) {
          throw new OidcFlowError(
            'Đăng nhập tập trung chưa được cấu hình đầy đủ trên máy chủ.',
          );
        }
        if (this.hasOidcCallback(configuration.oidc.callbackPath)) {
          await this.completeOidcLogin(configuration.oidc);
          return;
        }
        this.authState.set('anonymous');
        return;
      }

      const saved = this.readSavedApiKey();
      if (!saved) {
        this.authState.set('anonymous');
        return;
      }
      this.token.set(saved);
      await this.resolvePrincipal();
    } catch (error) {
      this.clearCredentials();
      this.startupErrorState.set(
        error instanceof OidcFlowError
          ? error.message
          : 'Không thể xác minh trạng thái API. Hãy kiểm tra backend và thử lại.',
      );
    }
  }

  async login(apiKey: string): Promise<Principal> {
    if (this.provider() !== 'api_key') {
      throw new Error('Máy chủ không cho phép đăng nhập bằng API key.');
    }
    const candidate = apiKey.trim();
    if (!candidate) throw new Error('API key là bắt buộc.');
    this.token.set(candidate);
    try {
      const principal = await this.resolvePrincipal();
      this.saveApiKey(candidate);
      return principal;
    } catch (error) {
      this.clearCredentials();
      throw error;
    }
  }

  async beginOidcLogin(returnUrl = '/dashboard'): Promise<void> {
    const oidc = this.configurationState()?.oidc;
    if (this.provider() !== 'oidc' || oidc === null || oidc === undefined) {
      throw new Error('Đăng nhập tập trung chưa sẵn sàng.');
    }
    if (!globalThis.crypto?.subtle) {
      throw new Error('Trình duyệt không hỗ trợ Web Crypto cần thiết cho đăng nhập PKCE.');
    }

    const state = this.randomBase64Url(32);
    const codeVerifier = this.randomBase64Url(48);
    const digest = await globalThis.crypto.subtle.digest(
      'SHA-256',
      new TextEncoder().encode(codeVerifier),
    );
    const codeChallenge = this.base64Url(new Uint8Array(digest));
    const redirectUri = new URL(oidc.callbackPath, window.location.origin).toString();
    const transaction: OidcTransaction = {
      state,
      codeVerifier,
      redirectUri,
      returnUrl: this.safeReturnUrl(returnUrl),
      createdAt: Date.now(),
    };
    try {
      sessionStorage.setItem(OIDC_TRANSACTION_KEY, JSON.stringify(transaction));
    } catch {
      throw new Error('Trình duyệt đang chặn session storage cần thiết cho đăng nhập an toàn.');
    }

    const authorizationUrl = new URL(oidc.authorizationEndpoint);
    authorizationUrl.searchParams.set('response_type', 'code');
    authorizationUrl.searchParams.set('client_id', oidc.clientId);
    authorizationUrl.searchParams.set('redirect_uri', redirectUri);
    authorizationUrl.searchParams.set('scope', oidc.scopes.join(' '));
    authorizationUrl.searchParams.set('state', state);
    authorizationUrl.searchParams.set('code_challenge', codeChallenge);
    authorizationUrl.searchParams.set('code_challenge_method', 'S256');
    window.location.assign(authorizationUrl.toString());
  }

  async continueDevelopment(): Promise<Principal> {
    this.token.set('');
    return this.resolvePrincipal();
  }

  logout(): boolean {
    const oidc = this.configurationState()?.oidc;
    const usesOidc = this.provider() === 'oidc';
    this.clearCredentials(true);
    if (!usesOidc || oidc?.endSessionEndpoint === null || oidc?.endSessionEndpoint === undefined) {
      return false;
    }
    const logoutUrl = new URL(oidc.endSessionEndpoint);
    logoutUrl.searchParams.set('client_id', oidc.clientId);
    logoutUrl.searchParams.set(
      'post_logout_redirect_uri',
      new URL('/login', window.location.origin).toString(),
    );
    window.location.assign(logoutUrl.toString());
    return true;
  }

  invalidate(): void {
    this.clearCredentials();
  }

  bearerToken(): string {
    return this.token();
  }

  apiKey(): string {
    return this.bearerToken();
  }

  hasAnyRole(...roles: UserRole[]): boolean {
    const role = this.principalState()?.role;
    return role !== undefined && roles.includes(role);
  }

  private async completeOidcLogin(oidc: OidcConsoleConfiguration): Promise<void> {
    const currentUrl = new URL(window.location.href);
    const oauthError = currentUrl.searchParams.get('error');
    if (oauthError !== null) {
      this.removeOidcTransaction();
      this.cleanOidcCallback('/login');
      const description = currentUrl.searchParams.get('error_description');
      throw new OidcFlowError(
        description?.trim() || `Nhà cung cấp danh tính từ chối đăng nhập (${oauthError}).`,
      );
    }

    const code = currentUrl.searchParams.get('code');
    const returnedState = currentUrl.searchParams.get('state');
    const transaction = this.readOidcTransaction();
    this.removeOidcTransaction();
    if (!code || code.length > 8192 || returnedState === null || transaction === null) {
      this.cleanOidcCallback('/login');
      throw new OidcFlowError('Phiên đăng nhập không hợp lệ hoặc đã được sử dụng.');
    }
    const transactionAge = Date.now() - transaction.createdAt;
    if (
      returnedState !== transaction.state ||
      !Number.isFinite(transactionAge) ||
      transactionAge < 0 ||
      transactionAge > OIDC_TRANSACTION_MAX_AGE_MS ||
      transaction.redirectUri !== new URL(oidc.callbackPath, window.location.origin).toString()
    ) {
      this.cleanOidcCallback('/login');
      throw new OidcFlowError('Phiên đăng nhập đã hết hạn hoặc không khớp yêu cầu ban đầu.');
    }

    let tokenResponse: OidcTokenResponse;
    try {
      tokenResponse = await firstValueFrom(
        this.api.exchangeOidcCode(oidc.tokenEndpoint, {
          code,
          clientId: oidc.clientId,
          codeVerifier: transaction.codeVerifier,
          redirectUri: transaction.redirectUri,
        }),
      );
    } catch {
      this.cleanOidcCallback('/login');
      throw new OidcFlowError('Không thể đổi mã đăng nhập với nhà cung cấp danh tính.');
    }
    const accessToken = tokenResponse.access_token?.trim();
    if (
      tokenResponse.token_type?.toLocaleLowerCase() !== 'bearer' ||
      !accessToken ||
      accessToken.length > 65_536 ||
      /\s/.test(accessToken)
    ) {
      this.cleanOidcCallback('/login');
      throw new OidcFlowError('Nhà cung cấp danh tính trả về access token không hợp lệ.');
    }

    this.token.set(accessToken);
    try {
      await this.resolvePrincipal();
    } catch {
      this.cleanOidcCallback('/login');
      throw new OidcFlowError('Backend từ chối danh tính hoặc vai trò được cấp.');
    }
    this.cleanOidcCallback(transaction.returnUrl);
  }

  private async resolvePrincipal(): Promise<Principal> {
    try {
      const principal = await firstValueFrom(this.api.currentPrincipal());
      this.principalState.set(principal);
      this.authState.set('authenticated');
      this.startupErrorState.set(null);
      return principal;
    } catch (error) {
      this.clearCredentials();
      throw error;
    }
  }

  private hasOidcCallback(callbackPath: string): boolean {
    const currentUrl = new URL(window.location.href);
    const expectedUrl = new URL(callbackPath, window.location.origin);
    return (
      currentUrl.origin === expectedUrl.origin &&
      currentUrl.pathname === expectedUrl.pathname &&
      (currentUrl.searchParams.has('code') ||
        currentUrl.searchParams.has('state') ||
        currentUrl.searchParams.has('error'))
    );
  }

  private readOidcTransaction(): OidcTransaction | null {
    let raw: string | null;
    try {
      raw = sessionStorage.getItem(OIDC_TRANSACTION_KEY);
    } catch {
      return null;
    }
    if (raw === null) return null;
    try {
      const value: unknown = JSON.parse(raw);
      if (typeof value !== 'object' || value === null) return null;
      const transaction = value as Partial<OidcTransaction>;
      if (
        typeof transaction.state !== 'string' ||
        typeof transaction.codeVerifier !== 'string' ||
        typeof transaction.redirectUri !== 'string' ||
        typeof transaction.returnUrl !== 'string' ||
        typeof transaction.createdAt !== 'number' ||
        !Number.isFinite(transaction.createdAt)
      ) {
        return null;
      }
      return transaction as OidcTransaction;
    } catch {
      return null;
    }
  }

  private clearCredentials(removeOidcTransaction = false): void {
    this.token.set('');
    this.principalState.set(null);
    this.authState.set('anonymous');
    this.removeSavedApiKey();
    if (removeOidcTransaction) this.removeOidcTransaction();
  }

  private readSavedApiKey(): string {
    try {
      return sessionStorage.getItem(API_KEY_SESSION_KEY) ?? '';
    } catch {
      return '';
    }
  }

  private saveApiKey(token: string): void {
    try {
      sessionStorage.setItem(API_KEY_SESSION_KEY, token);
    } catch {
      // The in-memory token still supports this tab when storage is unavailable.
    }
  }

  private removeSavedApiKey(): void {
    try {
      sessionStorage.removeItem(API_KEY_SESSION_KEY);
    } catch {
      // Storage can be disabled by browser privacy policy.
    }
  }

  private removeOidcTransaction(): void {
    try {
      sessionStorage.removeItem(OIDC_TRANSACTION_KEY);
    } catch {
      // A missing transaction will fail the callback closed.
    }
  }

  private cleanOidcCallback(target: string): void {
    window.history.replaceState({}, document.title, this.safeReturnUrl(target));
  }

  private safeReturnUrl(value: string): string {
    return value.startsWith('/') && !value.startsWith('//') && !value.includes('\\')
      ? value
      : '/dashboard';
  }

  private randomBase64Url(byteLength: number): string {
    const bytes = new Uint8Array(byteLength);
    globalThis.crypto.getRandomValues(bytes);
    return this.base64Url(bytes);
  }

  private base64Url(bytes: Uint8Array): string {
    let binary = '';
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
  }
}

export const authInterceptor: HttpInterceptorFn = (request, next) => {
  const auth = inject(AuthService);
  const token = auth.bearerToken();
  if (!request.url.startsWith('/api') || !token) return next(request);
  return next(
    request.clone({
      setHeaders: { Authorization: 'Bearer ' + token },
    }),
  ).pipe(
    catchError((error: unknown) => {
      if (typeof error === 'object' && error !== null && 'status' in error && error.status === 401) {
        auth.invalidate();
      }
      return throwError(() => error);
    }),
  );
};
