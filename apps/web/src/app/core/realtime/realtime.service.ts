import { Injectable, inject, signal } from '@angular/core';
import { Subject } from 'rxjs';

import { AuthService } from '../auth/auth.service';
import { RealtimeGap, VehicleEvent } from '../models/api.models';
import { parseRealtimeMessage } from '../utils/event-utils';

export type RealtimeConnectionState =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'unavailable';

@Injectable({ providedIn: 'root' })
export class RealtimeService {
  private readonly auth = inject(AuthService);
  private readonly eventSubject = new Subject<VehicleEvent>();
  private readonly gapSubject = new Subject<RealtimeGap>();
  private readonly recoverySubject = new Subject<void>();
  private readonly state = signal<RealtimeConnectionState>('idle');
  private socket: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private shouldRun = false;
  private reconnectAttempt = 0;
  private lastEventId: string | null = null;

  readonly connectionState = this.state.asReadonly();
  readonly events$ = this.eventSubject.asObservable();
  readonly gaps$ = this.gapSubject.asObservable();
  readonly recoveryRequested$ = this.recoverySubject.asObservable();

  connect(): void {
    this.shouldRun = true;
    if (
      this.socket &&
      (this.socket.readyState === WebSocket.OPEN ||
        this.socket.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    this.openSocket(this.reconnectAttempt > 0 ? 'reconnecting' : 'connecting');
  }

  disconnect(): void {
    this.shouldRun = false;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const socket = this.socket;
    this.socket = null;
    if (socket && socket.readyState < WebSocket.CLOSING) socket.close(1000, 'client disconnect');
    this.state.set('idle');
    this.reconnectAttempt = 0;
    this.lastEventId = null;
  }

  private openSocket(state: RealtimeConnectionState): void {
    if (!this.shouldRun) return;
    this.state.set(state);
    const url = new URL('/ws/events', window.location.href);
    url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    if (this.lastEventId) url.searchParams.set('lastEventId', this.lastEventId);

    const socket = new WebSocket(url.toString());
    this.socket = socket;
    socket.onopen = () => {
      const token = this.auth.apiKey();
      if (token) {
        socket.send(
          JSON.stringify({
            type: 'authenticate',
            token,
            ...(this.lastEventId ? { lastEventId: this.lastEventId } : {})
          })
        );
      }
    };
    socket.onmessage = (message) => this.handleMessage(String(message.data));
    socket.onerror = () => this.state.set('unavailable');
    socket.onclose = (event) => {
      if (this.socket === socket) this.socket = null;
      if (!this.shouldRun) return;
      if (event.code === 4401 || event.code === 4403) {
        this.shouldRun = false;
        this.auth.invalidate();
        this.state.set('unavailable');
        return;
      }
      this.scheduleReconnect();
    };
  }

  private handleMessage(raw: string): void {
    const parsed = parseRealtimeMessage(raw);
    if (parsed.kind === 'event') {
      this.lastEventId = parsed.envelope.id;
      this.eventSubject.next(parsed.envelope.data);
      return;
    }
    if (parsed.kind === 'gap') {
      if (parsed.envelope.data.lastAvailableEventId) {
        this.lastEventId = parsed.envelope.data.lastAvailableEventId;
      }
      this.gapSubject.next(parsed.envelope.data);
      this.recoverySubject.next();
      return;
    }
    if (parsed.kind === 'control' && parsed.envelope.type === 'system.realtime.ready') {
      this.reconnectAttempt = 0;
      this.state.set('connected');
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempt += 1;
    this.state.set('reconnecting');
    const delay = Math.min(30_000, 1_000 * 2 ** Math.min(this.reconnectAttempt - 1, 5));
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.openSocket('reconnecting');
    }, delay);
  }
}
