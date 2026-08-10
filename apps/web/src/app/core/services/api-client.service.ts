import { HttpClient, HttpParams, HttpResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  Alert,
  AlertPage,
  AlertStatus,
  Camera,
  CameraBatchResult,
  CameraConnectionTest,
  CameraCreateRequest,
  CameraHealth,
  EventFilters,
  EventMediaAccess,
  EventPage,
  LiveMonitorHealth,
  LiveMonitorState,
  ModelQualityReport,
  OnvifDiscoveryResult,
  DatasetSample,
  PlateReviewRequest,
  PlateReviewResponse,
  Principal,
  RealtimeHealth,
  Rule,
  RuleUpdateRequest,
  RuleWriteRequest,
  SystemHealth,
  VehicleSearchPage,
  VehicleIdentity,
  VehicleJourney,
  VehicleTimeline,
  WatchlistEntry,
  WatchlistType,
  WatchlistUpdateRequest,
  WatchlistWriteRequest
} from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ApiClientService {
  private readonly http = inject(HttpClient);

  systemHealth(): Observable<SystemHealth> {
    return this.http.get<SystemHealth>('/api/system/health');
  }

  currentPrincipal(): Observable<Principal> {
    return this.http.get<Principal>('/api/auth/me');
  }

  events(filters: EventFilters = {}): Observable<EventPage> {
    let params = new HttpParams();
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== null && value !== '') {
        params = params.set(key, String(value));
      }
    }
    return this.http.get<EventPage>('/api/events', { params });
  }

  event(eventId: string): Observable<import('../models/api.models').VehicleEvent> {
    return this.http.get<import('../models/api.models').VehicleEvent>(
      '/api/events/' + encodeURIComponent(eventId)
    );
  }

  eventMedia(eventId: string): Observable<EventMediaAccess> {
    return this.http.get<EventMediaAccess>(
      '/api/events/' + encodeURIComponent(eventId) + '/media'
    );
  }

  reviewPlate(
    eventId: string,
    request: PlateReviewRequest
  ): Observable<PlateReviewResponse> {
    return this.http.put<PlateReviewResponse>(
      '/api/events/' + encodeURIComponent(eventId) + '/plate-review',
      request
    );
  }

  datasetSamples(filters: {
    limit?: number;
    cursor?: string | null;
    type?: 'PLATE_OCR';
    status?: 'READY' | 'EXPORTING' | 'EXPORTED' | 'EXPORT_FAILED';
    reason?: 'HUMAN_CORRECTION' | 'HUMAN_CONFIRMATION';
    sourceEventId?: string;
  } = {}): Observable<{ items: DatasetSample[]; nextCursor: string | null }> {
    let params = new HttpParams();
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== null && value !== '') {
        params = params.set(key, String(value));
      }
    }
    return this.http.get<{ items: DatasetSample[]; nextCursor: string | null }>(
      '/api/dataset-samples',
      { params }
    );
  }

  modelQuality(from?: string, to?: string): Observable<ModelQualityReport> {
    let params = new HttpParams();
    if (from) params = params.set('from', from);
    if (to) params = params.set('to', to);
    return this.http.get<ModelQualityReport>('/api/model-quality', { params });
  }

  searchVehicleHistory(
    plate: string,
    limit = 50,
    cursor: string | null = null
  ): Observable<VehicleSearchPage> {
    let params = new HttpParams().set('plate', plate).set('limit', limit);
    if (cursor) params = params.set('cursor', cursor);
    return this.http.get<VehicleSearchPage>('/api/vehicles/search', { params });
  }

  vehicleIdentity(vehicleId: string): Observable<VehicleIdentity> {
    return this.http.get<VehicleIdentity>(
      '/api/vehicles/' + encodeURIComponent(vehicleId)
    );
  }

  vehicleTimeline(vehicleId: string, limit = 1000): Observable<VehicleTimeline> {
    return this.http.get<VehicleTimeline>(
      '/api/vehicles/' + encodeURIComponent(vehicleId) + '/timeline',
      { params: { limit } }
    );
  }

  vehicleJourney(vehicleId: string, limit = 1000): Observable<VehicleJourney> {
    return this.http.get<VehicleJourney>(
      '/api/vehicles/' + encodeURIComponent(vehicleId) + '/journey',
      { params: { limit } }
    );
  }

  cameras(enabledOnly = false): Observable<{ items: Camera[] }> {
    return this.http.get<{ items: Camera[] }>('/api/cameras', {
      params: { enabledOnly }
    });
  }

  cameraHealth(cameraId: string): Observable<CameraHealth> {
    return this.http.get<CameraHealth>(
      '/api/cameras/' + encodeURIComponent(cameraId) + '/health'
    );
  }

  liveMonitorState(cameraId: string): Observable<LiveMonitorState> {
    return this.http.get<LiveMonitorState>(
      '/api/cameras/' + encodeURIComponent(cameraId) + '/live'
    );
  }

  liveMonitorFrame(cameraId: string, sequence: number): Observable<HttpResponse<Blob>> {
    return this.http.get(
      '/api/cameras/' + encodeURIComponent(cameraId) + '/live/frame',
      {
        params: { sequence },
        observe: 'response',
        responseType: 'blob'
      }
    );
  }

  liveMonitorHealth(): Observable<LiveMonitorHealth> {
    return this.http.get<LiveMonitorHealth>('/api/live-monitor/health');
  }

  createCamera(request: CameraCreateRequest): Observable<Camera> {
    return this.http.post<Camera>('/api/cameras', request);
  }

  createCameraBatch(requests: CameraCreateRequest[]): Observable<CameraBatchResult> {
    return this.http.post<CameraBatchResult>('/api/cameras/batch', { items: requests });
  }

  discoverOnvifCameras(): Observable<OnvifDiscoveryResult> {
    return this.http.post<OnvifDiscoveryResult>('/api/cameras/discover', {});
  }

  setCameraEnabled(cameraId: string, enabled: boolean): Observable<Camera> {
    const action = enabled ? 'enable' : 'disable';
    return this.http.post<Camera>(
      '/api/cameras/' + encodeURIComponent(cameraId) + '/' + action,
      {}
    );
  }

  testCamera(cameraId: string): Observable<CameraConnectionTest> {
    return this.http.post<CameraConnectionTest>(
      '/api/cameras/' + encodeURIComponent(cameraId) + '/test-connection',
      {}
    );
  }

  alerts(filters: {
    limit?: number;
    cursor?: string | null;
    status?: AlertStatus | '';
    plate?: string;
    cameraId?: string;
    ruleId?: string;
  } = {}): Observable<AlertPage> {
    let params = new HttpParams();
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== null && value !== '') {
        params = params.set(key, String(value));
      }
    }
    return this.http.get<AlertPage>('/api/alerts', { params });
  }

  acknowledgeAlert(alertId: string): Observable<Alert> {
    return this.http.post<Alert>(
      '/api/alerts/' + encodeURIComponent(alertId) + '/acknowledge',
      {}
    );
  }

  resolveAlert(alertId: string): Observable<Alert> {
    return this.http.post<Alert>(
      '/api/alerts/' + encodeURIComponent(alertId) + '/resolve',
      {}
    );
  }

  realtimeHealth(): Observable<RealtimeHealth> {
    return this.http.get<RealtimeHealth>('/api/realtime/health');
  }

  watchlists(filters: {
    listType?: WatchlistType | '';
    enabled?: boolean;
    limit?: number;
  } = {}): Observable<{ items: WatchlistEntry[] }> {
    let params = new HttpParams();
    for (const [key, value] of Object.entries(filters)) {
      if (value !== undefined && value !== '') params = params.set(key, String(value));
    }
    return this.http.get<{ items: WatchlistEntry[] }>('/api/watchlists', { params });
  }

  createWatchlist(request: WatchlistWriteRequest): Observable<WatchlistEntry> {
    return this.http.post<WatchlistEntry>('/api/watchlists', request);
  }

  updateWatchlist(entryId: string, request: WatchlistUpdateRequest): Observable<WatchlistEntry> {
    return this.http.put<WatchlistEntry>(
      '/api/watchlists/' + encodeURIComponent(entryId),
      request
    );
  }

  deleteWatchlist(entryId: string): Observable<void> {
    return this.http.delete<void>('/api/watchlists/' + encodeURIComponent(entryId));
  }

  rules(enabledOnly = false, limit = 200): Observable<{ items: Rule[] }> {
    return this.http.get<{ items: Rule[] }>('/api/rules', {
      params: { enabledOnly, limit }
    });
  }

  createRule(request: RuleWriteRequest): Observable<Rule> {
    return this.http.post<Rule>('/api/rules', request);
  }

  updateRule(ruleId: string, request: RuleUpdateRequest): Observable<Rule> {
    return this.http.put<Rule>('/api/rules/' + encodeURIComponent(ruleId), request);
  }

  deleteRule(ruleId: string): Observable<void> {
    return this.http.delete<void>('/api/rules/' + encodeURIComponent(ruleId));
  }
}
