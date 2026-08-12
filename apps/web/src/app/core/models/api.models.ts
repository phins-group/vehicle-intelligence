export type UserRole = 'ADMIN' | 'OPERATOR' | 'VIEWER';
export type Direction = 'ENTER' | 'EXIT' | 'UNKNOWN';
export type EventType = 'VEHICLE_ENTER' | 'VEHICLE_EXIT' | 'VEHICLE_DETECTED';
export type EventStatus =
  | 'CONFIRMED'
  | 'LOW_CONFIDENCE'
  | 'NEEDS_REVIEW'
  | 'NO_PLATE'
  | 'UNREADABLE';
export type CameraStatus = 'CONNECTING' | 'ONLINE' | 'OFFLINE' | 'STOPPED';
export type AlertStatus = 'OPEN' | 'ACKNOWLEDGED' | 'RESOLVED';
export type AlertSeverity = 'INFO' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type WatchlistType =
  | 'WHITELIST'
  | 'BLACKLIST'
  | 'VIP'
  | 'STAFF'
  | 'CONTRACTOR'
  | 'DELIVERY';
export type RuleConditionOperator = 'EQ' | 'NEQ' | 'IN' | 'NOT_IN' | 'CONTAINS' | 'EXISTS';
export type RuleConditionField =
  | 'watchlist'
  | 'camera.id'
  | 'camera.zone'
  | 'direction'
  | 'eventType'
  | 'status'
  | 'plate.normalized'
  | 'vehicle.type'
  | 'vehicle.color';
export type RuleActionType =
  | 'OPEN_BARRIER'
  | 'CREATE_ALERT'
  | 'WEBHOOK'
  | 'HTTP_REQUEST'
  | 'NOTIFICATION'
  | 'LOG';

export interface Principal {
  id: string;
  displayName: string;
  role: UserRole;
  authenticationMethod: 'API_KEY' | 'OIDC' | 'SYSTEM' | 'DEVELOPMENT';
}

export interface SystemHealth {
  status: string;
  phase: string;
  authentication: 'enabled' | 'disabled';
  cameraManagement: 'available' | 'unavailable';
  onvifDiscovery: 'available' | 'disabled';
  policyEngine: string;
  auditLog: string;
  mediaAccess: 'available' | 'unavailable';
  humanReview: 'available' | 'unavailable';
  datasetReview: 'available' | 'disabled';
  datasetRegistry: 'available' | 'disabled';
  modelQuality: 'available' | 'unavailable';
  liveMonitor: 'STARTING' | 'ONLINE' | 'OFFLINE' | 'STOPPED' | 'DISABLED';
  realtime: string;
}

export interface Camera {
  id: string;
  schemaVersion: number;
  revision: number;
  name: string;
  stream: {
    fpsLimit: number;
    credentialsConfigured: boolean;
  };
  location: {
    name: string | null;
    zone: string | null;
  };
  direction: 'ENTRY' | 'EXIT' | 'BOTH';
  vision: {
    vehicleConfidence: number;
    plateConfidence: number;
  };
  geometry: {
    vehicleRoi: [number, number][] | null;
    crossingLine: [number, number][] | null;
    crossingPositiveToNegative: 'ENTER' | 'EXIT';
    finalizeOnCrossing: boolean;
  };
  enabled: boolean;
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface CameraHealth {
  cameraId: string;
  status: CameraStatus;
  sourceFps: number;
  decodeFps: number;
  queueSize: number;
  droppedFrames: number;
  reconnectCount: number;
  connectionFailures: number;
  streamEpoch: number;
  lastFrameAt: string | null;
  updatedAt: string;
  decodedFrames: number;
  sampledFrames: number;
  vehicleDetections: number;
  plateDetections: number;
  ocrRequests: number;
  ocrSuccess: number;
  eventsCreated: number;
  trackCount: number;
  inferenceFps: number;
  vehicleInferenceLatencyMs: number;
  plateInferenceLatencyMs: number;
  ocrLatencyMs: number;
}

export interface CameraConnectionTest {
  connected: boolean;
  latencyMs: number;
  testedAt: string;
  errorCode: string | null;
}

export interface OnvifDiscoveredDevice {
  endpointReference: string;
  serviceAddresses: string[];
  types: string[];
  scopes: string[];
  remoteAddress: string | null;
  name: string | null;
  hardware: string | null;
  locations: string[];
  metadataVersion: number | null;
  discoveredAt: string;
}

export interface OnvifDiscoveryResult {
  items: OnvifDiscoveredDevice[];
  count: number;
}

export type CameraBatchStatus = 'CREATED' | 'CONFLICT' | 'CAPACITY_REACHED';

export interface CameraBatchResult {
  items: Array<{
    cameraId: string;
    status: CameraBatchStatus;
    camera: Camera | null;
  }>;
  createdCount: number;
  conflictCount: number;
  capacityReachedCount: number;
}

export type LiveMonitorStatus = 'DISABLED' | 'WAITING' | 'LIVE' | 'STALE' | 'OFFLINE';
export type LiveMonitorSourceState = 'STARTING' | 'ONLINE' | 'OFFLINE' | 'STOPPED';

export interface LivePlateOverlay {
  bbox: [number, number, number, number];
  detectionConfidence: number;
  qualityScore: number | null;
  text: string | null;
  ocrConfidence: number | null;
}

export interface LiveVehicleOverlay {
  trackId: string;
  bbox: [number, number, number, number];
  confidence: number;
  vehicleType: string;
  direction: Direction;
  plate: LivePlateOverlay | null;
}

export interface LiveMonitorFrame {
  sequence: number;
  frameId: number;
  streamEpoch: number;
  capturedAt: string;
  receivedAt: string;
  sourceWidth: number;
  sourceHeight: number;
  previewWidth: number;
  previewHeight: number;
  vehicles: LiveVehicleOverlay[];
  vehicleRoi: [number, number][] | null;
  crossingLine: [number, number][] | null;
  frameUrl: string;
}

export interface LiveMonitorState {
  cameraId: string;
  status: LiveMonitorStatus;
  sourceState: LiveMonitorSourceState;
  latest: LiveMonitorFrame | null;
}

export interface LiveMonitorHealth {
  status: LiveMonitorSourceState | 'DISABLED' | 'UNAVAILABLE';
  camerasBuffered: number;
  framesReceived?: number;
  framesEvicted?: number;
  reconnectCount?: number;
  sourceFailures?: number;
  invalidMessages?: number;
  lastFrameAt?: string | null;
}

export interface CameraCreateRequest {
  id: string;
  name: string;
  stream: {
    rtspUrl: string;
    fpsLimit: number;
  };
  location: {
    name: string | null;
    zone: string | null;
  };
  direction: 'ENTRY' | 'EXIT' | 'BOTH';
  vision: {
    vehicleConfidence: number;
    plateConfidence: number;
  };
  enabled: boolean;
  metadata: Record<string, unknown>;
}

export interface VehicleEvent {
  _id: string;
  schemaVersion: number;
  camera: {
    id: string;
    name: string;
    zone: string | null;
  };
  trackId: string;
  vehicleId: string | null;
  eventType: EventType;
  direction: Direction;
  status: EventStatus;
  plate: {
    raw: string;
    normalized: string;
    confidence: number;
    observationCount: number;
    partial?: boolean;
    corrections: Array<{
      position: number;
      from: string;
      to: string;
      confidence: number;
    }>;
    prediction?: {
      raw: string;
      normalized: string;
      confidence: number;
      observationCount: number;
      partial?: boolean;
      corrections: Array<{
        position: number;
        from: string;
        to: string;
        confidence: number;
      }>;
    };
    review?: {
      normalized: string;
      revision: number;
      reviewedAt: string;
      reviewedBy: {
        id: string;
        displayName: string;
      };
      note: string | null;
    } | null;
    final?: string;
  } | null;
  vehicle: {
    type: string;
    confidence: number;
    color: string | null;
  };
  media: {
    snapshotKey: string | null;
    vehicleCropKey: string | null;
    plateCropKey: string | null;
    clipKey: string | null;
  };
  ai: Record<string, unknown>;
  occurredAt: string;
  createdAt: string;
  metadata: Record<string, unknown>;
}

export interface EventPage {
  items: VehicleEvent[];
  nextCursor: string | null;
}

export interface VehicleSearchPage {
  query: string;
  items: VehicleEvent[];
  nextCursor: string | null;
}

export type VehicleIdentityStatus = 'ACTIVE' | 'MERGED' | 'SPLIT' | 'ARCHIVED';

export interface VehicleIdentity {
  id: string;
  schemaVersion: number;
  revision: number;
  status: VehicleIdentityStatus;
  primaryPlate: string | null;
  plates: Array<{
    text: string;
    confidence: number;
    firstSeenAt: string;
    lastSeenAt: string;
  }>;
  attributes: {
    type: string | null;
    color: string | null;
  };
  firstSeenAt: string;
  lastSeenAt: string;
  observationCount: number;
  metadata: Record<string, unknown>;
  latestEvent: VehicleEvent | null;
}

export interface JourneyObservation {
  eventId: string;
  cameraId: string;
  cameraName: string;
  zone: string | null;
  occurredAt: string;
  eventType: EventType;
  direction: Direction;
  status: EventStatus;
  plate: string | null;
  vehicleType: string;
}

export interface JourneySegment {
  fromEventId: string;
  toEventId: string;
  fromCameraId: string;
  toCameraId: string;
  departedAt: string;
  arrivedAt: string;
  elapsedSeconds: number;
  topologyEdgeId: string | null;
  expectedMinimumSeconds: number | null;
  expectedMaximumSeconds: number | null;
  feasible: boolean | null;
}

export interface VehicleJourney {
  vehicleId: string;
  observations: JourneyObservation[];
  segments: JourneySegment[];
  startedAt: string | null;
  endedAt: string | null;
  truncated: boolean;
}

export interface VehicleTimeline {
  vehicleId: string;
  items: JourneyObservation[];
}

export interface PlateReviewRequest {
  text: string;
  expectedRevision: number;
  note?: string | null;
}

export interface PlateReviewResponse {
  event: VehicleEvent;
  changed: boolean;
  feedbackReason: 'HUMAN_CORRECTION' | 'HUMAN_CONFIRMATION';
  datasetSampleId: string | null;
}

export interface DatasetSample {
  _id: string;
  schemaVersion: number;
  type: 'PLATE_OCR';
  status: 'READY' | 'EXPORTING' | 'EXPORTED' | 'EXPORT_FAILED';
  sourceEventId: string;
  imageKey: string;
  prediction: {
    raw: string;
    normalized: string;
    confidence: number;
    model: {
      name: string;
      version: string;
      hash: string | null;
    } | null;
  };
  label: string;
  reason: 'HUMAN_CORRECTION' | 'HUMAN_CONFIRMATION';
  review: {
    revision: number;
    reviewedBy: {
      id: string;
      displayName: string;
    };
    reviewedAt: string;
  };
  export?: {
    id: string;
    attempts: number;
    claimedAt: string | null;
    exportedAt: string | null;
    manifestSha256: string | null;
    errorCode: string | null;
  } | null;
  createdAt: string;
}

export type DetectorReviewStatus =
  | 'PENDING_REVIEW'
  | 'APPROVED'
  | 'CORRECTED'
  | 'NEGATIVE'
  | 'REJECTED';
export type DetectorReviewAction = 'APPROVE' | 'CORRECT' | 'MARK_NEGATIVE' | 'REJECT';

export interface DetectorReviewBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectorReviewAnnotation {
  className: 'license_plate';
  bbox: DetectorReviewBox;
  attributes: Record<string, unknown>;
}

export interface DetectorReviewDecision {
  action: DetectorReviewAction;
  status: DetectorReviewStatus;
  annotations: DetectorReviewAnnotation[];
  revision: number;
  reviewedBy: {
    id: string;
    displayName: string;
  };
  reviewedAt: string;
  note: string | null;
}

export interface DetectorReviewItem {
  sourceId: string;
  reviewId: string;
  sourceImageSha256: string;
  sourceFilenameSha256: string;
  reason: string;
  status: DetectorReviewStatus;
  revision: number;
  suggestions: DetectorReviewAnnotation[];
  decision: DetectorReviewDecision | null;
  imageUrl: string;
  image?: {
    width: number;
    height: number;
  };
}

export interface DetectorReviewSource {
  sourceId: string;
  sourceManifestSha256: string;
  sourceType: string;
  collectionMethod: string;
  rightsStatus: string;
  promotionEligible: boolean;
  releaseEligible: boolean;
  distributionEligible: boolean;
  queueCount: number;
  statusCounts: Partial<Record<DetectorReviewStatus, number>>;
  reasonCounts: Record<string, number>;
  reviewedCount: number;
  pendingCount: number;
}

export interface DetectorReviewPage {
  items: DetectorReviewItem[];
  nextCursor: string | null;
}

export interface DetectorReviewRequest {
  action: DetectorReviewAction;
  expectedRevision: number;
  annotations: DetectorReviewBox[];
  note?: string | null;
}

export interface DetectorPromotionJob {
  id: string;
  sourceId: string;
  targetSourceId: string;
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  createdAt: string;
  updatedAt: string;
  requestedBy: string;
  reviewedSampleCount: number;
  pendingSampleCount: number;
  decisionSnapshotSha256: string;
  outputDirectory: string | null;
  manifestSha256: string | null;
  errorCode: string | null;
}

export type DatasetHubSyncStatus =
  | 'QUEUED'
  | 'PREPARING_EXPORT'
  | 'UPLOADING'
  | 'COMPLETED'
  | 'FAILED';

export interface DatasetHubSyncJob {
  id: string;
  sourceId: string;
  sourceManifestSha256: string;
  exportId: string;
  repoId: string;
  requestedRevision: string;
  status: DatasetHubSyncStatus;
  requestedBy: string;
  restrictedTransferConfirmed: boolean;
  createdAt: string;
  updatedAt: string;
  exportManifestSha256: string | null;
  hubCommitSha: string | null;
  hubUrl: string | null;
  reusedExport: boolean;
  errorCode: string | null;
}

export interface DetectorDatasetExport {
  exportId: string;
  manifestSha256: string;
  createdAt: string;
  sampleCount: number;
  annotationCount: number;
  negativeSampleCount: number;
  splitCounts: Record<string, number>;
  releaseEligible: boolean;
  distributionEligible: boolean;
  sourceManifestSha256: string;
}

export interface DetectorDatasetVersion {
  sourceId: string;
  sourceManifestSha256: string;
  createdAt: string;
  sampleCount: number;
  annotationCount: number;
  negativeSampleCount: number;
  reviewQueueCount: number;
  releaseEligible: boolean;
  distributionEligible: boolean;
  privacyClassification: string;
  parentSourceId: string | null;
  export: DetectorDatasetExport | null;
  latestSync: DatasetHubSyncJob | null;
}

export interface DatasetRegistryResponse {
  items: DetectorDatasetVersion[];
  hub: {
    enabled: boolean;
    hubEnabled: boolean;
    repoId: string | null;
    credentialsConfigured: boolean;
    restrictedPrivateSyncEnabled: boolean;
  };
}

export type DetectorDatasetSampleKind = 'ALL' | 'POSITIVE' | 'NEGATIVE';

export interface DetectorDatasetSampleBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectorDatasetSampleAnnotation {
  className: string;
  bbox: DetectorDatasetSampleBox;
}

export interface DetectorDatasetSamplePreview {
  sourceId: string;
  sampleId: string;
  imageSha256: string;
  cameraId: string;
  groupId: string;
  capturedAt: string;
  split: string | null;
  lighting: string;
  annotationStatus: string | null;
  negative: boolean;
  image: { width: number; height: number };
  annotations: DetectorDatasetSampleAnnotation[];
  imageUrl: string;
}

export interface DetectorDatasetSamplePage {
  items: DetectorDatasetSamplePreview[];
  nextCursor: string | null;
}

export interface ModelQualityMetrics {
  eventCount: number;
  readablePlateCount: number;
  confirmedCount: number;
  needsReviewCount: number;
  noPlateCount: number;
  unreadableCount: number;
  reviewedCount: number;
  correctedCount: number;
  ocrSuccessRate: number;
  unknownPlateRate: number;
  humanCorrectionRate: number;
  averagePlateConfidence: number | null;
}

export interface ModelQualityReport {
  schemaVersion: number;
  window: { from: string; to: string };
  generatedAt: string;
  totals: ModelQualityMetrics;
  models: Array<{
    model: { name: string; version: string; hash: string | null } | null;
    metrics: ModelQualityMetrics;
  }>;
  daily: Array<{ day: string; metrics: ModelQualityMetrics }>;
  feedback: {
    total: number;
    ready: number;
    exporting: number;
    exported: number;
    exportFailed: number;
    corrections: number;
    confirmations: number;
  };
  truncated: boolean;
}

export type MediaAccessStatus = 'AVAILABLE' | 'MISSING';

export interface SignedMediaAsset {
  key: string;
  url: string | null;
  contentType: 'image/jpeg' | 'video/mp4' | string;
  status: MediaAccessStatus;
}

export interface EventMediaAccess {
  eventId: string;
  expiresAt: string;
  media: {
    snapshot: SignedMediaAsset | null;
    vehicleCrop: SignedMediaAsset | null;
    plateCrop: SignedMediaAsset | null;
    clip: SignedMediaAsset | null;
  };
}

export interface EventFilters {
  limit?: number;
  cursor?: string | null;
  cameraId?: string;
  plate?: string;
  eventType?: EventType | '';
  direction?: Direction | '';
  status?: EventStatus | '';
  from?: string;
  to?: string;
}

export interface Alert {
  id: string;
  schemaVersion: number;
  revision: number;
  source: {
    eventId: string;
    executionId: string;
    actionId: string;
  };
  rule: {
    id: string;
    name: string;
  };
  camera: {
    id: string;
    name: string;
    zone: string | null;
  };
  eventType: EventType;
  direction: Direction;
  severity: AlertSeverity;
  status: AlertStatus;
  message: string;
  plate: string | null;
  vehicleType: string | null;
  occurredAt: string;
  createdAt: string;
  updatedAt: string;
  acknowledgedAt: string | null;
  acknowledgedBy: string | null;
  resolvedAt: string | null;
  resolvedBy: string | null;
  metadata: Record<string, unknown>;
}

export interface AlertPage {
  items: Alert[];
  nextCursor: string | null;
}

export interface WatchlistEntry {
  id: string;
  schemaVersion: number;
  revision: number;
  plate: string;
  listType: WatchlistType;
  enabled: boolean;
  validFrom: string | null;
  validUntil: string | null;
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface WatchlistWriteRequest {
  id?: string;
  plate: string;
  listType: WatchlistType;
  enabled: boolean;
  validFrom: string | null;
  validUntil: string | null;
  metadata: Record<string, unknown>;
}

export interface WatchlistUpdateRequest extends WatchlistWriteRequest {
  revision: number;
}

export interface RuleCondition {
  field: RuleConditionField;
  operator: RuleConditionOperator;
  value: unknown;
}

export interface RuleAction {
  id: string;
  type: RuleActionType;
  parameters: Record<string, unknown>;
}

export interface Rule {
  id: string;
  schemaVersion: number;
  revision: number;
  name: string;
  enabled: boolean;
  priority: number;
  conditions: RuleCondition[];
  actions: RuleAction[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface RuleWriteRequest {
  id?: string;
  name: string;
  enabled: boolean;
  priority: number;
  conditions: RuleCondition[];
  actions: RuleAction[];
  metadata: Record<string, unknown>;
}

export interface RuleUpdateRequest extends RuleWriteRequest {
  revision: number;
}

export interface RealtimeHealth {
  status: string;
  subscribers: number;
  eventsReceived?: number;
  eventsDistributed?: number;
  duplicateEvents?: number;
  clientEventsDropped?: number;
  reconnectCount?: number;
  sourceFailures?: number;
  invalidMessages?: number;
  lastEventAt?: string | null;
}

export interface RealtimeEnvelope<T = unknown> {
  id: string;
  type: string;
  schemaVersion: number;
  occurredAt: string;
  source: string;
  correlationId?: string | null;
  data: T;
}

export interface RealtimeGap {
  reason: string;
  droppedEvents: number;
  lastAvailableEventId: string | null;
  recoveryEndpoint: string;
}
