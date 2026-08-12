import {
  DatasetHubSyncStatus,
  DetectorDatasetVersion
} from '../models/api.models';

export function defaultDatasetExportId(sourceId: string): string {
  return sourceId.replace(/-source-(v\d+)$/i, '-$1');
}

export function isDatasetSyncActive(status: DatasetHubSyncStatus | undefined): boolean {
  return status === 'QUEUED' || status === 'PREPARING_EXPORT' || status === 'UPLOADING';
}

export function datasetReadiness(dataset: DetectorDatasetVersion):
  | 'REVIEW_PENDING'
  | 'READY_TO_EXPORT'
  | 'EXPORTED'
  | 'SYNCED' {
  if (dataset.reviewQueueCount > 0) return 'REVIEW_PENDING';
  if (dataset.latestSync?.status === 'COMPLETED') return 'SYNCED';
  if (dataset.export) return 'EXPORTED';
  return 'READY_TO_EXPORT';
}

