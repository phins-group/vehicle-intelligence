import {
  DetectorDatasetVersion,
  ModelTrainingRunStatus
} from '../models/api.models';

export interface TrainingDatasetReadiness {
  ready: boolean;
  blockers: string[];
}

export function isModelTrainingActive(status: ModelTrainingRunStatus | undefined): boolean {
  return (
    status === 'QUEUED' ||
    status === 'SUBMITTING' ||
    status === 'SCHEDULING' ||
    status === 'RUNNING'
  );
}

export function trainingDatasetReadiness(
  dataset: DetectorDatasetVersion | null,
  expectedRepoId: string
): TrainingDatasetReadiness {
  if (!dataset) return { ready: false, blockers: ['DATASET_NOT_SELECTED'] };
  const blockers: string[] = [];
  if (dataset.reviewQueueCount > 0) blockers.push('DATASET_REVIEW_PENDING');
  if (!dataset.releaseEligible) blockers.push('DATASET_NOT_RELEASE_ELIGIBLE');
  if (!dataset.export) blockers.push('DATASET_EXPORT_MISSING');
  if (dataset.latestSync?.status !== 'COMPLETED') blockers.push('DATASET_SYNC_INCOMPLETE');
  if (
    dataset.export &&
    dataset.latestSync?.status === 'COMPLETED' &&
    (dataset.export.sourceManifestSha256 !== dataset.sourceManifestSha256 ||
      dataset.latestSync.sourceManifestSha256 !== dataset.sourceManifestSha256 ||
      dataset.latestSync.exportManifestSha256 !== dataset.export.manifestSha256)
  ) {
    blockers.push('DATASET_REVISION_MISMATCH');
  }
  if (
    dataset.latestSync?.status === 'COMPLETED' &&
    dataset.latestSync.repoId !== expectedRepoId
  ) {
    blockers.push('DATASET_REPOSITORY_MISMATCH');
  }
  if (dataset.latestSync?.status === 'COMPLETED' && !dataset.latestSync.hubCommitSha) {
    blockers.push('DATASET_COMMIT_MISSING');
  }
  return { ready: blockers.length === 0, blockers };
}

export function modelTrainingBlockerMessage(code: string): string {
  const messages: Record<string, string> = {
      MODEL_TRAINING_DISABLED: 'Chức năng training đang tắt ở API.',
      HUGGING_FACE_DISABLED: 'Hugging Face integration đang tắt.',
      HUGGING_FACE_JOBS_DISABLED: 'Hugging Face Jobs chưa được bật trong model-training.yaml.',
      HUGGING_FACE_CREDENTIALS_MISSING: 'API chưa nhận HF_TOKEN có quyền tạo Job.',
      TRAINING_IMAGE_MISSING: 'Chưa cấu hình training image đã pin digest.',
      OUTPUT_BUCKET_MISSING: 'Chưa cấu hình private Hugging Face output bucket.',
      DATASET_NOT_SELECTED: 'Chưa chọn dataset.',
      DATASET_REVIEW_PENDING: 'Dataset vẫn còn mẫu chờ duyệt.',
      DATASET_NOT_RELEASE_ELIGIBLE: 'Dataset chưa đạt điều kiện release nội bộ.',
      DATASET_EXPORT_MISSING: 'Dataset chưa có COCO export đã xác minh.',
      DATASET_SYNC_INCOMPLETE: 'Dataset chưa sync thành công lên private Hugging Face Hub.',
      DATASET_REVISION_MISMATCH: 'Source, export và Hub sync không cùng một manifest.',
      DATASET_REPOSITORY_MISMATCH: 'Dataset được sync tới repository khác cấu hình training.',
      DATASET_COMMIT_MISSING: 'Hub sync chưa ghi nhận commit bất biến.'
  };
  return messages[code] ?? code;
}

export function trainingStatusLabel(status: ModelTrainingRunStatus): string {
  const labels: Record<ModelTrainingRunStatus, string> = {
      QUEUED: 'ĐANG XẾP HÀNG',
      SUBMITTING: 'ĐANG GỬI JOB',
      SCHEDULING: 'ĐANG CẤP GPU',
      RUNNING: 'ĐANG TRAIN',
      COMPLETED: 'TRAIN XONG',
      FAILED: 'THẤT BẠI',
      CANCELED: 'ĐÃ HỦY'
  };
  return labels[status];
}
