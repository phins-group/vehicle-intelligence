import { describe, expect, it } from 'vitest';

import { DetectorDatasetVersion } from '../models/api.models';
import {
  isModelTrainingActive,
  trainingDatasetReadiness
} from './model-training-utils';

describe('model training utilities', () => {
  it('accepts only a manifest-bound completed private Hub dataset revision', () => {
    const dataset = readyDataset();

    expect(trainingDatasetReadiness(dataset, 'phins-group/plate-dataset')).toEqual({
      ready: true,
      blockers: []
    });

    dataset.latestSync!.exportManifestSha256 = 'c'.repeat(64);
    expect(trainingDatasetReadiness(dataset, 'phins-group/plate-dataset')).toEqual({
      ready: false,
      blockers: ['DATASET_REVISION_MISMATCH']
    });
  });

  it('treats only pre-terminal run states as active', () => {
    expect(isModelTrainingActive('QUEUED')).toBe(true);
    expect(isModelTrainingActive('RUNNING')).toBe(true);
    expect(isModelTrainingActive('COMPLETED')).toBe(false);
    expect(isModelTrainingActive('CANCELED')).toBe(false);
  });
});

function readyDataset(): DetectorDatasetVersion {
  return {
    sourceId: 'phins-vn-plate',
    sourceManifestSha256: 'a'.repeat(64),
    createdAt: '2026-08-12T01:00:00Z',
    sampleCount: 38122,
    annotationCount: 54502,
    negativeSampleCount: 100,
    reviewQueueCount: 0,
    releaseEligible: true,
    distributionEligible: false,
    privacyClassification: 'RESTRICTED',
    parentSourceId: null,
    export: {
      exportId: 'phins-vn-plate',
      manifestSha256: 'b'.repeat(64),
      createdAt: '2026-08-12T01:00:00Z',
      sampleCount: 38122,
      annotationCount: 54502,
      negativeSampleCount: 100,
      splitCounts: { train: 26685, validation: 5718, test: 5719 },
      releaseEligible: true,
      distributionEligible: false,
      sourceManifestSha256: 'a'.repeat(64)
    },
    latestSync: {
      id: 'dataset-sync-1',
      sourceId: 'phins-vn-plate',
      sourceManifestSha256: 'a'.repeat(64),
      exportId: 'phins-vn-plate',
      repoId: 'phins-group/plate-dataset',
      requestedRevision: 'main',
      status: 'COMPLETED',
      requestedBy: 'admin',
      restrictedTransferConfirmed: true,
      createdAt: '2026-08-12T01:00:00Z',
      updatedAt: '2026-08-12T01:00:00Z',
      exportManifestSha256: 'b'.repeat(64),
      hubCommitSha: 'c'.repeat(40),
      hubUrl: null,
      reusedExport: false,
      errorCode: null
    }
  };
}
