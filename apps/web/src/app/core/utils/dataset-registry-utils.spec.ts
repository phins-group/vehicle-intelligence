import { describe, expect, it } from 'vitest';

import { DetectorDatasetVersion } from '../models/api.models';
import {
  datasetReadiness,
  defaultDatasetExportId,
  isDatasetSyncActive
} from './dataset-registry-utils';

describe('dataset registry utilities', () => {
  it('derives a stable export id from a promoted source id', () => {
    expect(defaultDatasetExportId('phins-vn-plate-production-source-v2')).toBe(
      'phins-vn-plate-production-v2'
    );
  });

  it('classifies source readiness and active sync states', () => {
    const dataset = {
      reviewQueueCount: 0,
      export: null,
      latestSync: null
    } as DetectorDatasetVersion;

    expect(datasetReadiness(dataset)).toBe('READY_TO_EXPORT');
    expect(isDatasetSyncActive('PREPARING_EXPORT')).toBe(true);
    expect(isDatasetSyncActive('COMPLETED')).toBe(false);
  });
});

