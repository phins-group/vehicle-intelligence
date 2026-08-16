import assert from 'node:assert/strict';
import test from 'node:test';

import { validateAuditReport } from './check_npm_audit_policy.mjs';

const allowedReport = () => ({
  auditReportVersion: 2,
  vulnerabilities: {
    '@angular-devkit/build-angular': {
      name: '@angular-devkit/build-angular',
      via: ['less', 'webpack-dev-server'],
    },
    less: { name: 'less', via: ['image-size'] },
    'image-size': {
      name: 'image-size',
      via: [
        {
          name: 'image-size',
          severity: 'high',
          url: 'https://github.com/advisories/GHSA-5p2g-fcmc-qvqq',
        },
        {
          name: 'image-size',
          severity: 'high',
          url: 'https://github.com/advisories/GHSA-w3rx-r6r6-pgpr',
        },
      ],
    },
    'webpack-dev-server': { name: 'webpack-dev-server', via: ['sockjs'] },
    sockjs: { name: 'sockjs', via: ['uuid'] },
    uuid: {
      name: 'uuid',
      via: [
        {
          name: 'uuid',
          severity: 'moderate',
          url: 'https://github.com/advisories/GHSA-w5hq-g745-h8pq',
        },
      ],
    },
  },
});

test('accepts only the bounded Angular build-tool exceptions', () => {
  const result = validateAuditReport(allowedReport(), new Date('2026-08-16T00:00:00Z'));
  assert.deepEqual(result.allowed, [
    'GHSA-5p2g-fcmc-qvqq',
    'GHSA-w3rx-r6r6-pgpr',
    'GHSA-w5hq-g745-h8pq',
  ]);
});

test('rejects a newly reported advisory', () => {
  const report = allowedReport();
  report.vulnerabilities.undici = {
    name: 'undici',
    via: [
      {
        name: 'undici',
        severity: 'high',
        url: 'https://github.com/advisories/GHSA-aaaa-bbbb-cccc',
      },
    ],
  };
  assert.throws(
    () => validateAuditReport(report, new Date('2026-08-16T00:00:00Z')),
    /unapproved npm advisory/,
  );
});

test('rejects an expired exception policy', () => {
  assert.throws(
    () => validateAuditReport(allowedReport(), new Date('2026-10-01T00:00:00Z')),
    /exception review expired/,
  );
});

test('rejects stale exceptions after an upstream fix', () => {
  const report = allowedReport();
  report.vulnerabilities.uuid.via = [];
  assert.throws(
    () => validateAuditReport(report, new Date('2026-08-16T00:00:00Z')),
    /remove stale npm advisory exceptions/,
  );
});
