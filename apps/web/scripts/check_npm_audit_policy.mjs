import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const POLICY_REVIEW_DATE = '2026-10-01';

// Build-time-only advisories with no non-breaking upstream fix as of 2026-08-16.
// Tracking: TODO(security-deps/angular-22). Do not add entries without a bounded
// review date, an exposure analysis, and a documented migration path.
const ALLOWED_ADVISORIES = new Map([
  ['GHSA-5p2g-fcmc-qvqq', { name: 'image-size', severity: 'high' }],
  ['GHSA-w3rx-r6r6-pgpr', { name: 'image-size', severity: 'high' }],
  ['GHSA-w5hq-g745-h8pq', { name: 'uuid', severity: 'moderate' }],
]);

function advisoryId(url) {
  if (typeof url !== 'string') {
    throw new Error('audit advisory is missing a URL');
  }
  const match = /\/advisories\/(GHSA-[a-z0-9-]+)$/.exec(url);
  if (match === null) {
    throw new Error(`unsupported advisory URL: ${url}`);
  }
  return match[1];
}

export function validateAuditReport(report, today = new Date()) {
  if (report?.auditReportVersion !== 2 || typeof report.vulnerabilities !== 'object') {
    throw new Error('expected npm audit report version 2');
  }

  const currentDate = today.toISOString().slice(0, 10);
  if (currentDate >= POLICY_REVIEW_DATE) {
    throw new Error(
      `npm advisory exception review expired on ${POLICY_REVIEW_DATE}; ` +
        'complete or re-evaluate TODO(security-deps/angular-22)',
    );
  }

  const seen = new Set();
  for (const vulnerability of Object.values(report.vulnerabilities)) {
    if (!Array.isArray(vulnerability?.via)) {
      throw new Error(`malformed vulnerability entry: ${vulnerability?.name ?? 'unknown'}`);
    }
    for (const via of vulnerability.via) {
      if (typeof via === 'string') {
        if (!(via in report.vulnerabilities)) {
          throw new Error(`unresolved npm audit dependency edge: ${via}`);
        }
        continue;
      }

      const id = advisoryId(via?.url);
      const allowed = ALLOWED_ADVISORIES.get(id);
      if (allowed === undefined) {
        throw new Error(`unapproved npm advisory: ${id}`);
      }
      if (via.name !== allowed.name || via.severity !== allowed.severity) {
        throw new Error(
          `npm advisory ${id} changed: expected ${allowed.name}/${allowed.severity}, ` +
            `received ${via.name}/${via.severity}`,
        );
      }
      seen.add(id);
    }
  }

  const stale = [...ALLOWED_ADVISORIES.keys()].filter((id) => !seen.has(id));
  if (stale.length > 0) {
    throw new Error(`remove stale npm advisory exceptions: ${stale.join(', ')}`);
  }

  return { allowed: [...seen].sort(), reviewDate: POLICY_REVIEW_DATE };
}

async function main() {
  const reportPath = process.argv[2];
  if (reportPath === undefined) {
    throw new Error('usage: node scripts/check_npm_audit_policy.mjs <npm-audit.json>');
  }
  const report = JSON.parse(await readFile(reportPath, 'utf8'));
  const result = validateAuditReport(report);
  console.log(
    `accepted ${result.allowed.length} build-time advisories; review before ${result.reviewDate}`,
  );
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
