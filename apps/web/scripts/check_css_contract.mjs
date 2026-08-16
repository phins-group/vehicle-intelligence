import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const DEFINITION = /(--[a-z0-9-]+)\s*:/gi;
const USAGE = /var\(\s*(--[a-z0-9-]+)/gi;

export function findUndefinedCustomProperties(styles) {
  const defined = new Set([...styles.matchAll(DEFINITION)].map((match) => match[1]));
  const used = new Set([...styles.matchAll(USAGE)].map((match) => match[1]));
  return [...used].filter((property) => !defined.has(property)).sort();
}

async function main() {
  const stylesheet = new URL('../src/styles.scss', import.meta.url);
  const styles = await readFile(stylesheet, 'utf8');
  const undefinedProperties = findUndefinedCustomProperties(styles);
  if (undefinedProperties.length > 0) {
    throw new Error(`undefined CSS custom properties: ${undefinedProperties.join(', ')}`);
  }
  console.log('CSS custom-property contract passed');
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  });
}
