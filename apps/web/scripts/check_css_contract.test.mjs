import assert from 'node:assert/strict';
import test from 'node:test';

import { findUndefinedCustomProperties } from './check_css_contract.mjs';

test('accepts custom properties declared anywhere in the stylesheet', () => {
  const styles = ':root { --surface: #fff; } .panel { color: var(--surface); }';
  assert.deepEqual(findUndefinedCustomProperties(styles), []);
});

test('reports every undefined custom property in stable order', () => {
  const styles = '.panel { color: var(--text); border-color: var(--border, #fff); }';
  assert.deepEqual(findUndefinedCustomProperties(styles), ['--border', '--text']);
});
