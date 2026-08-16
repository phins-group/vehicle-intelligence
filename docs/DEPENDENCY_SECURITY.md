# Dependency security policy

## Python cryptography baseline

The supported lock now requires `cryptography >=50,<51` on every platform.
Version 50 fixes PYSEC-2026-3552, and retaining the old macOS Intel wheel at
48.0.1 would also retain PYSEC-2026-3553 and PYSEC-2026-3554. CI therefore runs
`pip-audit` with no vulnerability exceptions.

PyCA does not publish a macOS x86_64 wheel for cryptography 50. Intel macOS
installations consequently build it from source and need Xcode command-line
tools, OpenSSL, and a Rust target matching the x86_64 Python interpreter. For a
Rust toolchain installed natively on Apple Silicon but a Python interpreter run
under Rosetta, install the cross target with:

```bash
rustup target add x86_64-apple-darwin
```

Prebuilt-wheel support for macOS Intel is dropped; the vulnerable 48.x fallback
must not be restored. Operators unable to provide the source-build toolchain
should use Linux or Apple Silicon.

## Web build-tool exceptions

Production web dependencies must pass `npm audit --omit=dev` at high severity.
The complete build graph is checked against an exact, expiring allowlist rather
than the previous broad critical-only gate. Two compatible overrides remove the
fixed advisories without changing Angular major versions:

- `undici` 7.29.0 under `@angular/build`;
- `webpack-dev-server` 5.2.6 under `@angular-devkit/build-angular`.

Three upstream build-time advisories remain until a compatible dependency is
published: GHSA-5p2g-fcmc-qvqq and GHSA-w3rx-r6r6-pgpr in optional
`image-size`, plus GHSA-w5hq-g745-h8pq in `uuid` through SockJS. They are absent
from the compiled Nginx runtime image. Build workers must remain isolated, the
development server must not be internet-exposed, and untrusted image files must
not be passed to the Less build pipeline. The policy expires on 2026-10-01 and
fails CI for any new advisory, changed severity/package, or stale exception.

Tracking item `TODO(security-deps/angular-22)` owns the migration:

1. Re-check for a fixed `image-size` release and a SockJS/webpack-dev-server
   release that consumes `uuid >=11.1.1`; prefer removing an exception without
   changing Angular.
2. In a dedicated branch, run the Angular updater for aligned Core, CLI, compiler
   and builder 22.x packages. Review its Node and TypeScript engine/peer changes;
   do not update only `@angular-devkit/build-angular`.
3. Remove overrides that the upgraded builder no longer needs, regenerate the
   lockfile, and require the full audit to reach zero exceptions.
4. Run unit tests, type checking, production build, non-root container smoke,
   REST/WebSocket dashboard acceptance, and a manual cross-browser pass before
   merging the major upgrade.
