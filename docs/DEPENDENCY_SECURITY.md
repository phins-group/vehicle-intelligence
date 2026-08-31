# Dependency security policy

## Python cryptography baseline

Python 3.12.14 is the minimum repository patch baseline. It is a Python 3.12
security release and replaces the frozen 3.12.12 container base. API and edge
images use the official `slim-trixie` multi-architecture image by immutable
digest; the CUDA training image builds the same version from the python.org
source tarball and verifies its SHA-256 checksum.

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

## Web dependency baseline

The console is aligned on Angular 22.1, the non-deprecated `@angular/build`
esbuild/Vite builder, TypeScript 6.0, Node 24.15, angular-eslint 22, and
Playwright 1.62.1. Less is pinned to 4.9.0 because it is a compatible optional
builder peer that removes the vulnerable legacy `image-size` dependency.
Migrating away from `@angular-devkit/build-angular` also removes the
Webpack/SockJS/legacy-UUID branch. The runtime is the stable unprivileged Nginx
1.30.4 image on Alpine 3.24, pinned by digest; Node and Playwright remain in the
build/test graph only.

Both the production-only and complete lockfile graphs must pass `npm audit` at
`--audit-level=low`; there are no advisory exceptions or dependency overrides.
CI also runs ESLint for TypeScript and Angular templates, strict type checking,
unit tests, browser-level OIDC/RBAC tests, the production build, and a non-root
Nginx container smoke test. Container images are scanned with digest-pinned
Trivy 0.74.0; Critical findings fail the build. The only exceptions are scoped
by package in `.trivyignore.yaml`, include an evidence-based statement, and
expire on 2026-11-30 so CI forces a fresh review. They cover Debian's Essential
`perl-base`: a 32-bit-only issue, an absent `Archive::Tar` module, and a Perl
regex path that the Python application does not invoke. Suppressed findings
remain visible in CI output. The database download timeout is 30 minutes so a
slow first mirror fetch cannot silently replace the scan.
