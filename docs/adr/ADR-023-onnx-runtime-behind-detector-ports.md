# ADR-023: Put optimized runtimes behind detector ports

- Status: accepted
- Date: 2026-08-10

## Context

Phase 4 requires measurable acceleration without coupling the domain pipeline to
ONNX Runtime or TensorRT. Hardware capabilities differ between edge nodes, and a
silent provider fallback would create misleading benchmark and capacity data.

## Decision

Select Ultralytics/ONNX Runtime/TensorRT-EP through a detector factory behind the
existing ports. Require explicit artifact identity and SHA-256 verification.
Require the requested execution provider to exist, then register only supported
fallbacks. Standardize repeatable JSON benchmark reports and threshold gates
before changing production provider configuration.

## Consequences

- Domain and application orchestration remain runtime-agnostic.
- CPU, CoreML, CUDA, and TensorRT paths use one decoded detector contract.
- Unsupported hardware fails visibly rather than being mislabeled.
- Each target host/model still needs representative accuracy and performance
  validation; one machine's speedup is not a universal deployment decision.
