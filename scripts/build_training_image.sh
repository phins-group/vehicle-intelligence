#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE_REPOSITORY="${TRAINING_IMAGE_REPOSITORY:-ghcr.io/phins-group/smart-cam-picodet-trainer}"
readonly IMAGE_TAG="${TRAINING_IMAGE_TAG:-v1}"
readonly PLATFORM="${TRAINING_IMAGE_PLATFORM:-linux/amd64}"
readonly IMAGE="${IMAGE_REPOSITORY}:${IMAGE_TAG}"

case "${IMAGE_REPOSITORY}" in
  *[!a-zA-Z0-9._/-]*|""|/*|*/|*//* )
    echo "TRAINING_IMAGE_REPOSITORY is invalid" >&2
    exit 2
    ;;
esac
case "${IMAGE_TAG}" in
  *[!a-zA-Z0-9._-]*|"" )
    echo "TRAINING_IMAGE_TAG is invalid" >&2
    exit 2
    ;;
esac
case "${PLATFORM}" in
  linux/amd64|linux/arm64) ;;
  *) echo "TRAINING_IMAGE_PLATFORM must be linux/amd64 or linux/arm64" >&2; exit 2 ;;
esac

command -v docker >/dev/null || { echo "docker is required" >&2; exit 2; }

if [[ "${1:-}" == "--push" ]]; then
  echo "Building and pushing ${IMAGE} for ${PLATFORM}"
  docker buildx build --platform "${PLATFORM}" --file Dockerfile.training \
    --tag "${IMAGE}" --provenance=mode=max --sbom=true --progress=plain --push .
  docker buildx imagetools inspect "${IMAGE}"
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--push]" >&2
  exit 2
else
  [[ "${PLATFORM}" == "linux/amd64" ]] || {
    echo "local load is supported only for linux/amd64 smoke testing" >&2
    exit 2
  }
  echo "Building ${IMAGE} for local smoke testing"
  docker buildx build --platform "${PLATFORM}" --file Dockerfile.training \
    --tag "${IMAGE}" --progress=plain --load .
fi
