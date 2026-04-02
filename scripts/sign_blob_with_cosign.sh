#!/usr/bin/env bash
set -euo pipefail

if ! command -v cosign >/dev/null 2>&1; then
  echo "cosign not found on PATH" >&2
  exit 2
fi

ARTIFACT="${1:-}"
KEY="${2:-}"

if [[ -z "${ARTIFACT}" || -z "${KEY}" ]]; then
  echo "Usage: $0 <artifact_path> <cosign_private_key_path>" >&2
  exit 2
fi

cosign sign-blob --key "${KEY}" "${ARTIFACT}"

