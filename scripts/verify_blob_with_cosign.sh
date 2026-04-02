#!/usr/bin/env bash
set -euo pipefail

if ! command -v cosign >/dev/null 2>&1; then
  echo "cosign not found on PATH" >&2
  exit 2
fi

ARTIFACT="${1:-}"
SIGNATURE="${2:-}"
PUBKEY="${3:-}"

if [[ -z "${ARTIFACT}" || -z "${SIGNATURE}" || -z "${PUBKEY}" ]]; then
  echo "Usage: $0 <artifact_path> <signature_path> <cosign_public_key_path>" >&2
  exit 2
fi

cosign verify-blob --key "${PUBKEY}" --signature "${SIGNATURE}" "${ARTIFACT}"

