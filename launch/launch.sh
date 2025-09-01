#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 ../app/launcher/entry.py
else
  echo "Python3 not found. Please install Python 3.8+ and retry." >&2
  exit 1
fi

