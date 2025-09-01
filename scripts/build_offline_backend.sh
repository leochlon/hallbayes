#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "Using Python: $PYTHON_BIN"

# Ensure build deps
$PYTHON_BIN -m pip install --upgrade pip wheel setuptools pyinstaller

# Ensure runtime deps are present in the build env so PyInstaller can collect them
$PYTHON_BIN -m pip install -r requirements.txt

# Build onefile binary
OUTDIR=bin
mkdir -p "$OUTDIR"
pyinstaller --noconfirm --onefile scripts/backend_entry.py \
  --add-data app:app \
  --add-data scripts:scripts

# Move artifact to bin/ (name uniformly for Electron)
if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* || "$OS" == "Windows_NT" ]]; then
  EXE="dist/backend_entry.exe"
  DEST="$OUTDIR/hallucination-backend.exe"
else
  EXE="dist/backend_entry"
  DEST="$OUTDIR/hallucination-backend"
fi

mv -f "$EXE" "$DEST"
echo "Built offline backend: $DEST"
