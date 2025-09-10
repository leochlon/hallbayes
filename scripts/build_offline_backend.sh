#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python3}
BUILD_VENV_DIR=${BUILD_VENV_DIR:-.build-venv}

echo "Using bootstrap Python: $PYTHON_BIN"
echo "Build venv: $BUILD_VENV_DIR"

# Create or reuse a local virtualenv to avoid system package mods
if [[ ! -d "$BUILD_VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$BUILD_VENV_DIR"
fi

VENV_PY="$BUILD_VENV_DIR/bin/python"
echo "Using venv Python: $VENV_PY"

# Ensure build deps (in venv)
"$VENV_PY" -m pip install --upgrade pip wheel setuptools
"$VENV_PY" -m pip install pyinstaller

# Ensure runtime deps are present in the build env so PyInstaller can collect them
"$VENV_PY" -m pip install -r requirements.txt

# Build onefile binary
OUTDIR=bin
mkdir -p "$OUTDIR"
"$VENV_PY" -m PyInstaller --noconfirm --onefile scripts/backend_entry.py \
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
