#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

version_from_pyproject() {
  "$PYTHON" - <<'PY'
from pathlib import Path
import re

text = Path("pyproject.toml").read_text(encoding="utf-8")
m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
if not m:
  raise SystemExit("version not found in pyproject.toml")
print(m.group(1))
PY
}

VERSION="${BERRY_VERSION:-}"
if [[ -z "$VERSION" ]]; then
  VERSION="$(cd "$ROOT" && version_from_pyproject)"
fi

ARCH="$(uname -m)"
IDENTIFIER="${BERRY_PKG_IDENTIFIER:-com.hassana.berry}"
OUT_DIR="${BERRY_PKG_OUT_DIR:-$ROOT/build/macos_pkg}"

echo "Building Berry macOS pkg"
echo "- version: $VERSION"
echo "- arch: $ARCH"
echo "- identifier: $IDENTIFIER"
echo "- out_dir: $OUT_DIR"

mkdir -p "$OUT_DIR"

if ! "$PYTHON" -c "import PyInstaller" >/dev/null 2>&1; then
  echo "PyInstaller not installed; installing into current environment..."
  "$PYTHON" -m pip install -U pyinstaller
fi

ENTRY="$ROOT/scripts/pyinstaller_entry.py"
WORK="$OUT_DIR/pyinstaller_work"
DIST="$OUT_DIR/pyinstaller_dist"
SPEC="$OUT_DIR/pyinstaller_spec"

rm -rf "$WORK" "$DIST" "$SPEC"

cd "$ROOT"
"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --noupx \
  --name berry \
  --paths "$ROOT/src" \
  --hidden-import httpx \
  --hidden-import certifi \
  --hidden-import httpcore \
  --hidden-import h11 \
  --hidden-import anyio \
  --hidden-import sniffio \
  --workpath "$WORK" \
  --distpath "$DIST" \
  --specpath "$SPEC" \
  "$ENTRY"

BIN="$DIST/berry"
if [[ ! -f "$BIN" ]]; then
  echo "Expected PyInstaller output missing: $BIN" >&2
  exit 1
fi

if [[ -n "${APPLE_APP_SIGN_IDENTITY:-}" ]]; then
  if [[ -f "$ROOT/scripts/apple_certs/developer_id_ca_g2.pem" ]]; then
    /usr/bin/openssl x509 -in "$ROOT/scripts/apple_certs/developer_id_ca_g2.pem" -outform der -out "$OUT_DIR/developer_id_ca_g2.cer" >/dev/null 2>&1 || true
    security import "$OUT_DIR/developer_id_ca_g2.cer" -k "$HOME/Library/Keychains/login.keychain-db" >/dev/null 2>&1 || true
  fi
  ENTITLEMENTS_PATH="${APPLE_APP_ENTITLEMENTS_PATH:-$ROOT/scripts/apple_certs/berry_cli_entitlements.plist}"
  if [[ ! -f "$ENTITLEMENTS_PATH" ]]; then
    echo "Entitlements file not found: $ENTITLEMENTS_PATH" >&2
    exit 1
  fi
  echo "Signing binary with codesign identity: $APPLE_APP_SIGN_IDENTITY"
  echo "- entitlements: $ENTITLEMENTS_PATH"
  /usr/bin/codesign --force --options runtime --timestamp --entitlements "$ENTITLEMENTS_PATH" --sign "$APPLE_APP_SIGN_IDENTITY" "$BIN"
fi

PKGROOT="$OUT_DIR/pkgroot"
rm -rf "$PKGROOT"
mkdir -p "$PKGROOT/usr/local/bin"
/usr/bin/install -m 0755 "$BIN" "$PKGROOT/usr/local/bin/berry"

SCRIPTS_DIR="$ROOT/scripts/macos_pkg_scripts"
if [[ ! -d "$SCRIPTS_DIR" ]]; then
  echo "Packaging scripts directory not found: $SCRIPTS_DIR" >&2
  exit 1
fi
if [[ ! -x "$SCRIPTS_DIR/postinstall" ]]; then
  echo "postinstall script missing or not executable: $SCRIPTS_DIR/postinstall" >&2
  exit 1
fi

# Avoid accidentally packaging AppleDouble (._*) files from extended attributes.
if [[ -x /usr/bin/xattr ]]; then
  /usr/bin/xattr -cr "$PKGROOT" || true
fi
find "$PKGROOT" -name '._*' -delete || true

UNSIGNED="$OUT_DIR/berry-${VERSION}-macos-${ARCH}-unsigned.pkg"
SIGNED="$OUT_DIR/berry-${VERSION}-macos-${ARCH}.pkg"

rm -f "$UNSIGNED" "$SIGNED"

/usr/bin/pkgbuild \
  --filter '(^|/)\._' \
  --filter '(^|/)\.DS_Store$' \
  --filter '(^|/)\.svn(/|$)' \
  --filter '(^|/)CVS(/|$)' \
  --root "$PKGROOT" \
  --ownership recommended \
  --scripts "$SCRIPTS_DIR" \
  --identifier "$IDENTIFIER" \
  --version "$VERSION" \
  --install-location / \
  "$UNSIGNED"

if [[ -n "${APPLE_INSTALLER_SIGN_IDENTITY:-}" ]]; then
  echo "Signing pkg with productsign identity: $APPLE_INSTALLER_SIGN_IDENTITY"
  /usr/bin/productsign --timestamp --sign "$APPLE_INSTALLER_SIGN_IDENTITY" "$UNSIGNED" "$SIGNED"
else
  cp "$UNSIGNED" "$SIGNED"
fi

NOTARY_KEY_PATH="${APPLE_NOTARY_KEY_PATH:-}"
if [[ -z "$NOTARY_KEY_PATH" && -n "${APPLE_NOTARY_KEY_P8_BASE64:-}" ]]; then
  NOTARY_KEY_PATH="$OUT_DIR/AuthKey.p8"
  echo "$APPLE_NOTARY_KEY_P8_BASE64" | /usr/bin/base64 --decode > "$NOTARY_KEY_PATH"
  chmod 0600 "$NOTARY_KEY_PATH"
fi

if [[ -n "${APPLE_NOTARY_KEY_ID:-}" && -n "${APPLE_NOTARY_ISSUER_ID:-}" && -n "$NOTARY_KEY_PATH" ]]; then
  echo "Notarizing pkg..."
  /usr/bin/xcrun notarytool submit \
    "$SIGNED" \
    --key "$NOTARY_KEY_PATH" \
    --key-id "$APPLE_NOTARY_KEY_ID" \
    --issuer "$APPLE_NOTARY_ISSUER_ID" \
    --wait
  echo "Stapling notarization ticket..."
  /usr/bin/xcrun stapler staple "$SIGNED"
fi

SHA256="$(/usr/bin/shasum -a 256 "$SIGNED" | awk '{print $1}')"
echo "Built: $SIGNED"
echo "SHA256: $SHA256"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  echo "pkg_path=$SIGNED" >> "$GITHUB_OUTPUT"
  echo "sha256=$SHA256" >> "$GITHUB_OUTPUT"
  echo "arch=$ARCH" >> "$GITHUB_OUTPUT"
  echo "version=$VERSION" >> "$GITHUB_OUTPUT"
fi
