#!/usr/bin/env bash
set -euo pipefail

# Validate a Berry macOS .pkg installer by inspecting and performing a smoke
# install on a macOS host.
#
# This is intended to run in CI on GitHub Actions macOS runners.

PKG_PATH="${1:-}"
if [[ -z "$PKG_PATH" ]]; then
  echo "usage: scripts/validate_macos_pkg.sh /path/to/berry.pkg" >&2
  exit 2
fi

if [[ ! -f "$PKG_PATH" ]]; then
  echo "pkg not found: $PKG_PATH" >&2
  exit 2
fi

echo "[validate] pkg: $PKG_PATH"

if ! command -v pkgutil >/dev/null 2>&1; then
  echo "pkgutil not found (this script must run on macOS)" >&2
  exit 2
fi

TMPDIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMPDIR" || true
  # If we installed a dummy cli into /usr/local/bin, clean it up.
  if [[ "${CREATED_DUMMY_CLAUDE:-0}" == "1" ]]; then
    sudo rm -f "${DUMMY_CLAUDE:-/usr/local/bin/claude}" || true
    sudo rm -f "${POST_CLAUDE_LOG:-/tmp/berry_validate_claude_postinstall.log}" || true
  fi
}
trap cleanup EXIT

echo "[validate] expanding pkg to inspect scripts..."
pkgutil --expand-full "$PKG_PATH" "$TMPDIR/expanded"

POSTINSTALL="$TMPDIR/expanded/Scripts/postinstall"
if [[ ! -f "$POSTINSTALL" ]]; then
  echo "ERROR: postinstall script missing from pkg" >&2
  find "$TMPDIR/expanded" -maxdepth 3 -type f -print >&2 || true
  exit 1
fi
if [[ ! -x "$POSTINSTALL" ]]; then
  echo "ERROR: postinstall exists but is not executable" >&2
  ls -l "$POSTINSTALL" >&2 || true
  exit 1
fi
echo "[validate] postinstall present: $POSTINSTALL"

# Validate that the pkg's postinstall hook performs best-effort integration
# (without requiring a real Claude installation).
POST_CLAUDE_LOG="/tmp/berry_validate_claude_postinstall.log"
DUMMY_CLAUDE="/usr/local/bin/claude"
CREATED_DUMMY_CLAUDE=0
if [[ ! -e "$DUMMY_CLAUDE" ]]; then
  echo "[validate] installing dummy claude CLI to validate postinstall integration"
  sudo rm -f "$POST_CLAUDE_LOG" || true
  sudo /bin/mkdir -p "/usr/local/bin"
  sudo /usr/bin/tee "$DUMMY_CLAUDE" >/dev/null <<'SH'
#!/usr/bin/env bash
set -euo pipefail
echo "$0 $*" >> "/tmp/berry_validate_claude_postinstall.log"
exit 0
SH
  sudo /bin/chmod 0755 "$DUMMY_CLAUDE"
  CREATED_DUMMY_CLAUDE=1
fi

echo "[validate] installing pkg..."
sudo installer -pkg "$PKG_PATH" -target / >/dev/null

if [[ "$CREATED_DUMMY_CLAUDE" == "1" ]]; then
  if [[ ! -f "$POST_CLAUDE_LOG" ]]; then
    echo "ERROR: postinstall did not invoke dummy claude (log missing)" >&2
    exit 1
  fi
  grep -q "claude mcp add berry" "$POST_CLAUDE_LOG"
  echo "[validate] postinstall integration invoked dummy claude"
fi

echo "[validate] verifying binary is on PATH..."
if ! command -v berry >/dev/null 2>&1; then
  echo "ERROR: berry not found on PATH after installation" >&2
  exit 1
fi

echo "[validate] berry version: $(berry version)"

echo "[validate] smoke test: berry init in a temp git repo"
REPO="$TMPDIR/repo"
mkdir -p "$REPO"
cd "$REPO"
git init -q
echo "hello" > README.md
berry init >/dev/null

test -f "$REPO/.cursor/mcp.json"
test -f "$REPO/.codex/config.toml"
test -f "$REPO/.mcp.json"
test -f "$REPO/.gemini/settings.json"

echo "[validate] smoke test: berry integrate with a dummy claude cli"
TESTBIN="$TMPDIR/testbin"
mkdir -p "$TESTBIN"
export PATH="$TESTBIN:$PATH"

# Dummy claude that records invocations.
export BERRY_TEST_CLAUDE_LOG="$TMPDIR/claude_invocations.txt"
cat > "$TESTBIN/claude" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
echo "$0 $*" >> "$BERRY_TEST_CLAUDE_LOG"
exit 0
SH
chmod +x "$TESTBIN/claude"

berry integrate --client claude --name berry --global >/dev/null

test -f "$BERRY_TEST_CLAUDE_LOG"
grep -q "claude mcp add berry" "$BERRY_TEST_CLAUDE_LOG"

echo "[validate] ok"