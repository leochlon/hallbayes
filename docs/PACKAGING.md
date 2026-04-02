# Packaging / distribution

Berry is a Python CLI, but we ship a macOS installer for “just works” installs and a Homebrew cask for easy upgrades.

## GitHub Actions release pipeline

Workflow: `.github/workflows/release-macos.yml`

On `v*` tags it:

1) Builds a standalone `berry` macOS executable with PyInstaller (arm64)
2) Codesigns the executable (Developer ID Application)
3) Builds a `.pkg` installer and signs it (Developer ID Installer)
4) Notarizes + staples the pkg
5) Installs the pkg on the macOS runner and runs a smoke‑test (`scripts/validate_macos_pkg.sh`)
6) Uploads the pkg to the GitHub release
7) Optionally updates the `hassana-labs/homebrew-tap` cask with new version + sha256 (or attach `berry.rb` for manual update)

## Required GitHub secrets (for signed + notarized `.pkg`)

Signing:

- `APPLE_KEYCHAIN_PASSWORD`
- `APPLE_APP_CERT_P12_BASE64`
- `APPLE_APP_CERT_P12_PASSWORD`
- `APPLE_APP_SIGN_IDENTITY` (e.g. `Developer ID Application: …`)
- `APPLE_INSTALLER_CERT_P12_BASE64`
- `APPLE_INSTALLER_CERT_P12_PASSWORD`
- `APPLE_INSTALLER_SIGN_IDENTITY` (e.g. `Developer ID Installer: …`)

Notes:
- The workflow imports the Developer ID **G2** intermediate certificate from `scripts/apple_certs/developer_id_ca_g2.pem` to ensure `codesign` can build a chain on runners.
- Identity strings should match the certificate “Common Name” exactly. You can list yours with `security find-identity -p codesigning`.

Notarization (App Store Connect API key):

- `APPLE_NOTARY_KEY_P8_BASE64`
- `APPLE_NOTARY_KEY_ID`
- `APPLE_NOTARY_ISSUER_ID`

Homebrew tap update (optional):

- `HOMEBREW_TAP_TOKEN` (must have write access to `hassana-labs/homebrew-tap`)

## No secrets? You still get artifacts

If Apple signing / notarization secrets are not configured, the release workflow still builds and uploads **unsigned** `.pkg` artifacts, and attaches a generated `berry.rb` cask file to the GitHub release.

Notes:
- Unsigned / un-notarized pkgs may trigger Gatekeeper warnings on install.
- Without `HOMEBREW_TAP_TOKEN`, the workflow won’t push to the tap repo; you can manually copy `berry.rb` into your tap as `Casks/berry.rb`.

## Cut a release

1) Update `version = "..."` in `pyproject.toml`
2) Push a matching tag:

```bash
git tag "vX.Y.Z"
git push origin "vX.Y.Z"
```

The workflow requires the tag version to match `pyproject.toml`.

## Local smoke build (unsigned)

On macOS:

```bash
python -m pip install -e ".[dev]"
python -m pip install -U pyinstaller
bash scripts/build_macos_pkg.sh
```

This produces an unsigned pkg in `./build/macos_pkg/` unless signing/notary env vars are provided.

## Postinstall behavior

The pkg includes a `postinstall` hook (`scripts/macos_pkg_scripts/postinstall`) that:

- Validates the `berry` binary is installed and runnable.
- Attempts *best‑effort* global MCP registration for CLI clients (currently Claude Code and Codex) by running
  `berry integrate` as the active console user.

Integration is non‑fatal: installation will not fail if the user does not have those client CLIs installed.

Repo‑scoped setup is still done with `berry init` in each repo.
