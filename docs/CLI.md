# Berry CLI reference

All commands are under the `berry` CLI.

## Core

### `berry version`
Prints the installed Berry version.

### `berry mcp`
Run the MCP server.

```
berry mcp [--server classic] [--transport stdio|sse|streamable-http] [--host HOST] [--port PORT] [--project-root PATH]
```

Notes:

- Berry ships a single MCP surface: **classic**.
- Older configs may still pass `--server science` or `--server forge`; those values are treated as aliases for `classic`.

### `berry init`
Create repo‑scoped MCP config files for supported clients.

```
berry init [--profile classic] [--project-root PATH] [--force] [--strict] [--no-claude-skill]
```

Options:

- `--strict` — also write `./.berry/config.json` with `enforce_verification=true`.
- (default) writes `./.claude/rules/berry.md` so Claude Code agents learn Berry's state machine.
- `--no-claude-skill` — skip writing `./.claude/rules/berry.md`.

Safety note:

- By default, `berry init` expects to run inside a git repo (it walks up to a `.git` directory).
  If no `.git` is found, pass `--project-root` explicitly (or set `BERRY_ALLOW_NON_GIT_ROOT=1` to
  treat the current directory as the project scope).

### `berry doctor`
Print health checks and basic environment info as JSON.

### `berry status`
Print the effective config as JSON.

## Config

### `berry config show`
Same as `berry status`.

### `berry config set`
Set a boolean config key.

```
berry config set allow_write true|false
berry config set enforce_verification true|false
berry config set diagnostics_opt_in true|false
berry config set audit_log_enabled true|false
berry config set paid_features_enabled true|false
```

### `berry config add-root`
Add a filesystem root to the allowed list.

```
berry config add-root /absolute/path
```

### `berry config remove-root`
Remove a filesystem root from the allowed list.

```
berry config remove-root /absolute/path
```

## Setup / API keys

### `berry setup`
Configure your verifier provider, model, and API key / access token.

This writes a JSON object to:

- `~/.berry/mcp_env.json` (or `$BERRY_HOME/mcp_env.json`)

That file is then:

- embedded into generated repo configs (`berry init`, `berry print-config`, etc.)
- applied at MCP server startup (without overriding already-set process env)

Usage:

```bash
# Guided setup (prompts for provider, model, and key)
berry setup

# Show current setup (no secrets)
berry setup status

# Clear saved verifier setup (keeps unrelated env keys)
berry setup clear

# Non-interactive (OpenAI)
echo -n "sk-..." | berry setup --provider openai --model gpt-4o-mini --stdin

# OpenRouter (OpenAI-compatible)
berry setup --provider openrouter --model openai/gpt-4o-mini

# Local vLLM (OpenAI-compatible server)
berry setup --provider vllm --base-url http://localhost:8000/v1 --model your-model-name

# Gemini Developer API (ai.google.dev)
berry setup --provider gemini --model gemini-2.0-flash

# Non-interactive Gemini
echo -n "AIza..." | berry setup --provider gemini --model gemini-2.0-flash --stdin

# Vertex AI (Gemini via Vertex generateContent)
berry setup --provider vertex --vertex-project your-gcp-project --vertex-location us-central1 --model gemini-2.0-flash-001

# Non-interactive Vertex (access token from gcloud)
gcloud auth print-access-token | berry setup --provider vertex --model projects/PROJECT/locations/LOCATION/publishers/google/models/gemini-2.0-flash-001 --stdin

# If you only want to write ~/.berry/mcp_env.json (and NOT touch any client configs)
berry setup --no-integrate

# Skip the live "does this model return logprobs" probe (not recommended)
berry setup --no-verify
```

## Support

### `berry support bundle`
Create a redacted support bundle ZIP.

```
berry support bundle [--out /path/to/bundle.zip]
```

### `berry support issue`
Create a support bundle and print a pasteable issue template.

```
berry support issue [--out /path/to/bundle.zip]
```

## Audit log

### `berry audit export`
Export the audit log as JSON.

```
berry audit export --out /path/to/audit.json
```

### `berry audit prune`
Prune audit log entries based on retention window.

```
berry audit prune
```

## Recipes

### `berry recipes list`
List built‑in recipes.

### `berry recipes export`
Export built‑in recipes to a JSON file.

```
berry recipes export --out /path/to/recipes.json
```

### `berry recipes install`
Install a built‑in recipe into the repo.

```
berry recipes install <name> [--force]
```

### `berry recipes import`
Import a recipe JSON file into the repo.

```
berry recipes import /path/to/recipe.json [--force]
```

## Licensing

### `berry license set`
Write a local license payload (paid layer scaffolding).

```
berry license set [--plan pro] [--features feature1,feature2]
```

### `berry license show`
Print the local license payload (JSON).

## Client setup helpers

### `berry quickstart`
Print the fastest path to first value.

### `berry instructions`
Print per‑client setup guidance.

```
berry instructions [--client cursor|codex|claude|gemini] [--name berry]
```

### `berry print-config`
Print the per‑client config (for copy/paste).

```
berry print-config cursor|codex|claude|gemini [--name berry]
```

### `berry deeplink`
Print a Cursor deep‑link for MCP install.

```
berry deeplink cursor [--name berry]
```

### `berry integrate`
Best‑effort global registration for CLI clients that support it.

This is primarily for **Claude Code** and **Codex** when you want Berry to show up
without committing repo‑scoped config files.

```
berry integrate [--client claude|codex] [--name berry] [--timeout 20] [--dry-run] [--json] [--managed] [--managed-only]
```

Options:

- `--managed` — Also write system-managed config files (requires admin/sudo rights).
- `--managed-only` — Only write system-managed config files (implies `--managed`).

Notes:

- This command **skips** clients whose CLIs are not installed.
- Repo‑scoped setup is still done via `berry init`.

#### Platform support for `--managed`

| Platform | Supported | Paths |
|----------|-----------|-------|
| macOS | Yes | `/Library/Application Support/ClaudeCode/managed-mcp.json`, `/Library/Application Support/GeminiCli/settings.json` |
| Linux | Yes | `/etc/claude-code/managed-mcp.json`, `/etc/gemini-cli/settings.json` |
| Windows | No | Returns "unsupported platform" and skips gracefully |

On Windows, the `--managed` and `--managed-only` flags will return a "skipped" status with the message "unsupported platform". Use the standard `berry integrate` command (without `--managed`) for Windows, which writes to user-level config files instead.

## Verification

### `berry verify`
Verify a signed artifact with cosign.

```
berry verify --artifact /path/to/file --signature /path/to/file.sig [--public-key /path/to/key.pub]
```
