# Berry assistant installer

`berry install` registers Berry with AI coding assistants using a platform-aware installer.

The installer writes three kinds of artifacts, depending on the platform:

1. a Berry skill file (`SKILL.md`) where the host supports skills;
2. an always-on instruction file or managed section (`AGENTS.md`, `CLAUDE.md`, `.cursor/rules/berry.mdc`, etc.);
3. MCP client config and hooks where the host exposes a stable config surface.

Generated artifacts embed the resolved Berry command path. Re-run `berry install` after reinstalling Berry, moving a `pipx`/`uv` environment, or changing Python versions. The installer will idempotently refresh the embedded path without duplicating Berry sections.

## Pick your platform

| Platform | Install command |
|---|---|
| Claude Code (Linux/Mac) | `berry install` |
| Claude Code (Windows) | `berry install` (auto-detected on Windows) or `berry install --platform windows` |
| CodeBuddy | `berry install --platform codebuddy` |
| Codex | `berry install --platform codex` |
| OpenCode | `berry install --platform opencode` |
| Kilo Code | `berry install --platform kilo` |
| GitHub Copilot CLI | `berry install --platform copilot` |
| VS Code Copilot Chat | `berry vscode install` |
| Aider | `berry install --platform aider` |
| OpenClaw | `berry install --platform claw` |
| Factory Droid | `berry install --platform droid` |
| Trae | `berry install --platform trae` |
| Trae CN | `berry install --platform trae-cn` |
| Gemini CLI | `berry install --platform gemini` |
| Hermes | `berry install --platform hermes` |
| Kimi Code | `berry install --platform kimi` |
| Amp | `berry amp install` |
| Kiro IDE/CLI | `berry kiro install` |
| Pi coding agent | `berry install --platform pi` |
| Cursor | `berry cursor install` |
| Devin CLI | `berry devin install` |
| Google Antigravity | `berry antigravity install` |

Codex installs also ensure `multi_agent = true` under `[features]` in `~/.codex/config.toml` or `.codex/config.toml` for project-scoped installs. CodeBuddy uses Claude-style `PreToolUse` hooks where available. Trae and Trae CN use `AGENTS.md` as the always-on mechanism because hooks are not assumed. Aider and OpenClaw get sequential always-on guidance only. Factory Droid guidance mentions Task-style parallel dispatch where the host supports it.

VS Code Copilot Chat installs use VS Code's native MCP shape: user-scope installs write the active VS Code user-profile `mcp.json`, and project-scope installs write `.vscode/mcp.json` with a top-level `servers` object.

## Scope

By default, `berry install` writes user-profile artifacts where the platform supports them. To install into the current repository instead, pass `--project`:

```bash
berry install --project --platform codex
berry cursor install --project
berry gemini install --project
```

Project installs require a git repository by default, matching `berry init` safety behavior. Pass `--project-root PATH` or set `BERRY_ALLOW_NON_GIT_ROOT=1` when intentionally installing into a non-git directory.

## Useful flags

```bash
berry install --list-platforms
berry install --platform codex --dry-run
berry install --platform gemini --json
berry install --platform claude --berry-command /opt/berry/bin/berry
berry install --platform claude --no-hooks
berry install --platform cursor --no-mcp
berry install codex gemini cursor
```

`--berry-command` is primarily for package maintainers and tests. Normal users should let the installer resolve the current Berry executable automatically.

## Safety model

- Writes are idempotent: Berry-owned Markdown sections are bounded by `<!-- berry:install:start -->` and `<!-- berry:install:end -->`.
- JSON writes preserve unrelated keys and fail closed on invalid JSON unless `--force` is provided.
- Text writes use an atomic temporary-file replace.
- `--dry-run` reports all planned actions without touching the filesystem.
- Repeated installs replace old Berry hooks/sections rather than appending duplicates.
