from __future__ import annotations

import json
import os
import platform as _platform
import re
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Callable, Iterable, Literal

from . import __version__
from .clients import McpServerSpec
from .integration import _upsert_codex_toml
from .mcp_env import load_mcp_env

Scope = Literal["user", "project"]

_BERRY_SECTION_START = "<!-- berry:install:start -->"
_BERRY_SECTION_END = "<!-- berry:install:end -->"
_SECTION_RE = re.compile(
    rf"\n?{re.escape(_BERRY_SECTION_START)}\n.*?\n{re.escape(_BERRY_SECTION_END)}\n?",
    re.DOTALL,
)


@dataclass(frozen=True)
class PlatformSpec:
    """Declarative install target for one assistant platform.

    Host-specific behavior lives in this table rather than in the CLI, so adding an
    assistant is a data change unless it needs a custom file format.
    """

    key: str
    display: str
    aliases: tuple[str, ...] = ()
    user_skill_path: str | None = None
    project_skill_path: str | None = None
    user_instruction_path: str | None = None
    project_instruction_path: str | None = None
    instruction_kind: str = "markdown"
    hook_settings_path: str | None = None
    mcp_client: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstallOptions:
    platform: str = "auto"
    scope: Scope = "user"
    project_dir: Path | None = None
    force: bool = False
    dry_run: bool = False
    berry_command: tuple[str, ...] | None = None
    name: str = "berry"
    install_hooks: bool = True
    install_mcp: bool = True


@dataclass(frozen=True)
class InstallAction:
    platform: str
    scope: str
    kind: str
    path: str
    status: str  # written | updated | unchanged | skipped | failed | dry-run
    message: str = ""


def _home() -> Path:
    return Path(os.path.expanduser("~")).resolve()


def _is_windows() -> bool:
    return sys.platform == "win32" or _platform.system().lower() == "windows"


def _template_context(
    invocation: tuple[str, ...], *, name: str, platform_key: str
) -> dict[str, str]:
    berry_cli = shlex.join(invocation)
    mcp_invocation = (*invocation, "mcp", "--server", "classic")
    return {
        "berry_cli": berry_cli,
        "berry_command_json": json.dumps(invocation[0]),
        "berry_args_json": json.dumps(list(invocation[1:])),
        "berry_mcp_cli": shlex.join(mcp_invocation),
        "berry_mcp_command_json": json.dumps(mcp_invocation[0]),
        "berry_mcp_args_json": json.dumps(list(mcp_invocation[1:])),
        "server_name": name,
        "platform": platform_key,
        "version": __version__,
    }


def _resolve_invocation(raw: str | None = None) -> tuple[str, ...]:
    """Return the command tuple to embed in generated config.

    Preference order:
    1. explicit --berry-command, parsed as a shell command;
    2. an absolute berry CLI discovered on PATH;
    3. sys.executable -m berry, which is safe for editable/source installs.
    """

    if raw is not None and raw.strip():
        parts = tuple(shlex.split(raw, posix=(os.name != "nt")))
        if not parts:
            raise ValueError("--berry-command parsed to an empty command")
        return parts
    found = shutil.which("berry")
    if found:
        return (str(Path(found).resolve()),)
    return (str(Path(sys.executable).resolve()), "-m", "berry")


def _mcp_spec(*, name: str, invocation: tuple[str, ...]) -> McpServerSpec:
    return McpServerSpec(
        name=name,
        command=invocation[0],
        args=[*invocation[1:], "mcp", "--server", "classic"],
        env=load_mcp_env(),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _write_if_changed(
    actions: list[InstallAction],
    *,
    platform_key: str,
    scope: Scope,
    kind: str,
    path: Path,
    content: str,
    dry_run: bool,
) -> None:
    old = _read_text(path)
    if old == content:
        actions.append(
            InstallAction(platform_key, scope, kind, str(path), "unchanged", "already current")
        )
        return
    status = "written" if not path.exists() else "updated"
    if dry_run:
        actions.append(InstallAction(platform_key, scope, kind, str(path), "dry-run", status))
        return
    _atomic_write_text(path, content)
    actions.append(InstallAction(platform_key, scope, kind, str(path), status, ""))


def _render_template(template: str, context: dict[str, str]) -> str:
    return Template(template).safe_substitute(context)


def _wrap_section(body: str) -> str:
    return f"{_BERRY_SECTION_START}\n{body.strip()}\n{_BERRY_SECTION_END}\n"


def _upsert_markdown_section(
    actions: list[InstallAction],
    *,
    platform_key: str,
    scope: Scope,
    path: Path,
    section: str,
    dry_run: bool,
) -> None:
    old = _read_text(path)
    wrapped = _wrap_section(section)
    start_count = old.count(_BERRY_SECTION_START)
    end_count = old.count(_BERRY_SECTION_END)
    if start_count != end_count:
        actions.append(
            InstallAction(
                platform_key,
                scope,
                "instructions",
                str(path),
                "failed",
                "malformed Berry section markers; repair the file or remove the partial section",
            )
        )
        return
    if start_count > 1:
        actions.append(
            InstallAction(
                platform_key,
                scope,
                "instructions",
                str(path),
                "failed",
                "multiple Berry-managed sections found; remove duplicates before reinstalling",
            )
        )
        return
    if _BERRY_SECTION_START in old or _BERRY_SECTION_END in old:
        new = _SECTION_RE.sub("\n" + wrapped, old).strip() + "\n"
    elif old.strip():
        new = old.rstrip() + "\n\n" + wrapped
    else:
        new = wrapped
    _write_if_changed(
        actions,
        platform_key=platform_key,
        scope=scope,
        kind="instructions",
        path=path,
        content=new,
        dry_run=dry_run,
    )


def _load_json_file(path: Path, *, force: bool) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, None
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}, None
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded, None
        if force:
            return {}, None
        return {}, "top-level JSON value is not an object"
    except json.JSONDecodeError as exc:
        if force:
            return {}, None
        return {}, f"invalid JSON: {exc}"
    except OSError as exc:
        return {}, str(exc)


def _strip_json_comments(raw: str) -> str:
    """Strip JSONC comments while preserving string contents."""

    out: list[str] = []
    in_string = False
    escaped = False
    line_comment = False
    block_comment = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        nxt = raw[i + 1] if i + 1 < len(raw) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
                out.append(ch)
            i += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
        out.append(ch)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _load_json_or_jsonc(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonc":
        raw = _strip_json_comments(raw)
    loaded = json.loads(raw) if raw.strip() else {}
    return loaded if isinstance(loaded, dict) else {}


def _upsert_json(
    actions: list[InstallAction],
    *,
    platform_key: str,
    scope: Scope,
    path: Path,
    force: bool,
    dry_run: bool,
    mutate: Callable[[dict], None],
    kind: str,
) -> None:
    payload, err = _load_json_file(path, force=force)
    if err:
        actions.append(InstallAction(platform_key, scope, kind, str(path), "failed", err))
        return
    old = _read_text(path)
    mutate(payload)
    new = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if old == new:
        actions.append(
            InstallAction(platform_key, scope, kind, str(path), "unchanged", "already current")
        )
        return
    status = "written" if not path.exists() else "updated"
    if dry_run:
        actions.append(InstallAction(platform_key, scope, kind, str(path), "dry-run", status))
        return
    _atomic_write_text(path, new)
    actions.append(InstallAction(platform_key, scope, kind, str(path), status, ""))


def _server_payload(spec: McpServerSpec) -> dict[str, object]:
    payload: dict[str, object] = {"command": spec.command, "args": spec.args}
    if spec.env:
        payload["env"] = spec.env
    return payload


def _upsert_mcp_json(
    actions: list[InstallAction],
    *,
    platform_key: str,
    scope: Scope,
    path: Path,
    spec: McpServerSpec,
    force: bool,
    dry_run: bool,
    key: str = "mcpServers",
) -> None:
    def mutate(payload: dict) -> None:
        servers = payload.get(key)
        if not isinstance(servers, dict):
            servers = {}
        servers[spec.name] = _server_payload(spec)
        payload[key] = servers

    _upsert_json(
        actions,
        platform_key=platform_key,
        scope=scope,
        path=path,
        force=force,
        dry_run=dry_run,
        mutate=mutate,
        kind="mcp-config",
    )


def _vscode_user_mcp_path() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base).expanduser().resolve() / "Code" / "User" / "mcp.json"
        return _home() / "AppData" / "Roaming" / "Code" / "User" / "mcp.json"
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
    return _home() / ".config" / "Code" / "User" / "mcp.json"


def _upsert_codex_features(path: Path, *, dry_run: bool) -> tuple[str, str]:
    """Ensure [features].multi_agent = true without taking a TOML dependency."""

    old = _read_text(path)
    lines = old.splitlines()
    out: list[str] = []
    in_features = False
    saw_features = False
    wrote_multi_agent = False
    changed = False

    def maybe_insert_feature() -> None:
        nonlocal wrote_multi_agent, changed
        if saw_features and not wrote_multi_agent:
            out.append("multi_agent = true")
            wrote_multi_agent = True
            changed = True

    for line in lines:
        stripped = line.strip()
        section_match = re.match(r"^\[([^\]]+)\]\s*$", stripped)
        if section_match:
            if in_features:
                maybe_insert_feature()
            section = section_match.group(1).strip()
            in_features = section == "features"
            saw_features = saw_features or in_features
            out.append(line)
            continue
        if in_features and re.match(r"^multi_agent\s*=", stripped):
            if stripped != "multi_agent = true":
                out.append("multi_agent = true")
                changed = True
            else:
                out.append(line)
            wrote_multi_agent = True
            continue
        out.append(line)
    if in_features:
        maybe_insert_feature()
    if not saw_features:
        if out and out[-1].strip():
            out.append("")
        out.append("[features]")
        out.append("multi_agent = true")
        changed = True
    new = "\n".join(out).rstrip() + "\n"
    if old == new:
        return "unchanged", old
    if dry_run:
        return "dry-run", new
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, new)
    return ("written" if not old else "updated"), new


def _install_codex_toml(
    actions: list[InstallAction],
    *,
    platform_key: str,
    scope: Scope,
    path: Path,
    spec: McpServerSpec,
    dry_run: bool,
) -> None:
    if dry_run:
        actions.append(
            InstallAction(
                platform_key, scope, "mcp-config", str(path), "dry-run", "upsert berry MCP server"
            )
        )
        actions.append(
            InstallAction(
                platform_key,
                scope,
                "features",
                str(path),
                "dry-run",
                "ensure [features].multi_agent = true",
            )
        )
        return
    try:
        before_exists = path.exists()
        before = _read_text(path)
        _upsert_codex_toml(path, spec)
        after = _read_text(path)
        if before == after:
            status = "unchanged"
            msg = "already current"
        else:
            status = "updated" if before_exists else "written"
            msg = ""
        actions.append(InstallAction(platform_key, scope, "mcp-config", str(path), status, msg))
        feat_status, _ = _upsert_codex_features(path, dry_run=False)
        actions.append(
            InstallAction(
                platform_key, scope, "features", str(path), feat_status, "multi_agent = true"
            )
        )
    except Exception as exc:
        actions.append(
            InstallAction(platform_key, scope, "mcp-config", str(path), "failed", str(exc))
        )


def _hook_entry(context: dict[str, str], *, matcher: str) -> dict[str, object]:
    message = (
        "Berry evidence verifier is installed. Prefer Berry MCP tools for factual claims and "
        f"verification. Embedded command: {context['berry_mcp_cli']}"
    )
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": message,
        }
    }
    command = "printf '%s\\n' " + shlex.quote(json.dumps(payload, separators=(",", ":")))
    return {
        "matcher": matcher,
        "hooks": [
            {
                "type": "command",
                "command": command,
                "berryManaged": True,
            }
        ],
    }


def _install_pre_tool_use_hooks(
    actions: list[InstallAction],
    *,
    platform_key: str,
    scope: Scope,
    path: Path,
    context: dict[str, str],
    force: bool,
    dry_run: bool,
) -> None:
    def mutate(payload: dict) -> None:
        hooks = payload.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}
        pre = hooks.get("PreToolUse")
        if not isinstance(pre, list):
            pre = []

        def is_berry_managed(hook: object) -> bool:
            if not isinstance(hook, dict):
                return False
            if hook.get("berryManaged") is True:
                return True
            text = json.dumps(hook).lower()
            return "berry evidence verifier" in text or "embedded command:" in text

        kept = [h for h in pre if not is_berry_managed(h)]
        kept.append(_hook_entry(context, matcher="Bash"))
        kept.append(_hook_entry(context, matcher="Read|Glob"))
        hooks["PreToolUse"] = kept
        payload["hooks"] = hooks

    _upsert_json(
        actions,
        platform_key=platform_key,
        scope=scope,
        path=path,
        force=force,
        dry_run=dry_run,
        mutate=mutate,
        kind="hooks",
    )


def _install_gemini_hook(
    actions: list[InstallAction],
    *,
    platform_key: str,
    scope: Scope,
    path: Path,
    context: dict[str, str],
    force: bool,
    dry_run: bool,
) -> None:
    message = (
        "Berry evidence verifier is installed. For codebase questions and factual claims, "
        f"use Berry MCP tools when available. Embedded command: {context['berry_mcp_cli']}"
    )
    hook = {
        "matcher": "read_file|list_directory|search_file_content|glob",
        "hooks": [
            {
                "type": "command",
                "command": "printf '%s\\n' "
                + shlex.quote(json.dumps({"decision": "allow", "additionalContext": message})),
                "berryManaged": True,
            }
        ],
    }

    def mutate(payload: dict) -> None:
        hooks = payload.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}
        before = hooks.get("BeforeTool")
        if not isinstance(before, list):
            before = []

        def is_berry_managed(hook: object) -> bool:
            if not isinstance(hook, dict):
                return False
            if hook.get("berryManaged") is True:
                return True
            text = json.dumps(hook).lower()
            return "berry evidence verifier" in text or "embedded command:" in text

        before = [h for h in before if not is_berry_managed(h)]
        before.append(hook)
        hooks["BeforeTool"] = before
        payload["hooks"] = hooks

    _upsert_json(
        actions,
        platform_key=platform_key,
        scope=scope,
        path=path,
        force=force,
        dry_run=dry_run,
        mutate=mutate,
        kind="hooks",
    )


_GENERIC_SKILL_TEMPLATE = """# Berry evidence verifier

Berry is a local MCP runtime and verifier for evidence-backed coding-agent work.

## Use this skill when
- The user asks for a factual answer about a repository, plan, design, bug, or generated code.
- You need to cite gathered evidence or verify that claims are supported by evidence spans.
- You are about to produce a plan or final answer that should not rely on unstated context.

## Operating rules
1. Prefer Berry MCP tools over free-form guessing.
2. Treat Berry evidence spans as the source of truth for factual claims.
3. When Berry returns `state=need_grant`, show the requested scopes and wait for explicit user approval before calling `berry_approve`.
4. When Berry returns `state=ask_user`, ask the returned questions verbatim and retry with the same `run_id` after the user answers.
5. When Berry returns `state=done`, use the verified answer or plan.
6. If the MCP server is unavailable, ask the user to run `$berry_mcp_cli` or reload MCP servers.

## Embedded command
- Berry CLI: `$berry_cli`
- Berry MCP: `$berry_mcp_cli`

Re-run `berry install` after moving/upgrading Berry, pipx, uv, or Python so generated configs refresh the embedded path.
"""

_ALWAYS_ON_TEMPLATE = """## Berry evidence verifier

Berry is installed for this project/platform.

- Use Berry MCP tools for evidence gathering and verification before making factual claims.
- Prefer `berry_solve` for questions and `berry_change` for code-change planning.
- Respect Berry's state machine: `need_grant` requires user approval, `ask_user` requires asking the returned questions, and `done` contains the verified output.
- Embedded Berry MCP command: `$berry_mcp_cli`.
- If Berry stops launching after an upgrade or environment move, re-run `berry install --platform $platform` to refresh this embedded path.
"""

_CURSOR_RULE_TEMPLATE = """---
description: Berry evidence verifier context
alwaysApply: true
---

Berry is installed for this workspace.

- For factual claims, repository questions, and code-change plans, use Berry MCP tools before answering.
- Prefer `berry_solve` for questions and `berry_change` for planned changes.
- Follow Berry's state machine exactly: ask the user for grants/questions when Berry requests it.
- Embedded Berry MCP command: `$berry_mcp_cli`.
- Re-run `berry install --platform cursor` after moving/upgrading Berry to refresh this path.
"""

_KILO_PLUGIN_TEMPLATE = """// Berry Kilo plugin. Generated by `berry install --platform kilo`.
// Fails open: if anything goes wrong, the user's tool call proceeds.
export const BerryPlugin = async () => {
  let reminded = false;
  return {
    "tool.execute.before": async (input, output) => {
      if (reminded) return;
      if (input && input.tool === "bash") {
        output.args.command = 'echo "[berry] Evidence verifier installed. Prefer Berry MCP tools for factual claims. Embedded MCP: $berry_mcp_cli" && ' + output.args.command;
        reminded = true;
      }
    },
  };
};
"""

_ANTIGRAVITY_WORKFLOW_TEMPLATE = """---
name: berry
description: Use Berry to gather evidence and verify factual claims
---

# Workflow: Berry evidence verification

Use the Berry skill and MCP tools for evidence-backed answers and code-change plans.

Embedded MCP command: `$berry_mcp_cli`

Re-run `berry install --platform antigravity` after moving/upgrading Berry to refresh this path.
"""

_KILO_COMMAND_TEMPLATE = """# /berry

Use Berry for evidence-backed answers and verification.

Embedded MCP command: `$berry_mcp_cli`
"""

_PLATFORM_CONFIG: dict[str, PlatformSpec] = {
    "claude": PlatformSpec(
        key="claude",
        display="Claude Code",
        aliases=("claude-code",),
        user_skill_path="~/.claude/skills/berry/SKILL.md",
        project_skill_path=".claude/skills/berry/SKILL.md",
        user_instruction_path="~/.claude/CLAUDE.md",
        project_instruction_path=".claude/CLAUDE.md",
        hook_settings_path=".claude/settings.json",
        mcp_client="claude",
    ),
    "windows": PlatformSpec(
        key="windows",
        display="Claude Code (Windows)",
        aliases=("claude-windows", "win"),
        user_skill_path="~/.claude/skills/berry/SKILL.md",
        project_skill_path=".claude/skills/berry/SKILL.md",
        user_instruction_path="~/.claude/CLAUDE.md",
        project_instruction_path=".claude/CLAUDE.md",
        hook_settings_path=".claude/settings.json",
        mcp_client="claude",
    ),
    "codebuddy": PlatformSpec(
        key="codebuddy",
        display="CodeBuddy",
        user_skill_path="~/.codebuddy/skills/berry/SKILL.md",
        project_skill_path=".codebuddy/skills/berry/SKILL.md",
        user_instruction_path="~/.codebuddy/CODEBUDDY.md",
        project_instruction_path="CODEBUDDY.md",
        hook_settings_path=".codebuddy/settings.json",
        notes=("Uses Claude-style PreToolUse hooks where available.",),
    ),
    "codex": PlatformSpec(
        key="codex",
        display="Codex",
        user_skill_path="~/.codex/skills/berry/SKILL.md",
        project_skill_path=".codex/skills/berry/SKILL.md",
        user_instruction_path="~/.codex/AGENTS.md",
        project_instruction_path="AGENTS.md",
        mcp_client="codex",
        notes=("Enables [features].multi_agent = true in config.toml.",),
    ),
    "opencode": PlatformSpec(
        key="opencode",
        display="OpenCode",
        aliases=("open-code",),
        user_skill_path="~/.config/opencode/skills/berry/SKILL.md",
        project_skill_path=".opencode/skills/berry/SKILL.md",
        user_instruction_path="~/.config/opencode/AGENTS.md",
        project_instruction_path="AGENTS.md",
    ),
    "kilo": PlatformSpec(
        key="kilo",
        display="Kilo Code",
        aliases=("kilo-code",),
        user_skill_path="~/.config/kilo/skills/berry/SKILL.md",
        project_skill_path=".kilo/skills/berry/SKILL.md",
        user_instruction_path="~/.config/kilo/AGENTS.md",
        project_instruction_path="AGENTS.md",
        notes=("Also writes a native Kilo command and project plugin when project scoped.",),
    ),
    "copilot": PlatformSpec(
        key="copilot",
        display="GitHub Copilot CLI",
        aliases=("github-copilot",),
        user_skill_path="~/.copilot/skills/berry/SKILL.md",
        project_skill_path=".copilot/skills/berry/SKILL.md",
        user_instruction_path="~/.copilot/AGENTS.md",
        project_instruction_path="AGENTS.md",
    ),
    "vscode": PlatformSpec(
        key="vscode",
        display="VS Code Copilot Chat",
        aliases=("vs-code", "copilot-chat"),
        user_skill_path="~/.copilot/skills/berry/SKILL.md",
        project_skill_path=".copilot/skills/berry/SKILL.md",
        project_instruction_path=".github/copilot-instructions.md",
        mcp_client="vscode",
    ),
    "aider": PlatformSpec(
        key="aider",
        display="Aider",
        user_skill_path="~/.aider/skills/berry/SKILL.md",
        project_skill_path=".aider/skills/berry/SKILL.md",
        user_instruction_path="~/.aider/AGENTS.md",
        project_instruction_path="AGENTS.md",
        notes=("Sequential agent extraction/verification guidance only; no hook is installed.",),
    ),
    "claw": PlatformSpec(
        key="claw",
        display="OpenClaw",
        aliases=("openclaw",),
        user_skill_path="~/.openclaw/skills/berry/SKILL.md",
        project_skill_path=".openclaw/skills/berry/SKILL.md",
        user_instruction_path="~/.openclaw/AGENTS.md",
        project_instruction_path="AGENTS.md",
    ),
    "droid": PlatformSpec(
        key="droid",
        display="Factory Droid",
        aliases=("factory", "factory-droid"),
        user_skill_path="~/.factory/skills/berry/SKILL.md",
        project_skill_path=".factory/skills/berry/SKILL.md",
        user_instruction_path="~/.factory/AGENTS.md",
        project_instruction_path="AGENTS.md",
        notes=("Guidance mentions Task-style parallel dispatch where the host supports it.",),
    ),
    "trae": PlatformSpec(
        key="trae",
        display="Trae",
        user_skill_path="~/.trae/skills/berry/SKILL.md",
        project_skill_path=".trae/skills/berry/SKILL.md",
        user_instruction_path="~/.trae/AGENTS.md",
        project_instruction_path="AGENTS.md",
        notes=("Trae does not support PreToolUse hooks; AGENTS.md is the always-on path.",),
    ),
    "trae-cn": PlatformSpec(
        key="trae-cn",
        display="Trae CN",
        aliases=("traecn",),
        user_skill_path="~/.trae-cn/skills/berry/SKILL.md",
        project_skill_path=".trae-cn/skills/berry/SKILL.md",
        user_instruction_path="~/.trae-cn/AGENTS.md",
        project_instruction_path="AGENTS.md",
        notes=("Trae CN uses the same always-on guidance as Trae.",),
    ),
    "gemini": PlatformSpec(
        key="gemini",
        display="Gemini CLI",
        user_skill_path="~/.gemini/skills/berry/SKILL.md",
        project_skill_path=".gemini/skills/berry/SKILL.md",
        user_instruction_path="~/.gemini/GEMINI.md",
        project_instruction_path="GEMINI.md",
        hook_settings_path=".gemini/settings.json",
        mcp_client="gemini",
    ),
    "hermes": PlatformSpec(
        key="hermes",
        display="Hermes",
        user_skill_path="~/.hermes/skills/berry/SKILL.md",
        project_skill_path=".hermes/skills/berry/SKILL.md",
        user_instruction_path="~/.hermes/AGENTS.md",
        project_instruction_path="AGENTS.md",
    ),
    "kimi": PlatformSpec(
        key="kimi",
        display="Kimi Code",
        aliases=("kimi-code",),
        user_skill_path="~/.kimi/skills/berry/SKILL.md",
        project_skill_path=".kimi/skills/berry/SKILL.md",
        user_instruction_path="~/.kimi/AGENTS.md",
        project_instruction_path="AGENTS.md",
    ),
    "amp": PlatformSpec(
        key="amp",
        display="Amp",
        user_skill_path="~/.config/agents/skills/berry/SKILL.md",
        project_skill_path=".agents/skills/berry/SKILL.md",
        user_instruction_path="~/.config/agents/AGENTS.md",
        project_instruction_path="AGENTS.md",
    ),
    "kiro": PlatformSpec(
        key="kiro",
        display="Kiro IDE/CLI",
        user_skill_path="~/.kiro/skills/berry/SKILL.md",
        project_skill_path=".kiro/skills/berry/SKILL.md",
        user_instruction_path="~/.kiro/steering/berry.md",
        project_instruction_path=".kiro/steering/berry.md",
        notes=("Writes Kiro steering guidance.",),
    ),
    "pi": PlatformSpec(
        key="pi",
        display="Pi coding agent",
        user_skill_path="~/.pi/agent/skills/berry/SKILL.md",
        project_skill_path=".pi/agent/skills/berry/SKILL.md",
        user_instruction_path="~/.pi/agent/AGENTS.md",
        project_instruction_path="AGENTS.md",
    ),
    "cursor": PlatformSpec(
        key="cursor",
        display="Cursor",
        project_instruction_path=".cursor/rules/berry.mdc",
        instruction_kind="cursor-rule",
        mcp_client="cursor",
        notes=("Cursor consumes the alwaysApply rule; no separate skill file is required.",),
    ),
    "devin": PlatformSpec(
        key="devin",
        display="Devin CLI",
        user_skill_path="~/.config/devin/skills/berry/SKILL.md",
        project_skill_path=".devin/skills/berry/SKILL.md",
        user_instruction_path="~/.config/devin/AGENTS.md",
        project_instruction_path=".windsurf/rules/berry.md",
    ),
    "antigravity": PlatformSpec(
        key="antigravity",
        display="Google Antigravity",
        aliases=("google-antigravity",),
        user_skill_path="~/.gemini/config/skills/berry/SKILL.md",
        project_skill_path=".agents/skills/berry/SKILL.md",
        user_instruction_path="~/.gemini/antigravity/AGENTS.md",
        project_instruction_path=".agents/rules/berry.md",
        mcp_client="gemini",
        notes=("Also writes a project workflow file under .agents/workflows/.",),
    ),
}

_ALIAS_TO_KEY: dict[str, str] = {}
for _key, _spec in _PLATFORM_CONFIG.items():
    _ALIAS_TO_KEY[_key] = _key
    for _alias in _spec.aliases:
        _ALIAS_TO_KEY[_alias] = _key


def platform_keys() -> tuple[str, ...]:
    return tuple(_PLATFORM_CONFIG.keys())


def resolve_platform(raw: str | None) -> str:
    s = (raw or "auto").strip().lower()
    if s == "auto":
        return "windows" if _is_windows() else "claude"
    key = _ALIAS_TO_KEY.get(s)
    if not key:
        raise ValueError(f"unknown platform {raw!r}; choose one of: {', '.join(platform_keys())}")
    return key


def _path(raw: str, *, scope: Scope, project_dir: Path) -> Path:
    if raw.startswith("~/") or raw == "~":
        return Path(os.path.expanduser(raw)).resolve()
    base = project_dir if scope == "project" else _home()
    return (base / raw).resolve()


def _skill_path(spec: PlatformSpec, *, scope: Scope, project_dir: Path) -> Path | None:
    raw = spec.project_skill_path if scope == "project" else spec.user_skill_path
    if not raw:
        return None
    return _path(raw, scope=scope, project_dir=project_dir)


def _instruction_path(spec: PlatformSpec, *, scope: Scope, project_dir: Path) -> Path | None:
    raw = spec.project_instruction_path if scope == "project" else spec.user_instruction_path
    if not raw:
        # Some platforms (e.g. VS Code/Cursor) are project-instruction only. If
        # a user-scope install is requested, still write the project rule in cwd
        # because that is the only host-supported always-on surface.
        raw = spec.project_instruction_path
        scope = "project"
    if not raw:
        return None
    return _path(raw, scope=scope, project_dir=project_dir)


def _settings_path(spec: PlatformSpec, *, project_dir: Path) -> Path | None:
    if not spec.hook_settings_path:
        return None
    # Hooks are workspace settings for these hosts. Even for user-scope installs,
    # writing hooks globally would be surprising, so keep them project-scoped.
    return (project_dir / spec.hook_settings_path).resolve()


def _render_skill(context: dict[str, str]) -> str:
    return _render_template(_GENERIC_SKILL_TEMPLATE, context)


def _render_instructions(spec: PlatformSpec, context: dict[str, str]) -> str:
    if spec.instruction_kind == "cursor-rule":
        return _render_template(_CURSOR_RULE_TEMPLATE, context)
    extra = ""
    if spec.key == "codex":
        extra = "\n- Codex users: `[features].multi_agent = true` is installed in config.toml for parallel agent work.\n"
    elif spec.key == "droid":
        extra = "\n- When Factory Droid exposes Task-style subagents, dispatch independent verification/extraction work in parallel.\n"
    elif spec.key in {"aider", "claw"}:
        extra = "\n- This host currently gets sequential always-on guidance; no PreToolUse hook is installed.\n"
    elif spec.key in {"trae", "trae-cn"}:
        extra = "\n- Trae does not expose PreToolUse hooks here; this AGENTS.md section is the always-on mechanism.\n"
    return _render_template(_ALWAYS_ON_TEMPLATE + extra, context)


def _install_extra_platform_files(
    actions: list[InstallAction],
    *,
    spec: PlatformSpec,
    scope: Scope,
    project_dir: Path,
    context: dict[str, str],
    dry_run: bool,
) -> None:
    if spec.key == "kilo":
        command_path = (
            _path("~/.config/kilo/command/berry.md", scope="user", project_dir=project_dir)
            if scope == "user"
            else (project_dir / ".kilo" / "command" / "berry.md").resolve()
        )
        _write_if_changed(
            actions,
            platform_key=spec.key,
            scope=scope,
            kind="command",
            path=command_path,
            content=_render_template(_KILO_COMMAND_TEMPLATE, context),
            dry_run=dry_run,
        )
        if scope == "project":
            plugin_path = (project_dir / ".kilo" / "plugins" / "berry.js").resolve()
            _write_if_changed(
                actions,
                platform_key=spec.key,
                scope=scope,
                kind="plugin",
                path=plugin_path,
                content=_render_template(_KILO_PLUGIN_TEMPLATE, context),
                dry_run=dry_run,
            )
            json_path = (project_dir / ".kilo" / "kilo.json").resolve()
            jsonc_path = (project_dir / ".kilo" / "kilo.jsonc").resolve()
            read_path = json_path if json_path.exists() else jsonc_path
            try:
                config = _load_json_or_jsonc(read_path)
                plugins = config.get("plugin")
                if not isinstance(plugins, list):
                    plugins = []
                config["plugin"] = plugins
                entry = plugin_path.as_uri()
                if entry not in plugins:
                    plugins.append(entry)
                new = json.dumps(config, indent=2, sort_keys=True) + "\n"
                _write_if_changed(
                    actions,
                    platform_key=spec.key,
                    scope=scope,
                    kind="plugin-config",
                    path=json_path,
                    content=new,
                    dry_run=dry_run,
                )
            except Exception as exc:
                actions.append(
                    InstallAction(
                        spec.key,
                        scope,
                        "plugin-config",
                        str(json_path),
                        "failed",
                        str(exc),
                    )
                )
    elif spec.key == "antigravity":
        workflow_path = (project_dir / ".agents" / "workflows" / "berry.md").resolve()
        _write_if_changed(
            actions,
            platform_key=spec.key,
            scope=scope,
            kind="workflow",
            path=workflow_path,
            content=_render_template(_ANTIGRAVITY_WORKFLOW_TEMPLATE, context),
            dry_run=dry_run,
        )


def install_platform(options: InstallOptions) -> list[InstallAction]:
    platform_key = resolve_platform(options.platform)
    spec = _PLATFORM_CONFIG[platform_key]
    project_dir = (options.project_dir or Path.cwd()).expanduser().resolve()
    invocation = options.berry_command or _resolve_invocation(None)
    context = _template_context(invocation, name=options.name, platform_key=platform_key)
    actions: list[InstallAction] = []

    skill_path = _skill_path(spec, scope=options.scope, project_dir=project_dir)
    if skill_path is not None:
        _write_if_changed(
            actions,
            platform_key=platform_key,
            scope=options.scope,
            kind="skill",
            path=skill_path,
            content=_render_skill(context),
            dry_run=options.dry_run,
        )
    else:
        actions.append(
            InstallAction(
                platform_key, options.scope, "skill", "", "skipped", "platform has no skill file"
            )
        )

    inst_path = _instruction_path(spec, scope=options.scope, project_dir=project_dir)
    if inst_path is not None:
        _upsert_markdown_section(
            actions,
            platform_key=platform_key,
            scope=options.scope,
            path=inst_path,
            section=_render_instructions(spec, context),
            dry_run=options.dry_run,
        )

    if options.install_hooks:
        settings_path = _settings_path(spec, project_dir=project_dir)
        if settings_path is not None:
            if platform_key == "gemini":
                _install_gemini_hook(
                    actions,
                    platform_key=platform_key,
                    scope=options.scope,
                    path=settings_path,
                    context=context,
                    force=options.force,
                    dry_run=options.dry_run,
                )
            else:
                _install_pre_tool_use_hooks(
                    actions,
                    platform_key=platform_key,
                    scope=options.scope,
                    path=settings_path,
                    context=context,
                    force=options.force,
                    dry_run=options.dry_run,
                )

    if options.install_mcp and spec.mcp_client:
        spec_obj = _mcp_spec(name=options.name, invocation=invocation)
        if spec.mcp_client == "codex":
            path = _path(
                "~/.codex/config.toml" if options.scope == "user" else ".codex/config.toml",
                scope=options.scope,
                project_dir=project_dir,
            )
            _install_codex_toml(
                actions,
                platform_key=platform_key,
                scope=options.scope,
                path=path,
                spec=spec_obj,
                dry_run=options.dry_run,
            )
        elif spec.mcp_client == "claude":
            path = (
                _path("~/.claude.json", scope="user", project_dir=project_dir)
                if options.scope == "user"
                else (project_dir / ".mcp.json").resolve()
            )
            _upsert_mcp_json(
                actions,
                platform_key=platform_key,
                scope=options.scope,
                path=path,
                spec=spec_obj,
                force=options.force,
                dry_run=options.dry_run,
            )
        elif spec.mcp_client == "gemini":
            path = _path(
                "~/.gemini/settings.json" if options.scope == "user" else ".gemini/settings.json",
                scope=options.scope,
                project_dir=project_dir,
            )
            _upsert_mcp_json(
                actions,
                platform_key=platform_key,
                scope=options.scope,
                path=path,
                spec=spec_obj,
                force=options.force,
                dry_run=options.dry_run,
            )
        elif spec.mcp_client == "cursor":
            path = _path(
                "~/.cursor/mcp.json" if options.scope == "user" else ".cursor/mcp.json",
                scope=options.scope,
                project_dir=project_dir,
            )
            _upsert_mcp_json(
                actions,
                platform_key=platform_key,
                scope=options.scope,
                path=path,
                spec=spec_obj,
                force=options.force,
                dry_run=options.dry_run,
            )
        elif spec.mcp_client == "vscode":
            path = (
                _vscode_user_mcp_path()
                if options.scope == "user"
                else (project_dir / ".vscode" / "mcp.json").resolve()
            )
            _upsert_mcp_json(
                actions,
                platform_key=platform_key,
                scope=options.scope,
                path=path,
                spec=spec_obj,
                force=options.force,
                dry_run=options.dry_run,
                key="servers",
            )

    _install_extra_platform_files(
        actions,
        spec=spec,
        scope=options.scope,
        project_dir=project_dir,
        context=context,
        dry_run=options.dry_run,
    )

    return actions


def install_many(
    platforms: Iterable[str],
    *,
    scope: Scope,
    project_dir: Path | None,
    force: bool,
    dry_run: bool,
    berry_command_raw: str | None,
    name: str,
    install_hooks: bool,
    install_mcp: bool,
) -> list[InstallAction]:
    invocation = _resolve_invocation(berry_command_raw)
    all_actions: list[InstallAction] = []
    for platform in platforms:
        all_actions.extend(
            install_platform(
                InstallOptions(
                    platform=platform,
                    scope=scope,
                    project_dir=project_dir,
                    force=force,
                    dry_run=dry_run,
                    berry_command=invocation,
                    name=name,
                    install_hooks=install_hooks,
                    install_mcp=install_mcp,
                )
            )
        )
    return all_actions


def actions_to_json(actions: Iterable[InstallAction]) -> str:
    return json.dumps([asdict(a) for a in actions], indent=2, sort_keys=True) + "\n"


def format_actions(actions: Iterable[InstallAction]) -> str:
    lines: list[str] = []
    for action in actions:
        path = f" -> {action.path}" if action.path else ""
        msg = f" ({action.message})" if action.message else ""
        lines.append(f"{action.platform}: {action.kind}: {action.status}{path}{msg}")
    return "\n".join(lines) + ("\n" if lines else "")
