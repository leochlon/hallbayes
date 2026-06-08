from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .clients import McpServerSpec, berry_server_spec


def _home() -> Path:
    return Path(os.path.expanduser("~")).resolve()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _merge_mcp_servers_json(path: Path, spec: McpServerSpec, *, key: str = "mcpServers") -> None:
    payload = _load_json(path)
    servers = payload.get(key)
    if not isinstance(servers, dict):
        servers = {}
    servers[spec.name] = {"command": spec.command, "args": spec.args}
    if spec.env:
        servers[spec.name]["env"] = spec.env
    payload[key] = servers
    _write_json(path, payload)


def _upsert_codex_toml(path: Path, spec: McpServerSpec) -> None:
    """Idempotently upsert a [mcp_servers.<name>] block into ~/.codex/config.toml.

    Avoids extra dependencies by using a simple section-replace strategy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    def _toml_str(s: str) -> str:
        return json.dumps(s)

    block_lines: list[str] = []
    block_lines.append(f"[mcp_servers.{spec.name}]")
    block_lines.append(f"command = {_toml_str(spec.command)}")
    block_lines.append("args = [" + ", ".join(_toml_str(a) for a in spec.args) + "]")
    block_lines.append("")
    if spec.env:
        block_lines.append(f"[mcp_servers.{spec.name}.env]")
        for k, v in spec.env.items():
            block_lines.append(f"{_toml_str(k)} = {_toml_str(v)}")
        block_lines.append("")
    block = "\n".join(block_lines)

    import re

    # Remove only Berry's own MCP sections. The previous pattern stopped only
    # at the next [mcp_servers.*] section, which could accidentally consume
    # unrelated TOML sections (for example [features]) that followed Berry's
    # block. Stop at any next TOML section instead.
    pat = re.compile(
        rf"^\[mcp_servers\.{re.escape(spec.name)}(?:\.env)?\]\s*\n(?:.*\n)*?(?=^\[|\Z)",
        re.M,
    )
    new_text = pat.sub("", text).rstrip()
    sep = "\n\n" if new_text else ""
    path.write_text(new_text + sep + block + "\n", encoding="utf-8")


@dataclass(frozen=True)
class IntegrationResult:
    client: str
    status: str  # ok | skipped | failed
    message: str
    command: List[str]
    returncode: Optional[int] = None


def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _run(cmd: List[str], *, timeout_s: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(timeout_s),
    )


def integrate_with_claude(
    *,
    spec: Optional[McpServerSpec] = None,
    timeout_s: int = 20,
    dry_run: bool = False,
) -> IntegrationResult:
    """Register Berry as a global MCP server in Claude Code (best-effort).

    Primary mechanism: merge into the user-level Claude Code config (~/.claude.json).
    Secondary mechanism (best-effort): use the `claude` CLI if present.
    """
    spec = spec or berry_server_spec()
    berry_cmd = _which(spec.command) or spec.command
    cmd = ["claude", "mcp", "add", spec.name, "--", berry_cmd, *spec.args]

    # 1) Always try file-based integration (this is what makes installers work).
    claude_json = _home() / ".claude.json"
    if dry_run:
        return IntegrationResult(
            client="claude",
            status="ok",
            message=f"dry-run: would write {claude_json}",
            command=cmd,
            returncode=0,
        )
    try:
        _merge_mcp_servers_json(claude_json, spec, key="mcpServers")
        file_msg = f"updated {claude_json}"
    except Exception as e:
        file_msg = f"failed to update {claude_json}: {e}"

    # 2) Best-effort: also register via CLI if installed (some Claude builds prefer it).
    exe = _which("claude")
    if not exe:
        status = "ok" if file_msg.startswith("updated") else "failed"
        return IntegrationResult(
            client="claude",
            status=status,
            message=file_msg + " (claude CLI not found)",
            command=cmd,
        )

    try:
        cp = _run(cmd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return IntegrationResult(
            client="claude",
            status="ok",
            message=file_msg + f"; cli timed out after {timeout_s}s",
            command=cmd,
        )
    except FileNotFoundError:
        return IntegrationResult(
            client="claude", status="ok", message=file_msg + "; cli not found", command=cmd
        )

    combined = f"{cp.stdout or ''}\n{cp.stderr or ''}".lower()
    if cp.returncode == 0 or (
        "already" in combined and ("exist" in combined or "added" in combined)
    ):
        return IntegrationResult(
            client="claude",
            status="ok",
            message=file_msg + "; cli ok",
            command=cmd,
            returncode=cp.returncode,
        )

    # If file update succeeded, treat CLI failure as non-fatal.
    if file_msg.startswith("updated"):
        return IntegrationResult(
            client="claude",
            status="ok",
            message=file_msg + "; cli failed (non-fatal)",
            command=cmd,
            returncode=cp.returncode,
        )

    msg = (cp.stderr or cp.stdout or "").strip() or f"exit {cp.returncode}"
    return IntegrationResult(
        client="claude",
        status="failed",
        message=file_msg + "; " + msg,
        command=cmd,
        returncode=cp.returncode,
    )


def integrate_with_codex(
    *,
    spec: Optional[McpServerSpec] = None,
    timeout_s: int = 20,
    dry_run: bool = False,
) -> IntegrationResult:
    """Register Berry as a global MCP server in Codex (best-effort).

    Primary mechanism: write/merge into ~/.codex/config.toml.
    Secondary mechanism (best-effort): use the `codex` CLI if present.
    """
    spec = spec or berry_server_spec()
    berry_cmd = _which(spec.command) or spec.command
    cmd = ["codex", "mcp", "add", spec.name, "--", berry_cmd, *spec.args]

    codex_toml = _home() / ".codex" / "config.toml"
    if dry_run:
        return IntegrationResult(
            client="codex",
            status="ok",
            message=f"dry-run: would write {codex_toml}",
            command=cmd,
            returncode=0,
        )
    try:
        _upsert_codex_toml(codex_toml, spec)
        file_msg = f"updated {codex_toml}"
    except Exception as e:
        file_msg = f"failed to update {codex_toml}: {e}"

    exe = _which("codex")
    if not exe:
        status = "ok" if file_msg.startswith("updated") else "failed"
        return IntegrationResult(
            client="codex", status=status, message=file_msg + " (codex CLI not found)", command=cmd
        )

    try:
        cp = _run(cmd, timeout_s=timeout_s)
    except subprocess.TimeoutExpired:
        return IntegrationResult(
            client="codex",
            status="ok",
            message=file_msg + f"; cli timed out after {timeout_s}s",
            command=cmd,
        )
    except FileNotFoundError:
        return IntegrationResult(
            client="codex", status="ok", message=file_msg + "; cli not found", command=cmd
        )

    combined = f"{cp.stdout or ''}\n{cp.stderr or ''}".lower()
    if cp.returncode == 0 or (
        "already" in combined and ("exist" in combined or "added" in combined)
    ):
        return IntegrationResult(
            client="codex",
            status="ok",
            message=file_msg + "; cli ok",
            command=cmd,
            returncode=cp.returncode,
        )

    if file_msg.startswith("updated"):
        return IntegrationResult(
            client="codex",
            status="ok",
            message=file_msg + "; cli failed (non-fatal)",
            command=cmd,
            returncode=cp.returncode,
        )

    msg = (cp.stderr or cp.stdout or "").strip() or f"exit {cp.returncode}"
    return IntegrationResult(
        client="codex",
        status="failed",
        message=file_msg + "; " + msg,
        command=cmd,
        returncode=cp.returncode,
    )


def integrate_with_cursor(
    *, spec: Optional[McpServerSpec] = None, dry_run: bool = False
) -> IntegrationResult:
    spec = spec or berry_server_spec()
    path = _home() / ".cursor" / "mcp.json"
    if dry_run:
        return IntegrationResult(
            client="cursor",
            status="ok",
            message=f"dry-run: would write {path}",
            command=[],
            returncode=0,
        )
    try:
        _merge_mcp_servers_json(path, spec, key="mcpServers")
        return IntegrationResult(
            client="cursor", status="ok", message=f"updated {path}", command=[], returncode=0
        )
    except Exception as e:
        return IntegrationResult(
            client="cursor", status="failed", message=f"failed to update {path}: {e}", command=[]
        )


def integrate_with_gemini(
    *, spec: Optional[McpServerSpec] = None, dry_run: bool = False
) -> IntegrationResult:
    spec = spec or berry_server_spec()
    path = _home() / ".gemini" / "settings.json"
    if dry_run:
        return IntegrationResult(
            client="gemini",
            status="ok",
            message=f"dry-run: would write {path}",
            command=[],
            returncode=0,
        )
    try:
        _merge_mcp_servers_json(path, spec, key="mcpServers")
        return IntegrationResult(
            client="gemini", status="ok", message=f"updated {path}", command=[], returncode=0
        )
    except Exception as e:
        return IntegrationResult(
            client="gemini", status="failed", message=f"failed to update {path}: {e}", command=[]
        )


def _system_paths() -> dict:
    import sys

    platform: str = sys.platform
    if platform == "darwin":
        return {
            "claude_managed": Path("/Library/Application Support/ClaudeCode/managed-mcp.json"),
            "gemini_system": Path("/Library/Application Support/GeminiCli/settings.json"),
        }
    if platform.startswith("linux"):
        return {
            "claude_managed": Path("/etc/claude-code/managed-mcp.json"),
            "gemini_system": Path("/etc/gemini-cli/settings.json"),
        }
    return {}


def integrate_with_claude_managed(
    *, spec: Optional[McpServerSpec] = None, dry_run: bool = False
) -> IntegrationResult:
    spec = spec or berry_server_spec()
    paths = _system_paths()
    path = paths.get("claude_managed")
    if not path:
        return IntegrationResult(
            client="claude-managed", status="skipped", message="unsupported platform", command=[]
        )
    if dry_run:
        return IntegrationResult(
            client="claude-managed",
            status="ok",
            message=f"dry-run: would write {path}",
            command=[],
            returncode=0,
        )
    try:
        _merge_mcp_servers_json(path, spec, key="mcpServers")
        return IntegrationResult(
            client="claude-managed",
            status="ok",
            message=f"updated {path}",
            command=[],
            returncode=0,
        )
    except Exception as e:
        return IntegrationResult(
            client="claude-managed",
            status="failed",
            message=f"failed to update {path}: {e}",
            command=[],
        )


def integrate_with_gemini_system(
    *, spec: Optional[McpServerSpec] = None, dry_run: bool = False
) -> IntegrationResult:
    spec = spec or berry_server_spec()
    paths = _system_paths()
    path = paths.get("gemini_system")
    if not path:
        return IntegrationResult(
            client="gemini-system", status="skipped", message="unsupported platform", command=[]
        )
    if dry_run:
        return IntegrationResult(
            client="gemini-system",
            status="ok",
            message=f"dry-run: would write {path}",
            command=[],
            returncode=0,
        )
    try:
        _merge_mcp_servers_json(path, spec, key="mcpServers")
        return IntegrationResult(
            client="gemini-system", status="ok", message=f"updated {path}", command=[], returncode=0
        )
    except Exception as e:
        return IntegrationResult(
            client="gemini-system",
            status="failed",
            message=f"failed to update {path}: {e}",
            command=[],
        )


def integrate(
    *,
    clients: Iterable[str],
    name: str = "berry",
    timeout_s: int = 20,
    dry_run: bool = False,
    managed: bool = False,
    managed_only: bool = False,
) -> List[IntegrationResult]:
    spec = berry_server_spec(name=str(name))
    out: List[IntegrationResult] = []
    if managed_only:
        if managed:
            out.append(integrate_with_claude_managed(spec=spec, dry_run=dry_run))
            out.append(integrate_with_gemini_system(spec=spec, dry_run=dry_run))
        return out
    for c in clients:
        if c == "claude":
            out.append(integrate_with_claude(spec=spec, timeout_s=timeout_s, dry_run=dry_run))
        elif c == "codex":
            out.append(integrate_with_codex(spec=spec, timeout_s=timeout_s, dry_run=dry_run))
        elif c == "cursor":
            out.append(integrate_with_cursor(spec=spec, dry_run=dry_run))
        elif c == "gemini":
            out.append(integrate_with_gemini(spec=spec, dry_run=dry_run))
        else:
            out.append(
                IntegrationResult(
                    client=str(c),
                    status="skipped",
                    message="unsupported client",
                    command=[],
                )
            )
    if managed:
        out.append(integrate_with_claude_managed(spec=spec, dry_run=dry_run))
        out.append(integrate_with_gemini_system(spec=spec, dry_run=dry_run))
    return out


def results_as_json(results: List[IntegrationResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2, sort_keys=True) + "\n"
