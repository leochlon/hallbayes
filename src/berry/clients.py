from __future__ import annotations

import base64
import json
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

from .mcp_env import load_mcp_env


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]


def berry_server_spec(*, name: str = "berry", server: str = "classic") -> McpServerSpec:
    """Return an MCP server spec. The server parameter is kept for backwards compat."""
    srv = str(server or "classic").strip().lower()
    if srv != "classic":
        srv = "classic"
    # Use the CLI entrypoint so users can `pipx install berry` and get a stable `berry` command on PATH.
    return McpServerSpec(
        name=name, command="berry", args=["mcp", "--server", srv], env=load_mcp_env()
    )


def berry_server_specs(*, profile: str = "classic", name: str = "berry") -> List[McpServerSpec]:
    """Return one or more MCP server specs. The profile parameter is kept for backwards compat."""
    _ = profile  # kept for API compatibility
    return [berry_server_spec(name=name, server="classic")]


def _normalize_specs(spec_or_specs: object | None) -> List[McpServerSpec]:
    if spec_or_specs is None:
        return [berry_server_spec()]
    if isinstance(spec_or_specs, McpServerSpec):
        return [spec_or_specs]
    # Duck-type iterable of specs.
    if isinstance(spec_or_specs, Iterable):
        items = cast(Iterable[object], spec_or_specs)
        out = [s for s in items if isinstance(s, McpServerSpec)]
        return out or [berry_server_spec()]
    return [berry_server_spec()]


def write_cursor_mcp_json(
    *, project_root: Path, spec: Optional[object] = None, force: bool = False
) -> Path:
    specs = _normalize_specs(spec)
    out = Path(project_root) / ".cursor" / "mcp.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite: {out} (use --force)")
    out.write_text(render_cursor_mcp_json(specs), encoding="utf-8")
    return out


def render_cursor_mcp_json(specs: object) -> str:
    ss = _normalize_specs(specs)
    payload = {
        "mcpServers": {s.name: {"command": s.command, "args": s.args, "env": s.env} for s in ss}
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_cursor_deeplink(spec: McpServerSpec) -> str:
    payload: Dict[str, object] = {"command": spec.command, "args": spec.args}
    if spec.env:
        payload["env"] = spec.env
    config_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    config_b64 = base64.b64encode(config_json).decode("ascii")
    name = urllib.parse.quote(spec.name, safe="")
    config = urllib.parse.quote(config_b64, safe="")
    return f"cursor://mcp/install?name={name}&config={config}"


def write_claude_mcp_json(
    *, project_root: Path, spec: Optional[object] = None, force: bool = False
) -> Path:
    specs = _normalize_specs(spec)
    out = Path(project_root) / ".mcp.json"
    if out.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite: {out} (use --force)")
    out.write_text(render_claude_mcp_json(specs), encoding="utf-8")
    return out


def render_claude_mcp_json(specs: object) -> str:
    ss = _normalize_specs(specs)
    payload = {
        "mcpServers": {s.name: {"command": s.command, "args": s.args, "env": s.env} for s in ss}
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_gemini_settings_json(
    *, project_root: Path, spec: Optional[object] = None, force: bool = False
) -> Path:
    specs = _normalize_specs(spec)
    out = Path(project_root) / ".gemini" / "settings.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite: {out} (use --force)")
    # Gemini CLI's `gemini mcp add` command writes to settings.json; we mirror a simple declarative shape.
    out.write_text(render_gemini_settings_json(specs), encoding="utf-8")
    return out


def render_gemini_settings_json(specs: object) -> str:
    ss = _normalize_specs(specs)
    # Gemini CLI uses a top-level `mcpServers` object in settings.json.
    # We mirror the declarative shape used by the CLI.
    payload = {
        "mcpServers": {s.name: {"command": s.command, "args": s.args, "env": s.env} for s in ss}
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_codex_config_toml(
    *, project_root: Path, spec: Optional[object] = None, force: bool = False
) -> Path:
    specs = _normalize_specs(spec)
    out = Path(project_root) / ".codex" / "config.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite: {out} (use --force)")

    out.write_text(render_codex_config_toml(specs), encoding="utf-8")
    return out


def render_codex_config_toml(specs: object) -> str:
    ss = _normalize_specs(specs)

    # Minimal TOML writer to avoid extra deps.
    def _toml_str(s: str) -> str:
        # json.dumps produces a quoted, escaped string that is compatible with TOML basic strings
        # for common characters (quotes, backslashes, newlines).
        return json.dumps(s)

    lines: list[str] = []
    for spec in ss:
        lines.append(f"[mcp_servers.{spec.name}]")
        lines.append(f"command = {_toml_str(spec.command)}")
        lines.append("args = [" + ", ".join(_toml_str(a) for a in spec.args) + "]")
        lines.append("")
        if spec.env:
            lines.append(f"[mcp_servers.{spec.name}.env]")
            for k, v in spec.env.items():
                # Quote keys defensively (TOML supports quoted keys).
                lines.append(f"{_toml_str(k)} = {_toml_str(v)}")
            lines.append("")
    return "\n".join(lines)
