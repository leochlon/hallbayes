from __future__ import annotations

import base64
import json
import urllib.parse
from pathlib import Path

from berry.clients import (
    McpServerSpec,
    berry_server_spec,
    render_claude_mcp_json,
    render_codex_config_toml,
    render_cursor_deeplink,
    render_cursor_mcp_json,
    render_gemini_settings_json,
    write_claude_mcp_json,
    write_codex_config_toml,
    write_cursor_mcp_json,
    write_gemini_settings_json,
)


def test_render_cursor_json_shape():
    spec = berry_server_spec()
    payload = json.loads(render_cursor_mcp_json(spec))
    assert "mcpServers" in payload
    assert spec.name in payload["mcpServers"]


def test_write_repo_scoped_configs(tmp_repo: Path):
    cursor = write_cursor_mcp_json(project_root=tmp_repo)
    codex = write_codex_config_toml(project_root=tmp_repo)
    claude = write_claude_mcp_json(project_root=tmp_repo)
    gemini = write_gemini_settings_json(project_root=tmp_repo)

    assert cursor.exists()
    assert codex.exists()
    assert claude.exists()
    assert gemini.exists()


def test_render_codex_toml_contains_mcp_servers_table():
    spec = berry_server_spec(name="berry")
    toml_text = render_codex_config_toml(spec)
    assert "[mcp_servers.berry]" in toml_text
    assert 'command = "berry"' in toml_text


def test_render_claude_json_shape():
    spec = berry_server_spec()
    payload = json.loads(render_claude_mcp_json(spec))
    assert "mcpServers" in payload


def test_render_gemini_json_shape():
    spec = berry_server_spec()
    payload = json.loads(render_gemini_settings_json(spec))
    # Gemini uses mcpServers (same as other clients)
    assert payload["mcpServers"][spec.name]["command"] == spec.command


def test_render_cursor_deeplink_encodes_spec():
    spec = McpServerSpec(
        name="berry",
        command="berry",
        args=["mcp", "--server", "science"],
        env={"BERRY_HOME": "/tmp/berry"},
    )
    deeplink = render_cursor_deeplink(spec)
    parsed = urllib.parse.urlparse(deeplink)
    assert parsed.scheme == "cursor"
    assert parsed.netloc == "mcp"
    assert parsed.path == "/install"

    query = urllib.parse.parse_qs(parsed.query)
    assert query["name"] == [spec.name]
    config_b64 = query["config"][0]
    config = json.loads(base64.b64decode(config_b64.encode("ascii")).decode("utf-8"))
    assert config["command"] == spec.command
    assert config["args"] == spec.args
    assert config["env"] == spec.env
