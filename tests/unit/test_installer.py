from __future__ import annotations

import json
from pathlib import Path

from berry.installer import actions_to_json, install_many, platform_keys

EXPECTED_PLATFORMS = {
    "claude",
    "windows",
    "codebuddy",
    "codex",
    "opencode",
    "kilo",
    "copilot",
    "vscode",
    "aider",
    "claw",
    "droid",
    "trae",
    "trae-cn",
    "gemini",
    "hermes",
    "kimi",
    "amp",
    "kiro",
    "pi",
    "cursor",
    "devin",
    "antigravity",
}


def test_platform_registry_surface() -> None:
    assert EXPECTED_PLATFORMS.issubset(set(platform_keys()))


def test_codex_install_embeds_and_refreshes_command_preserving_user_toml(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_config = tmp_path / ".codex" / "config.toml"
    codex_config.parent.mkdir(parents=True)
    codex_config.write_text(
        '[features]\nfoo = true\nmulti_agent = false\n\n[user]\nname = "me"\n',
        encoding="utf-8",
    )

    actions = install_many(
        ["codex"],
        scope="user",
        project_dir=None,
        force=False,
        dry_run=False,
        berry_command_raw="/opt/berry/bin/berry",
        name="berry",
        install_hooks=True,
        install_mcp=True,
    )

    assert all(a.status != "failed" for a in actions)
    assert (tmp_path / ".codex" / "skills" / "berry" / "SKILL.md").exists()
    agents = (tmp_path / ".codex" / "AGENTS.md").read_text(encoding="utf-8")
    assert "/opt/berry/bin/berry mcp --server classic" in agents
    config = codex_config.read_text(encoding="utf-8")
    assert "[features]" in config
    assert "foo = true" in config
    assert "multi_agent = true" in config
    assert "[user]" in config
    assert 'command = "/opt/berry/bin/berry"' in config

    install_many(
        ["codex"],
        scope="user",
        project_dir=None,
        force=False,
        dry_run=False,
        berry_command_raw="/new/location/berry",
        name="berry",
        install_hooks=True,
        install_mcp=True,
    )
    refreshed = codex_config.read_text(encoding="utf-8")
    refreshed_agents = (tmp_path / ".codex" / "AGENTS.md").read_text(encoding="utf-8")
    assert 'command = "/new/location/berry"' in refreshed
    assert "/opt/berry/bin/berry" not in refreshed
    assert "/new/location/berry mcp --server classic" in refreshed_agents
    assert "/opt/berry/bin/berry" not in refreshed_agents


def test_claude_project_install_upserts_hooks_and_mcp_without_destroying_existing_json(
    tmp_repo: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_repo)
    settings = tmp_repo / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps({"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]}, "x": 1}),
        encoding="utf-8",
    )

    actions = install_many(
        ["claude"],
        scope="project",
        project_dir=tmp_repo,
        force=False,
        dry_run=False,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=True,
    )

    assert all(a.status != "failed" for a in actions)
    assert (tmp_repo / ".claude" / "skills" / "berry" / "SKILL.md").exists()
    assert (tmp_repo / ".claude" / "CLAUDE.md").exists()
    payload = json.loads(settings.read_text(encoding="utf-8"))
    assert payload["x"] == 1
    pre = payload["hooks"]["PreToolUse"]
    assert len(pre) == 3  # original + Berry Bash + Berry Read|Glob
    assert any("/abs/berry mcp --server classic" in json.dumps(h) for h in pre)
    mcp = json.loads((tmp_repo / ".mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["berry"]["command"] == "/abs/berry"


def test_invalid_json_fails_closed_unless_force(tmp_repo: Path) -> None:
    settings = tmp_repo / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{ definitely not json", encoding="utf-8")

    actions = install_many(
        ["claude"],
        scope="project",
        project_dir=tmp_repo,
        force=False,
        dry_run=False,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=False,
    )
    failed = [a for a in actions if a.kind == "hooks"]
    assert failed and failed[0].status == "failed"
    assert settings.read_text(encoding="utf-8") == "{ definitely not json"

    repaired = install_many(
        ["claude"],
        scope="project",
        project_dir=tmp_repo,
        force=True,
        dry_run=False,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=False,
    )
    assert all(a.status != "failed" for a in repaired)
    assert json.loads(settings.read_text(encoding="utf-8"))["hooks"]["PreToolUse"]


def test_malformed_managed_markdown_section_fails_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    agents = tmp_path / ".codex" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("human text\n<!-- berry:install:start -->\npartial\n", encoding="utf-8")

    actions = install_many(
        ["codex"],
        scope="user",
        project_dir=None,
        force=False,
        dry_run=False,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=False,
    )

    failed = [a for a in actions if a.kind == "instructions"]
    assert failed and failed[0].status == "failed"
    assert (
        agents.read_text(encoding="utf-8") == "human text\n<!-- berry:install:start -->\npartial\n"
    )


def test_dry_run_does_not_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    actions = install_many(
        ["gemini"],
        scope="user",
        project_dir=None,
        force=False,
        dry_run=True,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=True,
    )
    assert actions
    assert {a.status for a in actions} == {"dry-run"}
    assert not (tmp_path / ".gemini").exists()
    # JSON serialization is part of the public CLI contract.
    parsed = json.loads(actions_to_json(actions))
    assert parsed[0]["platform"] == "gemini"


def test_cursor_user_install_writes_project_rule_but_user_mcp(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    monkeypatch.setenv("HOME", str(home))

    actions = install_many(
        ["cursor"],
        scope="user",
        project_dir=repo,
        force=False,
        dry_run=False,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=True,
    )
    assert all(a.status != "failed" for a in actions)
    assert (repo / ".cursor" / "rules" / "berry.mdc").exists()
    mcp = json.loads((home / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert mcp["mcpServers"]["berry"]["command"] == "/abs/berry"


def test_vscode_project_install_uses_vscode_mcp_shape(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    actions = install_many(
        ["vscode"],
        scope="project",
        project_dir=repo,
        force=False,
        dry_run=False,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=True,
    )
    assert all(a.status != "failed" for a in actions)
    assert (repo / ".github" / "copilot-instructions.md").exists()
    mcp = json.loads((repo / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    assert "servers" in mcp
    assert mcp["servers"]["berry"]["command"] == "/abs/berry"
    assert "mcpServers" not in mcp


def test_kilo_project_install_writes_and_registers_plugin(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".kilo").mkdir()
    (repo / ".kilo" / "kilo.jsonc").write_text(
        '{\n  // user plugin\n  "plugin": ["file:///existing.js"],\n}\n',
        encoding="utf-8",
    )

    actions = install_many(
        ["kilo"],
        scope="project",
        project_dir=repo,
        force=False,
        dry_run=False,
        berry_command_raw="/abs/berry",
        name="berry",
        install_hooks=True,
        install_mcp=True,
    )

    assert all(a.status != "failed" for a in actions)
    assert (repo / ".kilo" / "plugins" / "berry.js").exists()
    config = json.loads((repo / ".kilo" / "kilo.json").read_text(encoding="utf-8"))
    assert "file:///existing.js" in config["plugin"]
    assert (repo / ".kilo" / "plugins" / "berry.js").resolve().as_uri() in config["plugin"]
