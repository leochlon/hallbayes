from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _env(tmp_home: Path) -> dict[str, str]:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    return {
        **os.environ,
        "HOME": str(tmp_home),
        "BERRY_HOME": str(tmp_home / ".berry"),
        "BERRY_DISABLE_AUTO_AUTH": "1",
        "PYTHONPATH": str(src_dir) + (os.pathsep + os.environ.get("PYTHONPATH", ""))
        if os.environ.get("PYTHONPATH")
        else str(src_dir),
    }


def test_berry_install_codex_cli_refreshes_embedded_command(tmp_repo: Path, tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    env = _env(home)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "berry",
            "install",
            "--platform",
            "codex",
            "--berry-command",
            "/first/berry",
        ],
        cwd=tmp_repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "berry",
            "codex",
            "install",
            "--berry-command",
            "/second/berry",
        ],
        cwd=tmp_repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    config = (home / ".codex" / "config.toml").read_text(encoding="utf-8")
    agents = (home / ".codex" / "AGENTS.md").read_text(encoding="utf-8")
    assert 'command = "/second/berry"' in config
    assert "/second/berry mcp --server classic" in agents
    assert "/first/berry" not in config
    assert "/first/berry" not in agents


def test_berry_install_dry_run_json_has_no_side_effects(tmp_repo: Path, tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    env = _env(home)
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "berry",
            "install",
            "--platform",
            "gemini",
            "--dry-run",
            "--json",
            "--berry-command",
            "/abs/berry",
        ],
        cwd=tmp_repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    actions = json.loads(res.stdout)
    assert actions
    assert {a["status"] for a in actions} == {"dry-run"}
    assert not (home / ".gemini").exists()
