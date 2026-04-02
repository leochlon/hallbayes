from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_berry_init_creates_repo_scoped_files(tmp_repo: Path, tmp_berry_home: Path):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = {
        **os.environ,
        "BERRY_HOME": str(tmp_berry_home),
        "BERRY_DISABLE_AUTO_AUTH": "1",
        # Ensure `python -m berry ...` works from a source checkout (without requiring
        # an editable install).
        "PYTHONPATH": str(src_dir) + (os.pathsep + os.environ.get("PYTHONPATH", ""))
        if os.environ.get("PYTHONPATH")
        else str(src_dir),
    }
    subprocess.run([sys.executable, "-m", "berry", "init"], cwd=tmp_repo, env=env, check=True)
    assert (tmp_repo / ".cursor" / "mcp.json").exists()
    assert (tmp_repo / ".codex" / "config.toml").exists()
    assert (tmp_repo / ".mcp.json").exists()
    assert (tmp_repo / ".gemini" / "settings.json").exists()


def test_berry_support_issue_creates_bundle(tmp_repo: Path, tmp_berry_home: Path):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = {
        **os.environ,
        "BERRY_HOME": str(tmp_berry_home),
        "PYTHONPATH": str(src_dir) + (os.pathsep + os.environ.get("PYTHONPATH", ""))
        if os.environ.get("PYTHONPATH")
        else str(src_dir),
    }
    res = subprocess.run(
        [sys.executable, "-m", "berry", "support", "issue"],
        cwd=tmp_repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    first = (res.stdout.splitlines() or [""])[0]
    assert first.startswith("Support bundle: ")
    p = Path(first.removeprefix("Support bundle: ").strip())
    assert p.exists()
