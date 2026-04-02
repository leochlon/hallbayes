from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the src-layout package is importable when running `pytest` without
# installing the project (professional ergonomics).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    return repo


@pytest.fixture()
def tmp_berry_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "berry_home"
    home.mkdir()
    monkeypatch.setenv("BERRY_HOME", str(home))
    return home
