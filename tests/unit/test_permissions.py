from __future__ import annotations

from pathlib import Path

from berry.permissions import can_read_path, can_write_path


def test_can_read_with_project_root(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "a.txt"
    f.write_text("x", encoding="utf-8")

    dec = can_read_path(f, allowed_roots=[], project_root=repo)
    assert dec.allowed is True


def test_can_read_denies_outside_project_root(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x", encoding="utf-8")

    dec = can_read_path(outside, allowed_roots=[], project_root=repo)
    assert dec.allowed is False


def test_can_read_allows_allowed_roots(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    allow = tmp_path / "allow"
    allow.mkdir()
    f = allow / "x.txt"
    f.write_text("x", encoding="utf-8")

    dec = can_read_path(f, allowed_roots=[str(allow)], project_root=repo)
    assert dec.allowed is True


def test_can_read_denies_when_no_roots_configured(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")

    dec = can_read_path(f, allowed_roots=[], project_root=None)
    assert dec.allowed is False


def test_can_write_disabled_by_default(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    f = repo / "a.txt"

    dec = can_write_path(f, allow_write=False, allowed_roots=[], project_root=repo)
    assert dec.allowed is False
    assert "Writes are disabled" in dec.reason
