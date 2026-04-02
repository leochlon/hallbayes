from __future__ import annotations

from pathlib import Path


def test_goreleaser_config_contains_release_sections():
    text = Path(".goreleaser.yml").read_text(encoding="utf-8")
    for key in ("signs:", "notarize:"):
        assert key in text


def test_slsa_workflow_has_generator_and_triggers():
    text = Path(".github/workflows/slsa-provenance.yml").read_text(encoding="utf-8")
    assert "slsa-framework/slsa-github-generator" in text
    assert "workflow_dispatch:" in text
    assert "release:" in text


def test_release_check_workflow_runs_goreleaser():
    text = Path(".github/workflows/release-check.yml").read_text(encoding="utf-8")
    assert "goreleaser/goreleaser-action" in text
    assert "goreleaser" in text


def test_release_macos_workflow_updates_homebrew():
    text = Path(".github/workflows/release-macos.yml").read_text(encoding="utf-8")
    assert "render_homebrew_cask.py" in text
    assert "homebrew-tap" in text
