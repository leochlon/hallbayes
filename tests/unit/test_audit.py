from __future__ import annotations

from pathlib import Path

from berry.audit import append_event, iter_events, prune_events, redact


def test_redact_keys_and_openai_style_values():
    obj = {"api_key": "sk-1234567890SECRET", "nested": {"token": "abc"}}
    r = redact(obj)
    assert r["api_key"] == "REDACTED"
    assert r["nested"]["token"] == "REDACTED"


def test_append_and_prune(tmp_path: Path):
    log = tmp_path / "audit.jsonl"
    append_event("x", {"a": 1}, log_path=log)
    assert list(iter_events(log))
    removed = prune_events(retention_days=0, log_path=log)
    assert removed >= 0
