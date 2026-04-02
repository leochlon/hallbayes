from __future__ import annotations

import json
import zipfile
from pathlib import Path

from berry.audit import append_event
from berry.config import BerryConfig, save_global_config
from berry.support import create_support_bundle


def test_support_bundle_contains_diagnostics_and_audit(tmp_repo: Path, tmp_berry_home: Path):
    save_global_config(BerryConfig())
    append_event("tool_call", {"api_key": "sk-SECRET"}, log_path=tmp_berry_home / "audit.log.jsonl")
    out = create_support_bundle(project_root=tmp_repo)
    assert out.exists()

    with zipfile.ZipFile(out, "r") as z:
        diag = json.loads(z.read("diagnostics.json").decode("utf-8"))
        assert "berry_version" in diag
        assert "config" in diag
        assert z.read("audit.log.jsonl") is not None
