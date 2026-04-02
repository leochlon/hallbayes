from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path


def test_berry_deeplink_cursor_outputs_valid_url(tmp_repo: Path, tmp_berry_home: Path):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = {
        **os.environ,
        "BERRY_HOME": str(tmp_berry_home),
        "PYTHONPATH": str(src_dir) + (os.pathsep + os.environ.get("PYTHONPATH", ""))
        if os.environ.get("PYTHONPATH")
        else str(src_dir),
    }
    res = subprocess.run(
        [sys.executable, "-m", "berry", "deeplink", "cursor"],
        cwd=tmp_repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    deeplink = res.stdout.strip()
    parsed = urllib.parse.urlparse(deeplink)
    assert parsed.scheme == "cursor"
    assert parsed.netloc == "mcp"
    assert parsed.path == "/install"

    query = urllib.parse.parse_qs(parsed.query)
    assert "name" in query
    assert "config" in query
    config_b64 = query["config"][0]
    payload = json.loads(base64.b64decode(config_b64.encode("ascii")).decode("utf-8"))
    assert payload["command"] == "berry"
    assert payload["args"] == ["mcp", "--server", "classic"]
