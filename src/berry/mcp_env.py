from __future__ import annotations

import json
import os
from typing import Dict

from .paths import mcp_env_path


def load_mcp_env() -> Dict[str, str]:
    """Load env vars from mcp_env.json and BERRY_MCP_ENV_JSON (last wins)."""

    env: Dict[str, str] = {}

    # File-based defaults.
    try:
        p = mcp_env_path()
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    kk = str(k).strip()
                    if not kk:
                        continue
                    if v is None:
                        continue
                    env[kk] = str(v)
    except Exception:
        # Fail open: env injection is optional.
        pass

    # Explicit env override.
    try:
        j = os.environ.get("BERRY_MCP_ENV_JSON")
        if j:
            raw = json.loads(j)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    kk = str(k).strip()
                    if not kk:
                        continue
                    if v is None:
                        continue
                    env[kk] = str(v)
    except Exception:
        pass

    return env
