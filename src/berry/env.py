from __future__ import annotations

import os
from typing import Optional


def env_first(*keys: str, default: Optional[str] = None) -> Optional[str]:
    """Return the first non-empty environment variable among keys.

    This lets Berry support new branding (`STRAWBERRY_*`) while keeping
    legacy `BERRY_*` variables working.
    """

    for key in keys:
        value = os.environ.get(str(key), "")
        if value is not None:
            value = str(value).strip()
            if value != "":
                return value
    return default


_TRUTHY = {"1", "true", "yes", "y", "on"}


def env_truthy(*keys: str) -> bool:
    value = env_first(*keys)
    return str(value or "").strip().lower() in _TRUTHY
