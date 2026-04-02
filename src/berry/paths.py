from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def berry_home() -> Path:
    override = os.environ.get("BERRY_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".berry").resolve()


def ensure_berry_home() -> Path:
    home = berry_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def audit_log_path() -> Path:
    return berry_home() / "audit.log.jsonl"


def config_path() -> Path:
    return berry_home() / "config.json"


def license_path() -> Path:
    return berry_home() / "license.json"


def mcp_env_path() -> Path:
    """Path to optional MCP env defaults.

    This file is used when generating client configs (e.g., .cursor/mcp.json) so the
    Berry server can be launched with a consistent runtime environment.

    It is intentionally separate from `config.json` so it can be managed differently
    (it may contain secrets like API keys).
    """

    return berry_home() / "mcp_env.json"


def install_id_path() -> Path:
    """Path to a stable, locally-generated install id for anonymous onboarding."""

    return berry_home() / "install_id.json"


def support_bundle_dir() -> Path:
    return berry_home() / "support_bundles"


def resolve_user_path(path: Union[str, Path], *, project_root: Optional[Path] = None) -> Path:
    """Resolve a user-provided path.

    Berry should behave like repo-scoped tools:
    - "~" expands to the user's home.
    - Relative paths resolve against the configured project_root when available.
    - Otherwise, relative paths resolve against the current working directory.

    The returned path is fully-resolved (symlinks collapsed) via `Path.resolve()`.
    """

    p = Path(path).expanduser()
    if not p.is_absolute() and project_root is not None:
        p = Path(project_root) / p
    return p.resolve()
