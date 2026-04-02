from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str


def can_read_path(
    path: Path, *, allowed_roots: Iterable[str], project_root: Optional[Path]
) -> PermissionDecision:
    try:
        p = Path(path).expanduser().resolve()
    except Exception as e:
        return PermissionDecision(False, f"Invalid path: {e}")

    roots: list[Path] = []

    if project_root is not None:
        try:
            roots.append(Path(project_root).expanduser().resolve())
        except Exception as e:
            return PermissionDecision(False, f"Invalid project_root: {e}")

    for r in list(allowed_roots or []):
        try:
            rp = Path(str(r)).expanduser()
            if not rp.is_absolute() and project_root is not None:
                rp = Path(project_root) / rp
            roots.append(rp.resolve())
        except Exception:
            # Ignore invalid allowed_roots entries (fails closed).
            continue

    if not roots:
        return PermissionDecision(
            False, "File reads are disabled (no project root or allowed roots configured)"
        )

    for root in roots:
        if p == root or p.is_relative_to(root):
            return PermissionDecision(True, f"Read allowed within: {root}")

    return PermissionDecision(False, "Path is outside the project root and allowed_roots")


def can_write_path(
    path: Path,
    *,
    allow_write: bool,
    allowed_roots: Iterable[str],
    project_root: Optional[Path],
) -> PermissionDecision:
    if not allow_write:
        return PermissionDecision(False, "Writes are disabled by configuration")
    return PermissionDecision(True, "Write allowed")
