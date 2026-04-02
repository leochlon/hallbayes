from __future__ import annotations

import json
import platform
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from . import __version__
from .audit import redact
from .config import BerryConfig, load_config
from .paths import audit_log_path, support_bundle_dir


def create_support_bundle(*, project_root: Optional[Path], out_path: Optional[Path] = None) -> Path:
    bundle_dir = support_bundle_dir()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    p = out_path or (bundle_dir / f"berry-support-bundle-{ts}.zip")
    cfg: BerryConfig = load_config(project_root=project_root)

    diag = {
        "berry_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "project_root": str(project_root) if project_root else None,
        "config": redact(asdict(cfg)),
    }

    audit_p = audit_log_path()
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("diagnostics.json", json.dumps(diag, indent=2) + "\n")
        try:
            z.write(audit_p, arcname="audit.log.jsonl")
        except FileNotFoundError:
            z.writestr("audit.log.jsonl", "")
    return p
