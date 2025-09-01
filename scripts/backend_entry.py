"""
Offline Backend Entry (PyInstaller-friendly)
-------------------------------------------

Launches the Streamlit app without relying on pip at runtime.
Used by scripts/build_offline_backend.(sh|bat) to create a single-file binary.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _bundle_root() -> Path:
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline backend entry for Streamlit app")
    p.add_argument("--port", type=int, default=8501, help="Server port")
    p.add_argument("--headless", action="store_true", help="Run headless (no auto browser)")
    args = p.parse_args(argv)

    base = _bundle_root()
    # Structured path inside the bundle
    web_app = base / "app" / "web" / "web_app.py"
    if not web_app.exists():
        print(f"web_app.py not found at {web_app}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(base))

    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true" if args.headless else "false"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(web_app),
        "--server.port",
        str(args.port),
    ]
    if args.headless:
        cmd += ["--server.headless", "true"]

    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
# Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License - see LICENSE file for details
