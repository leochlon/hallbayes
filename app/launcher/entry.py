"""
One‑Click Launcher for the Streamlit Web UI (app path)
-----------------------------------------------------

Creates/uses a local virtualenv, installs dependencies (unless skipped),
then launches the Streamlit app. Mirrors top-level `launch.py`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[2]  # repo root
_env_venv = os.environ.get("LAUNCH_VENV_DIR", "").strip()
VENV_DIR = Path(_env_venv) if _env_venv else (HERE / ".venv")


def venv_python_path(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def ensure_venv(venv: Path) -> Path:
    py = venv_python_path(venv)
    if py.exists():
        return py
    print("Creating virtual environment in .venv …")
    import venv as venv_mod
    venv_mod.EnvBuilder(with_pip=True).create(str(venv))
    return venv_python_path(venv)


def pip_install(py: Path, *pkgs: str) -> None:
    cmd = [str(py), "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"]
    subprocess.check_call(cmd)
    req = HERE / "requirements.txt"
    if req.exists():
        subprocess.check_call([str(py), "-m", "pip", "install", "-r", str(req)])
    else:
        subprocess.check_call([str(py), "-m", "pip", "install", *pkgs])


def run_streamlit(py: Path, *, port: int | None = None, headless: bool = False) -> int:
    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true" if headless else "false"
    target = HERE / "app" / "web" / "web_app.py"
    cmd = [str(py), "-m", "streamlit", "run", str(target)]
    if port is not None:
        cmd += ["--server.port", str(port)]
    return subprocess.call(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch Streamlit app with local venv (app path)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default streamlit port)")
    parser.add_argument("--headless", action="store_true", help="Run Streamlit headless (do not auto-open browser)")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install step")
    args = parser.parse_args(argv)

    try:
        py = ensure_venv(VENV_DIR)
        print(f"Using Python: {py}")
        if not args.skip_install:
            print("Installing/updating dependencies (first run may take a minute)…")
            pip_install(py, "streamlit>=1.28.0", "openai>=1.0.0")
        print("Launching Streamlit…")
        code = run_streamlit(py, port=args.port, headless=bool(args.headless))
        if code != 0:
            print(f"Streamlit exited with code {code}")
        return code
    except subprocess.CalledProcessError as e:
        print("A command failed:", e)
        return e.returncode or 1
    except Exception as e:
        print("Unexpected error:", e)
        return 1
    finally:
        if sys.stdout and sys.stdout.isatty():
            try:
                input("\nPress Enter to close this window…")
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
# Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License - see LICENSE file for details
