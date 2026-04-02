from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _AnonAuthHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8") or "{}")
        type(self).requests.append({"path": self.path, "payload": payload})
        body = json.dumps(
            {
                "api_key": "sk-test-1234567890",
                "openai_base_url": "http://127.0.0.1:8001/v1",
                "berry_service_url": "http://127.0.0.1:8000",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def test_berry_init_auto_provisions_key_without_leaking_it_into_repo_configs(
    tmp_repo: Path, tmp_berry_home: Path
):
    src_dir = Path(__file__).resolve().parents[2] / "src"
    _AnonAuthHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AnonAuthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            **os.environ,
            "BERRY_HOME": str(tmp_berry_home),
            "BERRY_ANON_AUTH_URL": f"http://127.0.0.1:{server.server_port}/api/auth/anon",
            "PYTHONPATH": str(src_dir) + (os.pathsep + os.environ.get("PYTHONPATH", ""))
            if os.environ.get("PYTHONPATH")
            else str(src_dir),
        }
        res = subprocess.run(
            [sys.executable, "-m", "berry", "init"],
            cwd=tmp_repo,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    env_payload = json.loads((tmp_berry_home / "mcp_env.json").read_text(encoding="utf-8"))
    assert env_payload["OPENAI_API_KEY"] == "sk-test-1234567890"
    assert env_payload["OPENAI_BASE_URL"] == "http://127.0.0.1:8001/v1"
    assert env_payload["BERRY_SERVICE_URL"] == "http://127.0.0.1:8000"
    assert (tmp_berry_home / "install_id.json").exists()
    assert _AnonAuthHandler.requests
    assert _AnonAuthHandler.requests[0]["path"] == "/api/auth/anon"
    assert _AnonAuthHandler.requests[0]["payload"]["client"] == "berry"

    repo_files = [
        tmp_repo / ".cursor" / "mcp.json",
        tmp_repo / ".codex" / "config.toml",
        tmp_repo / ".mcp.json",
        tmp_repo / ".gemini" / "settings.json",
    ]
    for path in repo_files:
        text = path.read_text(encoding="utf-8")
        assert "sk-test-1234567890" not in text

    assert "provisioned hosted key" in res.stderr
