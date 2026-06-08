"""OpenAI-compatible -> Azure OpenAI pool proxy with round-robin concurrency.

Berry's openai backend speaks plain OpenAI (Bearer + base_url). Azure needs an
api-key header, an api-version query param, and the deployment in the path. This
proxy accepts OpenAI Chat Completions on localhost and forwards each request to
the next live Azure endpoint in the pool, so Berry's real verifier runs unmodified
with N-way concurrency.

Run: python bench/aoai_proxy.py --pool ~/Downloads/aiderB/aoai_pool.json --port 8900
Point Berry at it: OPENAI_BASE_URL=http://127.0.0.1:8900/v1  OPENAI_API_KEY=proxy
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_POOL: list[dict] = []
_CYCLE = None
_LOCK = threading.Lock()


def _next_endpoint() -> dict:
    with _LOCK:
        return next(_CYCLE)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):  # silence
        return

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404, "only /chat/completions")
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        payload, status = b'{"error":{"message":"proxy: no endpoint"}}', 502
        # retry transient failures by forwarding to the next endpoint in the pool
        for _ in range(min(4, len(_POOL))):
            ep = _next_endpoint()
            url = (
                ep["endpoint"].rstrip("/")
                + f"/openai/deployments/{ep['deployment']}/chat/completions"
                + f"?api-version={ep['apiVersion']}"
            )
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json", "api-key": ep["apiKey"]},
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    payload, status = r.read(), r.status
                break
            except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
                payload, status = e.read(), e.code
                if status < 500 and status != 429:
                    break  # genuine client error; do not retry
            except Exception as e:  # noqa: BLE001
                payload, status = json.dumps({"error": {"message": f"proxy: {e}"}}).encode(), 502
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--port", type=int, default=8900)
    args = ap.parse_args()
    global _POOL, _CYCLE
    _POOL = json.load(open(os.path.expanduser(args.pool)))
    _CYCLE = itertools.cycle(_POOL)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"proxy on http://127.0.0.1:{args.port}/v1 -> {len(_POOL)} endpoints", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
