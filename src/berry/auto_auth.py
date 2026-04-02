from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from .env import env_first, env_truthy
from .paths import ensure_berry_home, install_id_path, mcp_env_path

DEFAULT_BERRY_BASE_URL = "https://strawberry.hassana.io"
DEFAULT_ANON_AUTH_PATH = "/api/auth/anon"

# LiteLLM OpenAI-compatible gateway (for local verification calls).
DEFAULT_OPENAI_BASE_URL = "http://20.232.57.156/v1"
# Hosted verifier service (for audit/detect tools).
DEFAULT_BERRY_SERVICE_URL = "http://52.191.234.157:8000"


@dataclass(frozen=True)
class ProvisionResult:
    ok: bool
    changed: bool
    message: str = ""


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_json_dict_private(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        if os.name != "nt":
            path.chmod(0o600)
    except Exception:
        pass


def _get_or_create_install_id() -> str:
    ensure_berry_home()
    p = install_id_path()
    existing = _load_json_dict(p)
    raw = str(existing.get("install_id") or "").strip()
    if raw:
        try:
            # Normalize to canonical string form.
            return str(uuid.UUID(raw))
        except Exception:
            pass

    new_id = str(uuid.uuid4())
    _write_json_dict_private(p, {"install_id": new_id})
    return new_id


def _has_sk_key(value: str) -> bool:
    v = (value or "").strip()
    return v.startswith("sk-") and len(v) >= 12


def maybe_provision_anon_key(
    *,
    client: str = "berry",
    version: Optional[str] = None,
    timeout_s: float = 10.0,
) -> ProvisionResult:
    """Best-effort anonymous key provisioning for first-time users.

    Writes `~/.berry/mcp_env.json` with:
      - OPENAI_API_KEY (sk-...)
      - OPENAI_BASE_URL (LiteLLM gateway)
      - BERRY_SERVICE_URL / STRAWBERRY_SERVICE_URL (hosted verifier)

    This is intentionally:
      - opt-out via STRAWBERRY_DISABLE_AUTO_AUTH=1 (or BERRY_DISABLE_AUTO_AUTH=1)
      - no-op if a key is already configured
      - never prints the key value
    """

    if env_truthy("STRAWBERRY_DISABLE_AUTO_AUTH", "BERRY_DISABLE_AUTO_AUTH"):
        return ProvisionResult(ok=True, changed=False, message="auto-auth disabled")

    # Respect explicit process env; do not persist user-provided secrets.
    if _has_sk_key(os.environ.get("OPENAI_API_KEY", "")):
        return ProvisionResult(ok=True, changed=False)

    ensure_berry_home()
    env_path = mcp_env_path()
    existing_env = _load_json_dict(env_path)
    if _has_sk_key(str(existing_env.get("OPENAI_API_KEY") or "")):
        return ProvisionResult(ok=True, changed=False)

    install_id = _get_or_create_install_id()

    anon_url = (env_first("STRAWBERRY_ANON_AUTH_URL", "BERRY_ANON_AUTH_URL") or "").strip()
    if not anon_url:
        base = (
            env_first("STRAWBERRY_BASE_URL", "BERRY_BASE_URL") or ""
        ).strip() or DEFAULT_BERRY_BASE_URL
        anon_url = base.rstrip("/") + DEFAULT_ANON_AUTH_PATH

    payload: Dict[str, Any] = {"install_id": install_id, "client": str(client or "berry")}
    if version:
        payload["version"] = str(version)

    try:
        resp = httpx.post(
            anon_url,
            json=payload,
            timeout=float(timeout_s),
            headers={"Accept": "application/json"},
        )
    except Exception as e:
        return ProvisionResult(ok=False, changed=False, message=f"auto-auth request failed: {e}")

    if resp.status_code == 429:
        return ProvisionResult(ok=False, changed=False, message="auto-auth rate-limited (HTTP 429)")
    if resp.status_code != 200:
        body = (resp.text or "").strip()
        hint = f": {body[:200]}" if body else ""
        return ProvisionResult(
            ok=False, changed=False, message=f"auto-auth failed (HTTP {resp.status_code}){hint}"
        )

    try:
        data = resp.json() if resp.content else {}
    except Exception:
        data = {}

    api_key = str(data.get("api_key") or data.get("key") or "").strip()
    if not _has_sk_key(api_key):
        return ProvisionResult(
            ok=False, changed=False, message="auto-auth succeeded but returned no valid sk-* key"
        )

    openai_base_url = (
        str(data.get("openai_base_url") or data.get("base_url") or "").strip()
        or DEFAULT_OPENAI_BASE_URL
    )
    berry_service_url = (
        str(data.get("berry_service_url") or data.get("service_url") or "").strip()
        or DEFAULT_BERRY_SERVICE_URL
    )

    new_env = {**{str(k): str(v) for k, v in existing_env.items() if k}, **{}}
    new_env["OPENAI_API_KEY"] = api_key
    new_env["OPENAI_BASE_URL"] = openai_base_url
    new_env["BERRY_SERVICE_URL"] = berry_service_url
    new_env["STRAWBERRY_SERVICE_URL"] = berry_service_url

    _write_json_dict_private(env_path, new_env)

    return ProvisionResult(
        ok=True, changed=True, message=f"provisioned hosted key \u2192 {env_path}"
    )
