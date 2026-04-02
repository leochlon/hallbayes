from __future__ import annotations

import json
from pathlib import Path

from berry.auto_auth import maybe_provision_anon_key


class _DummyResponse:
    def __init__(
        self, *, status_code: int = 200, payload: dict | None = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.content = b"{}"

    def json(self) -> dict:
        return dict(self._payload)


def test_maybe_provision_anon_key_writes_mcp_env_and_install_id(tmp_berry_home: Path, monkeypatch):
    calls: list[dict] = []

    def _fake_post(url: str, json: dict, timeout: float, headers: dict):
        calls.append({"url": url, "json": dict(json), "timeout": timeout, "headers": dict(headers)})
        return _DummyResponse(
            payload={
                "api_key": "sk-test-1234567890",
                "openai_base_url": "http://127.0.0.1:8001/v1",
                "berry_service_url": "http://127.0.0.1:8000",
            }
        )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("berry.auto_auth.httpx.post", _fake_post)

    res = maybe_provision_anon_key(client="berry", version="1.0.1")
    assert res.ok is True
    assert res.changed is True

    env_payload = json.loads((tmp_berry_home / "mcp_env.json").read_text(encoding="utf-8"))
    assert env_payload["OPENAI_API_KEY"] == "sk-test-1234567890"
    assert env_payload["OPENAI_BASE_URL"] == "http://127.0.0.1:8001/v1"
    assert env_payload["BERRY_SERVICE_URL"] == "http://127.0.0.1:8000"
    assert env_payload["STRAWBERRY_SERVICE_URL"] == "http://127.0.0.1:8000"

    install_payload = json.loads((tmp_berry_home / "install_id.json").read_text(encoding="utf-8"))
    assert isinstance(install_payload.get("install_id"), str)
    assert calls and calls[0]["json"]["install_id"] == install_payload["install_id"]


def test_maybe_provision_anon_key_is_noop_when_existing_key_saved(
    tmp_berry_home: Path, monkeypatch
):
    (tmp_berry_home / "mcp_env.json").write_text(
        json.dumps({"OPENAI_API_KEY": "sk-existing-123456"}) + "\n",
        encoding="utf-8",
    )

    def _boom(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("httpx.post should not be called when a key is already saved")

    monkeypatch.setattr("berry.auto_auth.httpx.post", _boom)
    res = maybe_provision_anon_key(client="berry")
    assert res.ok is True
    assert res.changed is False


def test_maybe_provision_anon_key_respects_disable_flag(tmp_berry_home: Path, monkeypatch):
    monkeypatch.setenv("BERRY_DISABLE_AUTO_AUTH", "1")

    def _boom(*args, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("httpx.post should not be called when auto-auth is disabled")

    monkeypatch.setattr("berry.auto_auth.httpx.post", _boom)
    res = maybe_provision_anon_key(client="berry")
    assert res.ok is True
    assert res.changed is False
    assert res.message == "auto-auth disabled"
