from __future__ import annotations

import io
import json
import urllib.error
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from berry.hallucination_detector.backends import local_backend
from berry.hallucination_detector.backends.local_backend import call_text_chat_local
from berry.hallucination_detector.backends.openai_backend import TextResult


def _mk_response(payload: Dict[str, Any]) -> MagicMock:
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.read = MagicMock(return_value=body)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _mk_chat_payload(
    *,
    text: str = "YES",
    logprobs: Optional[Dict[str, Any]] = ...,  # type: ignore[assignment]
    response_id: str = "chatcmpl-local-1",
) -> Dict[str, Any]:
    choice: Dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": text},
        "finish_reason": "stop",
    }
    if logprobs is ...:
        choice["logprobs"] = {
            "content": [
                {
                    "token": "YES",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "YES", "logprob": -0.1},
                        {"token": "NO", "logprob": -2.0},
                        {"token": "UNSURE", "logprob": -3.0},
                    ],
                }
            ]
        }
    else:
        choice["logprobs"] = logprobs
    return {
        "id": response_id,
        "object": "chat.completion",
        "model": "local-model",
        "choices": [choice],
    }


def _mk_http_error(code: int, reason: str = "Server Error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://test/chat/completions",
        code=code,
        msg=reason,
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(b""),
    )


def test_local_backend_url_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: List[str] = []

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured.append(req.full_url)
        return _mk_response(_mk_chat_payload(logprobs=None))

    monkeypatch.setenv("BERRY_LOCAL_BASE_URL", "http://from-env:1111/v1")
    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        call_text_chat_local(prompt="x", model="local-model", base_url="http://from-param:2222/v1", retries=0)
    assert captured[-1].startswith("http://from-param:2222/v1")

    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        call_text_chat_local(prompt="x", model="local-model", retries=0)
    assert captured[-1].startswith("http://from-env:1111/v1")

    monkeypatch.delenv("BERRY_LOCAL_BASE_URL", raising=False)
    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        call_text_chat_local(prompt="x", model="local-model", retries=0)
    default_url = captured[-1]
    assert default_url.startswith("http://")
    assert "127.0.0.1" in default_url or "localhost" in default_url


def test_local_backend_request_shape() -> None:
    captured: Dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["headers"] = dict(req.header_items())
        raw = req.data
        captured["body"] = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        return _mk_response(_mk_chat_payload())

    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        call_text_chat_local(
            prompt="What is 2+2?",
            model="qwen2.5-coder",
            instructions="You are a calculator.",
            temperature=0.0,
            max_output_tokens=64,
            include_logprobs=True,
            top_logprobs=5,
            base_url="http://localhost:1234/v1",
            retries=0,
        )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    header_names = {k.lower() for k in captured["headers"]}
    assert "content-type" in header_names
    body = captured["body"]
    assert body["model"] == "qwen2.5-coder"
    assert isinstance(body["messages"], list) and len(body["messages"]) == 2
    assert body["messages"][0] == {"role": "system", "content": "You are a calculator."}
    assert body["messages"][1] == {"role": "user", "content": "What is 2+2?"}
    assert body["temperature"] == 0.0
    assert "max_tokens" in body and int(body["max_tokens"]) >= 16
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5


def test_local_backend_parses_logprobs() -> None:
    payload = _mk_chat_payload(text="YES")

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _mk_response(payload)

    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        res = call_text_chat_local(
            prompt="YES",
            model="local-model",
            include_logprobs=True,
            top_logprobs=3,
            base_url="http://localhost:1234/v1",
            retries=0,
        )

    assert isinstance(res, TextResult)
    assert res.text == "YES"
    assert res.response_id == "chatcmpl-local-1"
    assert res.logprobs is not None
    first = res.logprobs[0]
    assert first["token"] == "YES"
    assert first["logprob"] == -0.1
    assert isinstance(first["top_logprobs"], list)


def test_local_backend_handles_null_logprobs() -> None:
    payload = _mk_chat_payload(text="hello", logprobs=None)

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        return _mk_response(payload)

    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        res = call_text_chat_local(
            prompt="hi",
            model="local-model",
            include_logprobs=True,
            top_logprobs=3,
            base_url="http://localhost:1234/v1",
            retries=0,
        )

    assert isinstance(res, TextResult)
    assert res.text == "hello"
    assert res.logprobs is None


def test_local_backend_retries_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_backend.time, "sleep", lambda _s: None)
    calls: List[int] = []
    ok_payload = _mk_chat_payload(text="recovered", logprobs=None)

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        calls.append(1)
        if len(calls) == 1:
            raise _mk_http_error(503, "Service Unavailable")
        return _mk_response(ok_payload)

    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        res = call_text_chat_local(
            prompt="x",
            model="local-model",
            base_url="http://localhost:1234/v1",
            retries=3,
            retry_backoff_s=0.0,
        )

    assert len(calls) == 2
    assert res.text == "recovered"


def test_local_backend_raises_after_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_backend.time, "sleep", lambda _s: None)
    calls: List[int] = []

    def fake_urlopen(req, timeout=None):  # noqa: ARG001
        calls.append(1)
        raise _mk_http_error(503, "Service Unavailable")

    with patch.object(local_backend.urllib.request, "urlopen", side_effect=fake_urlopen):
        with pytest.raises(RuntimeError):
            call_text_chat_local(
                prompt="x",
                model="local-model",
                base_url="http://localhost:1234/v1",
                retries=2,
                retry_backoff_s=0.0,
            )

    assert len(calls) == 3
