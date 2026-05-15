"""Local OpenAI-compatible backend for Berry.

Targets local model runtimes that expose /v1/chat/completions:
  - LM Studio (default http://127.0.0.1:1234/v1)
  - llama.cpp server
  - vLLM (OpenAI-compatible mode)

Why bypass the `openai` SDK:
  - Avoid SDK quirks (header coercion, retry surprises, base_url precedence).
  - Keep this backend dependency-light (stdlib only).
  - Tolerate runtimes that aren't 100% openai-spec-compatible (e.g. slightly
    different field shapes) by parsing the raw JSON ourselves.

Response is normalized into the same `TextResult` shape as openai_backend so
callers can swap backends without branching on field layout.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .openai_backend import TextResult

_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
_DEFAULT_API_KEY = "local"  # most local runtimes accept any non-empty value


def _resolve_base_url(base_url: Optional[str]) -> str:
    if base_url:
        return base_url.rstrip("/")
    env = (os.environ.get("BERRY_LOCAL_BASE_URL") or "").strip()
    return (env or _DEFAULT_BASE_URL).rstrip("/")


def _resolve_api_key(api_key: Optional[str]) -> str:
    if api_key:
        return api_key
    env = (os.environ.get("BERRY_LOCAL_API_KEY") or "").strip()
    return env or _DEFAULT_API_KEY


def _is_retryable(exc: Exception) -> bool:
    """Retry on connection errors and 5xx HTTP responses."""
    if isinstance(exc, urllib.error.HTTPError):
        return 500 <= exc.code < 600
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


def _normalize_logprobs(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Convert raw /v1/chat/completions logprobs.content into our list-of-dicts shape.

    Matches the structure produced by openai_backend.call_text_chat:
      [{"token": str, "logprob": float, "top_logprobs": [{"token", "logprob"}, ...]}]
    """
    if not raw:
        return None
    content = raw.get("content") if isinstance(raw, dict) else None
    if not content:
        return None
    out: List[Dict[str, Any]] = []
    for token_info in content:
        if not isinstance(token_info, dict):
            continue
        entry: Dict[str, Any] = {
            "token": token_info.get("token", ""),
            "logprob": token_info.get("logprob", 0.0),
        }
        tops = token_info.get("top_logprobs") or []
        if tops:
            entry["top_logprobs"] = [
                {"token": t.get("token", ""), "logprob": t.get("logprob", 0.0)}
                for t in tops if isinstance(t, dict)
            ]
        out.append(entry)
    return out


def call_text_chat_local(
    *,
    prompt: str,
    model: str,
    instructions: str = "You are a helpful assistant.",
    temperature: float = 0.0,
    max_output_tokens: int = 64,
    include_logprobs: bool = False,
    top_logprobs: int = 0,
    retries: int = 3,
    retry_backoff_s: float = 1.5,
    timeout_s: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **_kwargs: Any,
) -> TextResult:
    """Call a local OpenAI-compatible chat completions endpoint."""
    if top_logprobs < 0 or top_logprobs > 20:
        raise ValueError("top_logprobs must be between 0 and 20")

    url = f"{_resolve_base_url(base_url)}/chat/completions"
    key = _resolve_api_key(api_key)

    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": max(16, int(max_output_tokens)),
    }
    if include_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = int(top_logprobs)

    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"local backend returned no choices: {raw[:200]}")
            choice = choices[0]
            message = choice.get("message") or {}
            out_text = message.get("content") or ""
            out_logprobs = _normalize_logprobs(choice.get("logprobs")) if include_logprobs else None
            return TextResult(
                text=str(out_text),
                response_id=data.get("id"),
                logprobs=out_logprobs,
            )
        except Exception as e:
            last_err = e
            if attempt >= retries or not _is_retryable(e):
                break
            time.sleep(float(retry_backoff_s) * (attempt + 1))

    raise RuntimeError(f"Local chat call to {url} failed after retries: {last_err}")
