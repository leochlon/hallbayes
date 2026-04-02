from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .openai_backend import TextResult

_thread_local = threading.local()


def _normalize_base_url(base_url: Optional[str]) -> str:
    if base_url is None:
        base_url = (os.environ.get("VERTEX_BASE_URL") or "").strip() or None
    return (base_url or "https://aiplatform.googleapis.com").rstrip("/")


def _get_access_token(access_token: Optional[str]) -> str:
    if access_token is None:
        access_token = (os.environ.get("VERTEX_ACCESS_TOKEN") or "").strip() or None
    if not access_token:
        raise ValueError("VERTEX_ACCESS_TOKEN is not set (required for Vertex backend)")
    return str(access_token)


def _normalize_model(model: str) -> str:
    m = str(model or "").strip()
    if not m:
        raise ValueError("model is required")
    if m.startswith("projects/"):
        return m

    project = (os.environ.get("VERTEX_PROJECT") or "").strip()
    location = (os.environ.get("VERTEX_LOCATION") or "").strip()
    if project and location:
        return f"projects/{project}/locations/{location}/publishers/google/models/{m}"

    raise ValueError(
        "Vertex model must be a full resource name (projects/.../models/... or projects/.../endpoints/...), "
        "or set VERTEX_PROJECT and VERTEX_LOCATION."
    )


def _get_client(
    *, timeout_s: Optional[float] = None, base_url: Optional[str] = None
) -> Tuple[str, httpx.Client]:
    base = _normalize_base_url(base_url)
    key = (base, float(timeout_s) if timeout_s is not None else None)
    cache = getattr(_thread_local, "clients", None)
    if cache is None:
        cache = {}
        _thread_local.clients = cache
    if key in cache:
        return base, cache[key]

    timeout = float(timeout_s) if timeout_s is not None else 30.0
    client = httpx.Client(timeout=timeout)
    cache[key] = client
    return base, client


def _get_dict(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}


def _extract_text(candidate: Any) -> str:
    c = _get_dict(candidate)
    content = _get_dict(c.get("content"))
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    out: List[str] = []
    for p in parts:
        if not isinstance(p, dict):
            continue
        t = p.get("text")
        if t is None:
            continue
        out.append(str(t))
    return "".join(out)


def _to_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _get_lp_candidate(d: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    tok = d.get("token")
    if tok is None:
        tok = d.get("text")
    lp = d.get("logProbability")
    if lp is None:
        lp = d.get("log_probability")
    return ("" if tok is None else str(tok), _to_float(lp))


def _convert_logprobs(logprobs_result: Any) -> List[Dict[str, Any]]:
    lr = _get_dict(logprobs_result)
    chosen = lr.get("chosenCandidates")
    if chosen is None:
        chosen = lr.get("chosen_candidates")
    top = lr.get("topCandidates")
    if top is None:
        top = lr.get("top_candidates")

    if not isinstance(chosen, list):
        return []
    top_list = top if isinstance(top, list) else []

    out: List[Dict[str, Any]] = []
    for i, ch in enumerate(chosen):
        chd = _get_dict(ch)
        tok, lp = _get_lp_candidate(chd)
        row: Dict[str, Any] = {"token": tok, "logprob": lp}

        if i < len(top_list) and isinstance(top_list[i], dict):
            cands = top_list[i].get("candidates")
            if cands is None:
                cands = top_list[i].get("Candidates")
            if isinstance(cands, list) and cands:
                tops: List[Dict[str, Any]] = []
                for cand in cands:
                    cd = _get_dict(cand)
                    t_tok, t_lp = _get_lp_candidate(cd)
                    if t_tok == "" or t_lp is None:
                        continue
                    tops.append({"token": t_tok, "logprob": t_lp})
                if tops:
                    row["top_logprobs"] = tops

        out.append(row)

    return out


def call_text_chat_vertex(
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
    access_token: Optional[str] = None,
    **_kwargs: Any,
) -> TextResult:
    """Call Vertex AI's generateContent (Gemini models) and return OpenAI-style token logprobs."""
    if top_logprobs < 0 or top_logprobs > 20:
        raise ValueError("top_logprobs must be between 0 and 20")

    vertex_model = _normalize_model(model)
    token = _get_access_token(access_token)
    base, client = _get_client(timeout_s=timeout_s, base_url=base_url)

    generation_config: Dict[str, Any] = {
        "temperature": float(temperature),
        "maxOutputTokens": int(max_output_tokens),
    }
    if include_logprobs:
        generation_config["responseLogprobs"] = True
        generation_config["logprobs"] = max(1, int(top_logprobs))

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": str(prompt)}]}],
        "generationConfig": generation_config,
    }
    if instructions:
        payload["systemInstruction"] = {"parts": [{"text": str(instructions)}]}

    url = f"{base}/v1/{vertex_model}:generateContent"
    headers = {"Authorization": f"Bearer {token}"}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Vertex generateContent failed ({resp.status_code}): {resp.text}"
                )

            data = resp.json()
            candidates = data.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise RuntimeError("Vertex generateContent response missing candidates")

            first = candidates[0] if isinstance(candidates[0], dict) else {}
            out_text = _extract_text(first)

            out_logprobs = None
            if include_logprobs:
                lr = first.get("logprobsResult")
                if lr is None:
                    lr = first.get("logprobs_result")
                if lr is None:
                    raise RuntimeError(
                        "Vertex response missing logprobsResult (logprobs were requested)"
                    )
                out_logprobs = _convert_logprobs(lr)
                if not out_logprobs:
                    raise RuntimeError("Vertex response logprobsResult was empty/unparseable")

            return TextResult(text=str(out_text), response_id=None, logprobs=out_logprobs)

        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(float(retry_backoff_s) * (attempt + 1))

    raise RuntimeError(f"Vertex generateContent call failed after retries: {last_err}")
