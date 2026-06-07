from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from openai import OpenAI
else:
    try:
        from openai import OpenAI
    except Exception:
        OpenAI = None


_thread_local = threading.local()


def _get_client(
    *,
    timeout_s: Optional[float] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    if OpenAI is None:
        raise ImportError("openai package not installed. Run: pip install openai")

    if base_url is None:
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    if api_key is None:
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set (required for OpenAI backend)")

    key = (timeout_s, base_url, api_key)
    cache = getattr(_thread_local, "clients", None)
    if cache is None:
        cache = {}
        _thread_local.clients = cache
    if key in cache:
        return cache[key]

    kwargs: Dict[str, Any] = {}
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key

    client = OpenAI(**kwargs) if kwargs else OpenAI()
    cache[key] = client
    return client


@dataclass
class TextResult:
    text: str
    response_id: Optional[str] = None
    logprobs: Optional[Any] = None


def call_text_chat(
    *,
    prompt: str,
    model: str = "gpt-4o-mini",
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
    **_kwargs: Any,  # ignore extra args like 'reasoning'
) -> TextResult:
    """Call an OpenAI model using the Chat Completions API (supports logprobs)."""
    client = _get_client(timeout_s=timeout_s, base_url=base_url, api_key=api_key)

    if top_logprobs < 0 or top_logprobs > 20:
        raise ValueError("top_logprobs must be between 0 and 20")

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            api_kwargs: Dict[str, Any] = {}
            if include_logprobs:
                api_kwargs["logprobs"] = True
                api_kwargs["top_logprobs"] = int(top_logprobs)

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(temperature),
                max_tokens=int(max_output_tokens),
                **api_kwargs,
            )

            choice = resp.choices[0]
            out_text = choice.message.content or ""

            out_logprobs = None
            if (
                include_logprobs
                and getattr(choice, "logprobs", None)
                and getattr(choice.logprobs, "content", None)
            ):
                out_logprobs = []
                for token_info in choice.logprobs.content:
                    token_data = {
                        "token": token_info.token,
                        "logprob": token_info.logprob,
                    }
                    if token_info.top_logprobs:
                        token_data["top_logprobs"] = [
                            {"token": t.token, "logprob": t.logprob}
                            for t in token_info.top_logprobs
                        ]
                    out_logprobs.append(token_data)

            return TextResult(
                text=str(out_text), response_id=getattr(resp, "id", None), logprobs=out_logprobs
            )

        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            time.sleep(float(retry_backoff_s) * (attempt + 1))

    raise RuntimeError(f"OpenAI chat call failed after retries: {last_err}")
