from __future__ import annotations

import concurrent.futures as cf
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .gemini_backend import call_text_chat_gemini
from .openai_backend import TextResult, call_text_chat
from .vertex_backend import call_text_chat_vertex


@dataclass
class BackendConfig:
    kind: str = "openai"
    max_concurrency: int = 20
    timeout_s: Optional[float] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class OpenAIBackend:
    """Thin wrapper around openai backend with batch helpers (thread-pool parallelism)."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

    def call_text(self, **kwargs: Any) -> TextResult:
        return call_text_chat(
            **kwargs,
            timeout_s=self.cfg.timeout_s,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )

    def call_text_batch(self, *, prompts: Sequence[str], **kwargs: Any) -> List[TextResult]:
        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_text, prompt=p, **kwargs) for p in prompts]
            return [f.result() for f in futs]

    def reset_state(self) -> None:
        import gc

        gc.collect()


class VertexBackend:
    """Thin wrapper around Vertex backend with batch helpers (thread-pool parallelism)."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

    def call_text(self, **kwargs: Any) -> TextResult:
        return call_text_chat_vertex(
            **kwargs,
            timeout_s=self.cfg.timeout_s,
            base_url=self.cfg.base_url,
            access_token=self.cfg.api_key,
        )

    def call_text_batch(self, *, prompts: Sequence[str], **kwargs: Any) -> List[TextResult]:
        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_text, prompt=p, **kwargs) for p in prompts]
            return [f.result() for f in futs]

    def reset_state(self) -> None:
        import gc

        gc.collect()


class GeminiBackend:
    """Thin wrapper around Gemini backend with batch helpers (thread-pool parallelism)."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

    def call_text(self, **kwargs: Any) -> TextResult:
        return call_text_chat_gemini(
            **kwargs,
            timeout_s=self.cfg.timeout_s,
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )

    def call_text_batch(self, *, prompts: Sequence[str], **kwargs: Any) -> List[TextResult]:
        max_workers = max(1, int(self.cfg.max_concurrency))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(self.call_text, prompt=p, **kwargs) for p in prompts]
            return [f.result() for f in futs]

    def reset_state(self) -> None:
        import gc

        gc.collect()


def make_backend(cfg: BackendConfig):
    kind = (cfg.kind or "openai").lower().strip()
    if kind == "openai":
        return OpenAIBackend(cfg)
    if kind == "gemini":
        return GeminiBackend(cfg)
    if kind == "vertex":
        return VertexBackend(cfg)
    if kind == "dummy":
        return DummyBackend()
    raise ValueError(f"Unknown backend kind: {cfg.kind!r}")


class DummyBackend:
    """Deterministic backend for tests/offline development (returns stable logprobs)."""

    def __init__(self, p_yes: float = 0.55):
        self.p_yes = float(p_yes)

    def call_text(self, **_kwargs: Any) -> TextResult:
        p_yes = min(max(self.p_yes, 1e-6), 1.0 - 1e-6)
        p_other = (1.0 - p_yes) / 2.0
        logprobs = [
            {
                "token": "YES",
                "logprob": math.log(p_yes),
                "top_logprobs": [
                    {"token": "YES", "logprob": math.log(p_yes)},
                    {"token": "NO", "logprob": math.log(p_other)},
                    {"token": "UNSURE", "logprob": math.log(p_other)},
                ],
            }
        ]
        return TextResult(text="YES", response_id="dummy", logprobs=logprobs)

    def call_text_batch(self, *, prompts: Sequence[str], **kwargs: Any) -> List[TextResult]:
        return [self.call_text(prompt=p, **kwargs) for p in prompts]

    def reset_state(self) -> None:
        return None
