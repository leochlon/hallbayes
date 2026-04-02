from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    out: Dict[str, Any] = {}
    for k in dir(x):
        if k.startswith("_"):
            continue
        try:
            v = getattr(x, k)
        except Exception:
            continue
        if callable(v):
            continue
        out[k] = v
    return out


def _get_token(obj: Any) -> str:
    d = _as_dict(obj)
    tok = d.get("token")
    if tok is None:
        tok = d.get("text")
    return "" if tok is None else str(tok)


def _get_logprob(obj: Any) -> Optional[float]:
    d = _as_dict(obj)
    lp = d.get("logprob")
    if lp is None:
        lp = d.get("log_prob")
    try:
        return None if lp is None else float(lp)
    except Exception:
        return None


def _get_top_logprobs(obj: Any) -> List[Any]:
    d = _as_dict(obj)
    top = d.get("top_logprobs")
    if top is None:
        top = d.get("top_log_probs")
    if top is None:
        return []
    return list(top)


@dataclass
class TokenTopK:
    """Top-K distribution at the answer-start token position."""

    pos: int
    generated_token: str
    generated_logprob: float
    topk_logprobs: Dict[str, float]
    kth_logprob: Optional[float]


def extract_answer_topk(logprobs: Any) -> TokenTopK:
    """Extract a top-K distribution for the first non-whitespace output token."""
    if logprobs is None:
        raise ValueError("logprobs is None; call the API with logprobs enabled")

    seq = list(logprobs)
    if not seq:
        raise ValueError("empty logprobs list")

    pos = 0
    for i, tokinfo in enumerate(seq):
        tok = _get_token(tokinfo)
        if tok.strip() != "":
            pos = i
            break

    tokinfo = seq[pos]
    gen_tok = _get_token(tokinfo)
    gen_lp = _get_logprob(tokinfo)
    if gen_lp is None:
        raise ValueError("missing logprob for generated token")

    top_list = _get_top_logprobs(tokinfo)
    topk: Dict[str, float] = {}
    for t in top_list:
        tt = _get_token(t)
        lp = _get_logprob(t)
        if lp is None:
            continue
        key = tt.lstrip()
        if key == "":
            continue
        topk[key] = max(topk.get(key, -math.inf), float(lp))

    kth = None
    if top_list:
        lps = [lp for lp in ([_get_logprob(t) for t in top_list]) if lp is not None]
        kth = min(lps) if lps else None

    return TokenTopK(
        pos=int(pos),
        generated_token=str(gen_tok),
        generated_logprob=float(gen_lp),
        topk_logprobs=topk,
        kth_logprob=kth,
    )
