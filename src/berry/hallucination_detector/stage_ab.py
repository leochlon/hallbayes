from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

_LABEL_ALIASES = {
    "YES": {"Y", "YES"},
    "NO": {"N", "NO"},
    "UNSURE": {"U", "UNSURE", "UNKNOWN"},
}
_LABEL_EDGE_RE = re.compile(r"^[^A-Za-z]*([A-Za-z]+)[^A-Za-z]*$")


def canonical_answer_label(token: Any) -> Optional[str]:
    """Map a verifier answer token to YES/NO/UNSURE.

    The grouped verifier prompt uses one-character labels (Y/N/U) so each
    claim-level answer is likely to occupy a single generated token. The legacy
    single-claim prompt still uses YES/NO/UNSURE. This helper accepts both forms
    and conservatively rejects decoded tokens that contain multiple alphabetic
    runs, because those cannot provide one independent logprob distribution per
    claim.
    """

    raw = str(token or "").strip()
    if not raw:
        return None
    m = _LABEL_EDGE_RE.match(raw)
    if not m:
        return None
    text = m.group(1).upper()
    for label, aliases in _LABEL_ALIASES.items():
        if text in aliases:
            return label
    return None


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
    """Top-K distribution at one generated answer-label token position."""

    pos: int
    generated_token: str
    generated_logprob: float
    topk_logprobs: Dict[str, float]
    kth_logprob: Optional[float]


def _token_topk_at(seq: Sequence[Any], pos: int) -> TokenTopK:
    tokinfo = seq[pos]
    gen_tok = _get_token(tokinfo)
    gen_lp = _get_logprob(tokinfo)
    if gen_lp is None:
        raise ValueError(f"missing logprob for generated token at position {pos}")

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

    return _token_topk_at(seq, pos)


def extract_label_topks(
    logprobs: Any,
    *,
    labels: Sequence[str],
    expected_count: Optional[int] = None,
    exact: bool = False,
) -> List[TokenTopK]:
    """Extract top-K distributions at generated label-token positions.

    Grouped verifier prompts emit one answer label per claim. The model may add
    benign separators such as whitespace, bullets, or numbering, so this scans
    the generated token stream and returns positions whose decoded token is one
    of ``labels`` after conservative normalization.

    With ``exact=True``, extra label-looking tokens are treated as an error.
    That fail-closed behavior is important: if a grouped answer is verbose or
    echoes labels from the prompt, callers should retry single-claim prompts
    rather than risk shifting labels across claims.
    """

    if logprobs is None:
        raise ValueError("logprobs is None; call the API with logprobs enabled")

    seq = list(logprobs)
    if not seq:
        raise ValueError("empty logprobs list")

    wanted: Set[str] = set()
    for label in labels:
        canonical = canonical_answer_label(label)
        if canonical is not None:
            wanted.add(canonical)
    if not wanted:
        raise ValueError("labels must contain at least one valid answer label")

    out: List[TokenTopK] = []
    for pos, tokinfo in enumerate(seq):
        canonical = canonical_answer_label(_get_token(tokinfo))
        if canonical not in wanted:
            continue
        out.append(_token_topk_at(seq, pos))

    if expected_count is not None:
        n = int(expected_count)
        if n < 0:
            raise ValueError("expected_count must be non-negative")
        if exact and len(out) != n:
            raise ValueError(f"expected exactly {n} label tokens, found {len(out)}")
        if len(out) < n:
            raise ValueError(f"expected at least {n} label tokens, found {len(out)}")
        return out[:n]

    return out


def extract_answer_topks(logprobs: Any, expected: int) -> List[TokenTopK]:
    """Extract one answer-label top-K distribution per claim in a grouped reply."""

    if int(expected) < 1:
        raise ValueError("expected must be positive")
    return extract_label_topks(
        logprobs,
        labels=["Y", "N", "U", "YES", "NO", "UNSURE"],
        expected_count=int(expected),
        exact=True,
    )
