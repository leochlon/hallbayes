from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .backends.base import BackendConfig
from .trace_budget import build_trace_budget_prompts, score_trace_budget

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_CITE_RE = re.compile(r"\[(?P<id>[A-Za-z]\w*|\d+)\]")

_LN2 = math.log(2.0)

_NO_LOGPROBS_MARKERS = (
    "logprobs is None",
    "empty logprobs list",
    "missing logprob for generated token",
)


def _no_logprobs_verdict() -> Dict[str, Any]:
    return {
        "flagged": True,
        "under_budget": True,
        "error": (
            "Backend did not return logprobs - Berry requires top_logprobs "
            "support. Check backend."
        ),
        "error_type": "no_logprobs_from_backend",
        "details": [],
    }


def _is_no_logprobs_error(exc: BaseException) -> bool:
    msg = str(exc) or ""
    return any(marker in msg for marker in _NO_LOGPROBS_MARKERS)


@dataclass
class Span:
    sid: str
    text: str


@dataclass
class Step:
    idx: int
    claim: str
    cites: List[str]
    confidence: float


@dataclass
class Trace:
    steps: List[Step]
    spans: List[Span]


def _to_bits(nats: float) -> float:
    return float(nats) / _LN2


_SPAN_TEXT_FALLBACK_KEYS = ("text", "snippet", "content", "body", "passage")


def _normalize_spans(spans: List[Dict[str, str]]) -> List[Span]:
    """Coerce loose MCP span dicts into Span objects.

    - Auto-assigns ``sid`` as ``s{i+1}`` when caller omitted it (1-indexed).
    - Accepts text under fallback keys: text, snippet, content, body, passage.
    - Silently skips entries with no usable text; raises ``ValueError`` with a
      concrete diagnostic if a non-empty input list yields zero valid spans
      (so the caller can distinguish malformed input from empty input).
    """
    out: List[Span] = []
    skipped: List[str] = []
    raw = spans or []
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            skipped.append(f"index {i}: not a dict (got {type(s).__name__})")
            continue
        sid = str(s.get("sid", "") or "").strip()
        text = ""
        for k in _SPAN_TEXT_FALLBACK_KEYS:
            v = s.get(k)
            if v is None:
                continue
            cand = str(v).strip()
            if cand:
                text = cand
                break
        if not text:
            present = sorted(k for k in s.keys() if k != "sid")
            skipped.append(f"index {i}: no usable text (keys present: {present or 'none'}; expected one of {list(_SPAN_TEXT_FALLBACK_KEYS)})")
            continue
        if not sid:
            sid = f"s{i+1}"
        out.append(Span(sid=sid, text=text))
    if raw and not out:
        raise ValueError("All spans malformed: " + "; ".join(skipped))
    return out


def _normalize_steps(steps: List[Dict[str, Any]], default_target: float) -> List[Step]:
    out: List[Step] = []
    for i, st in enumerate(steps or []):
        claim = str(st.get("claim", "")).strip()
        if not claim:
            continue
        idx = int(st.get("idx", i))
        cites = [str(c).strip() for c in (st.get("cites") or []) if str(c).strip()]
        conf = float(st.get("confidence", default_target) or default_target)
        out.append(Step(idx=idx, claim=claim, cites=cites, confidence=conf))
    out.sort(key=lambda x: x.idx)
    return out


def _extract_cites(text: str, cite_re: re.Pattern) -> List[str]:
    return [m.group("id") for m in cite_re.finditer(text or "")]


def _split_claims(answer: str, mode: str, max_claims: int) -> List[str]:
    a = (answer or "").strip()
    if not a:
        return []

    if mode == "lines":
        raw = [ln.strip() for ln in a.splitlines() if ln.strip()]
    else:
        raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(a) if s.strip()]

    cite_prefix_re = re.compile(r"^\s*(?:\[(?:[A-Za-z]\w*|\d+)\]\s*)+")

    merged: List[str] = []
    for seg in raw:
        # If a segment begins with citations and then continues with content,
        # move the citation prefix back onto the previous claim.
        if merged:
            m = cite_prefix_re.match(seg)
            if m:
                prefix = seg[: m.end()].strip()
                rest = seg[m.end() :].strip()
                if prefix and rest:
                    merged[-1] = (merged[-1] + " " + prefix).strip()
                    seg = rest

        remainder = _DEFAULT_CITE_RE.sub("", seg)
        remainder = re.sub(r"[\\s,;:.\\-–—!?]+", "", remainder)
        cites_only = (remainder == "") and bool(_DEFAULT_CITE_RE.search(seg))
        if cites_only and merged:
            merged[-1] = (merged[-1] + " " + seg).strip()
        else:
            merged.append(seg)

    return merged[: max(1, int(max_claims))]


def _map_cites_to_known_ids(cites: List[str], known: set) -> List[str]:
    mapped: List[str] = []
    for c in cites:
        if c in known:
            mapped.append(c)
            continue

        if c.isdigit():
            n = int(c)
            if f"S{n}" in known:
                mapped.append(f"S{n}")
                continue
            if n > 0 and f"S{n - 1}" in known:
                mapped.append(f"S{n - 1}")
                continue

        if c.startswith("S") and c[1:].isdigit():
            tail = c[1:]
            if tail in known:
                mapped.append(tail)
                continue

        mapped.append(c)

    seen = set()
    out = []
    for c in mapped:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _format_result(r, units: str) -> Dict[str, Any]:
    if units == "bits":
        req_min = _to_bits(r.required_bits_min)
        req_max = _to_bits(r.required_bits_max)
        obs_min = _to_bits(r.observed_bits_min)
        obs_max = _to_bits(r.observed_bits_max)
        gap_min = _to_bits(r.budget_gap_min)
        gap_max = _to_bits(r.budget_gap_max)
    else:
        req_min, req_max = r.required_bits_min, r.required_bits_max
        obs_min, obs_max = r.observed_bits_min, r.observed_bits_max
        gap_min, gap_max = r.budget_gap_min, r.budget_gap_max

    return {
        "idx": r.idx,
        "claim": r.claim,
        "cites": r.cites,
        "target": r.target,
        "prior_yes": {
            "p_lower": r.prior_yes.p_yes_lower,
            "p_upper": r.prior_yes.p_yes_upper,
            "generated": r.prior_yes.generated,
            "topk": r.prior_yes.topk,
        },
        "post_yes": {
            "p_lower": r.post_yes.p_yes_lower,
            "p_upper": r.post_yes.p_yes_upper,
            "generated": r.post_yes.generated,
            "topk": r.post_yes.topk,
        },
        "required": {"min": req_min, "max": req_max, "units": units},
        "observed": {"min": obs_min, "max": obs_max, "units": units},
        "budget_gap": {"min": gap_min, "max": gap_max, "units": units},
        "flagged": bool(r.flagged),
        "has_any_citations": bool(r.cites),
        "missing_citations": False,
    }


def run_detect_hallucination(
    *,
    answer: str,
    spans: List[Dict[str, str]],
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,
    placeholder: str = "[REDACTED]",
    max_claims: int = 25,
    claim_split: str = "sentences",
    citation_regex: Optional[str] = None,
    temperature: float = 0.0,
    top_logprobs: int = 10,
    max_concurrency: int = 8,
    timeout_s: Optional[float] = 30.0,
    units: str = "bits",
    context_mode: str = "all",
    require_citations: bool = False,
    include_prompts: bool = False,
    max_prompt_chars: int = 3000,
    pool_json_path: Optional[str] = None,
    local_llm_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        span_objs = _normalize_spans(spans)
    except ValueError as e:
        return {
            "flagged": True,
            "under_budget": True,
            "error": str(e),
            "error_type": "malformed_spans",
            "details": [],
        }
    if not span_objs:
        return {
            "flagged": True,
            "under_budget": True,
            "error": "No spans provided (cannot verify citations).",
            "error_type": "no_spans",
            "details": [],
        }

    if pool_json_path or local_llm_model_path:
        return {
            "flagged": True,
            "under_budget": True,
            "error": "Only the OpenAI backend is supported in Berry right now (AOAI/local LLM not implemented).",
            "details": [],
        }

    cite_re = re.compile(citation_regex) if citation_regex else _DEFAULT_CITE_RE
    known_ids = {s.sid for s in span_objs}
    claims = _split_claims(answer, mode=claim_split, max_claims=max_claims)

    steps: List[Step] = []
    for i, cl in enumerate(claims):
        cites = _extract_cites(cl, cite_re=cite_re)
        cites = _map_cites_to_known_ids(cites, known=known_ids)
        steps.append(Step(idx=i, claim=cl, cites=cites, confidence=float(default_target)))

    trace = Trace(steps=steps, spans=span_objs)

    prompts = None
    if include_prompts:
        prompts = build_trace_budget_prompts(
            trace=trace,
            placeholder=str(placeholder),
            context_mode=str(context_mode),
        )

    backend_kind = (os.environ.get("BERRY_VERIFIER_BACKEND") or "openai").strip().lower()
    cfg = BackendConfig(
        kind=backend_kind, max_concurrency=int(max_concurrency), timeout_s=timeout_s
    )
    try:
        results = score_trace_budget(
            trace=trace,
            verifier_model=verifier_model,
            backend_cfg=cfg,
            default_target=float(default_target),
            temperature=float(temperature),
            top_logprobs=int(top_logprobs),
            placeholder=str(placeholder),
            context_mode=str(context_mode),
            reasoning=None,
        )
    except ValueError as e:
        if _is_no_logprobs_error(e):
            return _no_logprobs_verdict()
        raise

    details = [_format_result(r, units) for r in results]

    if require_citations:
        for d in details:
            if not bool(d.get("has_any_citations")):
                d["missing_citations"] = True
                d["flagged"] = True

    if prompts:
        cap = max(100, int(max_prompt_chars or 3000))
        for d, pp in zip(details, prompts):
            prior = str(pp.get("prior_prompt") or "")
            post = str(pp.get("post_prompt") or "")
            if len(prior) > cap:
                prior = prior[:cap] + "\n...[TRUNCATED prior_prompt]"
            if len(post) > cap:
                post = post[:cap] + "\n...[TRUNCATED post_prompt]"
            d["prior_prompt"] = prior
            d["post_prompt"] = post

    flagged = any(d["flagged"] for d in details)
    flagged_idxs = [d["idx"] for d in details if d["flagged"]]

    return {
        "flagged": flagged,
        "under_budget": flagged,
        "summary": {
            "claims_scored": len(details),
            "flagged_claims": len(flagged_idxs),
            "flagged_idxs": flagged_idxs[:50],
            "units": units,
            "verifier_model": verifier_model,
            "backend": backend_kind,
        },
        "details": details,
    }


def run_audit_trace_budget(
    *,
    steps: List[Dict[str, Any]],
    spans: List[Dict[str, str]],
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,
    placeholder: str = "[REDACTED]",
    temperature: float = 0.0,
    top_logprobs: int = 10,
    max_concurrency: int = 8,
    timeout_s: Optional[float] = 30.0,
    units: str = "bits",
    context_mode: str = "all",
    require_citations: bool = False,
    include_prompts: bool = False,
    max_prompt_chars: int = 3000,
    pool_json_path: Optional[str] = None,
    local_llm_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    span_objs = _normalize_spans(spans)
    step_objs = _normalize_steps(steps, default_target=float(default_target))
    trace = Trace(steps=step_objs, spans=span_objs)

    prompts = None
    if include_prompts:
        prompts = build_trace_budget_prompts(
            trace=trace,
            placeholder=str(placeholder),
            context_mode=str(context_mode),
        )

    if pool_json_path or local_llm_model_path:
        return {
            "flagged": True,
            "under_budget": True,
            "error": "Only the OpenAI backend is supported in Berry right now (AOAI/local LLM not implemented).",
            "details": [],
        }

    backend_kind = (os.environ.get("BERRY_VERIFIER_BACKEND") or "openai").strip().lower()
    cfg = BackendConfig(
        kind=backend_kind, max_concurrency=int(max_concurrency), timeout_s=timeout_s
    )
    try:
        results = score_trace_budget(
            trace=trace,
            verifier_model=verifier_model,
            backend_cfg=cfg,
            default_target=float(default_target),
            temperature=float(temperature),
            top_logprobs=int(top_logprobs),
            placeholder=str(placeholder),
            context_mode=str(context_mode),
            reasoning=None,
        )
    except ValueError as e:
        if _is_no_logprobs_error(e):
            return _no_logprobs_verdict()
        raise

    out = []
    for r in results:
        if units == "bits":
            out.append(
                {
                    "idx": r.idx,
                    "claim": r.claim,
                    "cites": r.cites,
                    "flagged": bool(r.flagged),
                    "required": {
                        "min": _to_bits(r.required_bits_min),
                        "max": _to_bits(r.required_bits_max),
                        "units": "bits",
                    },
                    "observed": {
                        "min": _to_bits(r.observed_bits_min),
                        "max": _to_bits(r.observed_bits_max),
                        "units": "bits",
                    },
                    "budget_gap": {
                        "min": _to_bits(r.budget_gap_min),
                        "max": _to_bits(r.budget_gap_max),
                        "units": "bits",
                    },
                }
            )
        else:
            out.append(
                {
                    "idx": r.idx,
                    "claim": r.claim,
                    "cites": r.cites,
                    "flagged": bool(r.flagged),
                    "required": {
                        "min": r.required_bits_min,
                        "max": r.required_bits_max,
                        "units": "nats",
                    },
                    "observed": {
                        "min": r.observed_bits_min,
                        "max": r.observed_bits_max,
                        "units": "nats",
                    },
                    "budget_gap": {
                        "min": r.budget_gap_min,
                        "max": r.budget_gap_max,
                        "units": "nats",
                    },
                }
            )

    if require_citations:
        for x in out:
            if not (x.get("cites") or []):
                x["missing_citations"] = True
                x["flagged"] = True

    if prompts:
        cap = max(100, int(max_prompt_chars or 3000))
        for x, pp in zip(out, prompts):
            prior = str(pp.get("prior_prompt") or "")
            post = str(pp.get("post_prompt") or "")
            if len(prior) > cap:
                prior = prior[:cap] + "\n...[TRUNCATED prior_prompt]"
            if len(post) > cap:
                post = post[:cap] + "\n...[TRUNCATED post_prompt]"
            x["prior_prompt"] = prior
            x["post_prompt"] = post

    flagged = any(x["flagged"] for x in out)
    return {
        "flagged": flagged,
        "under_budget": flagged,
        "summary": {
            "steps_scored": len(out),
            "flagged_steps": sum(1 for x in out if x["flagged"]),
            "units": units,
            "verifier_model": verifier_model,
            "backend": backend_kind,
        },
        "details": out,
    }
