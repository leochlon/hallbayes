from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backends.base import BackendConfig, make_backend
from .stage_ab import extract_answer_topk

logger = logging.getLogger(__name__)


def _trace_debug_enabled() -> bool:
    return os.environ.get("BERRY_TRACE_BUDGET_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


@dataclass
class YesProb:
    p_yes_lower: float
    p_yes_upper: float
    generated: str
    generated_logprob: float
    kth_logprob: Optional[float]
    topk: Dict[str, float]


def _safe_clip(p: float, eps: float = 1e-12) -> float:
    return min(max(float(p), eps), 1.0 - eps)


def kl_bernoulli(a: float, b: float) -> float:
    a = _safe_clip(a)
    b = _safe_clip(b)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


_WH_Q_RE = re.compile(r"^(who|what|which|when|where|why|how)\b", re.IGNORECASE)
_AUX_Q_RE = re.compile(
    r"^(is|are|was|were|do|does|did|can|could|should|would|will|may|might|have|has|had)\b",
    re.IGNORECASE,
)


def _span_kind(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    if not t:
        return "EMPTY"

    lo = t.lower()
    if lo.startswith(("question:", "q:", "prompt:", "task:")):
        return "QUESTION"
    if lo.startswith(
        (
            "reply with",
            "respond with",
            "choose",
            "select",
            "options:",
            "answers:",
            "a)",
            "b)",
            "c)",
            "d)",
            "(a)",
            "(b)",
            "(c)",
            "(d)",
        )
    ):
        return "INSTRUCTION"

    if t.endswith("?"):
        return "QUESTION"
    if _WH_Q_RE.match(lo):
        return "QUESTION"
    if _AUX_Q_RE.match(lo) and ("?" in t or t.count(" ") < 12):
        return "QUESTION"

    return "ASSERTION"


def _spans_block(spans: Sequence[Any], *, mask_nonassertive: bool = True) -> str:
    lines: List[str] = []
    for s in spans:
        sid = getattr(s, "sid")
        text = str(getattr(s, "text"))
        kind = _span_kind(text)
        if mask_nonassertive and kind in {"QUESTION", "INSTRUCTION", "EMPTY"}:
            shown = f"[NON-EVIDENCE:{kind}]"
        else:
            shown = text
        lines.append(f"[{sid}] {shown}")
    return "\n".join(lines).strip()


def scrub_spans_by_id(
    spans: Sequence[Any], cites: Sequence[str], *, placeholder: str = "[REDACTED]"
) -> List[Any]:
    cset = {str(c) for c in cites}
    out: List[Any] = []
    for s in spans:
        sid = str(getattr(s, "sid"))
        text = str(getattr(s, "text"))
        if sid in cset:
            try:
                out.append(type(s)(sid=sid, text=placeholder))
            except Exception:
                out.append({"sid": sid, "text": placeholder})
        else:
            try:
                out.append(type(s)(sid=sid, text=text))
            except Exception:
                out.append({"sid": sid, "text": text})
    return out


def _select_context_spans(*, spans: Sequence[Any], cites: Sequence[str], mode: str) -> List[Any]:
    m = (mode or "all").strip().lower()
    if m in {"all", "full"}:
        return list(spans)
    if m == "auto":
        if cites:
            cset = {str(c) for c in cites}
            return [s for s in spans if str(getattr(s, "sid")) in cset]
        return list(spans)
    if m in {"cited", "cite", "citations", "cites"}:
        cset = {str(c) for c in cites}
        return [s for s in spans if str(getattr(s, "sid")) in cset]
    raise ValueError(f"Unknown context_mode: {mode!r}")


def build_yes_prompt(*, spans: Sequence[Any], claim: str) -> str:
    ctx = _spans_block(spans, mask_nonassertive=True)
    return f"""
You are a **strict textual entailment** verifier.

Definitions:
- Only **declarative assertions** in the CONTEXT can entail facts.
- **Questions, prompts, headings, and instructions do NOT assert facts** and do NOT entail their presuppositions.
- Do **not** use world knowledge or plausibility; judge only whether the CLAIM follows from the asserted text.

Decision rule:
- Reply YES only if the CLAIM is explicitly stated or logically implied by at least one ASSERTION span.
- Reply NO only if the CONTEXT explicitly contradicts the CLAIM.
- Otherwise reply UNSURE (including when the CONTEXT contains only questions/instructions).

CONTEXT SPANS:
{ctx}

CLAIM:
{claim.strip()}

Question: Is the CLAIM entailed by the CONTEXT?

Reply with EXACTLY one of these tokens (no punctuation, no extra text):
YES
NO
UNSURE
""".strip()


def build_trace_budget_prompts(
    *,
    trace: Any,
    placeholder: str = "[REDACTED]",
    context_mode: str = "all",
) -> List[Dict[str, str]]:
    """Reconstruct the exact prior/post prompts used for score_trace_budget.

    This is intended for diagnostics (e.g. comparing verifier backends).
    """

    steps = list(getattr(trace, "steps", []) or [])
    spans = list(getattr(trace, "spans", []) or [])
    out: List[Dict[str, str]] = []
    if not steps:
        return out

    for st in steps:
        claim = str(getattr(st, "claim"))
        cites = [str(c) for c in getattr(st, "cites", [])]
        ctx_spans = _select_context_spans(spans=spans, cites=cites, mode=context_mode)
        post_prompt = build_yes_prompt(spans=ctx_spans, claim=claim)

        if cites:
            null_spans = scrub_spans_by_id(ctx_spans, cites, placeholder=placeholder)
        else:
            null_spans = scrub_spans_by_id(
                ctx_spans, [getattr(s, "sid") for s in ctx_spans], placeholder=placeholder
            )

        prior_prompt = build_yes_prompt(spans=null_spans, claim=claim)
        out.append({"post_prompt": post_prompt, "prior_prompt": prior_prompt})

    return out


def yesprob_from_logprobs(logprobs: Any) -> YesProb:
    topk = extract_answer_topk(logprobs)
    yes_lps = [
        float(lp)
        for tok, lp in (topk.topk_logprobs or {}).items()
        if str(tok).strip().upper() == "YES"
    ]

    if yes_lps:
        p = sum(math.exp(lp) for lp in yes_lps)
        p = min(max(float(p), 0.0), 1.0)
        p_lower = p
        p_upper = p
    else:
        p_lower = 0.0
        if topk.kth_logprob is not None and math.isfinite(float(topk.kth_logprob)):
            p_upper = math.exp(float(topk.kth_logprob))
        else:
            p_upper = 1.0

    return YesProb(
        p_yes_lower=float(p_lower),
        p_yes_upper=float(p_upper),
        generated=str(topk.generated_token),
        generated_logprob=float(topk.generated_logprob),
        kth_logprob=float(topk.kth_logprob) if topk.kth_logprob is not None else None,
        topk=dict(topk.topk_logprobs),
    )


@dataclass
class BudgetResult:
    idx: int
    claim: str
    cites: List[str]
    target: float
    prior_yes: YesProb
    post_yes: YesProb
    required_bits_min: float
    required_bits_max: float
    observed_bits_min: float
    observed_bits_max: float
    budget_gap_min: float
    budget_gap_max: float
    flagged: bool


def _budget_from_intervals(
    *,
    target: float,
    p0_lo: float,
    p0_hi: float,
    p1_lo: float,
    p1_hi: float,
) -> Tuple[float, float, float, float, float, float, bool]:
    req_min = kl_bernoulli(target, p0_hi)
    req_max = kl_bernoulli(target, max(p0_lo, 1e-12))

    corners = [(p1_lo, p0_lo), (p1_lo, p0_hi), (p1_hi, p0_lo), (p1_hi, p0_hi)]
    obs_vals = [kl_bernoulli(p1, p0) for (p1, p0) in corners]
    obs_min, obs_max = float(min(obs_vals)), float(max(obs_vals))

    gap_min = req_min - obs_max
    gap_max = req_max - obs_min
    flagged = req_min > obs_max
    return req_min, req_max, obs_min, obs_max, gap_min, gap_max, flagged


def score_trace_budget(
    *,
    trace: Any,
    verifier_model: str,
    backend_cfg: Optional[BackendConfig] = None,
    default_target: float = 0.95,
    temperature: float = 0.0,
    top_logprobs: int = 10,
    placeholder: str = "[REDACTED]",
    context_mode: str = "all",
    reasoning: Optional[Dict[str, Any]] = None,
) -> List[BudgetResult]:
    steps = list(getattr(trace, "steps", []))
    spans = list(getattr(trace, "spans", []))
    if not steps:
        return []

    cfg = backend_cfg or BackendConfig(kind="openai")
    backend = make_backend(cfg)

    post_prompts: List[str] = []
    prior_prompts: List[str] = []
    targets: List[float] = []
    cites_list: List[List[str]] = []
    idxs: List[int] = []
    claims: List[str] = []

    if _trace_debug_enabled():
        logger.debug("DEBUG [trace_budget]: %s claims to verify", len(steps))
        logger.debug(
            "DEBUG [trace_budget]: %s total spans, context_mode=%r", len(spans), context_mode
        )

    for st in steps:
        idx = int(getattr(st, "idx"))
        claim = str(getattr(st, "claim"))
        cites = [str(c) for c in getattr(st, "cites", [])]
        target = float(getattr(st, "confidence", default_target) or default_target)
        idxs.append(idx)
        claims.append(claim)
        cites_list.append(cites)
        targets.append(target)

        ctx_spans = _select_context_spans(spans=spans, cites=cites, mode=context_mode)
        post_prompts.append(build_yes_prompt(spans=ctx_spans, claim=claim))

        if cites:
            null_spans = scrub_spans_by_id(ctx_spans, cites, placeholder=placeholder)
        else:
            null_spans = scrub_spans_by_id(
                ctx_spans, [getattr(s, "sid") for s in ctx_spans], placeholder=placeholder
            )

        prior_prompts.append(build_yes_prompt(spans=null_spans, claim=claim))

    post_results = backend.call_text_batch(
        prompts=post_prompts,
        model=verifier_model,
        instructions="Reply with exactly one token: YES, NO, or UNSURE.",
        temperature=temperature,
        max_output_tokens=5,
        include_logprobs=True,
        top_logprobs=int(top_logprobs),
        reasoning=reasoning,
    )

    if hasattr(backend, "reset_state"):
        backend.reset_state()

    prior_results = backend.call_text_batch(
        prompts=prior_prompts,
        model=verifier_model,
        instructions="Reply with exactly one token: YES, NO, or UNSURE.",
        temperature=temperature,
        max_output_tokens=5,
        include_logprobs=True,
        top_logprobs=int(top_logprobs),
        reasoning=reasoning,
    )

    out: List[BudgetResult] = []
    for idx, claim, cites, target, post_tr, prior_tr in zip(
        idxs, claims, cites_list, targets, post_results, prior_results
    ):
        post_yes = yesprob_from_logprobs(post_tr.logprobs)
        prior_yes = yesprob_from_logprobs(prior_tr.logprobs)

        req_min, req_max, obs_min, obs_max, gap_min, gap_max, flagged = _budget_from_intervals(
            target=target,
            p0_lo=prior_yes.p_yes_lower,
            p0_hi=prior_yes.p_yes_upper,
            p1_lo=post_yes.p_yes_lower,
            p1_hi=post_yes.p_yes_upper,
        )

        out.append(
            BudgetResult(
                idx=int(idx),
                claim=str(claim),
                cites=list(cites),
                target=float(target),
                prior_yes=prior_yes,
                post_yes=post_yes,
                required_bits_min=float(req_min),
                required_bits_max=float(req_max),
                observed_bits_min=float(obs_min),
                observed_bits_max=float(obs_max),
                budget_gap_min=float(gap_min),
                budget_gap_max=float(gap_max),
                flagged=bool(flagged),
            )
        )

    return out
