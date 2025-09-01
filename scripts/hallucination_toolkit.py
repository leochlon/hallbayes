
"""
Hallucination Risk Calculator (OpenAI-only) — Closed-Book Ready
================================================================

This update adds a robust **closed-book mode** so you can assess hallucination risk
on *any* input prompt (variable length, no evidence). It preserves the EDFL/B2T/ISR
framework while replacing the "evidence-erase" skeletons with **semantic masking**
skeletons that (a) remove *specific content hints* (entities, numbers, quoted titles)
and (b) keep task structure intact. This avoids collapsing the conservative prior q_lo
to ~0 for ordinary questions, enabling useful Δ̄ and bounded RoH without user-supplied
evidence.

Key additions
-------------
- `skeleton_policy`: "auto" | "evidence_erase" | "closed_book"
  * auto: uses evidence erasure if evidence fields are present or configured;
          otherwise uses closed-book masking.
- Closed-book skeletons: progressive **entity/number masking** (regex, seeded).
- Decision head updated for closed-book: “answer if you know from pretrained knowledge,
  otherwise refuse.” (No hard evidence requirement.)
- `q_floor`: optional prior floor to avoid pathological q_lo→0 in closed-book;
  defaults to Laplace floor 1/(n_samples+2) (can be increased).
- Higher default `B_clip=12.0` and recommended `clip_mode="one-sided"` to reflect
  larger Δ̄ needed when q_lo is small but nonzero.
- Backwards compatible with evidence-providing workflows.

All info measures remain in **nats**. OpenAI API only.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------------
# OpenAI Backend
# ------------------------------------------------------------------------------------

try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    OpenAI = None


class OpenAIBackend:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        request_timeout: float = 60.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError("Install `openai>=1.0.0` and set OPENAI_API_KEY.")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.request_timeout = float(request_timeout)

    def chat_create(self, messages: List[Dict], **kwargs):
        params = dict(model=self.model, messages=messages, max_tokens=8, temperature=0.7)
        params.update(kwargs)
        if "timeout" in params:
            params["request_timeout"] = params.pop("timeout")
        return self.client.chat.completions.create(**params)

    def multi_choice(self, messages: List[Dict], n: int = 1, **kwargs):
        try:
            resp = self.chat_create(messages, n=n, **kwargs)
            choices = getattr(resp, "choices", None) or []
            if len(choices) == n:
                return choices
        except Exception:
            pass
        choices = []
        for _ in range(n):
            resp = self.chat_create(messages, **kwargs)
            choices.append(resp.choices[0])
        return choices


# ------------------------------------------------------------------------------------
# Core math (EDFL etc., nats)
# ------------------------------------------------------------------------------------

EPS = 1e-12


def _clamp01(x: float, eps: float = EPS) -> float:
    return min(1.0 - eps, max(eps, x))


def _safe_log(x: float) -> float:
    return math.log(max(x, EPS))


def harmonic_number(n: int) -> float:
    if n < 1:
        return 0.0
    if n < 100000:
        return sum(1.0 / r for r in range(1, n + 1))
    gamma = 0.5772156649015328606
    return math.log(n) + gamma + 1.0/(2*n) - 1.0/(12*n*n)


def expected_harmonic_distance(n: int) -> float:
    return harmonic_number(n) - 1.5


def martingale_violation_bound(n: int, L: float, C: float, alpha: float = 1.0) -> float:
    if n <= 1:
        return 0.0
    if alpha == 1.0:
        return (L*C/8.0) * (expected_harmonic_distance(n))
    elif 0.0 < alpha < 1.0:
        return (L*C/8.0) * (n**(1.0 - alpha)) / (1.0 - alpha)
    else:
        return (L*C/8.0)


def kl_bernoulli(p: float, q: float) -> float:
    p = _clamp01(p); q = _clamp01(q)
    return p*(_safe_log(p) - _safe_log(q)) + (1.0-p)*(_safe_log(1.0-p) - _safe_log(1.0-q))


def inv_kl_bernoulli_upper(q: float, delta: float, tol: float = 1e-12, max_iter: int = 10000) -> float:
    q = _clamp01(q); delta = max(0.0, float(delta))
    hi = 1.0 - EPS
    if kl_bernoulli(hi, q) <= delta:
        return hi
    lo = q
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        if kl_bernoulli(mid, q) <= delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def inv_kl_bernoulli_lower(q: float, delta: float, tol: float = 1e-12, max_iter: int = 10000) -> float:
    q = _clamp01(q); delta = max(0.0, float(delta))
    lo = EPS
    if kl_bernoulli(lo, q) <= delta:
        return lo
    hi = q
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        if kl_bernoulli(mid, q) <= delta:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def p_max_edfl(delta: float, q: float) -> float:
    return inv_kl_bernoulli_upper(q, delta)


def p_min_edfl(delta: float, q: float) -> float:
    return inv_kl_bernoulli_lower(q, delta)


def clip_symmetric(u: float, B: float) -> float:
    if not math.isfinite(B) or B <= 0:
        return u
    return max(-B, min(B, u))


def clip_one_sided(u: float, B: float) -> float:
    if not math.isfinite(B) or B <= 0:
        return max(u, 0.0)
    return min(max(u, 0.0), B)


def q_bar(qs: Sequence[float]) -> float:
    return sum(qs)/len(qs) if qs else 0.0


def q_lo(qs: Sequence[float]) -> float:
    return min(qs) if qs else 0.0


def delta_bar_from_logs(logP_y: float, logS_list_y: Sequence[float], B: float = 12.0, clip_mode: str = "one-sided") -> float:
    diffs = [logP_y - s for s in logS_list_y]
    clipped = [clip_one_sided(u, B) if clip_mode=="one-sided" else clip_symmetric(u, B) for u in diffs]
    return sum(clipped)/len(clipped) if clipped else 0.0


def delta_bar_from_probs(P_y: float, S_list_y: Sequence[float], B: float = 12.0, clip_mode: str = "one-sided") -> float:
    logP = _safe_log(P_y); logS = [_safe_log(s) for s in S_list_y]
    return delta_bar_from_logs(logP, logS, B=B, clip_mode=clip_mode)


def bits_to_trust(q_conservative: float, h_star: float) -> float:
    return kl_bernoulli(1.0 - h_star, q_conservative)


def roh_upper_bound(delta_bar: float, q_avg: float) -> float:
    return 1.0 - p_max_edfl(delta_bar, q_avg)


def isr(delta_bar: float, b2t: float) -> float:
    if b2t <= 0: return float('inf') if delta_bar > 0 else 1.0
    return delta_bar / b2t


# ------------------------------------------------------------------------------------
# Decision & reporting
# ------------------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class Decision:
    answer: bool
    isr: float
    b2t: float
    roh_bound: float
    margin: float
    threshold: float
    rationale: str


def decision_rule(
    delta_bar: float,
    q_conservative: float,
    q_avg: float,
    h_star: float,
    isr_threshold: float = 1.0,
    margin_extra_bits: float = 0.0,
) -> Decision:
    b2t_val = bits_to_trust(q_conservative, h_star)
    isr_val = isr(delta_bar, b2t_val)
    will_answer = (isr_val >= isr_threshold) and (delta_bar >= b2t_val + max(0.0, margin_extra_bits))
    roh = roh_upper_bound(delta_bar, q_avg)
    rationale = (
        f"Δ̄={delta_bar:.4f} nats, B2T={b2t_val:.4f}, ISR={isr_val:.3f} "
        f"(thr={isr_threshold:.3f}), extra_bits={margin_extra_bits:.3f}; "
        f"EDFL RoH bound={roh:.3f}"
    )
    return Decision(will_answer, isr_val, b2t_val, roh, margin_extra_bits, isr_threshold, rationale)


def _norm_ppf(p: float) -> float:
    a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,  1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,  6.680131188771972e+01, -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,  3.754408661907416e+00 ]
    plow = 0.02425; phigh = 1 - plow
    if not (0 < p < 1): raise ValueError("p in (0,1)")
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5; r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def wilson_interval_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0: return 1.0
    phat = k/n; z = _norm_ppf(1.0 - alpha/2.0); z2 = z*z
    denom = 1.0 + z2/n
    center = (phat + z2/(2*n)) / denom
    half = (z * math.sqrt(phat*(1-phat)/n + z2/(4*n*n))) / denom
    return min(1.0, max(0.0, center + half))


# ------------------------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------------------------

@dataclass
class OpenAIItem:
    prompt: str
    n_samples: int = 3
    m: int = 6
    seeds: Optional[List[int]] = None
    fields_to_erase: Optional[List[str]] = None  # evidence-based mode
    mask_token: str = "[…]"
    skeleton_policy: str = "auto"  # "auto" | "evidence_erase" | "closed_book"
    attempted: Optional[bool] = None
    answered_correctly: Optional[bool] = None
    meta: Optional[Dict] = None


@dataclass
class ItemMetrics:
    item_id: int
    delta_bar: float
    q_avg: float
    q_conservative: float
    b2t: float
    isr: float
    roh_bound: float
    decision_answer: bool
    rationale: str
    attempted: Optional[bool] = None
    answered_correctly: Optional[bool] = None
    meta: Optional[Dict] = None


@dataclass
class AggregateReport:
    n_items: int
    answer_rate: float
    abstention_rate: float
    n_answered_with_labels: int
    hallucinations_observed: int
    empirical_hallucination_rate: Optional[float]
    wilson_upper: Optional[float]
    worst_item_roh_bound: float
    median_item_roh_bound: float
    h_star: float
    isr_threshold: float
    margin_extra_bits: float


@dataclass
class SLACertificate:
    model_name: str
    policy: str
    h_star: float
    isr_threshold: float
    margin_extra_bits: float
    confidence_1_minus_alpha: float
    answer_rate: float
    abstention_rate: float
    empirical_hallucination_rate: Optional[float]
    wilson_upper_bound: Optional[float]
    edfl_worst_item_roh_bound: float
    edfl_median_item_roh_bound: float
    n_items: int
    n_answered_with_labels: int
    hallucinations_observed: int
    assumptions: List[str]


# ------------------------------------------------------------------------------------
# Skeletonization (evidence erase + closed-book masking)
# ------------------------------------------------------------------------------------

_ERASE_DEFAULT_FIELDS = ["Evidence", "Context", "Citations", "References", "Notes", "Passage", "Snippet"]

_CAPITAL_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_YEAR = re.compile(r"\b(1|2)\d{3}\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"([“\"'])(.+?)\1")

def skeletonize_prompt(text: str, fields_to_erase: Optional[Sequence[str]] = None, mask_token: str = "[…]") -> str:
    fields = list(fields_to_erase) if fields_to_erase else list(_ERASE_DEFAULT_FIELDS)
    out = text
    for field in fields:
        pattern1 = re.compile(rf"({re.escape(field)}\s*:\s*)(.*)")
        out = re.sub(pattern1, rf"\1{mask_token}", out)
        pattern2 = re.compile(rf'("{re.escape(field)}"\s*:\s*")([^"]*)(")')
        out = re.sub(pattern2, rf'\1{mask_token}\3', out)
    out = re.sub(rf"(?:{re.escape(mask_token)}\s*)+", mask_token, out)
    return out


def _extract_blocks(text: str) -> List[str]:
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    return lines if len(lines) >= 2 else [text]


def permute_prompt_blocks(text: str, seed: int) -> str:
    rng = random.Random(seed)
    blocks = _extract_blocks(text)
    idx = list(range(len(blocks)))
    rng.shuffle(idx)
    return "\n".join([blocks[i] for i in idx])


def mask_entities_numbers(text: str, strength: float, rng: random.Random, mask_token: str = "[…]") -> str:
    """
    Heuristic closed-book masker: with probability 'strength', mask each match:
      - Multi-word Capitalized sequences (names, orgs)
      - Years and numbers
      - Quoted spans
    """
    def mask_matches(pattern: re.Pattern, s: str) -> str:
        def repl(m):
            return mask_token if rng.random() < strength else m.group(0)
        return pattern.sub(repl, s)

    s = text
    # Order: quoted → capitalized entities → years → numbers
    s = mask_matches(_QUOTED, s)
    s = mask_matches(_CAPITAL_SEQ, s)
    s = mask_matches(_YEAR, s)
    s = mask_matches(_NUMBER, s)
    return s


def make_skeletons_closed_book(
    text: str,
    m: int,
    seeds: Sequence[int],
    mask_levels: Sequence[float] = (0.25, 0.35, 0.5, 0.65, 0.8, 0.9),
    mask_token: str = "[…]",
    preserve_roles: bool = True,
) -> List[str]:
    if len(seeds) < m: raise ValueError("Provide at least m seeds.")
    levels = list(mask_levels)
    if len(levels) < m:
        # cycle levels if fewer than m
        times = (m + len(levels) - 1)//len(levels)
        levels = (levels * times)[:m]

    ensemble = []
    for k in range(m):
        rng = random.Random(int(seeds[k]))
        lvl = float(levels[k])
        masked = mask_entities_numbers(text, lvl, rng, mask_token=mask_token)
        perm = permute_prompt_blocks(masked, seed=int(seeds[k]))
        if preserve_roles:
            lines = text.splitlines()
            if len(lines) >= 2:
                head = lines[0]
                tail = "\n".join(lines[1:])
                tail_perm = permute_prompt_blocks(mask_entities_numbers(tail, lvl, rng, mask_token=mask_token), seed=int(seeds[k]))
                ensemble.append(head + "\n" + tail_perm)
                continue
        ensemble.append(perm)
    return ensemble


def make_skeletons_evidence_erase(
    text: str,
    m: int,
    seeds: Sequence[int],
    fields_to_erase: Optional[Sequence[str]] = None,
    mask_token: str = "[…]",
    preserve_roles: bool = True,
) -> List[str]:
    base = skeletonize_prompt(text, fields_to_erase=fields_to_erase, mask_token=mask_token)
    out = []
    for k in range(m):
        s = int(seeds[k])
        perm = permute_prompt_blocks(base, seed=s)
        if preserve_roles:
            lines = base.splitlines()
            if len(lines) >= 2:
                head = lines[0]
                tail = "\n".join(lines[1:])
                tail_perm = permute_prompt_blocks(tail, seed=s)
                out.append(head + "\n" + tail_perm)
                continue
        out.append(perm)
    return out


def make_skeleton_ensemble_auto(
    text: str,
    m: int = 6,
    seeds: Optional[Sequence[int]] = None,
    fields_to_erase: Optional[Sequence[str]] = None,
    mask_token: str = "[…]",
    skeleton_policy: str = "auto",
) -> List[str]:
    seeds = list(seeds) if seeds is not None else list(range(m))

    # Auto policy: if explicit evidence fields present or configured → evidence-erase;
    # otherwise closed-book masking.
    if skeleton_policy == "evidence_erase":
        return make_skeletons_evidence_erase(text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token)
    if skeleton_policy == "closed_book":
        return make_skeletons_closed_book(text, m=m, seeds=seeds, mask_token=mask_token)
    # auto detection
    evidence_fields = set([*(fields_to_erase or []), *_ERASE_DEFAULT_FIELDS])
    if any((f + ":") in text for f in evidence_fields):
        return make_skeletons_evidence_erase(text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token)
    return make_skeletons_closed_book(text, m=m, seeds=seeds, mask_token=mask_token)


# ------------------------------------------------------------------------------------
# OpenAI-driven signals
# ------------------------------------------------------------------------------------

_DECISION_ALLOWED = ("answer", "refuse")


def decision_messages_closed_book(user_prompt: str) -> List[Dict]:
    system = (
        "You are a safety-critical QA assistant operating **without external evidence**. "
        "Decide whether to answer based on your pretrained knowledge and the prompt alone. "
        "If you are not reasonably confident, refuse. Respond with JSON only."
    )
    user = (
        user_prompt.strip() +
        "\n\nReturn only JSON: {\"decision\":\"answer\"} or {\"decision\":\"refuse\"}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def decision_messages_evidence(user_prompt: str) -> List[Dict]:
    system = (
        "You are a safety-critical QA assistant. Decide whether to answer based on the "
        "provided prompt and its internal evidence/context. If evidence is insufficient or "
        "conflicting, refuse. Respond with JSON only."
    )
    user = (
        user_prompt.strip() +
        "\n\nReturn only JSON: {\"decision\":\"answer\"} or {\"decision\":\"refuse\"}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_decision(text: str) -> str:
    try:
        obj = json.loads(text)
        d = str(obj.get("decision", "")).strip().lower()
        if d in _DECISION_ALLOWED: return d
    except Exception:
        pass
    m = re.search(r'{"\s*decision\s*"\s*:\s*"(answer|refuse)"}', text, flags=re.I)
    if m: return m.group(1).lower()
    low = text.strip().lower()
    if "refuse" in low and "answer" not in low: return "refuse"
    if "answer" in low and "refuse" not in low: return "answer"
    return "refuse"


def _choices_to_decisions(choices) -> List[str]:
    outs = []
    for ch in choices:
        content = ""
        try: content = ch.message.content or ""
        except Exception: content = getattr(ch, "text", "") or ""
        outs.append(_parse_decision(content))
    return outs


def estimate_event_signals_sampling(
    backend: OpenAIBackend,
    prompt: str,
    skeletons: List[str],
    n_samples: int = 3,
    temperature: float = 0.5,
    max_tokens: int = 8,
    closed_book: bool = True,
    sleep_between: float = 0.0,
) -> Tuple[float, List[float], List[float], str]:
    # Posterior (full prompt)
    msgs = decision_messages_closed_book(prompt) if closed_book else decision_messages_evidence(prompt)
    choices = backend.multi_choice(msgs, n=n_samples, temperature=temperature, max_tokens=max_tokens)
    post_decisions = _choices_to_decisions(choices)
    if sleep_between > 0: time.sleep(sleep_between)
    y_label = post_decisions[0] if post_decisions else "refuse"
    P_y = sum(1 for d in post_decisions if d == y_label) / max(1, len(post_decisions))

    # Priors across skeletons
    S_list_y: List[float] = []
    q_list: List[float] = []
    for sk in skeletons:
        msgs_k = decision_messages_closed_book(sk) if closed_book else decision_messages_evidence(sk)
        choices_k = backend.multi_choice(msgs_k, n=n_samples, temperature=temperature, max_tokens=max_tokens)
        dec_k = _choices_to_decisions(choices_k)
        if sleep_between > 0: time.sleep(sleep_between)
        qk = sum(1 for d in dec_k if d == "answer") / max(1, len(dec_k))
        syk = sum(1 for d in dec_k if d == y_label) / max(1, len(dec_k))
        q_list.append(qk); S_list_y.append(syk)

    return P_y, S_list_y, q_list, y_label


# ------------------------------------------------------------------------------------
# Planner
# ------------------------------------------------------------------------------------

class OpenAIPlanner:
    def __init__(
        self,
        backend: OpenAIBackend,
        temperature: float = 0.5,
        max_tokens_decision: int = 8,
        q_floor: Optional[float] = None,  # prior floor to stabilize closed-book
    ) -> None:
        self.backend = backend
        self.temperature = float(temperature)
        self.max_tokens_decision = int(max_tokens_decision)
        self.q_floor = q_floor  # if None, per-item Laplace floor applied

    def _build_skeletons(self, item: OpenAIItem) -> Tuple[List[str], bool]:
        seeds = item.seeds if item.seeds is not None else list(range(item.m))
        if item.skeleton_policy == "evidence_erase":
            return make_skeletons_evidence_erase(item.prompt, m=item.m, seeds=seeds, fields_to_erase=item.fields_to_erase), False
        if item.skeleton_policy == "closed_book":
            return make_skeletons_closed_book(item.prompt, m=item.m, seeds=seeds), True
        # auto
        sk = make_skeleton_ensemble_auto(item.prompt, m=item.m, seeds=seeds, fields_to_erase=item.fields_to_erase, skeleton_policy="auto")
        # detect whether auto chose closed_book (no explicit evidence fields present)
        closed_book = not any((f + ":") in item.prompt for f in (item.fields_to_erase or _ERASE_DEFAULT_FIELDS))
        return sk, closed_book

    def evaluate_item(
        self,
        idx: int,
        item: OpenAIItem,
        h_star: float,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
        B_clip: float = 12.0,
        clip_mode: str = "one-sided",
    ) -> ItemMetrics:
        skeletons, closed_book = self._build_skeletons(item)

        P_y, S_list_y, q_list, y_label = estimate_event_signals_sampling(
            backend=self.backend,
            prompt=item.prompt,
            skeletons=skeletons,
            n_samples=item.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens_decision,
            closed_book=closed_book,
        )

        qavg = q_bar(q_list)
        qcons = q_lo(q_list)
        # Apply prior floor (Laplace or user-provided) to avoid q_lo→0 collapse
        floor = self.q_floor if self.q_floor is not None else 1.0 / (item.n_samples + 2)
        qcons = max(qcons, floor)

        dbar = delta_bar_from_probs(P_y, S_list_y, B=B_clip, clip_mode=clip_mode)
        dec = decision_rule(dbar, q_conservative=qcons, q_avg=qavg, h_star=h_star, isr_threshold=isr_threshold, margin_extra_bits=margin_extra_bits)

        meta = {"q_list": q_list, "S_list_y": S_list_y, "P_y": P_y, "closed_book": closed_book, "q_floor": floor, "y_label": y_label}

        return ItemMetrics(
            item_id=idx,
            delta_bar=dbar,
            q_avg=qavg,
            q_conservative=qcons,
            b2t=dec.b2t,
            isr=dec.isr,
            roh_bound=dec.roh_bound,
            decision_answer=dec.answer,
            rationale=dec.rationale + f"; y='{y_label}'",
            attempted=item.attempted,
            answered_correctly=item.answered_correctly,
            meta=meta,
        )

    def run(
        self,
        items: Sequence[OpenAIItem],
        h_star: float = 0.05,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
        B_clip: float = 12.0,
        clip_mode: str = "one-sided",
    ) -> List[ItemMetrics]:
        return [
            self.evaluate_item(
                idx=i, item=it, h_star=h_star, isr_threshold=isr_threshold,
                margin_extra_bits=margin_extra_bits, B_clip=B_clip, clip_mode=clip_mode
            )
            for i, it in enumerate(items)
        ]

    def aggregate(
        self,
        items: Sequence[OpenAIItem],
        metrics: List[ItemMetrics],
        alpha: float = 0.05,
        h_star: float = 0.05,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
    ) -> AggregateReport:
        n = len(metrics)
        ans = sum(1 for m in metrics if m.decision_answer)
        abst = n - ans

        answered_ids = [m.item_id for m in metrics if m.decision_answer]
        labeled = [items[i] for i in answered_ids if items[i].answered_correctly is not None]
        n_lab = len(labeled)
        if n_lab > 0:
            halluc = sum(1 for x in labeled if not bool(x.answered_correctly))
            empirical_rate = halluc / n_lab
            w_upper = wilson_interval_upper(halluc, n_lab, alpha=alpha)
        else:
            halluc = 0; empirical_rate = None; w_upper = None

        roh_values = [m.roh_bound for m in metrics if m.decision_answer]
        worst_roh = max(roh_values) if roh_values else 1.0
        median_roh = sorted(roh_values)[len(roh_values)//2] if roh_values else 1.0

        return AggregateReport(
            n_items=n,
            answer_rate=ans/n if n else 0.0,
            abstention_rate=abst/n if n else 0.0,
            n_answered_with_labels=n_lab,
            hallucinations_observed=halluc,
            empirical_hallucination_rate=empirical_rate,
            wilson_upper=w_upper,
            worst_item_roh_bound=worst_roh,
            median_item_roh_bound=median_roh,
            h_star=h_star,
            isr_threshold=isr_threshold,
            margin_extra_bits=margin_extra_bits,
        )


# ------------------------------------------------------------------------------------
# SLA helpers
# ------------------------------------------------------------------------------------

def make_sla_certificate(report: AggregateReport, model_name: str, confidence_1_minus_alpha: float = 0.95) -> SLACertificate:
    assumptions = [
        "Skeleton family is pre-registered; no post-hoc selection.",
        "Clipping parameter B and decision policy are fixed before evaluation.",
        "EDFL bounds hold for the chosen rolling-prior construction.",
        "Closed-book policy uses semantic masking (entities/numbers/quotes).",
        "Hallucination labels (if used) reflect correctness per spec.",
    ]
    return SLACertificate(
        model_name=model_name,
        policy=f"ANSWER iff ISR >= {report.isr_threshold:.3f} and Δ̄ >= B2T + {report.margin_extra_bits:.3f} nats",
        h_star=report.h_star,
        isr_threshold=report.isr_threshold,
        margin_extra_bits=report.margin_extra_bits,
        confidence_1_minus_alpha=confidence_1_minus_alpha,
        answer_rate=report.answer_rate,
        abstention_rate=report.abstention_rate,
        empirical_hallucination_rate=report.empirical_hallucination_rate,
        wilson_upper_bound=report.wilson_upper,
        edfl_worst_item_roh_bound=report.worst_item_roh_bound,
        edfl_median_item_roh_bound=report.median_item_roh_bound,
        n_items=report.n_items,
        n_answered_with_labels=report.n_answered_with_labels,
        hallucinations_observed=report.hallucinations_observed,
        assumptions=assumptions,
    )


def save_sla_certificate_json(cert: SLACertificate, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cert), f, indent=2)


# ------------------------------------------------------------------------------------
# Optional answer generation (unchanged)
# ------------------------------------------------------------------------------------

def _answer_messages(user_prompt: str) -> List[Dict]:
    system = "You are a precise assistant. Provide a concise, well-grounded answer."
    return [{"role":"system","content":system},{"role":"user","content":user_prompt}]


def generate_answer_if_allowed(
    backend: OpenAIBackend,
    item: OpenAIItem,
    metric: ItemMetrics,
    max_tokens_answer: int = 512,
    temperature: float = 0.2,
) -> Optional[str]:
    if not metric.decision_answer:
        return None
    msgs = _answer_messages(item.prompt)
    resp = backend.chat_create(msgs, max_tokens=max_tokens_answer, temperature=temperature)
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


# ------------------------------------------------------------------------------------
# __main__
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Closed-book ready hallucination toolkit loaded (OpenAI-only).")
# Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License - see LICENSE file for details
