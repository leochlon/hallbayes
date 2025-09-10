# -*- coding: utf-8 -*-
"""
Hallucination Toolkit — Semantic‑Logprob Variant (v2)
====================================================

What’s new in v2 (this release)
------------------
1) **Concurrency** (faster end‑to‑end):
   • Parallel skeleton sampling via `ThreadPoolExecutor` (configurable with
     `max_workers_skeletons`).  
   • Optional parallel item evaluation via `run(..., max_workers_items=...)`.
   • Thread‑safe entailment providers (`OpenAILLMEntailer`, `BudgetedEntailer`).

2) **Safer gate on made‑up prompts**:
   • Optional **top‑meaning LLR budget** (`use_llr_top_meaning_budget=True`) to
     avoid multi‑class KL blow‑ups.  
   • **Posterior‑dominance latch**: require `P* ≥ 1 - h*` in the EDFL branch.  
   • **KL smoothing** for the multi‑class budget (default `eps=1e-2`).  
   • Keeps the **prior‑sufficiency gate** from v2.3.

This version keeps the “semantic entropy over meanings” estimator and the
EDFL/B2T/ISR decision head, but hardens it against counterfactual prompts and
adds concurrency for speed.  The design follows the Nature paper (“semantic
entropy”; sampling → bidirectional‑entailment clustering → entropy over meanings)
and the closed‑book EDFL rationale from v2.3/v2.1.  fileciteturn0file1 fileciteturn0file0

Usage (unchanged API surface)
-----------------------------
from scripts.hallucination_toolkit_semantic import OpenAIChatBackend, SemanticPlanner, OpenAIItem

backend = OpenAIChatBackend(model="gpt-4o-mini", debug=True, strict=True)
planner = SemanticPlanner(
    backend,
    temperature=1.0, top_p=0.9,
    max_tokens_answer=48,
    max_entail_calls=120, entailment_timeout_s=15.0,
    use_llr_top_meaning_budget=True,        # <-- strongly recommended
    max_workers_skeletons=6,                 # <-- concurrency over skeletons
    kl_smoothing_eps=1e-2,                   # <-- smoothing for KL budget
)

items = [OpenAIItem(prompt="Who won the Nobel Prize in physics in 2014?",
                    M_full=10, m_skeletons=6, n_samples_per_skeleton=6)]
metrics = planner.run(items, h_star=0.05, isr_threshold=1.0)[0]
print(metrics)

To evaluate many items faster:
metrics = planner.run(items, h_star=0.05, isr_threshold=1.0, max_workers_items=8)

Notes
-----
• Concurrency is conservative: it parallelizes **skeleton sampling** (the dominant
  cost) and optional **item‑level** evaluation.  Entailment calls are cached and guarded
  with locks; sampling calls are batched (`n` choices per API call) and parallelized
  across skeletons.  fileciteturn0file0
• The **posterior‑dominance latch** ensures we never answer when the top meaning
  itself is not sufficiently probable, even if Δ̄ is large.  This restores conservative
  behavior on fabricated prompts without re‑introducing “ask the model to decide
  to answer”.  fileciteturn0file0
"""
__version__ = "2.0.0"

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Iterable, Protocol, runtime_checkable

# -----------------------------------------------------------------------------
# Core math (nats)
# -----------------------------------------------------------------------------

EPS = 1e-12

def _clamp01(x: float, eps: float = EPS) -> float:
    return min(1.0 - eps, max(eps, x))

def _safe_log(x: float) -> float:
    return math.log(max(x, EPS))

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
        if hi - lo < tol: break
    return lo

def p_max_edfl(delta: float, q: float) -> float:
    return inv_kl_bernoulli_upper(q, delta)

def roh_upper_bound(delta_bar: float, q_avg: float) -> float:
    return 1.0 - p_max_edfl(delta_bar, q_avg)

def bits_to_trust(q_conservative: float, h_star: float) -> float:
    """
    EDFL lower bound: Δ̄ ≥ KL(Ber(1-h*) || Ber(q)). Interpret *bits to trust* as
    the *extra evidence required when q < 1-h*; return 0 when q ≥ 1-h*.
    This prevents spurious abstentions when the prior already certifies
    the target reliability.  fileciteturn0file0
    """
    target = 1.0 - h_star
    if q_conservative >= target:
        return 0.0
    return kl_bernoulli(target, q_conservative)

def isr(delta_bar: float, b2t: float) -> float:
    if b2t <= 0: return float('inf') if delta_bar > 0 else float('inf')
    return delta_bar / b2t

def _kl_discrete(P: List[float], Q: List[float], eps: float = 1e-2) -> float:
    """
    Smoothed discrete KL: Q' = (1-eps)·Q + eps/K  (prevents blow‑ups when Q has zeros).
    """
    assert len(P) == len(Q) and len(P) > 0
    K = len(P)
    s = 0.0
    for p, q in zip(P, Q):
        if p <= 0: continue
        q_s = (1.0 - eps)*q + eps/float(K)
        s += p * (_safe_log(p) - _safe_log(q_s))
    return s


# -----------------------------------------------------------------------------
# OpenAI backend with robust text/logprob extraction + debug + health‑check
# -----------------------------------------------------------------------------

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None

class OpenAIChatBackend:
    """
    Thin wrapper around the OpenAI Chat Completions API.
    - Requests token logprobs when supported; otherwise falls back to text only.
    - Guarantees non‑empty samples via a resample fallback.
    - Adds `health_check()` and `debug` logging to make API traffic visible.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        request_timeout: float = 60.0,
        debug: bool = False,
        strict: bool = False,
    ) -> None:
        if OpenAI is None:
            raise ImportError("Install `openai>=1.0.0` and set OPENAI_API_KEY.")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.request_timeout = float(request_timeout)
        self.debug = bool(debug)
        self.strict = bool(strict)

    # --- logging helpers ---
    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[OpenAIChatBackend] {msg}", flush=True)

    def health_check(self) -> None:
        self._dbg(f"health_check(): model={self.model}")
        msgs = [{"role":"system","content":"Say OK."},{"role":"user","content":"Say OK, nothing else."}]
        try:
            r = self.client.chat.completions.create(model=self.model, messages=msgs, max_tokens=1, temperature=0.0, timeout=self.request_timeout)
            txt = ""
            try: txt = r.choices[0].message.content or ""
            except Exception: txt = getattr(r.choices[0], "text", "") or ""
            self._dbg(f"health_check(): ok, text='{txt[:64]}'")
        except Exception as e:
            self._dbg(f"health_check(): FAILED: {type(e).__name__}: {e}")
            raise

    def chat(self, messages: List[Dict], **kwargs):
        params = dict(model=self.model, messages=messages, max_tokens=64, temperature=1.0)
        params.update(kwargs)
        if "timeout" in params:
            params["timeout"] = params.pop("timeout")
        if self.debug:
            self._dbg(f"chat(): max_tokens={params.get('max_tokens')} temp={params.get('temperature')} logprobs={params.get('logprobs', False)} top_logprobs={params.get('top_logprobs', None)}")
        return self.client.chat.completions.create(**params)

    def _coerce_text_from_choice(self, choice) -> str:
        try:
            msg = getattr(choice, "message", None)
            if msg is None:
                txt = getattr(choice, "text", "") or ""
                return txt.strip()
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for part in content:
                    t = getattr(part, "text", None)
                    if t is None and isinstance(part, dict):
                        t = part.get("text")
                    if t: parts.append(str(t))
                return " ".join(parts).strip()
            if isinstance(msg, dict):
                txt = msg.get("content", "")
                if isinstance(txt, str):
                    return txt.strip()
                if isinstance(txt, list):
                    return " ".join([str(getattr(p, "text", "") or (p.get("text","") if isinstance(p, dict) else "")) for p in txt]).strip()
        except Exception:
            pass
        try:
            return (getattr(choice, "text", "") or "").strip()
        except Exception:
            return ""

    def _extract_token_logprobs(self, choice) -> Optional[List[float]]:
        try:
            lp_content = getattr(choice, "logprobs", None)
            if not lp_content:
                return None
            content = getattr(lp_content, "content", None)
            if not content:
                return None
            vals: List[float] = []
            for seg in content:
                lp = getattr(seg, "logprob", None)
                if lp is None and isinstance(seg, dict):
                    lp = seg.get("logprob")
                if lp is not None:
                    try: vals.append(float(lp))
                    except Exception: pass
            return vals if len(vals) > 0 else None
        except Exception:
            return None

    def _sample_once(self, messages: List[Dict], **kwargs) -> List[Tuple[str, Optional[List[float]]]]:
        resp = self.chat(messages, **kwargs)
        out: List[Tuple[str, Optional[List[float]]]] = []
        for ch in getattr(resp, "choices", []) or []:
            text = self._coerce_text_from_choice(ch)
            tok_lps = self._extract_token_logprobs(ch)
            out.append((text, tok_lps))
        self._dbg(f"_sample_once(): got {len(out)} choices")
        return out

    def sample_with_logprobs(
        self,
        messages: List[Dict],
        n: int = 10,
        temperature: float = 1.0,
        max_tokens: int = 48,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        logprobs: bool = True,
        top_logprobs: int = 5,
        timeout: Optional[float] = None,
    ) -> List[Tuple[str, Optional[List[float]]]]:
        try:
            out = self._sample_once(
                messages,
                n=n,
                temperature=temperature,
                max_tokens=max(8, int(max_tokens)),
                top_p=top_p,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                timeout=timeout or self.request_timeout,
            )
        except Exception as e:
            self._dbg(f"sample_with_logprobs(): first attempt failed: {type(e).__name__}: {e}")
            if self.strict: raise
            out = []
        if not out or all((t.strip() == "") for (t, _) in out):
            try:
                out = self._sample_once(
                    messages,
                    n=n,
                    temperature=0.2,
                    max_tokens=max(12, int(max_tokens)),
                    top_p=1.0,
                    logprobs=False,
                    timeout=timeout or self.request_timeout,
                )
            except Exception as e:
                self._dbg(f"sample_with_logprobs(): second attempt failed: {type(e).__name__}: {e}")
                if self.strict: raise
                out = []
        if not out:
            if self.strict:
                raise RuntimeError("OpenAI sampling failed twice; strict=True.")
            out = [("Unknown", None) for _ in range(n)]
        else:
            out = [(t if t.strip() else "Unknown", lp) for (t, lp) in out]
        return out


# -----------------------------------------------------------------------------
# Entailment providers (thread‑safe; budgeted)
# -----------------------------------------------------------------------------

@runtime_checkable
class EntailmentProvider(Protocol):
    def entails(self, question: str, a: str, b: str) -> bool: ...

def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ \t\r\n]+", " ", s)
    s = s.strip(" .,;:!?“”'\"()[]")
    return s

def _digits_set(s: str) -> Tuple[str, Tuple[str, ...]]:
    ds = tuple(re.findall(r"\d+", s))
    return (";".join(ds), ds)

def _token_set(s: str) -> set:
    s = _normalize_text(s)
    toks = re.findall(r"[a-z0-9]+", s)
    return set(toks)

def cheap_semantic_equiv(question: str, a: str, b: str) -> bool:
    na, nb = _normalize_text(a), _normalize_text(b)
    if not na or not nb: return False
    if na == nb: return True
    la, lb = len(na), len(nb)
    if (na in nb and la/max(lb,1) >= 0.6) or (nb in na and lb/max(la,1) >= 0.6):
        return True
    da, _ = _digits_set(na); db, _ = _digits_set(nb)
    if da and da == db: return True
    ta, tb = _token_set(na), _token_set(nb)
    if ta and tb:
        j = len(ta & tb) / max(1, len(ta | tb))
        if j >= 0.8: return True
    return False

class OpenAILLMEntailer(EntailmentProvider):
    def __init__(self, backend: OpenAIChatBackend, request_timeout: float = 15.0):
        self.backend = backend
        self._cache: Dict[Tuple[str, str, str], bool] = {}
        self._lock = threading.RLock()
        self.request_timeout = float(request_timeout)

    @staticmethod
    def _prompt(q: str, a: str, b: str) -> List[Dict]:
        sys = "You judge semantic entailment precisely."
        user = (
            f"We are evaluating answers to the question:\n{q}\n"
            f"Possible Answer 1: {a}\n"
            f"Possible Answer 2: {b}\n"
            "Does Answer 1 semantically entail Answer 2 in the context of the question? "
            "Respond with exactly one word: entailment, contradiction, or neutral."
        )
        return [{"role":"system","content":sys},{"role":"user","content":user}]

    def entails(self, question: str, a: str, b: str) -> bool:
        if cheap_semantic_equiv(question, a, b):
            return True
        key = (question, a, b)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        # Do the API call outside the lock for concurrency
        try:
            r = self.backend.chat(self._prompt(question, a, b), temperature=0.0, max_tokens=2, timeout=self.request_timeout)
            lab = self.backend._coerce_text_from_choice(r.choices[0]).strip().lower()
            ans = "entailment" in lab
        except Exception:
            ans = False
        with self._lock:
            self._cache[key] = ans
        return ans

class BudgetedEntailer(EntailmentProvider):
    def __init__(self, base: EntailmentProvider, max_calls: int = 120):
        self.base = base
        self.max_calls = int(max_calls)
        self.calls = 0
        self._lock = threading.Lock()

    def entails(self, question: str, a: str, b: str) -> bool:
        if cheap_semantic_equiv(question, a, b):
            return True
        with self._lock:
            if self.calls >= self.max_calls:
                return False
            self.calls += 1
        try:
            return self.base.entails(question, a, b)
        except Exception:
            return False


# -----------------------------------------------------------------------------
# Clustering and probabilities
# -----------------------------------------------------------------------------

@dataclass
class Cluster:
    rep: str
    members: List[str]

def _stable_unique(seq: Sequence[str]) -> List[str]:
    seen = set(); out = []
    for s in seq:
        key = _normalize_text(s)
        if key not in seen:
            seen.add(key); out.append(s)
    return out

def cluster_by_bidirectional_entailment(
    question: str,
    answers: List[str],
    entailer: EntailmentProvider,
    max_clusters: int = 50,
) -> List[Cluster]:
    answers = _stable_unique([s for s in answers if s.strip()])
    clusters: List[Cluster] = []
    for s in answers:
        placed = False
        for c in clusters:
            rep = c.rep
            if cheap_semantic_equiv(question, s, rep):
                c.members.append(s); placed = True; break
            if entailer.entails(question, s, rep) and entailer.entails(question, rep, s):
                c.members.append(s); placed = True; break
        if not placed:
            clusters.append(Cluster(rep=s, members=[s]))
            if len(clusters) >= max_clusters:
                break
    if not clusters:
        clusters = [Cluster(rep="Unknown", members=[])]
    return clusters

def length_normalized_weight(token_logprobs: Optional[List[float]]) -> float:
    if not token_logprobs:
        return 1.0
    N = max(1, len(token_logprobs))
    avg_lp = sum(token_logprobs)/N
    try:
        return float(math.exp(avg_lp))
    except OverflowError:
        return 0.0

def _best_lexical_cluster_index(question: str, txt: str, clusters: List[Cluster], jaccard_threshold: float = 0.65) -> Optional[int]:
    cand = None; best = 0.0
    T = _token_set(txt)
    for i, c in enumerate(clusters):
        J = len(T & _token_set(c.rep)) / max(1, len(T | _token_set(c.rep)))
        if J > best:
            best = J; cand = i
    return cand if best >= jaccard_threshold else None

def cluster_probs_from_samples(
    clusters: List[Cluster],
    answers_with_weights: List[Tuple[str, float]],
    question: str,
    entailer: EntailmentProvider,
    include_other: bool = False,
) -> Tuple[List[float], List[str]]:
    K = len(clusters)
    sums = [0.0 for _ in range(K)]
    labels = [c.rep for c in clusters]
    other_sum = 0.0

    for txt, w in answers_with_weights:
        if not txt or w <= 0: continue
        assigned = False
        for i, c in enumerate(clusters):
            rep = c.rep
            if cheap_semantic_equiv(question, txt, rep) or (entailer.entails(question, txt, rep) and entailer.entails(question, rep, txt)):
                sums[i] += w; assigned = True; break
        if not assigned:
            j = _best_lexical_cluster_index(question, txt, clusters, jaccard_threshold=0.65)
            if j is not None:
                sums[j] += w; assigned = True
        if not assigned and include_other:
            other_sum += w

    if include_other:
        sums.append(other_sum); labels.append("OTHER")

    total = sum(sums)
    if total <= 0:
        if sums:
            sums[0] = 1.0; total = 1.0
    probs = [s/total for s in sums] if total > 0 else [1.0/len(sums) for _ in sums]
    return probs, labels

def probs_from_full_prompt(
    question: str,
    samples: List[Tuple[str, Optional[List[float]]]],
    entailer: EntailmentProvider
) -> Tuple[List[float], List[Cluster], List[str], float, int, str]:
    answers = [s for (s, _) in samples if s]
    clusters = cluster_by_bidirectional_entailment(question, answers, entailer)
    aw = [(s, length_normalized_weight(lp)) for (s, lp) in samples if s]
    probs, labels = cluster_probs_from_samples(clusters, aw, question, entailer, include_other=False)
    H = -sum(p*_safe_log(p) for p in probs) if probs else 0.0
    i_star = int(max(range(len(probs)), key=lambda i: probs[i])) if probs else 0
    C_star = labels[i_star] if labels else "Unknown"
    return probs, clusters, labels, H, i_star, C_star


# -----------------------------------------------------------------------------
# Skeletonization (same as v2.3)  fileciteturn0file0
# -----------------------------------------------------------------------------

_ERASE_DEFAULT_FIELDS = ["Evidence", "Context", "Citations", "References", "Notes", "Passage", "Snippet"]
_CAPITAL_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_YEAR = re.compile(r"\b(1|2)\d{3}\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"([“\"'])(.+?)\1")

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
    def mask_matches(pattern: re.Pattern, s: str) -> str:
        def repl(m):
            return mask_token if rng.random() < strength else m.group(0)
        return pattern.sub(repl, s)
    s = text
    s = mask_matches(_QUOTED, s)
    s = mask_matches(_CAPITAL_SEQ, s)
    s = mask_matches(_YEAR, s)
    s = mask_matches(_NUMBER, s)
    return s

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
                ensemble.append(head + "\n" + tail_perm); continue
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
                out.append(head + "\n" + tail_perm); continue
        out.append(perm)
    return out

def make_skeleton_ensemble_auto(
    text: str,
    m: int,
    seeds: Optional[Sequence[int]] = None,
    fields_to_erase: Optional[Sequence[str]] = None,
    mask_token: str = "[…]",
    skeleton_policy: str = "auto",
) -> Tuple[List[str], bool]:
    seeds = list(seeds) if seeds is not None else list(range(m))
    if skeleton_policy == "evidence_erase":
        return make_skeletons_evidence_erase(text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token), False
    if skeleton_policy == "closed_book":
        return make_skeletons_closed_book(text, m=m, seeds=seeds, mask_token=mask_token), True
    evidence_fields = set([*(fields_to_erase or []), *_ERASE_DEFAULT_FIELDS])
    if any((f + ":") in text for f in evidence_fields):
        return make_skeletons_evidence_erase(text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token), False
    return make_skeletons_closed_book(text, m=m, seeds=seeds, mask_token=mask_token), True


# -----------------------------------------------------------------------------
# Prompt for short answers
# -----------------------------------------------------------------------------

def short_answer_messages(question: str) -> List[Dict]:
    sys = "Answer concisely; give only the answer (no preamble)."
    usr = f"Question: {question}\nAnswer:"
    return [{"role":"system","content":sys},{"role":"user","content":usr}]


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class OpenAIItem:
    prompt: str
    n_samples_per_skeleton: int = 6
    M_full: int = 10
    m_skeletons: int = 6
    seeds: Optional[List[int]] = None
    fields_to_erase: Optional[List[str]] = None
    mask_token: str = "[…]"
    skeleton_policy: str = "auto"
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
    H_sem: float
    top_meaning: str
    P_full_top: float
    attempted: Optional[bool] = None
    answered_correctly: Optional[bool] = None
    meta: Optional[Dict] = None


# -----------------------------------------------------------------------------
# Refusal‑like detector
# -----------------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    r"\bi (?:do|do\s*not|don't|cannot|can't|cannot)\s+(?:have|provide|access)\b",
    r"\bnot (?:available|provided|enough|sufficient)\b",
    r"\b(?:as an ai|as a language model)\b",
    r"\bknowledge (?:cut[\- ]?off|cutoff)\b",
    r"\bno (?:information|data)\b",
    r"\bunable to (?:answer|provide)\b",
    r"\bi (?:cannot|can't)\s+answer\b",
    r"\bi (?:do|don't)\s+know\b",
]

def looks_like_refusal(text: str) -> bool:
    t = _normalize_text(text)
    return any(re.search(p, t) for p in _REFUSAL_PATTERNS)


# -----------------------------------------------------------------------------
# Planner (semantic entropy + prior‑sufficiency + concurrency)
# -----------------------------------------------------------------------------

class SemanticPlanner:
    def __init__(
        self,
        backend: OpenAIChatBackend,
        entailer: Optional["EntailmentProvider"] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        max_tokens_answer: int = 48,
        m: int = 6,
        mask_token: str = "[…]",
        q_floor: Optional[float] = None,
        use_llr_top_meaning_budget: bool = False,
        max_entail_calls: int = 120,
        entailment_timeout_s: float = 15.0,
        # v2.4 concurrency/safety knobs
        max_workers_skeletons: int = 6,
        kl_smoothing_eps: float = 1e-2,
    ) -> None:
        self.backend = backend
        self.entailer = entailer or OpenAILLMEntailer(backend, request_timeout=entailment_timeout_s)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens_answer = int(max_tokens_answer)
        self.m_default = int(m)
        self.mask_token = mask_token
        self.q_floor = q_floor
        self.use_llr = bool(use_llr_top_meaning_budget)
        self.max_entail_calls = int(max_entail_calls)
        self.entail_timeout = float(entailment_timeout_s)
        self.max_workers_skeletons = int(max_workers_skeletons)
        self.kl_smoothing_eps = float(kl_smoothing_eps)

    def _sample_and_project_single(self, sk: str, clusters: List[Cluster], item: OpenAIItem, budgeted_entailer: EntailmentProvider, question: str) -> Tuple[List[float], float]:
        msgs_k = short_answer_messages(sk)
        samp_k = self.backend.sample_with_logprobs(
            msgs_k, n=item.n_samples_per_skeleton, temperature=self.temperature,
            max_tokens=self.max_tokens_answer, top_p=self.top_p,
            logprobs=True, top_logprobs=5, timeout=self.entail_timeout,
        )
        answers_k = [(s, length_normalized_weight(lp)) for (s, lp) in samp_k if s]
        probs_k, labels_k = cluster_probs_from_samples(clusters, answers_k, question, budgeted_entailer, include_other=True)
        if len(labels_k) and labels_k[-1] == "OTHER":
            probs_k = probs_k[:-1]
        return probs_k, 0.0  # legacy second value placeholder

    def evaluate_item(
        self,
        idx: int,
        item: OpenAIItem,
        h_star: float = 0.05,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
        skeleton_policy: Optional[str] = None,
    ) -> ItemMetrics:
        budgeted_entailer = BudgetedEntailer(self.entailer, max_calls=self.max_entail_calls)

        # 1) Posterior over meanings (full prompt)
        full_msgs = short_answer_messages(item.prompt)
        full_samples = self.backend.sample_with_logprobs(
            full_msgs, n=item.M_full, temperature=self.temperature,
            max_tokens=self.max_tokens_answer, top_p=self.top_p,
            logprobs=True, top_logprobs=5, timeout=self.entail_timeout,
        )
        P_full, clusters, labels, H_sem, i_star, C_star_label = probs_from_full_prompt(item.prompt, full_samples, budgeted_entailer)
        if not P_full or sum(P_full) <= 0:
            P_full = [1.0]; clusters = [Cluster(rep="Unknown", members=[])] ; labels = ["Unknown"]; H_sem = 0.0; i_star = 0; C_star_label="Unknown"
        P_star = float(P_full[i_star])

        # 2) Skeletons → priors (with concurrency over skeleton sampling)
        m = int(item.m_skeletons or self.m_default)
        seeds = item.seeds if item.seeds is not None else list(range(m))
        policy = skeleton_policy or item.skeleton_policy or "auto"
        skeletons, closed_book = make_skeleton_ensemble_auto(
            item.prompt, m=m, seeds=seeds, fields_to_erase=item.fields_to_erase,
            mask_token=item.mask_token, skeleton_policy=policy
        )

        S_list: List[List[float]] = []
        S_star_list: List[float] = []

        def worker(sk_: str):
            try:
                pvec, _ = self._sample_and_project_single(sk_, clusters, item, budgeted_entailer, item.prompt)
                # normalize and align
                if len(pvec) < len(P_full):
                    pvec = pvec + [0.0]*(len(P_full) - len(pvec))
                s = sum(pvec)
                if s <= 0:
                    pvec = [1.0/len(P_full) for _ in P_full]
                else:
                    pvec = [p/s for p in pvec]
                return pvec
            except Exception:
                # fall back to uniform prior on error
                return [1.0/len(P_full) for _ in P_full]

        max_workers = max(1, self.max_workers_skeletons)
        if max_workers == 1:
            for sk in skeletons:
                pvec = worker(sk)
                S_list.append(pvec)
                S_star_list.append(float(pvec[i_star]))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(worker, sk) for sk in skeletons]
                for f in as_completed(futs):
                    pvec = f.result()
                    S_list.append(pvec)
                    S_star_list.append(float(pvec[i_star]))

        # 3) Info budget + priors → EDFL (with prior‑sufficiency + posterior latch)
        if not self.use_llr:
            deltas = [_kl_discrete(P_full, Sk, eps=self.kl_smoothing_eps) for Sk in S_list]
            delta_bar = sum(deltas) / max(1, len(deltas))
        else:
            logP = _safe_log(P_star)
            deltas = [max(0.0, logP - _safe_log((1.0 - self.kl_smoothing_eps)*Sk[i_star] + self.kl_smoothing_eps/len(P_full))) for Sk in S_list]
            delta_bar = sum(deltas) / max(1, len(deltas))

        q_avg = sum(S_star_list) / max(1, len(S_star_list))
        q_min = min(S_star_list) if S_star_list else 0.0
        floor = self.q_floor if self.q_floor is not None else 1.0 / (item.n_samples_per_skeleton + 2)
        q_conservative = max(q_min, floor)

        target = 1.0 - h_star
        refusal_flag = looks_like_refusal(C_star_label)
        gating_mode = "edfl_gate"

        # **PRIOR‑SUFFICIENCY GATE** (from v2.3)  fileciteturn0file0
        if (q_conservative >= target) and (P_star >= target) and (not refusal_flag):
            b2t = 0.0
            isr_val = float('inf') if delta_bar > 0 else float('inf')
            roh = roh_upper_bound(delta_bar, q_avg)
            will_answer = True
            gating_mode = "prior_sufficiency"
            rationale = (
                f"[prior-sufficiency] q_lo={q_conservative:.3f} and P*={P_star:.3f} ≥ {target:.3f}; "
                f"B2T=0.0000, ISR=∞. Δ̄={delta_bar:.4f} nats; EDFL RoH bound={roh:.3f}; "
                f"H_sem={H_sem:.3f}; top='{C_star_label}' P_full={P_star:.3f}; "
                f"entail_calls_used={(budgeted_entailer.calls)}/{self.max_entail_calls}"
            )
        else:
            # EDFL branch with **posterior‑dominance latch**
            b2t = bits_to_trust(q_conservative, h_star)
            isr_val = isr(delta_bar, b2t)
            roh = roh_upper_bound(delta_bar, q_avg)
            will_answer = (isr_val >= isr_threshold) and (delta_bar >= b2t + max(0.0, margin_extra_bits)) and (not refusal_flag) and (P_star >= target)
            rationale = (
                f"Δ̄={delta_bar:.4f} nats (over meanings), B2T={b2t:.4f}, ISR={isr_val:.3f} "
                f"(thr={isr_threshold:.3f}), extra_bits={margin_extra_bits:.3f}; "
                f"EDFL RoH bound={roh:.3f}; H_sem={H_sem:.3f}; top='{C_star_label}' P_full={P_star:.3f}; "
                f"entail_calls_used={(budgeted_entailer.calls)}/{self.max_entail_calls}"
            )
            if refusal_flag:
                rationale += " [blocked: refusal-like top meaning]"
            if P_star < target:
                rationale += f" [blocked: P* < {target:.3f}]"

        meta = {
            "P_full": P_full,
            "S_list": S_list,
            "S_star_list": S_star_list,
            "clusters": [dataclasses.asdict(c) for c in clusters],
            "labels": labels,
            "closed_book": closed_book,
            "q_floor": floor,
            "policy": policy,
            "use_llr": self.use_llr,
            "entail_calls_used": getattr(budgeted_entailer, "calls", 0),
            "gating_mode": gating_mode,
            "refusal_like": refusal_flag,
            "max_workers_skeletons": self.max_workers_skeletons,
        }

        return ItemMetrics(
            item_id=idx,
            delta_bar=delta_bar,
            q_avg=q_avg,
            q_conservative=q_conservative,
            b2t=b2t,
            isr=isr_val,
            roh_bound=roh,
            decision_answer=will_answer,
            rationale=rationale,
            H_sem=H_sem,
            top_meaning=C_star_label,
            P_full_top=P_star,
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
        max_workers_items: int = 1,
    ) -> List[ItemMetrics]:
        """
        Evaluate a list of items. Set `max_workers_items>1` to evaluate several items
        concurrently (each item already parallelizes skeleton sampling).
        """
        if max_workers_items <= 1 or len(items) <= 1:
            return [self.evaluate_item(i, it, h_star=h_star, isr_threshold=isr_threshold, margin_extra_bits=margin_extra_bits) for i, it in enumerate(items)]

        out: List[Optional[ItemMetrics]] = [None for _ in items]
        def _job(i: int, it: OpenAIItem):
            out[i] = self.evaluate_item(i, it, h_star=h_star, isr_threshold=isr_threshold, margin_extra_bits=margin_extra_bits)

        with ThreadPoolExecutor(max_workers=max_workers_items) as ex:
            futs = [ex.submit(_job, i, it) for i, it in enumerate(items)]
            for _ in as_completed(futs):
                pass
        return [m for m in out if m is not None]

# ---- end of module ----
