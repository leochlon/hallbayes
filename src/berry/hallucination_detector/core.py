from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .backends.base import BackendConfig
from .trace_budget import BudgetResult, score_trace_budget

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_CITE_RE = re.compile(r"\[(?P<id>[A-Za-z]\w*|\d+)\]")
_VALID_CLAIM_SPLITS = {"sentences", "lines"}
_VALID_CONTEXT_MODE_ALIASES = {
    "all": "all",
    "full": "all",
    "auto": "auto",
    "cited": "cited",
    "cite": "cited",
    "citations": "cited",
    "cites": "cited",
}
_LN2 = math.log(2.0)
_DEFAULT_GROUP_CLAIMS = True
_DEFAULT_MAX_GROUP_SIZE = 8
_DEFAULT_MAX_GROUP_PROMPT_CHARS = 24000


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
    raw_cites: Optional[List[str]] = None
    raw_claim: Optional[str] = None
    unknown_citations: List[str] = field(default_factory=list)
    citation_normalizations: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Trace:
    steps: List[Step]
    spans: List[Span]


def _to_bits(nats: float) -> float:
    return float(nats) / _LN2


def _normalize_units(units: str) -> str:
    u = (units or "bits").strip().lower()
    if u not in {"bits", "nats"}:
        raise ValueError("units must be 'bits' or 'nats'")
    return u


def _validate_probability(value: Any, *, name: str) -> float:
    try:
        p = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite probability in (0, 1), got {value!r}") from exc
    if not math.isfinite(p) or not (0.0 < p < 1.0):
        raise ValueError(f"{name} must be a finite probability in (0, 1), got {value!r}")
    return p


def _backend_kind() -> str:
    return (os.environ.get("BERRY_VERIFIER_BACKEND") or "openai").strip().lower()


def _normalize_spans(spans: List[Dict[str, str]]) -> List[Span]:
    out: List[Span] = []
    for raw in spans or []:
        if not isinstance(raw, dict):
            continue
        sid = str(raw.get("sid", "")).strip()
        text = str(raw.get("text", "")).strip()
        if not sid or not text:
            continue
        out.append(Span(sid=sid, text=text))
    return out


def _duplicate_span_ids(spans: Sequence[Span]) -> List[str]:
    seen: Set[str] = set()
    duplicates: List[str] = []
    for span in spans:
        if span.sid in seen and span.sid not in duplicates:
            duplicates.append(span.sid)
        seen.add(span.sid)
    return duplicates


def _dedupe_preserving_order(items: Sequence[Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        s = str(item).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _map_cites_with_normalizations(
    raw_cites: Sequence[Any], known: Set[str]
) -> Tuple[List[str], List[Dict[str, str]]]:
    mapped: List[str] = []
    normalizations: List[Dict[str, str]] = []
    for raw in raw_cites:
        c = str(raw).strip()
        if not c:
            continue
        resolved = c
        if c in known:
            mapped.append(c)
            continue
        if c.isdigit():
            n = int(c)
            candidates = [c, f"S{n}"]
            if n > 0:
                candidates.append(f"S{n - 1}")
            hit = next((candidate for candidate in candidates if candidate in known), None)
            if hit is not None:
                resolved = hit
        elif c.startswith("S") and c[1:].isdigit():
            tail = c[1:]
            if tail in known:
                resolved = tail
        if resolved != c:
            normalizations.append({"raw": c, "resolved": resolved})
        mapped.append(resolved)
    return _dedupe_preserving_order(mapped), normalizations


def _map_cites_to_known_ids(cites: List[str], known: Set[str]) -> List[str]:
    mapped, _normalizations = _map_cites_with_normalizations(cites, known)
    return mapped


def _normalize_steps(steps: List[Dict[str, Any]], default_target: float, known_ids: Set[str]) -> List[Step]:
    out: List[Step] = []
    for i, raw_step in enumerate(steps or []):
        if not isinstance(raw_step, dict):
            continue
        claim = str(raw_step.get("claim", "")).strip()
        if not claim:
            continue
        idx = int(raw_step.get("idx", i))
        raw_cites = [str(c).strip() for c in (raw_step.get("cites") or []) if str(c).strip()]
        cites, normalizations = _map_cites_with_normalizations(raw_cites, known_ids)
        explicit_unknown = raw_step.get("unknown_citations", raw_step.get("unknown_cites", [])) or []
        unknown = _dedupe_preserving_order(list(explicit_unknown) + [cite for cite in cites if cite not in known_ids])
        conf = _validate_probability(raw_step.get("confidence", default_target) or default_target, name="confidence")
        out.append(
            Step(
                idx=idx,
                claim=claim,
                cites=cites,
                confidence=conf,
                raw_cites=list(raw_cites),
                raw_claim=str(raw_step.get("raw_claim", claim)).strip() or claim,
                unknown_citations=unknown,
                citation_normalizations=normalizations,
            )
        )
    out.sort(key=lambda item: item.idx)
    return out


def _extract_cites(text: str, cite_re: re.Pattern) -> List[str]:
    return [match.group("id") for match in cite_re.finditer(text or "")]


def _strip_citations(text: str, cite_re: re.Pattern) -> str:
    stripped = cite_re.sub("", text or "")
    stripped = re.sub(r"\s+([,.;:!?])", r"\1", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped.strip()


def _split_claims(answer: str, mode: str, max_claims: Optional[int] = None) -> List[str]:
    a = (answer or "").strip()
    if not a:
        return []
    m = (mode or "sentences").strip().lower()
    if m in {"sentence"}:
        m = "sentences"
    if m in {"line"}:
        m = "lines"
    if m not in _VALID_CLAIM_SPLITS:
        raise ValueError(f"claim_split must be one of {sorted(_VALID_CLAIM_SPLITS)}")

    if m == "lines":
        raw = [line.strip() for line in a.splitlines() if line.strip()]
    else:
        raw = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(a) if segment.strip()]

    cite_prefix_re = re.compile(r"^\s*(?:\[(?:[A-Za-z]\w*|\d+)\]\s*)+")
    merged: List[str] = []
    for segment in raw:
        if merged:
            prefix_match = cite_prefix_re.match(segment)
            if prefix_match:
                prefix = segment[: prefix_match.end()].strip()
                rest = segment[prefix_match.end() :].strip()
                if prefix and rest:
                    merged[-1] = (merged[-1] + " " + prefix).strip()
                    segment = rest
        remainder = _DEFAULT_CITE_RE.sub("", segment)
        remainder = re.sub(r"[\s,;:.\-–—!?]+", "", remainder)
        cites_only = (remainder == "") and bool(_DEFAULT_CITE_RE.search(segment))
        if cites_only and merged:
            merged[-1] = (merged[-1] + " " + segment).strip()
        else:
            merged.append(segment)

    if max_claims is None:
        return merged
    return merged[: max(1, int(max_claims))]


def _prob_dict(prob: Any) -> Dict[str, Any]:
    return {
        "p_lower": prob.p_yes_lower,
        "p_upper": prob.p_yes_upper,
        "p_no_lower": getattr(prob, "p_no_lower", 0.0),
        "p_no_upper": getattr(prob, "p_no_upper", 1.0),
        "p_unsure_lower": getattr(prob, "p_unsure_lower", 0.0),
        "p_unsure_upper": getattr(prob, "p_unsure_upper", 1.0),
        "generated": prob.generated,
        "topk": prob.topk,
        "labels": {
            "YES": {"lower": prob.p_yes_lower, "upper": prob.p_yes_upper},
            "NO": {"lower": getattr(prob, "p_no_lower", 0.0), "upper": getattr(prob, "p_no_upper", 1.0)},
            "UNSURE": {
                "lower": getattr(prob, "p_unsure_lower", 0.0),
                "upper": getattr(prob, "p_unsure_upper", 1.0),
            },
        },
    }


def _cap_prompt(text: Optional[str], cap: int, label: str) -> Optional[str]:
    if text is None:
        return None
    out = str(text)
    if len(out) > cap:
        out = out[:cap] + f"\n...[TRUNCATED {label}]"
    return out


def _unit_value(value: Optional[float], units: str) -> Optional[float]:
    if value is None:
        return None
    return _to_bits(float(value)) if units == "bits" else float(value)


def _format_result(
    result: BudgetResult,
    units: str,
    *,
    include_prompts: bool = False,
    max_prompt_chars: int = 3000,
) -> Dict[str, Any]:
    gain_min = _unit_value(getattr(result, "evidence_log_odds_gain_min", None), units)
    gain_max = _unit_value(getattr(result, "evidence_log_odds_gain_max", None), units)
    posterior_supported = bool(getattr(result, "posterior_supported", result.post_supports_target))
    prior_supported = bool(getattr(result, "prior_supported", not result.prior_below_target))
    kl_under_budget = bool(getattr(result, "kl_under_budget", not result.kl_budget_sufficient))
    gain_required = _unit_value(result.min_log_odds_gain, units)
    detail: Dict[str, Any] = {
        "idx": result.idx,
        "claim": result.claim,
        "raw_claim": result.raw_claim if result.raw_claim is not None else result.claim,
        "cites": list(result.cites),
        "raw_cites": list(result.raw_cites),
        "citation_normalizations": list(result.citation_normalizations),
        "target": result.target,
        "status": result.status,
        "reasons": list(result.reasons),
        "prior_yes": _prob_dict(result.prior_yes),
        "post_yes": _prob_dict(result.post_yes),
        "required": {
            "min": _unit_value(result.required_bits_min, units),
            "max": _unit_value(result.required_bits_max, units),
            "units": units,
        },
        "observed": {
            "min": _unit_value(result.observed_bits_min, units),
            "max": _unit_value(result.observed_bits_max, units),
            "units": units,
        },
        "budget_gap": {
            "min": _unit_value(result.budget_gap_min, units),
            "max": _unit_value(result.budget_gap_max, units),
            "units": units,
        },
        "evidence_gain": {"min": gain_min, "max": gain_max, "required_min": gain_required, "units": units},
        "evidence_log_odds_gain": {"min": gain_min, "max": gain_max, "required_min": gain_required, "units": units},
        "flagged": bool(result.flagged),
        "has_any_citations": bool(result.raw_cites or result.cites),
        "missing_citations": bool(result.missing_citations),
        "unknown_citations": list(result.unknown_citations),
        "empty_context": bool(result.empty_context),
        "no_spans": bool(result.no_spans),
        "skipped_verifier": bool(result.skipped_verifier),
        "verification_skipped": bool(result.verification_skipped),
        "verifier_calls": int(result.verifier_calls),
        "verification_calls": int(result.verifier_calls),
        "verifier_api_call_share": float(getattr(result, "verifier_api_call_share", result.verifier_calls)),
        "verification_api_call_share": float(getattr(result, "verifier_api_call_share", result.verifier_calls)),
        "grouped_verifier": bool(getattr(result, "grouped_verifier", False)),
        "post_grouped": bool(getattr(result, "post_grouped", False)),
        "prior_grouped": bool(getattr(result, "prior_grouped", False)),
        "post_group_size": int(getattr(result, "post_group_size", 1)),
        "prior_group_size": int(getattr(result, "prior_group_size", 0)),
        "group_fallback": bool(getattr(result, "group_fallback", False)),
        "prior_skipped": bool(result.prior_skipped),
        "post_supports_target": bool(result.post_supports_target),
        "posterior_supported": posterior_supported,
        "prior_below_target": bool(result.prior_below_target),
        "prior_supported": prior_supported,
        "evidence_dependent": bool(result.evidence_dependent),
        "prior_leak": bool(result.prior_leak),
        "kl_budget_sufficient": bool(result.kl_budget_sufficient),
        "kl_under_budget": kl_under_budget,
        "budget_evaluated": bool(not result.skipped_verifier and not result.prior_skipped and result.error is None),
        "min_log_odds_gain": _unit_value(result.min_log_odds_gain, units),
    }
    fallback_reason = getattr(result, "group_fallback_reason", None)
    if fallback_reason:
        detail["group_fallback_reason"] = str(fallback_reason)[:2000]
    if result.error:
        detail["error"] = result.error

    if include_prompts:
        cap = max(100, int(max_prompt_chars or 3000))
        prior = _cap_prompt(result.prior_prompt, cap, "prior_prompt")
        post = _cap_prompt(result.post_prompt, cap, "post_prompt")
        if prior is not None:
            detail["prior_prompt"] = prior
        if post is not None:
            detail["post_prompt"] = post
        post_group = _cap_prompt(getattr(result, "post_group_prompt", None), cap, "post_group_prompt")
        prior_group = _cap_prompt(getattr(result, "prior_group_prompt", None), cap, "prior_group_prompt")
        if post_group is not None:
            detail["post_group_prompt"] = post_group
        if prior_group is not None:
            detail["prior_group_prompt"] = prior_group
    return detail


def _error_response(*, message: str, units: str = "bits", verifier_model: str = "", backend: str = "") -> Dict[str, Any]:
    return {
        "flagged": True,
        "under_budget": True,
        "error": message,
        "summary": {
            "units": units,
            "verifier_model": verifier_model,
            "backend": backend,
            "flagged_idxs": [],
            "verifier_calls": 0,
            "verifier_calls_planned": 0,
            "verification_calls": 0,
            "verification_calls_planned": 0,
            "verifier_api_calls_estimated": 0,
            "verifier_api_calls_planned": 0,
            "verification_api_calls_estimated": 0,
            "verification_api_calls_planned": 0,
            "verifier_calls_saved_by_grouping": 0,
            "grouped_results": 0,
            "group_fallbacks": 0,
        },
        "details": [],
    }


def _summary(
    *,
    details: List[Dict[str, Any]],
    units: str,
    verifier_model: str,
    backend_kind: str,
    context_mode: str,
    require_citations: bool,
    min_log_odds_gain: float,
    score_key: str,
    total_key: str,
    total: int,
    max_claims: Optional[int] = None,
    truncated: Optional[bool] = None,
    group_claims: Optional[bool] = None,
    max_group_size: Optional[int] = None,
    max_group_prompt_chars: Optional[int] = None,
) -> Dict[str, Any]:
    flagged_idxs = [detail["idx"] for detail in details if detail["flagged"]]
    statuses: Dict[str, int] = {}
    for detail in details:
        status = str(detail.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    logical_calls = sum(int(detail.get("verifier_calls", 0)) for detail in details)
    api_call_share = sum(float(detail.get("verifier_api_call_share", detail.get("verifier_calls", 0))) for detail in details)
    out: Dict[str, Any] = {
        total_key: int(total),
        score_key: len(details),
        "flagged_claims" if score_key == "claims_scored" else "flagged_steps": len(flagged_idxs),
        "flagged_idxs": flagged_idxs[:50],
        "units": units,
        "verifier_model": verifier_model,
        "backend": backend_kind,
        "context_mode": context_mode,
        "require_citations": bool(require_citations),
        "min_log_odds_gain": float(min_log_odds_gain),
        "verifier_calls": logical_calls,
        "verifier_calls_planned": logical_calls,
        "verification_calls": logical_calls,
        "verification_calls_planned": logical_calls,
        "verifier_api_calls_estimated": round(api_call_share, 6),
        "verifier_api_calls_planned": round(api_call_share, 6),
        "verification_api_calls_estimated": round(api_call_share, 6),
        "verification_api_calls_planned": round(api_call_share, 6),
        "verifier_calls_saved_by_grouping": round(max(0.0, float(logical_calls) - api_call_share), 6),
        "grouped_results": sum(1 for detail in details if detail.get("grouped_verifier")),
        "group_fallbacks": sum(1 for detail in details if detail.get("group_fallback")),
        "statuses": statuses,
    }
    if max_claims is not None:
        out["max_claims"] = int(max_claims)
    if group_claims is not None:
        out["group_claims"] = bool(group_claims)
    if max_group_size is not None:
        out["max_group_size"] = int(max_group_size)
    if max_group_prompt_chars is not None:
        out["max_group_prompt_chars"] = int(max_group_prompt_chars)
    if truncated is not None:
        out["truncated"] = bool(truncated)
    return out


def _top_level_verifier_error(details: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Preserve the legacy top-level error contract when every scored item failed in the verifier layer."""

    if not details or any(str(detail.get("status")) != "verifier_error" for detail in details):
        return None
    messages: List[str] = []
    for detail in details:
        if detail.get("error"):
            messages.append(str(detail["error"]))
            continue
        for reason in detail.get("reasons") or []:
            if reason:
                messages.append(str(reason))
                break
    joined = "; ".join(messages) if messages else "all verifier calls failed"
    return joined[:2000]


def _normalize_context_mode(mode: str) -> str:
    m = str(mode or "cited").strip().lower()
    if m not in _VALID_CONTEXT_MODE_ALIASES:
        allowed = ", ".join(sorted(_VALID_CONTEXT_MODE_ALIASES))
        raise ValueError(f"context_mode must be one of: {allowed}")
    return _VALID_CONTEXT_MODE_ALIASES[m]


def _validate_top_logprobs(value: Any) -> int:
    try:
        k = int(value)
    except Exception as exc:
        raise ValueError(f"top_logprobs must be an integer in [1, 20], got {value!r}") from exc
    if not (1 <= k <= 20):
        raise ValueError(f"top_logprobs must be an integer in [1, 20], got {value!r}")
    return k


def _validate_common(
    *,
    units: str,
    default_target: float,
    context_mode: str,
    top_logprobs: int,
    max_concurrency: int,
    min_log_odds_gain: float,
) -> Tuple[str, float, str, int, int, float]:
    units = _normalize_units(units)
    default_target = _validate_probability(default_target, name="default_target")
    context_mode = _normalize_context_mode(context_mode)
    top_logprobs = _validate_top_logprobs(top_logprobs)
    try:
        max_concurrency = int(max_concurrency or 1)
    except Exception as exc:
        raise ValueError(f"max_concurrency must be an integer in [1, 64], got {max_concurrency!r}") from exc
    if not (1 <= max_concurrency <= 64):
        raise ValueError(f"max_concurrency must be an integer in [1, 64], got {max_concurrency!r}")
    min_log_odds_gain = float(min_log_odds_gain or 0.0)
    if not math.isfinite(min_log_odds_gain):
        raise ValueError("min_log_odds_gain must be finite")
    return units, default_target, context_mode, top_logprobs, max_concurrency, min_log_odds_gain


def run_detect_hallucination(
    *,
    answer: str,
    spans: List[Dict[str, str]],
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,
    placeholder: str = "[CITED_EVIDENCE_REMOVED]",
    max_claims: int = 25,
    claim_split: str = "sentences",
    citation_regex: Optional[str] = None,
    temperature: float = 0.0,
    top_logprobs: int = 5,
    max_concurrency: int = 8,
    timeout_s: Optional[float] = 30.0,
    units: str = "bits",
    context_mode: str = "cited",
    require_citations: bool = False,
    include_prompts: bool = False,
    max_prompt_chars: int = 3000,
    min_log_odds_gain: float = 0.0,
    use_cache: bool = True,
    group_claims: bool = _DEFAULT_GROUP_CLAIMS,
    max_group_size: int = _DEFAULT_MAX_GROUP_SIZE,
    max_group_prompt_chars: int = _DEFAULT_MAX_GROUP_PROMPT_CHARS,
    pool_json_path: Optional[str] = None,
    local_llm_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    backend_kind = _backend_kind()
    try:
        units, default_target, context_mode, top_logprobs, max_concurrency, min_log_odds_gain = _validate_common(
            units=units,
            default_target=default_target,
            context_mode=context_mode,
            top_logprobs=top_logprobs,
            max_concurrency=max_concurrency,
            min_log_odds_gain=min_log_odds_gain,
        )
        max_claims_eff = max(1, int(max_claims or 25))
    except Exception as exc:
        return _error_response(message=str(exc), units=str(units or "bits"), verifier_model=verifier_model, backend=backend_kind)

    if pool_json_path or local_llm_model_path:
        return _error_response(
            message="AOAI pool/local LLM verifier paths are not implemented in this runtime; configure BERRY_VERIFIER_BACKEND instead.",
            units=units,
            verifier_model=verifier_model,
            backend=backend_kind,
        )

    try:
        cite_re = re.compile(citation_regex) if citation_regex else _DEFAULT_CITE_RE
        if "id" not in cite_re.groupindex:
            raise ValueError("citation_regex must define a named capture group called 'id'")
        span_objs = _normalize_spans(spans)
        duplicates = _duplicate_span_ids(span_objs)
        if duplicates:
            raise ValueError(f"duplicate span ids are not allowed: {', '.join(duplicates)}")
        known_ids = {span.sid for span in span_objs}
        all_claims = _split_claims(answer, mode=claim_split, max_claims=None)
    except Exception as exc:
        return _error_response(message=str(exc), units=units, verifier_model=verifier_model, backend=backend_kind)

    claims = all_claims[:max_claims_eff]
    truncated = len(all_claims) > len(claims)
    steps: List[Step] = []
    for idx, raw_claim in enumerate(claims):
        raw_cites = _extract_cites(raw_claim, cite_re=cite_re)
        cites, normalizations = _map_cites_with_normalizations(raw_cites, known_ids)
        claim = _strip_citations(raw_claim, cite_re=cite_re) or raw_claim.strip()
        unknown = [cite for cite in cites if cite not in known_ids]
        steps.append(
            Step(
                idx=idx,
                claim=claim,
                cites=cites,
                confidence=default_target,
                raw_cites=raw_cites,
                raw_claim=raw_claim,
                unknown_citations=unknown,
                citation_normalizations=normalizations,
            )
        )

    trace = Trace(steps=steps, spans=span_objs)
    cfg = BackendConfig(kind=backend_kind, max_concurrency=max_concurrency, timeout_s=timeout_s)
    try:
        results = score_trace_budget(
            trace=trace,
            verifier_model=verifier_model,
            backend_cfg=cfg,
            default_target=default_target,
            temperature=float(temperature),
            top_logprobs=top_logprobs,
            placeholder=str(placeholder),
            context_mode=context_mode,
            require_citations=bool(require_citations),
            min_log_odds_gain=float(min_log_odds_gain),
            include_prompts=bool(include_prompts),
            use_cache=bool(use_cache),
            group_claims=bool(group_claims),
            max_group_size=max_group_size,
            max_group_prompt_chars=max_group_prompt_chars,
            reasoning=None,
        )
    except Exception as exc:
        return _error_response(message=str(exc), units=units, verifier_model=verifier_model, backend=backend_kind)

    details = [
        _format_result(result, units, include_prompts=include_prompts, max_prompt_chars=max_prompt_chars)
        for result in results
    ]
    flagged = any(detail["flagged"] for detail in details) or bool(truncated)
    response = {
        "flagged": flagged,
        "under_budget": flagged,
        "summary": _summary(
            details=details,
            units=units,
            verifier_model=verifier_model,
            backend_kind=backend_kind,
            context_mode=context_mode,
            require_citations=bool(require_citations),
            min_log_odds_gain=float(min_log_odds_gain),
            score_key="claims_scored",
            total_key="claims_total",
            total=len(all_claims),
            max_claims=max_claims_eff,
            truncated=truncated,
            group_claims=bool(group_claims),
            max_group_size=max_group_size,
            max_group_prompt_chars=max_group_prompt_chars,
        ),
        "details": details,
    }
    verifier_error = _top_level_verifier_error(details)
    if verifier_error:
        response["error"] = verifier_error
    return response


def run_audit_trace_budget(
    *,
    steps: List[Dict[str, Any]],
    spans: List[Dict[str, str]],
    verifier_model: str = "gpt-4o-mini",
    default_target: float = 0.95,
    placeholder: str = "[CITED_EVIDENCE_REMOVED]",
    temperature: float = 0.0,
    top_logprobs: int = 5,
    max_concurrency: int = 8,
    timeout_s: Optional[float] = 30.0,
    units: str = "bits",
    context_mode: str = "cited",
    require_citations: bool = False,
    include_prompts: bool = False,
    max_prompt_chars: int = 3000,
    min_log_odds_gain: float = 0.0,
    use_cache: bool = True,
    group_claims: bool = _DEFAULT_GROUP_CLAIMS,
    max_group_size: int = _DEFAULT_MAX_GROUP_SIZE,
    max_group_prompt_chars: int = _DEFAULT_MAX_GROUP_PROMPT_CHARS,
    pool_json_path: Optional[str] = None,
    local_llm_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    backend_kind = _backend_kind()
    try:
        units, default_target, context_mode, top_logprobs, max_concurrency, min_log_odds_gain = _validate_common(
            units=units,
            default_target=default_target,
            context_mode=context_mode,
            top_logprobs=top_logprobs,
            max_concurrency=max_concurrency,
            min_log_odds_gain=min_log_odds_gain,
        )
    except Exception as exc:
        return _error_response(message=str(exc), units=str(units or "bits"), verifier_model=verifier_model, backend=backend_kind)

    if pool_json_path or local_llm_model_path:
        return _error_response(
            message="AOAI pool/local LLM verifier paths are not implemented in this runtime; configure BERRY_VERIFIER_BACKEND instead.",
            units=units,
            verifier_model=verifier_model,
            backend=backend_kind,
        )

    try:
        span_objs = _normalize_spans(spans)
        duplicates = _duplicate_span_ids(span_objs)
        if duplicates:
            raise ValueError(f"duplicate span ids are not allowed: {', '.join(duplicates)}")
        known_ids = {span.sid for span in span_objs}
        step_objs = _normalize_steps(steps, default_target=default_target, known_ids=known_ids)
        trace = Trace(steps=step_objs, spans=span_objs)
        cfg = BackendConfig(kind=backend_kind, max_concurrency=max_concurrency, timeout_s=timeout_s)
        results = score_trace_budget(
            trace=trace,
            verifier_model=verifier_model,
            backend_cfg=cfg,
            default_target=default_target,
            temperature=float(temperature),
            top_logprobs=top_logprobs,
            placeholder=str(placeholder),
            context_mode=context_mode,
            require_citations=bool(require_citations),
            min_log_odds_gain=float(min_log_odds_gain),
            include_prompts=bool(include_prompts),
            use_cache=bool(use_cache),
            group_claims=bool(group_claims),
            max_group_size=max_group_size,
            max_group_prompt_chars=max_group_prompt_chars,
            reasoning=None,
        )
    except Exception as exc:
        return _error_response(message=str(exc), units=units, verifier_model=verifier_model, backend=backend_kind)

    details = [
        _format_result(result, units, include_prompts=include_prompts, max_prompt_chars=max_prompt_chars)
        for result in results
    ]
    flagged = any(detail["flagged"] for detail in details)
    response = {
        "flagged": flagged,
        "under_budget": flagged,
        "summary": _summary(
            details=details,
            units=units,
            verifier_model=verifier_model,
            backend_kind=backend_kind,
            context_mode=context_mode,
            require_citations=bool(require_citations),
            min_log_odds_gain=float(min_log_odds_gain),
            score_key="steps_scored",
            total_key="steps_total",
            total=len(step_objs),
            group_claims=bool(group_claims),
            max_group_size=max_group_size,
            max_group_prompt_chars=max_group_prompt_chars,
        ),
        "details": details,
    }
    verifier_error = _top_level_verifier_error(details)
    if verifier_error:
        response["error"] = verifier_error
    return response
