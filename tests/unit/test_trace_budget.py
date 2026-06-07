from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from berry.hallucination_detector import trace_budget as tb
from berry.hallucination_detector.backends.base import BackendConfig
from berry.hallucination_detector.backends.openai_backend import TextResult
from berry.hallucination_detector.core import Span, Step, Trace


class ScriptedBackend:
    def __init__(self, responses: Sequence[TextResult]):
        self.responses = list(responses)
        self.batches: list[list[str]] = []
        self.kwargs: list[dict[str, Any]] = []
        self.reset_count = 0

    def call_text_batch(self, *, prompts: Sequence[str], **kwargs: Any) -> list[TextResult]:
        self.batches.append(list(prompts))
        self.kwargs.append(dict(kwargs))
        if len(self.responses) < len(prompts):
            raise AssertionError("not enough scripted verifier responses")
        out = self.responses[: len(prompts)]
        self.responses = self.responses[len(prompts) :]
        return out

    def reset_state(self) -> None:
        self.reset_count += 1


@dataclass
class TraceFixture:
    trace: Trace


@pytest.fixture(autouse=True)
def clear_trace_budget_cache() -> None:
    tb.clear_verifier_cache()
    yield
    tb.clear_verifier_cache()


def _text_result(*, yes: float, no: float | None = None, unsure: float | None = None, token: str = "YES") -> TextResult:
    if no is None or unsure is None:
        other = (1.0 - yes) / 2.0
        no = other if no is None else no
        unsure = other if unsure is None else unsure
    probs = {"YES": yes, "NO": float(no), "UNSURE": float(unsure)}
    # Normalize defensively for hand-written fixtures.
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    top = [
        {"token": label, "logprob": math.log(max(prob, 1e-12))}
        for label, prob in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    ]
    generated = token
    return TextResult(
        text=generated,
        response_id="scripted",
        logprobs=[
            {
                "token": generated,
                "logprob": math.log(max(probs[generated], 1e-12)),
                "top_logprobs": top,
            }
        ],
    )


def _group_text_result(labels: Sequence[Any]) -> TextResult:
    """Return logprobs for a grouped Y/N/U verifier response.

    Each tuple is (generated_label, p_yes, p_no, p_unsure), where generated_label
    may be YES/NO/UNSURE. The grouped prompt asks for Y/N/U to keep each answer
    on one token, while the scorer canonicalizes those aliases back to labels.
    """

    alias = {"YES": "Y", "NO": "N", "UNSURE": "U"}
    token_probs = {"YES": "Y", "NO": "N", "UNSURE": "U"}
    rows: list[dict[str, Any]] = []
    text_tokens: list[str] = []
    for i, item in enumerate(labels):
        if isinstance(item, dict):
            label_raw = item.get("token", item.get("label", ""))
            yes = item.get("yes", item.get("p_yes", 0.0))
            no = item.get("no", item.get("p_no", 0.0))
            unsure = item.get("unsure", item.get("p_unsure", 0.0))
        else:
            label_raw, yes, no, unsure = item
        label = str(label_raw).strip().upper()
        if label not in alias:
            raise ValueError(f"unknown label in grouped fixture: {label_raw!r}")
        if i:
            rows.append({"token": "\n", "logprob": 0.0, "top_logprobs": []})
            text_tokens.append("\n")
        probs = {"YES": float(yes), "NO": float(no), "UNSURE": float(unsure)}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        token = alias[label]
        text_tokens.append(token)
        rows.append(
            {
                "token": token,
                "logprob": math.log(max(probs[label], 1e-12)),
                "top_logprobs": [
                    {"token": token_probs[top_label], "logprob": math.log(max(prob, 1e-12))}
                    for top_label, prob in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                ],
            }
        )
    return TextResult(text="".join(text_tokens), response_id="scripted-group", logprobs=rows)


def _trace(claim: str = "The service supports Gemini.", cites: list[str] | None = None) -> Trace:
    return Trace(
        steps=[Step(idx=0, claim=claim, cites=["S0"] if cites is None else cites, confidence=0.95)],
        spans=[Span(sid="S0", text="The service supports Gemini.")],
    )


def _score(monkeypatch, backend: ScriptedBackend, trace: Trace, **kwargs: Any) -> list[tb.BudgetResult]:
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    return tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        default_target=0.95,
        context_mode="cited",
        top_logprobs=5,
        use_cache=False,
        **kwargs,
    )


def test_interval_budget_is_target_directed_not_symmetric_kl() -> None:
    required_min, required_max, observed_min, observed_max, gap_min, gap_max, flagged = tb._budget_from_intervals(
        target=0.95,
        p0_lo=0.99,
        p0_hi=0.99,
        p1_lo=0.01,
        p1_hi=0.01,
    )

    assert required_min > 0
    assert observed_max > required_max
    assert gap_min < 0  # legacy KL-only scoring would have passed this case.
    assert gap_max < 0
    assert flagged is True


def test_contradicted_claim_fails_after_posterior_gate(monkeypatch) -> None:
    backend = ScriptedBackend([_text_result(yes=0.01, no=0.98, unsure=0.01, token="NO")])

    [res] = _score(monkeypatch, backend, _trace())

    assert res.flagged is True
    assert res.status == "contradicted"
    assert res.prior_skipped is True
    assert res.verifier_calls == 1
    assert len(backend.batches) == 1
    assert backend.kwargs[0]["max_output_tokens"] == 1


def test_prior_leak_fails_even_when_posterior_is_high(monkeypatch) -> None:
    backend = ScriptedBackend([
        _text_result(yes=0.97, no=0.01, unsure=0.02, token="YES"),
        _text_result(yes=0.96, no=0.02, unsure=0.02, token="YES"),
    ])

    [res] = _score(monkeypatch, backend, _trace())

    assert res.flagged is True
    assert res.status == "prior_leak"
    assert "prior_at_or_above_target" in res.reasons
    assert res.post_supports_target is True
    assert res.prior_below_target is False
    assert len(backend.batches) == 2


def test_supported_evidence_passes(monkeypatch) -> None:
    backend = ScriptedBackend([
        _text_result(yes=0.97, no=0.01, unsure=0.02, token="YES"),
        _text_result(yes=0.05, no=0.10, unsure=0.85, token="UNSURE"),
    ])

    [res] = _score(monkeypatch, backend, _trace())

    assert res.flagged is False
    assert res.status == "passed"
    assert res.post_supports_target is True
    assert res.prior_below_target is True
    assert res.evidence_log_odds_gain_min > 0


def test_cited_context_excludes_uncited_spans(monkeypatch) -> None:
    backend = ScriptedBackend([_text_result(yes=0.10, no=0.05, unsure=0.85, token="UNSURE")])
    trace = Trace(
        steps=[Step(idx=0, claim="The service supports Gemini.", cites=["S0"], confidence=0.95)],
        spans=[
            Span(sid="S0", text="The service supports Gemini."),
            Span(sid="S1", text="This uncited span must not be visible to the verifier."),
        ],
    )

    _score(monkeypatch, backend, trace)

    assert "This uncited span" not in backend.batches[0][0]
    assert 'id="S0"' in backend.batches[0][0]
    assert 'id="S1"' not in backend.batches[0][0]


def test_missing_citations_skip_verifier(monkeypatch) -> None:
    backend = ScriptedBackend([])
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = _trace(cites=[])

    [res] = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        require_citations=True,
        context_mode="cited",
        use_cache=False,
    )

    assert res.flagged is True
    assert res.status == "missing_citations"
    assert res.skipped_verifier is True
    assert res.verifier_calls == 0
    assert backend.batches == []


def test_unknown_citations_skip_verifier(monkeypatch) -> None:
    backend = ScriptedBackend([])
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = _trace(cites=["S404"])

    [res] = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        context_mode="all",
        use_cache=False,
    )

    assert res.flagged is True
    assert res.status == "unknown_citations"
    assert res.unknown_citations == ["S404"]
    assert res.skipped_verifier is True
    assert backend.batches == []


def test_empty_cited_context_skip_verifier_when_citations_are_optional(monkeypatch) -> None:
    backend = ScriptedBackend([])
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = _trace(cites=[])

    [res] = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        require_citations=False,
        context_mode="cited",
        use_cache=False,
    )

    assert res.flagged is True
    assert res.status == "empty_context"
    assert res.empty_context is True
    assert res.skipped_verifier is True
    assert backend.batches == []


def test_identical_prompts_are_deduplicated_in_cache(monkeypatch) -> None:
    tb.clear_verifier_cache()
    backend = ScriptedBackend([
        _text_result(yes=0.97, no=0.01, unsure=0.02, token="YES"),
        _text_result(yes=0.05, no=0.10, unsure=0.85, token="UNSURE"),
    ])
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = Trace(
        steps=[
            Step(idx=0, claim="The service supports Gemini.", cites=["S0"], confidence=0.95),
            Step(idx=1, claim="The service supports Gemini.", cites=["S0"], confidence=0.95),
        ],
        spans=[Span(sid="S0", text="The service supports Gemini.")],
    )

    results = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        context_mode="cited",
        top_logprobs=5,
        use_cache=True,
        group_claims=False,
    )

    assert [r.status for r in results] == ["passed", "passed"]
    assert len(backend.batches) == 2
    assert len(backend.batches[0]) == 1
    assert len(backend.batches[1]) == 1

    # Second identical audit is served entirely from cache.
    again = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        context_mode="cited",
        top_logprobs=5,
        use_cache=True,
        group_claims=False,
    )
    assert [r.status for r in again] == ["passed", "passed"]
    assert len(backend.batches) == 2
    tb.clear_verifier_cache()




def test_grouped_claims_same_context_reduce_api_calls(monkeypatch) -> None:
    backend = ScriptedBackend(
        [
            _group_text_result([
                ("YES", 0.97, 0.01, 0.02),
                ("YES", 0.98, 0.01, 0.01),
            ]),
            _group_text_result([
                ("UNSURE", 0.04, 0.08, 0.88),
                ("UNSURE", 0.03, 0.07, 0.90),
            ]),
        ]
    )
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = Trace(
        steps=[
            Step(idx=0, claim="The service supports Gemini.", cites=["S0"], confidence=0.95),
            Step(idx=1, claim="The service supports Vertex.", cites=["S0"], confidence=0.95),
        ],
        spans=[Span(sid="S0", text="The service supports Gemini and Vertex.")],
    )

    results = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        context_mode="cited",
        use_cache=False,
        group_claims=True,
        max_group_size=8,
    )

    assert [res.status for res in results] == ["passed", "passed"]
    assert len(backend.batches) == 2
    assert [len(batch) for batch in backend.batches] == [1, 1]
    assert "CLAIMS, in order" in backend.batches[0][0]
    assert all(res.post_grouped for res in results)
    assert all(res.prior_grouped for res in results)
    assert [res.post_group_size for res in results] == [2, 2]
    assert [res.prior_group_size for res in results] == [2, 2]
    assert sum(res.verifier_calls for res in results) == 4
    assert sum(res.verifier_api_call_share for res in results) == pytest.approx(2.0)


def test_grouped_posterior_gate_only_runs_prior_for_supported_claims(monkeypatch) -> None:
    backend = ScriptedBackend(
        [
            _group_text_result([
                ("YES", 0.97, 0.01, 0.02),
                ("NO", 0.01, 0.98, 0.01),
            ]),
            _text_result(yes=0.05, no=0.10, unsure=0.85, token="UNSURE"),
        ]
    )
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = Trace(
        steps=[
            Step(idx=0, claim="The service supports Gemini.", cites=["S0"], confidence=0.95),
            Step(idx=1, claim="The service supports Claude.", cites=["S0"], confidence=0.95),
        ],
        spans=[Span(sid="S0", text="The service supports Gemini and does not support Claude.")],
    )

    results = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        context_mode="cited",
        use_cache=False,
        group_claims=True,
    )

    assert [res.status for res in results] == ["passed", "contradicted"]
    assert results[0].post_grouped is True
    assert results[0].prior_grouped is False
    assert results[1].post_grouped is True
    assert results[1].prior_skipped is True
    assert len(backend.batches) == 2
    assert [len(batch) for batch in backend.batches] == [1, 1]
    assert sum(res.verifier_api_call_share for res in results) == pytest.approx(2.0)


def test_grouped_parse_failure_falls_back_to_single_claim_prompts(monkeypatch) -> None:
    backend = ScriptedBackend(
        [
            # The grouped posterior response has one answer for two claims, so it
            # must not be trusted; the scorer should retry both claims singly.
            _text_result(yes=0.97, no=0.01, unsure=0.02, token="YES"),
            _text_result(yes=0.97, no=0.01, unsure=0.02, token="YES"),
            _text_result(yes=0.98, no=0.01, unsure=0.01, token="YES"),
            _group_text_result([
                ("UNSURE", 0.04, 0.08, 0.88),
                ("UNSURE", 0.03, 0.07, 0.90),
            ]),
        ]
    )
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = Trace(
        steps=[
            Step(idx=0, claim="The service supports Gemini.", cites=["S0"], confidence=0.95),
            Step(idx=1, claim="The service supports Vertex.", cites=["S0"], confidence=0.95),
        ],
        spans=[Span(sid="S0", text="The service supports Gemini and Vertex.")],
    )

    results = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        context_mode="cited",
        use_cache=False,
        group_claims=True,
    )

    assert [res.status for res in results] == ["passed", "passed"]
    assert [len(batch) for batch in backend.batches] == [1, 2, 1]
    assert all(res.group_fallback for res in results)
    assert all(res.group_fallback_reason for res in results)
    assert all(not res.post_grouped for res in results)
    assert all(res.prior_grouped for res in results)
    assert sum(res.verifier_api_call_share for res in results) == pytest.approx(4.0)


def test_interval_budget_is_directional_for_contradictions() -> None:
    *_, flagged = tb._budget_from_intervals(
        target=0.95,
        p0_lo=0.99,
        p0_hi=0.99,
        p1_lo=0.01,
        p1_hi=0.01,
    )

    assert flagged is True


def test_unknown_citations_take_precedence_over_missing_citations(monkeypatch) -> None:
    backend = ScriptedBackend([])
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)
    trace = Trace(
        steps=[
            Step(
                idx=0,
                claim="The service supports Gemini.",
                cites=[],
                confidence=0.95,
                unknown_citations=["S404"],
            )
        ],
        spans=[Span(sid="S0", text="The service supports Gemini.")],
    )

    [res] = tb.score_trace_budget(
        trace=trace,
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        require_citations=True,
        context_mode="cited",
        use_cache=False,
    )

    assert res.status == "unknown_citations"
    assert res.unknown_citations == ["S404"]
    assert res.skipped_verifier is True
    assert backend.batches == []


def test_score_trace_budget_accepts_dict_like_trace(monkeypatch) -> None:
    backend = ScriptedBackend([
        _text_result(yes=0.97, no=0.01, unsure=0.02, token="YES"),
        _text_result(yes=0.05, no=0.10, unsure=0.85, token="UNSURE"),
    ])
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)

    [res] = tb.score_trace_budget(
        trace={
            "steps": [
                {
                    "idx": 7,
                    "claim": "The service supports Gemini.",
                    "cites": ["S0"],
                    "confidence": 0.95,
                }
            ],
            "spans": [{"sid": "S0", "text": "The service supports Gemini."}],
        },
        verifier_model="verifier",
        backend_cfg=BackendConfig(kind="scripted"),
        context_mode="cited",
        use_cache=False,
    )

    assert res.idx == 7
    assert res.status == "passed"
    assert backend.batches


def test_group_claims_false_preserves_single_claim_prompt_shape(monkeypatch) -> None:
    backend = ScriptedBackend(
        [
            _text_result(yes=0.97, no=0.01, unsure=0.02, token="YES"),
            _text_result(yes=0.98, no=0.01, unsure=0.01, token="YES"),
            _text_result(yes=0.05, no=0.10, unsure=0.85, token="UNSURE"),
            _text_result(yes=0.04, no=0.10, unsure=0.86, token="UNSURE"),
        ]
    )
    trace = Trace(
        steps=[
            Step(idx=0, claim="The service supports Gemini.", cites=["S0"], confidence=0.95),
            Step(idx=1, claim="The service supports OpenAI.", cites=["S0"], confidence=0.95),
        ],
        spans=[Span(sid="S0", text="The service supports Gemini and OpenAI.")],
    )

    results = _score(monkeypatch, backend, trace, group_claims=False)

    assert [result.status for result in results] == ["passed", "passed"]
    assert [len(batch) for batch in backend.batches] == [2, 2]
    assert all("CLAIMS, in order" not in prompt for batch in backend.batches for prompt in batch)
    assert all(not result.grouped_verifier for result in results)
    assert all(result.verifier_api_call_share == pytest.approx(2.0) for result in results)


def test_invalid_grouping_parameters_fail_closed_before_backend(monkeypatch) -> None:
    backend = ScriptedBackend([])
    monkeypatch.setattr(tb, "make_backend", lambda _cfg: backend)

    with pytest.raises(ValueError, match="max_group_size"):
        tb.score_trace_budget(
            trace=_trace(),
            verifier_model="verifier",
            backend_cfg=BackendConfig(kind="scripted"),
            context_mode="cited",
            use_cache=False,
            max_group_size=0,
        )

    with pytest.raises(ValueError, match="max_group_prompt_chars"):
        tb.score_trace_budget(
            trace=_trace(),
            verifier_model="verifier",
            backend_cfg=BackendConfig(kind="scripted"),
            context_mode="cited",
            use_cache=False,
            max_group_prompt_chars=999,
        )

    assert backend.batches == []
