from __future__ import annotations

from berry.hallucination_detector.core import run_audit_trace_budget, run_detect_hallucination


def test_detect_missing_citations_fails_without_backend(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "openai")

    out = run_detect_hallucination(
        answer="The service supports Gemini.",
        spans=[{"sid": "S0", "text": "The service supports Gemini."}],
        require_citations=True,
        context_mode="cited",
    )

    assert out["flagged"] is True
    assert "error" not in out
    assert out["details"][0]["status"] == "missing_citations"
    assert out["summary"]["verifier_calls_planned"] == 0


def test_detect_unknown_citation_fails_without_backend(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "openai")

    out = run_detect_hallucination(
        answer="The service supports Gemini [S404].",
        spans=[{"sid": "S0", "text": "The service supports Gemini."}],
        require_citations=False,
        context_mode="all",
    )

    assert out["flagged"] is True
    assert "error" not in out
    assert out["details"][0]["status"] == "unknown_citations"
    assert out["details"][0]["unknown_citations"] == ["S404"]
    assert out["summary"]["verifier_calls_planned"] == 0


def test_detect_reports_truncation_and_strips_cites(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "openai")

    out = run_detect_hallucination(
        answer="A is true. B is true. C is true.",
        spans=[{"sid": "S0", "text": "A is true."}],
        require_citations=True,
        context_mode="cited",
        max_claims=2,
    )

    assert out["summary"]["claims_total"] == 3
    assert out["summary"]["claims_scored"] == 2
    assert out["summary"]["truncated"] is True
    assert out["flagged"] is True
    assert [d["status"] for d in out["details"]] == ["missing_citations", "missing_citations"]


def test_detect_strips_citations_from_claim_text(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "openai")

    out = run_detect_hallucination(
        answer="The service supports Gemini [S404].",
        spans=[{"sid": "S0", "text": "The service supports Gemini."}],
        context_mode="all",
    )

    assert out["details"][0]["claim"] == "The service supports Gemini."


def test_audit_trace_budget_fails_closed_without_spans() -> None:
    out = run_audit_trace_budget(
        steps=[{"claim": "The service supports Gemini.", "cites": ["S0"]}],
        spans=[],
    )

    assert out["flagged"] is True
    assert "error" not in out
    assert out["details"][0]["status"] == "no_spans"
    assert out["details"][0]["no_spans"] is True
    assert out["summary"]["verifier_calls_planned"] == 0


def test_audit_trace_budget_uses_consistent_detail_schema(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "openai")

    out = run_audit_trace_budget(
        steps=[{"claim": "The service supports Gemini.", "cites": []}],
        spans=[{"sid": "S0", "text": "The service supports Gemini."}],
        require_citations=True,
        context_mode="cited",
    )

    detail = out["details"][0]
    assert detail["status"] == "missing_citations"
    assert "prior_yes" in detail
    assert "post_yes" in detail
    assert "evidence_log_odds_gain" in detail
    assert out["summary"]["verifier_calls_planned"] == 0


def test_audit_trace_budget_rejects_duplicate_span_ids_without_backend(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "openai")

    out = run_audit_trace_budget(
        steps=[{"claim": "The service supports Gemini.", "cites": ["S0"]}],
        spans=[
            {"sid": "S0", "text": "The service supports Gemini."},
            {"sid": "S0", "text": "A conflicting duplicate span."},
        ],
    )

    assert out["flagged"] is True
    assert "duplicate span ids" in out["error"]
    assert out["details"] == []


def test_detect_validates_context_mode_and_top_logprobs_before_backend(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "openai")

    bad_context = run_detect_hallucination(
        answer="The service supports Gemini [S0].",
        spans=[{"sid": "S0", "text": "The service supports Gemini."}],
        context_mode="everything",
    )
    bad_topk = run_detect_hallucination(
        answer="The service supports Gemini [S0].",
        spans=[{"sid": "S0", "text": "The service supports Gemini."}],
        top_logprobs=0,
    )

    assert bad_context["flagged"] is True
    assert "context_mode" in bad_context["error"]
    assert bad_context["details"] == []
    assert bad_topk["flagged"] is True
    assert "top_logprobs" in bad_topk["error"]
    assert bad_topk["details"] == []
