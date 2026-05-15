"""Tests for _normalize_spans auto-sid + loud-failure behavior."""

from __future__ import annotations

import pytest

from berry.hallucination_detector.core import (
    Span,
    _normalize_spans,
    run_detect_hallucination,
)


def test_normalize_spans_empty_list_returns_empty() -> None:
    assert _normalize_spans([]) == []


def test_normalize_spans_none_returns_empty() -> None:
    assert _normalize_spans(None) == []  # type: ignore[arg-type]


def test_normalize_spans_explicit_sid_and_text_preserved() -> None:
    spans = _normalize_spans([{"sid": "s1", "text": "x"}])
    assert len(spans) == 1
    assert isinstance(spans[0], Span)
    assert spans[0].sid == "s1"
    assert spans[0].text == "x"


def test_normalize_spans_auto_assigns_sid_when_missing() -> None:
    spans = _normalize_spans([{"text": "x"}])
    assert len(spans) == 1
    assert spans[0].sid == "s1"
    assert spans[0].text == "x"


def test_normalize_spans_auto_sid_is_sequential() -> None:
    spans = _normalize_spans(
        [
            {"text": "alpha"},
            {"text": "beta"},
            {"text": "gamma"},
        ]
    )
    assert [s.sid for s in spans] == ["s1", "s2", "s3"]
    assert [s.text for s in spans] == ["alpha", "beta", "gamma"]


def test_normalize_spans_snippet_fallback() -> None:
    spans = _normalize_spans([{"snippet": "x"}])
    assert len(spans) == 1
    assert spans[0].sid == "s1"
    assert spans[0].text == "x"


def test_normalize_spans_text_wins_over_snippet() -> None:
    spans = _normalize_spans([{"text": "real", "snippet": "ignored"}])
    assert len(spans) == 1
    assert spans[0].text == "real"


def test_normalize_spans_empty_span_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _normalize_spans([{}])


def test_normalize_spans_whitespace_only_text_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _normalize_spans([{"text": "   "}])


def _use_dummy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BERRY_VERIFIER_BACKEND", "dummy")


def test_run_detect_hallucination_no_spans_returns_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_dummy_backend(monkeypatch)
    result = run_detect_hallucination(answer="Foo.", spans=[])
    assert isinstance(result, dict)
    assert result.get("error_type") == "no_spans"
    assert result.get("flagged") is True
    assert result.get("under_budget") is True
    assert "error" in result
    assert result.get("details") == []


def test_run_detect_hallucination_malformed_spans_returns_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_dummy_backend(monkeypatch)
    result = run_detect_hallucination(answer="Foo.", spans=[{}])
    assert isinstance(result, dict)
    assert result.get("error_type") == "malformed_spans"
    assert result.get("flagged") is True
    assert result.get("under_budget") is True
    assert result.get("details") == []
