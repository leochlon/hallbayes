from __future__ import annotations

import json

import httpx
import pytest

from berry.hallucination_detector.backends import vertex_backend
from berry.hallucination_detector.backends.vertex_backend import call_text_chat_vertex


def test_vertex_generate_content_parses_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "YES"}]},
                "logprobsResult": {
                    "chosenCandidates": [{"token": "YES", "tokenId": 42, "logProbability": -0.1}],
                    "topCandidates": [
                        {
                            "candidates": [
                                {"token": "YES", "tokenId": 42, "logProbability": -0.1},
                                {"token": "NO", "tokenId": 43, "logProbability": -2.0},
                            ]
                        }
                    ],
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert (
            str(request.url)
            == "https://aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini-2.0-flash-001:generateContent"
        )
        assert request.headers.get("Authorization") == "Bearer tok"
        body = json.loads(request.content)
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][0]["parts"][0]["text"] == "YES"
        assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
        assert body["generationConfig"]["responseLogprobs"] is True
        assert body["generationConfig"]["logprobs"] == 5
        return httpx.Response(200, json=fixture)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    def fake_get_client(*, timeout_s=None, base_url=None):
        return "https://aiplatform.googleapis.com", client

    monkeypatch.setattr(vertex_backend, "_get_client", fake_get_client)

    res = call_text_chat_vertex(
        prompt="YES",
        model="projects/p/locations/l/publishers/google/models/gemini-2.0-flash-001",
        instructions="SYSTEM",
        temperature=0,
        max_output_tokens=1,
        include_logprobs=True,
        top_logprobs=5,
        access_token="tok",
        base_url="https://aiplatform.googleapis.com",
    )

    assert res.text == "YES"
    assert res.logprobs is not None
    assert res.logprobs[0]["token"] == "YES"
    assert res.logprobs[0]["logprob"] == -0.1
    assert res.logprobs[0]["top_logprobs"][0]["token"] == "YES"
    assert res.logprobs[0]["top_logprobs"][0]["logprob"] == -0.1
    assert res.logprobs[0]["top_logprobs"][1]["token"] == "NO"
    assert res.logprobs[0]["top_logprobs"][1]["logprob"] == -2.0

    client.close()


def test_vertex_generate_content_missing_logprobs_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = {"candidates": [{"content": {"role": "model", "parts": [{"text": "YES"}]}}]}

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=fixture)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    def fake_get_client(*, timeout_s=None, base_url=None):
        return "https://aiplatform.googleapis.com", client

    monkeypatch.setattr(vertex_backend, "_get_client", fake_get_client)

    with pytest.raises(RuntimeError, match="logprobsResult"):
        call_text_chat_vertex(
            prompt="YES",
            model="projects/p/locations/l/publishers/google/models/gemini-2.0-flash-001",
            instructions="SYSTEM",
            temperature=0,
            max_output_tokens=1,
            include_logprobs=True,
            top_logprobs=5,
            access_token="tok",
            base_url="https://aiplatform.googleapis.com",
        )

    client.close()


def test_vertex_short_model_uses_project_and_location(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "YES"}]},
                "logprobsResult": {
                    "chosenCandidates": [{"token": "YES", "tokenId": 42, "logProbability": -0.1}],
                    "topCandidates": [
                        {"candidates": [{"token": "YES", "tokenId": 42, "logProbability": -0.1}]}
                    ],
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert "projects/p/locations/l/publishers/google/models/gemini-2.0-flash-001" in str(
            request.url
        )
        return httpx.Response(200, json=fixture)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    def fake_get_client(*, timeout_s=None, base_url=None):
        return "https://aiplatform.googleapis.com", client

    monkeypatch.setattr(vertex_backend, "_get_client", fake_get_client)
    monkeypatch.setenv("VERTEX_PROJECT", "p")
    monkeypatch.setenv("VERTEX_LOCATION", "l")

    res = call_text_chat_vertex(
        prompt="YES",
        model="gemini-2.0-flash-001",
        instructions="SYSTEM",
        temperature=0,
        max_output_tokens=1,
        include_logprobs=True,
        top_logprobs=1,
        access_token="tok",
        base_url="https://aiplatform.googleapis.com",
    )

    assert res.text == "YES"

    client.close()
