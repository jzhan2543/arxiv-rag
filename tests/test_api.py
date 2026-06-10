"""Unit tests for the FastAPI app. The agent is a fake injected via create_app()."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app

GOOD_RESULT: dict[str, Any] = {
    "answer": "Reflection, planning, tool use, multi-agent collaboration [2501.09136].",
    "citations": [{"arxiv_id": "2501.09136", "title": "Agentic RAG Survey"}],
    "steps": [
        {"tool": "retrieve", "args": {"query": "patterns"}, "result_summary": "5 chunks"},
        {"tool": "synthesize", "args": {}, "result_summary": "answer with 1 citations"},
    ],
    "truncated": False,
}


class FakeAgent:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or GOOD_RESULT
        self.questions: list[str] = []

    def ask(self, question: str) -> dict[str, Any]:
        self.questions.append(question)
        return self.result


def _client(agent: FakeAgent) -> TestClient:
    return TestClient(create_app(agent))


# -- /healthz ---------------------------------------------------------------------


def test_healthz() -> None:
    with _client(FakeAgent()) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# -- /ask -------------------------------------------------------------------------


def test_ask_returns_section3_schema() -> None:
    agent = FakeAgent()
    with _client(agent) as client:
        resp = client.post("/ask", json={"question": "What patterns?"})

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"answer", "citations", "steps", "truncated"}
    assert body["answer"].startswith("Reflection")
    assert body["citations"] == [{"arxiv_id": "2501.09136", "title": "Agentic RAG Survey"}]
    assert body["steps"][0] == {
        "tool": "retrieve",
        "args": {"query": "patterns"},
        "result_summary": "5 chunks",
    }
    assert body["truncated"] is False
    assert agent.questions == ["What patterns?"]


def test_ask_empty_question_is_422() -> None:
    with _client(FakeAgent()) as client:
        resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_missing_question_is_422() -> None:
    with _client(FakeAgent()) as client:
        resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_validates_agent_output_against_contract() -> None:
    """An agent result that violates the §3 schema must not silently pass through."""
    broken = FakeAgent({"answer": "no citations key", "truncated": False})
    with _client(broken) as client, pytest.raises(ValidationError):
        client.post("/ask", json={"question": "q"})


def test_truncated_flag_passes_through() -> None:
    result = dict(GOOD_RESULT, truncated=True)
    with _client(FakeAgent(result)) as client:
        resp = client.post("/ask", json={"question": "q"})
    assert resp.json()["truncated"] is True


# -- Import-time posture -------------------------------------------------------------


def test_module_level_app_imports_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`uvicorn app.main:app` must be importable on a box with no env vars set;
    adapters are only constructed at lifespan startup."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    import importlib

    import app.main

    module = importlib.reload(app.main)
    assert module.app.title == "arxiv-rag"
