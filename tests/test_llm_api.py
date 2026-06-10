"""Unit tests for AnthropicLLM. No live network — uses an injected fake client.

The opt-in integration test at the bottom calls the real API when
RUN_INTEGRATION=1 and ANTHROPIC_API_KEY are set.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic import APIStatusError, RateLimitError

from app.llm_api import DEFAULT_MODEL, AnthropicLLM


def _text_block(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id: str, name: str, args: dict[str, Any]) -> Any:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=args)


def _message(*blocks: Any) -> Any:
    return SimpleNamespace(content=list(blocks))


class _FakeMessages:
    def __init__(
        self,
        responses: list[Any] | None = None,
        raise_each: list[Exception | None] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._raise_each = list(raise_each or [])
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raise_each:
            exc = self._raise_each.pop(0)
            if exc is not None:
                raise exc
        return self._responses.pop(0)


class _FakeAnthropic:
    def __init__(
        self,
        responses: list[Any] | None = None,
        raise_each: list[Exception | None] | None = None,
    ) -> None:
        self.messages = _FakeMessages(responses, raise_each)


def _rate_limit_error() -> RateLimitError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req)
    return RateLimitError("rate limited", response=resp, body=None)


def _server_error(status: int = 503) -> APIStatusError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req)
    return APIStatusError("server error", response=resp, body=None)


def _bad_request_error() -> APIStatusError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(400, request=req)
    return APIStatusError("bad request", response=resp, body=None)


# -- Construction --------------------------------------------------------------


def test_requires_key_when_no_client() -> None:
    with pytest.raises(ValueError, match="api_key required"):
        AnthropicLLM()


def test_uses_injected_client() -> None:
    client = _FakeAnthropic([_message(_text_block("ok"))])
    llm = AnthropicLLM(client=client)  # type: ignore[arg-type]
    result = llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"


# -- complete() shape ----------------------------------------------------------


def test_returns_text_content_when_no_tool_calls() -> None:
    client = _FakeAnthropic([_message(_text_block("hello"))])
    llm = AnthropicLLM(client=client)  # type: ignore[arg-type]
    result = llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert result == {"content": "hello", "tool_calls": []}
    assert client.messages.calls[0]["model"] == DEFAULT_MODEL
    assert "system" not in client.messages.calls[0]


def test_extracts_system_message_to_top_level_field() -> None:
    client = _FakeAnthropic([_message(_text_block("ack"))])
    llm = AnthropicLLM(client=client)  # type: ignore[arg-type]
    llm.complete(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    sent = client.messages.calls[0]
    assert sent["system"] == "be terse"
    assert all(m["role"] != "system" for m in sent["messages"])


def test_concatenates_multiple_text_blocks() -> None:
    client = _FakeAnthropic([_message(_text_block("part one "), _text_block("part two"))])
    llm = AnthropicLLM(client=client)  # type: ignore[arg-type]
    result = llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "part one part two"


def test_surfaces_tool_calls() -> None:
    client = _FakeAnthropic(
        [
            _message(
                _text_block("let me look that up"),
                _tool_use_block("toolu_1", "retrieve", {"query": "agentic RAG"}),
            )
        ]
    )
    llm = AnthropicLLM(client=client)  # type: ignore[arg-type]
    tools = [{"name": "retrieve", "description": "...", "input_schema": {"type": "object"}}]
    result = llm.complete(
        messages=[{"role": "user", "content": "find papers"}],
        tools=tools,
    )
    assert result["content"] == "let me look that up"
    assert result["tool_calls"] == [
        {"id": "toolu_1", "name": "retrieve", "arguments": {"query": "agentic RAG"}},
    ]
    assert client.messages.calls[0]["tools"] == tools


def test_tool_use_without_text_returns_none_content() -> None:
    client = _FakeAnthropic([_message(_tool_use_block("toolu_1", "retrieve", {"q": "x"}))])
    llm = AnthropicLLM(client=client)  # type: ignore[arg-type]
    result = llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert result["content"] is None
    assert len(result["tool_calls"]) == 1


# -- Retry behavior ------------------------------------------------------------


def test_retries_on_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm_api.time.sleep", lambda _: None)
    client = _FakeAnthropic(
        responses=[_message(_text_block("ok"))],
        raise_each=[_rate_limit_error(), None],
    )
    llm = AnthropicLLM(client=client, max_retries=3)  # type: ignore[arg-type]
    result = llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"
    assert len(client.messages.calls) == 2


def test_retries_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm_api.time.sleep", lambda _: None)
    client = _FakeAnthropic(
        responses=[_message(_text_block("ok"))],
        raise_each=[_server_error(503), None],
    )
    llm = AnthropicLLM(client=client, max_retries=3)  # type: ignore[arg-type]
    result = llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert result["content"] == "ok"
    assert len(client.messages.calls) == 2


def test_does_not_retry_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm_api.time.sleep", lambda _: None)
    client = _FakeAnthropic(raise_each=[_bad_request_error()])
    llm = AnthropicLLM(client=client, max_retries=3)  # type: ignore[arg-type]
    with pytest.raises(APIStatusError):
        llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert len(client.messages.calls) == 1


def test_exhausts_retries_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm_api.time.sleep", lambda _: None)
    client = _FakeAnthropic(
        raise_each=[_rate_limit_error(), _rate_limit_error(), _rate_limit_error()],
    )
    llm = AnthropicLLM(client=client, max_retries=3)  # type: ignore[arg-type]
    with pytest.raises(RateLimitError):
        llm.complete(messages=[{"role": "user", "content": "hi"}])
    assert len(client.messages.calls) == 3


# -- Integration (opt-in) ------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="RUN_INTEGRATION=1 and ANTHROPIC_API_KEY required",
)
def test_integration_messages_create() -> None:
    llm = AnthropicLLM(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=10,
    )
    result = llm.complete(messages=[{"role": "user", "content": "reply with exactly: ok"}])
    assert isinstance(result["content"], str)
    assert result["tool_calls"] == []
