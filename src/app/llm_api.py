# Owner: Workstream A
"""Anthropic Messages API implementation of the LLM protocol.

Surfaces `tool_use` response blocks as `tool_calls` so the WS-D bounded
loop can drive multi-step tool calling. System messages embedded in the
input message list are extracted to Anthropic's top-level `system` field.
"""

from __future__ import annotations

import time
from typing import Any, cast

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    RateLimitError,
)
from anthropic.types import Message

# Verified available in account on 2026-06-09. The Haiku tier is the cheap
# default; callers can pass model="claude-sonnet-4-6" for stronger reasoning.
DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicLLM:
    """LLM Protocol adapter against the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        max_retries: int = 4,
        timeout: float = 60.0,
        client: Anthropic | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key required when no pre-built client is provided")
            client = Anthropic(api_key=api_key, timeout=timeout)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        system, msgs = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = self._call_with_retry(kwargs)
        return _normalize(resp)

    def _call_with_retry(self, kwargs: dict[str, Any]) -> Message:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return cast(Message, self._client.messages.create(**kwargs))
            except RateLimitError as e:
                last_exc = e
            except APIConnectionError as e:
                last_exc = e
            except APIStatusError as e:
                # Only retry transient 5xx; surface 4xx (auth, bad request) immediately.
                if not (500 <= e.status_code < 600):
                    raise
                last_exc = e
            if attempt < self._max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        assert last_exc is not None
        raise last_exc


def _split_system(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Pull system messages out of the list; Anthropic wants them as a top-level field."""
    sys_blocks: list[str] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system":
            content = m.get("content", "")
            if isinstance(content, str):
                sys_blocks.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        sys_blocks.append(str(block.get("text", "")))
        else:
            rest.append(m)
    return "\n\n".join(sys_blocks), rest


def _normalize(resp: Message) -> dict[str, Any]:
    """Map an Anthropic Message into the LLM protocol shape."""
    content_text: str | None = None
    tool_calls: list[dict[str, Any]] = []
    for block in resp.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "")
            content_text = (content_text or "") + text
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "arguments": getattr(block, "input", {}),
                }
            )
    return {"content": content_text, "tool_calls": tool_calls}
