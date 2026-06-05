"""Shared Protocol contracts. WS0 owns this file; all other workstreams treat it as read-only."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypedDict


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    @property
    def dim(self) -> int: ...


class LLM(Protocol):
    # complete() returns {"content": str | None, "tool_calls": [...]}
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


class Chunk(TypedDict):
    id: str           # "{arxiv_id}#{n}"
    text: str
    arxiv_id: str
    title: str
    score: float


class VectorStore(Protocol):
    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None: ...
    def search(self, vector: list[float], k: int) -> list[Chunk]: ...


class Tool(Protocol):
    name: str
    def spec(self) -> dict[str, Any]: ...  # JSON schema for tool-calling
    def __call__(self, **kwargs: Any) -> dict[str, Any]: ...
