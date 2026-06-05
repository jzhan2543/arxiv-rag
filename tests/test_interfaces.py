"""Smoke tests: protocols import and have the agreed-upon shape (frozen contract for WS A-F)."""

from typing import get_type_hints

from app.interfaces import LLM, Chunk, Embedder, Tool, VectorStore


def test_protocols_importable() -> None:
    assert Embedder is not None
    assert LLM is not None
    assert VectorStore is not None
    assert Tool is not None


def test_chunk_shape() -> None:
    hints = get_type_hints(Chunk)
    assert set(hints.keys()) == {"id", "text", "arxiv_id", "title", "score"}


def test_embedder_protocol_has_required_members() -> None:
    assert hasattr(Embedder, "embed")
    assert hasattr(Embedder, "dim")


def test_llm_protocol_has_complete() -> None:
    assert hasattr(LLM, "complete")


def test_vectorstore_protocol_has_upsert_and_search() -> None:
    assert hasattr(VectorStore, "upsert")
    assert hasattr(VectorStore, "search")


def test_tool_protocol_has_spec_and_call() -> None:
    assert hasattr(Tool, "spec")
    assert "__call__" in Tool.__dict__
