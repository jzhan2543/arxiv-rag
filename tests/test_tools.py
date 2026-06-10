"""Unit tests for the three agent tools. All dependencies are fakes — no network."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.ingest import PaperMeta
from app.interfaces import Chunk
from app.tools import ArxivSearchTool, RetrieveTool, SynthesizeTool

# -- Fakes ----------------------------------------------------------------------


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    @property
    def dim(self) -> int:
        return 3


class FakeStore:
    def __init__(self, results: list[Chunk] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[list[float], int]] = []

    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], k: int) -> list[Chunk]:
        self.calls.append((vector, k))
        return self.results[:k]


class FakeLLM:
    def __init__(self, content: str = "fake answer") -> None:
        self.content = content
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((messages, tools))
        return {"content": self.content, "tool_calls": []}


def _chunk(arxiv_id: str, n: int = 0, score: float = 0.9) -> Chunk:
    return Chunk(
        id=f"{arxiv_id}#{n}",
        text=f"text of {arxiv_id}",
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        score=score,
    )


# -- Spec shape (all three) -------------------------------------------------------


def test_specs_are_anthropic_shaped() -> None:
    tools: list[Any] = [
        RetrieveTool(FakeEmbedder(), FakeStore()),
        ArxivSearchTool(fetch=lambda q, n: []),
        SynthesizeTool(FakeLLM()),
    ]
    assert [t.name for t in tools] == ["retrieve", "arxiv_search", "synthesize"]
    for t in tools:
        spec = t.spec()
        assert spec["name"] == t.name
        assert spec["description"]
        assert spec["input_schema"]["type"] == "object"
        assert "required" in spec["input_schema"]


# -- RetrieveTool -----------------------------------------------------------------


def test_retrieve_embeds_query_and_searches() -> None:
    embedder, store = FakeEmbedder(), FakeStore([_chunk("2501.09136")])
    tool = RetrieveTool(embedder, store)

    result = tool(query="agentic rag patterns")

    assert embedder.calls == [["agentic rag patterns"]]
    assert store.calls == [([0.1, 0.2, 0.3], 5)]  # default k=5
    assert result == {"chunks": [dict(_chunk("2501.09136"))]}


def test_retrieve_k_passthrough_and_default() -> None:
    store = FakeStore([_chunk(f"id{i}") for i in range(10)])
    tool = RetrieveTool(FakeEmbedder(), store, default_k=7)
    tool(query="q")
    assert store.calls[-1][1] == 7
    tool(query="q", k=2)
    assert store.calls[-1][1] == 2


def test_retrieve_empty_store_returns_empty_chunks() -> None:
    tool = RetrieveTool(FakeEmbedder(), FakeStore([]))
    assert tool(query="q") == {"chunks": []}


# -- ArxivSearchTool ---------------------------------------------------------------


def test_arxiv_search_maps_papers() -> None:
    captured: dict[str, Any] = {}

    def fetch(query: str, max_results: int) -> Iterable[PaperMeta]:
        captured["args"] = (query, max_results)
        return [PaperMeta(arxiv_id="2310.11511", title="Self-RAG", abstract="Reflection...")]

    tool = ArxivSearchTool(fetch=fetch, max_results=3)
    result = tool(query="self-rag")

    assert captured["args"] == ("self-rag", 3)
    assert result == {
        "papers": [
            {"arxiv_id": "2310.11511", "title": "Self-RAG", "abstract": "Reflection..."}
        ]
    }


def test_arxiv_search_empty() -> None:
    tool = ArxivSearchTool(fetch=lambda q, n: [])
    assert tool(query="nothing") == {"papers": []}


# -- SynthesizeTool ----------------------------------------------------------------


def test_synthesize_answers_and_cites() -> None:
    llm = FakeLLM(content="Grounded answer [2501.09136].")
    tool = SynthesizeTool(llm)

    result = tool(
        question="What patterns?",
        chunks=[
            {"arxiv_id": "2501.09136", "title": "Survey", "text": "patterns..."},
            {"arxiv_id": "2310.11511", "title": "Self-RAG", "text": "reflection..."},
        ],
    )

    assert result["answer"] == "Grounded answer [2501.09136]."
    assert result["citations"] == [
        {"arxiv_id": "2501.09136", "title": "Survey"},
        {"arxiv_id": "2310.11511", "title": "Self-RAG"},
    ]


def test_synthesize_prompt_contains_question_and_sources() -> None:
    llm = FakeLLM()
    SynthesizeTool(llm)(
        question="What four patterns?",
        chunks=[{"arxiv_id": "2501.09136", "title": "Survey", "text": "reflection, planning"}],
    )
    messages, tools = llm.calls[0]
    assert tools is None  # synthesize never exposes tools
    user_msg = messages[-1]["content"]
    assert "What four patterns?" in user_msg
    assert "[2501.09136]" in user_msg
    assert "reflection, planning" in user_msg


def test_synthesize_dedupes_citations_preserving_order() -> None:
    tool = SynthesizeTool(FakeLLM())
    result = tool(
        question="q",
        chunks=[
            {"arxiv_id": "B", "title": "tB", "text": ""},
            {"arxiv_id": "A", "title": "tA", "text": ""},
            {"arxiv_id": "B", "title": "tB-dup", "text": ""},
        ],
    )
    assert [c["arxiv_id"] for c in result["citations"]] == ["B", "A"]


def test_synthesize_empty_chunks() -> None:
    result = SynthesizeTool(FakeLLM())(question="q", chunks=[])
    assert result["citations"] == []
    # The prompt still goes out with an explicit empty-sources marker.
    assert isinstance(result["answer"], str)


def test_synthesize_accepts_abstract_key_for_live_papers() -> None:
    """arxiv_search results carry 'abstract' instead of 'text'; both must ground."""
    llm = FakeLLM()
    SynthesizeTool(llm)(
        question="q",
        chunks=[{"arxiv_id": "X", "title": "T", "abstract": "from live arxiv"}],
    )
    user_msg = llm.calls[0][0][-1]["content"]
    assert "from live arxiv" in user_msg
