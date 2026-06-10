# Owner: Workstream D
"""Agent tools: arxiv_search, retrieve, synthesize.

Each implements the Tool protocol (name / spec() / __call__) and returns a
JSON-serializable dict. spec() emits Anthropic tool-use JSON schema, which the
AnthropicLLM adapter passes through to the Messages API verbatim.

synthesize is the terminal tool: it returns {"answer", "citations"} and the
agent loop stops when it runs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from app.ingest import PaperMeta, fetch_papers, make_client
from app.interfaces import LLM, Embedder, VectorStore

logger = logging.getLogger(__name__)

# One process-wide arXiv client so the package's delay_seconds throttle spans
# successive arxiv_search calls (CLAUDE.md §8: single connection, >=3s apart).
_shared_arxiv_client: Any = None


def _default_arxiv_fetch(query: str, max_results: int) -> Iterable[PaperMeta]:
    global _shared_arxiv_client
    if _shared_arxiv_client is None:  # lazy: keeps import + cold start cheap
        _shared_arxiv_client = make_client()
    return fetch_papers(
        _shared_arxiv_client, query=query, seed_ids=(), max_results=max_results
    )


class RetrieveTool:
    """Vector search over the indexed corpus."""

    name = "retrieve"

    def __init__(self, embedder: Embedder, store: VectorStore, default_k: int = 5) -> None:
        self._embedder = embedder
        self._store = store
        self._default_k = default_k

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Search the locally indexed corpus of agentic-RAG arXiv abstracts "
                "by semantic similarity. Use this first for every question."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "natural-language search query"},
                    "k": {
                        "type": "integer",
                        "description": f"number of chunks to return (default {self._default_k})",
                    },
                },
                "required": ["query"],
            },
        }

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        query: str = kwargs["query"]
        k: int = int(kwargs.get("k", self._default_k))
        vector = self._embedder.embed([query])[0]
        chunks = self._store.search(vector, k)
        return {"chunks": [dict(c) for c in chunks]}


class ArxivSearchTool:
    """Live arXiv metadata lookup, politeness-throttled. Does NOT write the index."""

    name = "arxiv_search"

    def __init__(
        self,
        fetch: Callable[[str, int], Iterable[PaperMeta]] | None = None,
        max_results: int = 5,
    ) -> None:
        self._fetch = fetch or _default_arxiv_fetch
        self._max_results = max_results

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Search arXiv live for papers not in the local corpus. Slow "
                "(rate-limited); use only when retrieve returns nothing relevant."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            'arXiv API query, e.g. \'abs:"agentic RAG" AND cat:cs.CL\' '
                            "or plain keywords"
                        ),
                    },
                },
                "required": ["query"],
            },
        }

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        query: str = kwargs["query"]
        papers = list(self._fetch(query, self._max_results))
        return {
            "papers": [
                {"arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract}
                for p in papers
            ]
        }


_SYNTH_INSTRUCTIONS = (
    "Answer the question using ONLY the provided source chunks. Cite the arXiv "
    "id inline in square brackets after each claim, e.g. [2501.09136]. If the "
    "chunks do not contain the answer, say so plainly. Be concise."
)


class SynthesizeTool:
    """Terminal tool: produce the final grounded answer from gathered chunks."""

    name = "synthesize"

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Produce the final cited answer from the chunks gathered so far. "
                "Terminal: the conversation ends after this tool."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "the user's question"},
                    "chunks": {
                        "type": "array",
                        "description": "the relevant chunks/papers gathered from prior tool calls",
                        "items": {
                            "type": "object",
                            "properties": {
                                "arxiv_id": {"type": "string"},
                                "title": {"type": "string"},
                                "text": {"type": "string"},
                            },
                            "required": ["arxiv_id"],
                        },
                    },
                },
                "required": ["question", "chunks"],
            },
        }

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        question: str = kwargs["question"]
        chunks: list[dict[str, Any]] = list(kwargs.get("chunks", []))

        sources = "\n\n".join(
            f"[{c.get('arxiv_id', '?')}] {c.get('title', '')}\n"
            f"{c.get('text', c.get('abstract', ''))}"
            for c in chunks
        )
        resp = self._llm.complete(
            [
                {"role": "system", "content": _SYNTH_INSTRUCTIONS},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nSources:\n{sources or '(none)'}",
                },
            ]
        )
        answer = resp.get("content") or ""

        citations: list[dict[str, str]] = []
        seen: set[str] = set()
        for c in chunks:
            arxiv_id = str(c.get("arxiv_id", ""))
            if arxiv_id and arxiv_id not in seen:
                seen.add(arxiv_id)
                citations.append({"arxiv_id": arxiv_id, "title": str(c.get("title", ""))})
        return {"answer": answer, "citations": citations}
