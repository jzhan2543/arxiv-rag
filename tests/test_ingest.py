"""Unit tests for the ingest pipeline. run()/fetch_papers() are exercised with
fakes — no live network. The opt-in test at the bottom hits the real arXiv API
(one polite request, no API keys needed) when RUN_INTEGRATION=1.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from app.ingest import (
    SEED_IDS,
    PaperMeta,
    _versionless,
    fetch_papers,
    run,
)

# -- Fakes ----------------------------------------------------------------------


class FakeEmbedder:
    """Embedder that records inputs and returns deterministic 4-dim vectors."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

    @property
    def dim(self) -> int:
        return 4


class FakeStore:
    """VectorStore that records upserts into a dict keyed by chunk id."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], dict[str, Any]]] = {}
        self.upsert_calls = 0

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        assert len(ids) == len(vectors) == len(metadata)
        self.upsert_calls += 1
        for i, chunk_id in enumerate(ids):
            self.rows[chunk_id] = (vectors[i], metadata[i])

    def search(self, vector: list[float], k: int) -> list[Any]:
        raise NotImplementedError("not used in ingest tests")


def _paper(n: int) -> PaperMeta:
    return PaperMeta(arxiv_id=f"2501.{n:05d}", title=f"Title {n}", abstract=f"Abstract {n}")


class _FakeArxivResult:
    def __init__(self, short_id: str, title: str, summary: str) -> None:
        self._short_id = short_id
        self.title = title
        self.summary = summary

    def get_short_id(self) -> str:
        return self._short_id


class _FakeArxivClient:
    """Maps each search to a canned list of results, in submission order."""

    def __init__(self, by_search: list[list[_FakeArxivResult]]) -> None:
        self._by_search = list(by_search)
        self.searches: list[Any] = []

    def results(self, search: Any) -> Iterator[_FakeArxivResult]:
        self.searches.append(search)
        yield from self._by_search.pop(0)


# -- run() ----------------------------------------------------------------------


def test_run_embeds_and_upserts() -> None:
    embedder, store = FakeEmbedder(), FakeStore()
    papers = [_paper(1), _paper(2)]

    total = run(embedder, store, papers)

    assert total == 2
    assert set(store.rows) == {"2501.00001#0", "2501.00002#0"}
    _, meta = store.rows["2501.00001#0"]
    assert meta == {"arxiv_id": "2501.00001", "title": "Title 1", "text": "Abstract 1"}


def test_run_embeds_title_plus_abstract() -> None:
    embedder, store = FakeEmbedder(), FakeStore()
    run(embedder, store, [_paper(1)])
    assert embedder.batches == [["Title 1\n\nAbstract 1"]]


def test_run_batches_by_batch_size() -> None:
    embedder, store = FakeEmbedder(), FakeStore()
    total = run(embedder, store, [_paper(i) for i in range(5)], batch_size=2)
    assert total == 5
    assert store.upsert_calls == 3  # 2 + 2 + 1
    assert [len(b) for b in embedder.batches] == [2, 2, 1]


def test_run_empty_iterable_is_no_op() -> None:
    embedder, store = FakeEmbedder(), FakeStore()
    assert run(embedder, store, []) == 0
    assert store.upsert_calls == 0
    assert embedder.batches == []


def test_run_consumes_lazily_from_iterator() -> None:
    """run() must stream from a generator, not materialize it."""
    embedder, store = FakeEmbedder(), FakeStore()

    def gen() -> Iterator[PaperMeta]:
        yield _paper(1)
        yield _paper(2)
        yield _paper(3)

    assert run(embedder, store, gen(), batch_size=2) == 3
    assert store.upsert_calls == 2


# -- fetch_papers() ---------------------------------------------------------------


def test_fetch_papers_yields_seeds_then_query_deduped() -> None:
    seed = _FakeArxivResult("2501.09136v1", "Agentic RAG Survey", "Survey abstract.")
    q1 = _FakeArxivResult("2501.09136v1", "Agentic RAG Survey", "Survey abstract.")
    q2 = _FakeArxivResult("2502.11111v3", "Other Paper", "Other abstract.")
    client = _FakeArxivClient([[seed], [q1, q2]])

    papers = list(fetch_papers(client, query="q", seed_ids=["2501.09136"], max_results=10))

    # Duplicate of the seed in query results is dropped; version suffixes stripped.
    assert [p.arxiv_id for p in papers] == ["2501.09136", "2502.11111"]
    assert len(client.searches) == 2


def test_fetch_papers_no_seeds_runs_single_search() -> None:
    client = _FakeArxivClient([[_FakeArxivResult("2502.22222v1", "T", "A")]])
    papers = list(fetch_papers(client, query="q", seed_ids=[], max_results=10))
    assert [p.arxiv_id for p in papers] == ["2502.22222"]
    assert len(client.searches) == 1


def test_fetch_papers_cleans_hard_wrapped_text() -> None:
    raw = _FakeArxivResult("2502.33333v1", "A  Title\n  Wrapped", "An\nabstract   with\nbreaks")
    client = _FakeArxivClient([[raw]])
    papers = list(fetch_papers(client, query="q", seed_ids=[], max_results=10))
    assert papers[0].title == "A Title Wrapped"
    assert papers[0].abstract == "An abstract with breaks"


def test_default_seed_is_the_singh_survey() -> None:
    assert "2501.09136" in SEED_IDS


# -- _versionless -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("short_id", "expected"),
    [
        ("2501.09136v1", "2501.09136"),
        ("2501.09136v12", "2501.09136"),
        ("2501.09136", "2501.09136"),
        ("math.GT/0309136v2", "math.GT/0309136"),  # pre-2007 id scheme
    ],
)
def test_versionless(short_id: str, expected: str) -> None:
    assert _versionless(short_id) == expected


# -- Integration (opt-in, network but no keys) -------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="RUN_INTEGRATION=1 required (live arXiv API, ~1 polite request)",
)
def test_integration_fetch_seed_and_populate_index(tmp_path: Any) -> None:
    """End-to-end against live arXiv: fetch the seed paper only (no topic query),
    embed with a fake, and write a real sqlite-vec index file."""
    from app.ingest import make_client
    from app.store_sqlitevec import SqliteVecStore

    papers = list(
        fetch_papers(make_client(), query="", seed_ids=list(SEED_IDS), max_results=0)
    )
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2501.09136"
    assert "agentic" in paper.title.lower() or "agentic" in paper.abstract.lower()

    embedder = FakeEmbedder()
    store = SqliteVecStore(tmp_path / "live.db", dim=embedder.dim)
    try:
        assert run(embedder, store, papers) == 1
    finally:
        store.close()
    assert (tmp_path / "live.db").exists()
