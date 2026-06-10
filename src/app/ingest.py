# Owner: Workstream C
"""arXiv ingestion: query API -> chunk abstracts -> embed -> upsert.

v0 scope (CLAUDE.md §1): abstracts only, 1 chunk per paper, chunk id
"{arxiv_id}#0". The embedding input is "{title}\n\n{abstract}" (titles carry
retrieval signal); the stored chunk text is the abstract alone.

arXiv politeness (CLAUDE.md §8, load-bearing): Client(page_size=100,
delay_seconds=3.0, num_retries=5), descriptive User-Agent, never parallelize.

CLI: `python -m app.ingest [--query ...] [--max-results N]` — the composition
root for the ingestion runtime shape; it alone imports concrete adapters.
`run()` and `fetch_papers()` depend only on the Protocols and are unit-tested
with fakes.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import batched
from typing import Any

import arxiv

from app.interfaces import Embedder, VectorStore

logger = logging.getLogger(__name__)

USER_AGENT = "arxiv-rag/0.0 (+https://github.com/jzhan2543/arxiv-rag)"

# Canonical agentic-RAG survey (Singh et al.); always ingested even if the
# topic query misses it. The arXiv API has no citation graph, so "and its
# references" from the source plan is approximated by the topic query.
SEED_IDS: tuple[str, ...] = ("2501.09136",)

DEFAULT_QUERY = (
    '(abs:"agentic RAG" OR abs:"agentic retrieval-augmented generation" '
    'OR abs:"self-reflective retrieval" OR (ti:"retrieval-augmented" AND abs:agent)) '
    "AND (cat:cs.CL OR cat:cs.IR OR cat:cs.AI)"
)
DEFAULT_MAX_RESULTS = 300
EMBED_BATCH_SIZE = 64


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str  # versionless, e.g. "2501.09136"
    title: str
    abstract: str


def make_client() -> arxiv.Client:
    """arXiv client tuned to the politeness rule in CLAUDE.md §8."""
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    try:
        # The arxiv package keeps its requests.Session private; setting a
        # descriptive User-Agent is required by arXiv's ToU, so we reach in.
        client._session.headers.update({"User-Agent": USER_AGENT})
    except AttributeError:  # pragma: no cover - depends on arxiv pkg internals
        logger.warning("could not set User-Agent on arxiv.Client; using package default")
    return client


def fetch_papers(
    client: arxiv.Client,
    query: str = DEFAULT_QUERY,
    seed_ids: Sequence[str] = SEED_IDS,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> Iterator[PaperMeta]:
    """Yield seed papers (fetched by id) then topic-query results, deduped.

    Ids are normalized to versionless form so re-ingesting a revised paper
    overwrites its chunk instead of duplicating it.
    """
    searches: list[Any] = []
    if seed_ids:
        searches.append(arxiv.Search(id_list=list(seed_ids)))
    if query and max_results > 0:
        searches.append(
            arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
        )

    seen: set[str] = set()
    for search in searches:
        for result in client.results(search):
            arxiv_id = _versionless(result.get_short_id())
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            yield PaperMeta(
                arxiv_id=arxiv_id,
                title=_clean(result.title),
                abstract=_clean(result.summary),
            )


def run(
    embedder: Embedder,
    store: VectorStore,
    papers: Iterable[PaperMeta],
    batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """Chunk, embed, and upsert papers. Returns the number of chunks written.

    Streams in batches so a full ingest holds at most `batch_size` papers in
    memory and makes one Embedder call per batch.
    """
    total = 0
    for batch in batched(papers, batch_size):
        ids = [f"{p.arxiv_id}#0" for p in batch]
        vectors = embedder.embed([_embed_text(p) for p in batch])
        metadata: list[dict[str, Any]] = [
            {"arxiv_id": p.arxiv_id, "title": p.title, "text": p.abstract} for p in batch
        ]
        store.upsert(ids, vectors, metadata)
        total += len(batch)
        logger.info("ingested batch of %d (total %d)", len(batch), total)
    return total


def _embed_text(paper: PaperMeta) -> str:
    return f"{paper.title}\n\n{paper.abstract}"


def _versionless(short_id: str) -> str:
    """'2501.09136v2' -> '2501.09136'; already-versionless ids pass through."""
    return re.sub(r"v\d+$", "", short_id)


def _clean(text: str) -> str:
    """Collapse arXiv's hard-wrapped whitespace into single spaces."""
    return " ".join(text.split())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Ingest arXiv agentic-RAG abstracts into the sqlite-vec index.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="arXiv API query string")
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"cap on topic-query results (default {DEFAULT_MAX_RESULTS})",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Composition root: the one place in the ingest runtime shape that may
    # import concrete adapters (CLAUDE.md §2 rule 2).
    from app.config import Settings
    from app.embed_api import VoyageEmbedder
    from app.store_sqlitevec import SqliteVecStore

    settings = Settings.from_env()
    embedder = VoyageEmbedder(api_key=settings.voyage_api_key, input_type="document")
    store = SqliteVecStore(settings.index_path, dim=embedder.dim)
    papers = fetch_papers(make_client(), query=args.query, max_results=args.max_results)
    total = run(embedder, store, papers)
    print(f"ingested {total} chunks into {settings.index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
