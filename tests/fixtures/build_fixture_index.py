"""Build the committed fixture index used by hermetic CI tests.

Two modes:

  default (hash)   Deterministic SHA-256-chained pseudo-embeddings over the
                   paraphrased texts below. Offline, reproducible, NOT
                   semantic — exercises only the storage + search path.
  --live           Fetches the same 10 papers' real titles/abstracts from
                   arXiv (one polite request) and embeds them with Voyage
                   (requires VOYAGE_API_KEY). Semantic: real query embeddings
                   rank correctly against it, which the WS-E eval gate needs.

The COMMITTED index.db is built with --live so `python -m app.eval` gives
meaningful recall in CI. WS-B's storage tests pass against either mode.

Run from repo root:
    uv run python tests/fixtures/build_fixture_index.py [--live]
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from app.store_sqlitevec import SqliteVecStore

DIM = 1024

# 10 chunks anchored on real arXiv ids in the agentic-RAG canon — text is
# paraphrased from the public abstracts. This file is meant for storage tests,
# not factual eval.
CHUNKS: list[tuple[str, str, str, str]] = [
    (
        "2501.09136#0",
        "2501.09136",
        "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG",
        "Agentic RAG leverages agentic design patterns — reflection, planning, "
        "tool use, and multi-agent collaboration — to dynamically manage "
        "retrieval strategies, iteratively refine contextual understanding, and "
        "adapt workflows.",
    ),
    (
        "2310.11511#0",
        "2310.11511",
        "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "Self-RAG improves an LLM's quality and factuality through on-demand "
        "retrieval and self-reflection using reflection tokens.",
    ),
    (
        "2005.11401#0",
        "2005.11401",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "RAG combines pre-trained parametric and non-parametric memory for "
        "knowledge-intensive language generation, retrieving from Wikipedia at "
        "inference time.",
    ),
    (
        "2305.14283#0",
        "2305.14283",
        "Query Rewriting for Retrieval-Augmented Large Language Models",
        "A trainable rewrite-retrieve-read framework adapts the retriever and "
        "reader to LLM-friendly queries.",
    ),
    (
        "2401.18059#0",
        "2401.18059",
        "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "RAPTOR builds a recursive tree of document summaries, enabling "
        "retrieval at multiple levels of abstraction.",
    ),
    (
        "2312.10997#0",
        "2312.10997",
        "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "Survey of RAG paradigms — Naive RAG, Advanced RAG, and Modular RAG — "
        "covering retrieval, generation, augmentation, and evaluation.",
    ),
    (
        "2402.03367#0",
        "2402.03367",
        "Self-Discover: Large Language Models Self-Compose Reasoning Structures",
        "LLMs self-compose task-specific reasoning structures from atomic "
        "reasoning modules to improve performance on complex problems.",
    ),
    (
        "2303.17580#0",
        "2303.17580",
        "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face",
        "A planner LLM orchestrates expert models from a model hub to solve "
        "multimodal tasks, decomposing user requests into subtasks.",
    ),
    (
        "2210.03629#0",
        "2210.03629",
        "ReAct: Synergizing Reasoning and Acting in Language Models",
        "ReAct interleaves reasoning traces and task-specific actions, "
        "improving performance on language and decision-making benchmarks.",
    ),
    (
        "2303.11366#0",
        "2303.11366",
        "Reflexion: Language Agents with Verbal Reinforcement Learning",
        "Reflexion uses verbal self-reflection on prior failures to improve "
        "language agent performance across decision-making, programming, and "
        "reasoning tasks.",
    ),
]


def embed(text: str, dim: int = DIM) -> list[float]:
    """Deterministic SHA-256-chained pseudo-embedding, L2-normalized."""
    out: list[float] = []
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    counter = 0
    while len(out) < dim:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for i in range(0, len(block), 2):
            n = int.from_bytes(block[i : i + 2], "big", signed=True)
            out.append(n / 32768.0)
            if len(out) >= dim:
                break
        counter += 1
    norm = math.sqrt(sum(f * f for f in out))
    return [f / norm for f in out] if norm > 0 else [0.0] * dim


def _rows_hash() -> tuple[list[str], list[list[float]], list[dict[str, str]]]:
    ids = [c[0] for c in CHUNKS]
    vectors = [embed(c[3]) for c in CHUNKS]
    metadata = [{"arxiv_id": c[1], "title": c[2], "text": c[3]} for c in CHUNKS]
    return ids, vectors, metadata


def _rows_live() -> tuple[list[str], list[list[float]], list[dict[str, str]]]:
    """Real abstracts from arXiv + real Voyage embeddings (same 10 papers)."""
    from app.config import Settings
    from app.embed_api import VoyageEmbedder
    from app.ingest import fetch_papers, make_client

    paper_ids = [c[1] for c in CHUNKS]
    papers = list(
        fetch_papers(make_client(), query="", seed_ids=paper_ids, max_results=0)
    )
    if len(papers) != len(paper_ids):
        fetched = {p.arxiv_id for p in papers}
        raise RuntimeError(f"arXiv returned {len(papers)}/{len(paper_ids)}; "
                           f"missing {set(paper_ids) - fetched}")
    embedder = VoyageEmbedder(
        api_key=Settings.from_env().voyage_api_key, input_type="document"
    )
    ids = [f"{p.arxiv_id}#0" for p in papers]
    # Mirror app.ingest._embed_text: title + abstract go into the embedding.
    vectors = embedder.embed([f"{p.title}\n\n{p.abstract}" for p in papers])
    metadata = [
        {"arxiv_id": p.arxiv_id, "title": p.title, "text": p.abstract} for p in papers
    ]
    return ids, vectors, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="fetch real abstracts from arXiv and embed with Voyage (needs VOYAGE_API_KEY)",
    )
    args = parser.parse_args()

    ids, vectors, metadata = _rows_live() if args.live else _rows_hash()

    out_path = Path(__file__).parent / "index.db"
    if out_path.exists():
        out_path.unlink()
    store = SqliteVecStore(out_path, dim=DIM)
    try:
        store.upsert(ids, vectors, metadata)
    finally:
        store.close()
    mode = "live (arXiv + Voyage)" if args.live else "hash (offline)"
    print(f"wrote {len(ids)} chunks to {out_path} [{mode}]")


if __name__ == "__main__":
    main()
