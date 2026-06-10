"""Build the committed fixture index used by hermetic CI tests.

Deterministic — same input text always yields the same vector via a SHA-256
chain. The "embeddings" here are NOT semantic; they only exercise the storage
+ search path. WS-E's eval gate must mock retrieval or build its own
semantic-embedded fixture if it needs meaningful ranking.

Run from repo root:
    uv run python tests/fixtures/build_fixture_index.py
"""

from __future__ import annotations

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


def main() -> None:
    out_path = Path(__file__).parent / "index.db"
    if out_path.exists():
        out_path.unlink()
    store = SqliteVecStore(out_path, dim=DIM)
    try:
        ids = [c[0] for c in CHUNKS]
        vectors = [embed(c[3]) for c in CHUNKS]
        metadata = [
            {"arxiv_id": c[1], "title": c[2], "text": c[3]} for c in CHUNKS
        ]
        store.upsert(ids, vectors, metadata)
    finally:
        store.close()
    print(f"wrote {len(CHUNKS)} chunks to {out_path}")


if __name__ == "__main__":
    main()
