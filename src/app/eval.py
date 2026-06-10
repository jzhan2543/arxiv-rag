# Owner: Workstream E
"""Eval harness: recall@k + citation accuracy (deterministic) + faithfulness (LLM judge).

Metric definitions (per item, averaged over the golden set):
  recall@k          1.0 if any expected_paper_id appears among the arxiv_ids of
                    the top-k retrieved chunks (a direct Embedder->VectorStore
                    pass, independent of agent behavior), else 0.0.
  citation_accuracy |returned citations ∩ must_cite| / |must_cite| — "did the
                    agent cite what it was required to cite". Extra citations
                    are not penalized in v0.
  faithfulness      LLM-judged fraction of answer claims supported by the
                    retrieved context (same metric RAGAS computes; implemented
                    against our own LLM protocol instead of RAGAS to avoid the
                    langchain glue dependency — see PR #5). Skippable; None
                    when no judge is provided.

The harness is the integration contract: recall exercises Embedder->VectorStore,
citation/faithfulness exercise the full agent path.

CLI: `python -m app.eval [--gate --min-recall X --min-citation-acc Y
--min-faithfulness Z] [--skip-faithfulness] [--golden PATH] [--k N]`.
`--gate` exits 1 when any thresholded metric falls below its minimum.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.interfaces import LLM, Embedder, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = Path("tests/golden/qa.jsonl")
DEFAULT_K = 5


class AskAgent(Protocol):
    def ask(self, question: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class GoldenItem:
    q: str
    expected_paper_ids: list[str]
    reference_answer: str
    must_cite: list[str]


@dataclass
class ItemResult:
    question: str
    recall: float
    citation_accuracy: float
    faithfulness: float | None
    answer: str
    cited: list[str]
    retrieved: list[str]


@dataclass
class EvalReport:
    items: list[ItemResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def recall(self) -> float:
        return _mean([i.recall for i in self.items])

    @property
    def citation_accuracy(self) -> float:
        return _mean([i.citation_accuracy for i in self.items])

    @property
    def faithfulness(self) -> float | None:
        scores = [i.faithfulness for i in self.items if i.faithfulness is not None]
        return _mean(scores) if scores else None


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        try:
            items.append(
                GoldenItem(
                    q=raw["q"],
                    expected_paper_ids=list(raw["expected_paper_ids"]),
                    reference_answer=raw["reference_answer"],
                    must_cite=list(raw["must_cite"]),
                )
            )
        except KeyError as e:
            raise ValueError(f"{path}:{line_no} missing field {e}") from e
    return items


# -- Faithfulness judge -----------------------------------------------------------

_JUDGE_INSTRUCTIONS = (
    "You grade whether an answer is supported by source documents. Break the "
    "answer into atomic factual claims, then count how many are directly "
    "supported by the sources. Respond with ONLY a JSON object: "
    '{"supported": <int>, "total": <int>}. No other text.'
)


def judge_faithfulness(judge: LLM, answer: str, context: str) -> float:
    """Fraction of answer claims supported by context, per the judge LLM.

    Malformed judge output scores 0.0 (conservative). An answer with no
    factual claims (total=0) scores 1.0 (vacuously faithful).
    """
    resp = judge.complete(
        [
            {"role": "system", "content": _JUDGE_INSTRUCTIONS},
            {"role": "user", "content": f"Sources:\n{context}\n\nAnswer:\n{answer}"},
        ]
    )
    content = resp.get("content") or ""
    match = re.search(r"\{[^{}]*\}", content)
    if not match:
        logger.warning("judge returned no JSON: %r", content[:200])
        return 0.0
    try:
        verdict = json.loads(match.group(0))
        supported, total = int(verdict["supported"]), int(verdict["total"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("judge returned malformed JSON: %r", content[:200])
        return 0.0
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, supported / total))


# -- Core harness -------------------------------------------------------------------


def evaluate(
    agent: AskAgent,
    embedder: Embedder,
    store: VectorStore,
    items: Sequence[GoldenItem],
    *,
    k: int = DEFAULT_K,
    judge: LLM | None = None,
) -> EvalReport:
    report = EvalReport()
    # One batched call for all question embeddings (recall pass): cheaper and
    # far friendlier to Voyage rate limits than per-item embeds.
    question_vectors = embedder.embed([item.q for item in items]) if items else []
    for item, vector in zip(items, question_vectors, strict=True):
        chunks = store.search(vector, k)
        retrieved_ids = [c["arxiv_id"] for c in chunks]
        recall = 1.0 if set(item.expected_paper_ids) & set(retrieved_ids) else 0.0

        result = agent.ask(item.q)
        cited = [c.get("arxiv_id", "") for c in result.get("citations", [])]
        if item.must_cite:
            citation_acc = len(set(cited) & set(item.must_cite)) / len(set(item.must_cite))
        else:
            citation_acc = 1.0

        faithfulness: float | None = None
        if judge is not None:
            context = "\n\n".join(
                f"[{c['arxiv_id']}] {c['title']}\n{c['text']}" for c in chunks
            )
            faithfulness = judge_faithfulness(judge, result.get("answer", ""), context)

        report.items.append(
            ItemResult(
                question=item.q,
                recall=recall,
                citation_accuracy=citation_acc,
                faithfulness=faithfulness,
                answer=result.get("answer", ""),
                cited=cited,
                retrieved=retrieved_ids,
            )
        )
        logger.info(
            "evaluated %r recall=%.0f cit=%.2f faith=%s",
            item.q[:60],
            recall,
            citation_acc,
            "skip" if faithfulness is None else f"{faithfulness:.2f}",
        )
    return report


# -- Reporting & gate ----------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    min_recall: float | None = None
    min_citation_acc: float | None = None
    min_faithfulness: float | None = None


def format_table(report: EvalReport, thresholds: Thresholds, k: int) -> str:
    rows: list[tuple[str, str, str, str]] = []

    def row(name: str, value: float | None, minimum: float | None) -> None:
        val = "skipped" if value is None else f"{value:.3f}"
        thr = "-" if minimum is None else f"{minimum:.2f}"
        if value is None or minimum is None:
            status = "-"
        else:
            status = "PASS" if value >= minimum else "FAIL"
        rows.append((name, val, thr, status))

    row(f"recall@{k}", report.recall, thresholds.min_recall)
    row("citation_accuracy", report.citation_accuracy, thresholds.min_citation_acc)
    row("faithfulness", report.faithfulness, thresholds.min_faithfulness)

    header = ("metric", "value", "min", "status")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(4)]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
        "  ".join("-" * widths[i] for i in range(4)),
    ]
    lines += ["  ".join(r[i].ljust(widths[i]) for i in range(4)) for r in rows]
    lines.append(f"({report.n} golden items)")
    return "\n".join(lines)


def gate_failures(report: EvalReport, thresholds: Thresholds) -> list[str]:
    """Names of metrics below their threshold. Skipped metrics never fail."""
    failures: list[str] = []
    if thresholds.min_recall is not None and report.recall < thresholds.min_recall:
        failures.append("recall")
    if (
        thresholds.min_citation_acc is not None
        and report.citation_accuracy < thresholds.min_citation_acc
    ):
        failures.append("citation_accuracy")
    if (
        thresholds.min_faithfulness is not None
        and report.faithfulness is not None
        and report.faithfulness < thresholds.min_faithfulness
    ):
        failures.append("faithfulness")
    return failures


# -- CLI ------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.eval")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--index", type=Path, default=None, help="override INDEX_PATH")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--gate", action="store_true", help="exit 1 on threshold failure")
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--min-citation-acc", type=float, default=None)
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument(
        "--skip-faithfulness",
        action="store_true",
        help="skip the LLM-judged metric (deterministic metrics only)",
    )
    parser.add_argument("--max-items", type=int, default=None, help="evaluate a subset")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Composition root for the eval runtime (CLAUDE.md §2 rule 2 carve-out).
    from app.agent import Agent
    from app.config import Settings
    from app.embed_api import VoyageEmbedder
    from app.llm_api import AnthropicLLM
    from app.store_sqlitevec import SqliteVecStore

    settings = Settings.from_env()
    index_path = args.index if args.index is not None else settings.index_path
    llm = AnthropicLLM(api_key=settings.anthropic_api_key)
    # max_retries=8: eval is batch work, so patience is free — the backoff
    # rides out Voyage's 3 RPM cap on accounts without a payment method.
    embedder = VoyageEmbedder(
        api_key=settings.voyage_api_key, input_type="query", max_retries=8
    )
    store = SqliteVecStore(index_path, dim=embedder.dim)
    agent = Agent(llm, embedder, store)
    judge: LLM | None = None if args.skip_faithfulness else llm

    items = load_golden(args.golden)
    if args.max_items is not None:
        items = items[: args.max_items]

    report = evaluate(agent, embedder, store, items, k=args.k, judge=judge)
    thresholds = Thresholds(
        min_recall=args.min_recall,
        min_citation_acc=args.min_citation_acc,
        min_faithfulness=args.min_faithfulness,
    )
    print(format_table(report, thresholds, args.k))

    if args.gate:
        failures = gate_failures(report, thresholds)
        if failures:
            print(f"GATE FAILED: {', '.join(failures)}", file=sys.stderr)
            return 1
        print("gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
