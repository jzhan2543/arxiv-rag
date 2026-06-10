"""Unit tests for the eval harness. All LLM/store/embedder dependencies are fakes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.eval import (
    EvalReport,
    GoldenItem,
    ItemResult,
    Thresholds,
    evaluate,
    format_table,
    gate_failures,
    judge_faithfulness,
    load_golden,
)
from app.interfaces import Chunk

# -- Fakes ----------------------------------------------------------------------


class ConstEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    @property
    def dim(self) -> int:
        return 2


class QueueStore:
    """search() pops canned result lists in call order."""

    def __init__(self, result_sets: list[list[Chunk]]) -> None:
        self._sets = list(result_sets)

    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], k: int) -> list[Chunk]:
        return self._sets.pop(0)[:k]


class ScriptedAgent:
    """ask() returns canned results keyed by call order."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = list(results)

    def ask(self, question: str) -> dict[str, Any]:
        return self._results.pop(0)


class JudgeLLM:
    """LLM whose complete() pops canned content strings."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(messages)
        return {"content": self._contents.pop(0), "tool_calls": []}


def _chunk(arxiv_id: str) -> Chunk:
    return Chunk(
        id=f"{arxiv_id}#0",
        text=f"text {arxiv_id}",
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        score=0.9,
    )


def _item(q: str, expected: list[str], must_cite: list[str]) -> GoldenItem:
    return GoldenItem(q=q, expected_paper_ids=expected, reference_answer="ref", must_cite=must_cite)


def _agent_result(cited: list[str], answer: str = "answer") -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": [{"arxiv_id": c, "title": f"Title {c}"} for c in cited],
        "steps": [],
        "truncated": False,
    }


# -- load_golden -------------------------------------------------------------------


def test_load_golden_real_file_is_well_formed() -> None:
    golden = Path(__file__).resolve().parent / "golden" / "qa.jsonl"
    items = load_golden(golden)
    assert len(items) >= 20
    for item in items:
        assert item.q and item.reference_answer
        assert item.expected_paper_ids and item.must_cite
        # Golden ids must be versionless arXiv ids (match the index's id scheme).
        for pid in item.expected_paper_ids + item.must_cite:
            assert "v" not in pid.split(".")[-1], f"versioned id in golden set: {pid}"


def test_load_golden_parses_fields(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps(
            {
                "q": "q1",
                "expected_paper_ids": ["A"],
                "reference_answer": "r",
                "must_cite": ["A"],
            }
        )
        + "\n\n"  # blank lines tolerated
    )
    items = load_golden(p)
    assert items == [
        GoldenItem(q="q1", expected_paper_ids=["A"], reference_answer="r", must_cite=["A"])
    ]


# -- evaluate: recall ---------------------------------------------------------------


def test_recall_hit_and_miss() -> None:
    items = [_item("q1", ["A"], ["A"]), _item("q2", ["B"], ["B"])]
    store = QueueStore([[_chunk("A")], [_chunk("X")]])  # hit for q1, miss for q2
    agent = ScriptedAgent([_agent_result(["A"]), _agent_result(["B"])])

    report = evaluate(agent, ConstEmbedder(), store, items, k=5)

    assert report.items[0].recall == 1.0
    assert report.items[1].recall == 0.0
    assert report.recall == 0.5


def test_recall_any_expected_id_counts() -> None:
    items = [_item("q", ["A", "B"], ["A"])]
    store = QueueStore([[_chunk("B")]])  # any-of semantics
    agent = ScriptedAgent([_agent_result(["A"])])
    report = evaluate(agent, ConstEmbedder(), store, items)
    assert report.recall == 1.0


# -- evaluate: citation accuracy -----------------------------------------------------


def test_citation_accuracy_full_and_zero() -> None:
    items = [_item("q1", ["A"], ["A"]), _item("q2", ["B"], ["B"])]
    store = QueueStore([[_chunk("A")], [_chunk("B")]])
    agent = ScriptedAgent([_agent_result(["A"]), _agent_result([])])

    report = evaluate(agent, ConstEmbedder(), store, items)

    assert report.items[0].citation_accuracy == 1.0
    assert report.items[1].citation_accuracy == 0.0
    assert report.citation_accuracy == 0.5


def test_citation_accuracy_partial_and_extra_not_penalized() -> None:
    items = [_item("q", ["A"], ["A", "B"])]
    store = QueueStore([[_chunk("A")]])
    # Cites A (required), misses B (required), adds Z (extra, not penalized).
    agent = ScriptedAgent([_agent_result(["A", "Z"])])

    report = evaluate(agent, ConstEmbedder(), store, items)

    assert report.items[0].citation_accuracy == 0.5


# -- evaluate: faithfulness -----------------------------------------------------------


def test_faithfulness_skipped_without_judge() -> None:
    items = [_item("q", ["A"], ["A"])]
    report = evaluate(
        ScriptedAgent([_agent_result(["A"])]), ConstEmbedder(), QueueStore([[_chunk("A")]]), items
    )
    assert report.items[0].faithfulness is None
    assert report.faithfulness is None


def test_faithfulness_judged_with_context() -> None:
    items = [_item("q", ["A"], ["A"])]
    judge = JudgeLLM(['{"supported": 3, "total": 4}'])
    report = evaluate(
        ScriptedAgent([_agent_result(["A"], answer="the answer")]),
        ConstEmbedder(),
        QueueStore([[_chunk("A")]]),
        items,
        judge=judge,
    )
    assert report.items[0].faithfulness == 0.75
    # Judge saw both the retrieved context and the answer.
    user_msg = judge.calls[0][-1]["content"]
    assert "text A" in user_msg
    assert "the answer" in user_msg


def test_judge_malformed_output_scores_zero() -> None:
    judge = JudgeLLM(["I think it is mostly fine."])
    assert judge_faithfulness(judge, "answer", "context") == 0.0


def test_judge_no_claims_is_vacuously_faithful() -> None:
    judge = JudgeLLM(['{"supported": 0, "total": 0}'])
    assert judge_faithfulness(judge, "", "context") == 1.0


def test_judge_clamps_overcount() -> None:
    judge = JudgeLLM(['{"supported": 9, "total": 4}'])
    assert judge_faithfulness(judge, "answer", "context") == 1.0


def test_judge_extracts_json_from_chatter() -> None:
    judge = JudgeLLM(['Here is my verdict: {"supported": 1, "total": 2} as requested.'])
    assert judge_faithfulness(judge, "answer", "context") == 0.5


# -- Gate & table ----------------------------------------------------------------------


def _report(recall: float, cit: float, faith: float | None) -> EvalReport:
    return EvalReport(
        items=[
            ItemResult(
                question="q",
                recall=recall,
                citation_accuracy=cit,
                faithfulness=faith,
                answer="a",
                cited=[],
                retrieved=[],
            )
        ]
    )


def test_gate_passes_at_threshold() -> None:
    report = _report(0.7, 0.8, 0.85)
    t = Thresholds(min_recall=0.7, min_citation_acc=0.8, min_faithfulness=0.85)
    assert gate_failures(report, t) == []


def test_gate_fails_below_threshold() -> None:
    report = _report(0.5, 0.9, 0.2)
    t = Thresholds(min_recall=0.7, min_citation_acc=0.8, min_faithfulness=0.85)
    assert gate_failures(report, t) == ["recall", "faithfulness"]


def test_gate_ignores_unthresholded_and_skipped_metrics() -> None:
    report = _report(0.0, 0.0, None)
    assert gate_failures(report, Thresholds()) == []
    # faithfulness threshold set but metric skipped -> not a failure.
    assert gate_failures(report, Thresholds(min_faithfulness=0.9)) == []


def test_format_table_shows_status() -> None:
    report = _report(0.9, 0.5, None)
    table = format_table(report, Thresholds(min_recall=0.7, min_citation_acc=0.8), k=5)
    assert "recall@5" in table
    assert "PASS" in table
    assert "FAIL" in table
    assert "skipped" in table
    assert "(1 golden items)" in table
