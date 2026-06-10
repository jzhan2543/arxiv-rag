"""Unit tests for the bounded agent loop, driven by a scripted fake LLM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent import NO_SOURCES_FALLBACK, SYSTEM_PROMPT, Agent
from app.ingest import PaperMeta
from app.interfaces import Chunk

# -- Fakes ----------------------------------------------------------------------


class ScriptedLLM:
    """Returns canned complete() responses in order; records every call."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self._script = list(script)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((list(messages), tools))
        return self._script.pop(0)


class FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    @property
    def dim(self) -> int:
        return 2


class FakeStore:
    """search() pops result sets in order; repeats the last one when exhausted."""

    def __init__(self, result_sets: list[list[Chunk]]) -> None:
        self._sets = list(result_sets)

    def upsert(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> None:
        raise NotImplementedError

    def search(self, vector: list[float], k: int) -> list[Chunk]:
        if len(self._sets) > 1:
            return self._sets.pop(0)[:k]
        return self._sets[0][:k]


def _chunk(arxiv_id: str, score: float = 0.9) -> Chunk:
    return Chunk(
        id=f"{arxiv_id}#0",
        text=f"text {arxiv_id}",
        arxiv_id=arxiv_id,
        title=f"Title {arxiv_id}",
        score=score,
    )


def _tool_call(name: str, args: dict[str, Any], call_id: str = "t1") -> dict[str, Any]:
    return {"content": None, "tool_calls": [{"id": call_id, "name": name, "arguments": args}]}


def _text(content: str) -> dict[str, Any]:
    return {"content": content, "tool_calls": []}


SURVEY_CHUNK = {"arxiv_id": "2501.09136", "title": "Survey", "text": "four patterns"}


# -- Happy path -------------------------------------------------------------------


def test_happy_path_retrieve_then_synthesize() -> None:
    llm = ScriptedLLM(
        [
            _tool_call("retrieve", {"query": "agentic patterns"}),
            _tool_call("synthesize", {"question": "What patterns?", "chunks": [SURVEY_CHUNK]}),
            _text("Reflection, planning, tool use, multi-agent [2501.09136]."),  # inside synth
        ]
    )
    agent = Agent(llm, FakeEmbedder(), FakeStore([[_chunk("2501.09136")]]))

    result = agent.ask("What patterns?")

    assert result["truncated"] is False
    assert result["answer"] == "Reflection, planning, tool use, multi-agent [2501.09136]."
    assert result["citations"] == [{"arxiv_id": "2501.09136", "title": "Survey"}]
    assert [s["tool"] for s in result["steps"]] == ["retrieve", "synthesize"]
    assert result["steps"][0]["result_summary"].startswith("1 chunks")


def test_system_prompt_sent_verbatim_and_tools_exposed() -> None:
    llm = ScriptedLLM([_text("direct")])
    Agent(llm, FakeEmbedder(), FakeStore([[]])).ask("q")

    messages, tools = llm.calls[0]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "q"}
    assert tools is not None
    assert [t["name"] for t in tools] == ["retrieve", "arxiv_search", "synthesize"]


def test_tool_results_fed_back_as_anthropic_blocks() -> None:
    llm = ScriptedLLM(
        [
            _tool_call("retrieve", {"query": "q"}, call_id="call_abc"),
            _text("done"),
        ]
    )
    Agent(llm, FakeEmbedder(), FakeStore([[_chunk("X")]])).ask("q")

    second_call_messages = llm.calls[1][0]
    assistant = second_call_messages[-2]
    tool_result = second_call_messages[-1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["id"] == "call_abc"
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "call_abc"
    assert "X#0" in tool_result["content"][0]["content"]


# -- Zero-retrieval fallback --------------------------------------------------------


def test_zero_retrieval_falls_back_to_arxiv_search() -> None:
    live_paper = PaperMeta(arxiv_id="2310.11511", title="Self-RAG", abstract="reflection")
    llm = ScriptedLLM(
        [
            _tool_call("retrieve", {"query": "q"}),
            _tool_call("arxiv_search", {"query": "self-rag"}),
            _tool_call(
                "synthesize",
                {
                    "question": "q",
                    "chunks": [
                        {"arxiv_id": "2310.11511", "title": "Self-RAG", "abstract": "reflection"}
                    ],
                },
            ),
            _text("Self-RAG uses reflection [2310.11511]."),  # inside synthesize
        ]
    )
    agent = Agent(
        llm,
        FakeEmbedder(),
        FakeStore([[]]),  # local corpus has nothing
        arxiv_fetch=lambda q, n: [live_paper],
    )

    result = agent.ask("q")

    assert [s["tool"] for s in result["steps"]] == ["retrieve", "arxiv_search", "synthesize"]
    assert result["steps"][0]["result_summary"] == "0 chunks"
    assert result["steps"][1]["result_summary"] == "1 papers"
    assert result["truncated"] is False
    assert result["citations"] == [{"arxiv_id": "2310.11511", "title": "Self-RAG"}]


# -- Step cap ----------------------------------------------------------------------


def test_step_cap_forces_synthesize_and_sets_truncated() -> None:
    # Model never synthesizes: 5 retrieve rounds (the cap), each finding a chunk.
    llm = ScriptedLLM(
        [_tool_call("retrieve", {"query": f"q{i}"}) for i in range(5)]
        + [_text("Forced grounded answer [A].")]  # the forced synthesize's inner call
    )
    store = FakeStore([[_chunk("A")], [_chunk("B")], [_chunk("C")], [_chunk("D")], [_chunk("E")]])
    agent = Agent(llm, FakeEmbedder(), store, max_steps=5)

    result = agent.ask("q")

    assert result["truncated"] is True
    assert result["answer"] == "Forced grounded answer [A]."
    assert {c["arxiv_id"] for c in result["citations"]} == {"A", "B", "C", "D", "E"}
    tools_used = [s["tool"] for s in result["steps"]]
    assert tools_used == ["retrieve"] * 5 + ["synthesize"]
    assert result["steps"][-1]["args"] == {"forced_after_step_cap": True}
    # 5 loop rounds + 1 synthesize-internal call.
    assert len(llm.calls) == 6


def test_step_cap_with_no_sources_returns_fallback() -> None:
    llm = ScriptedLLM([_tool_call("retrieve", {"query": "q"}) for _ in range(3)])
    agent = Agent(llm, FakeEmbedder(), FakeStore([[]]), max_steps=3)

    result = agent.ask("q")

    assert result["truncated"] is True
    assert result["answer"] == NO_SOURCES_FALLBACK
    assert result["citations"] == []
    assert [s["tool"] for s in result["steps"]] == ["retrieve"] * 3


# -- Robustness --------------------------------------------------------------------


def test_direct_text_answer_without_tools() -> None:
    llm = ScriptedLLM([_text("Direct answer, no tools.")])
    result = Agent(llm, FakeEmbedder(), FakeStore([[]])).ask("q")
    assert result == {
        "answer": "Direct answer, no tools.",
        "citations": [],
        "steps": [],
        "truncated": False,
    }


def test_unknown_tool_is_reported_and_loop_continues() -> None:
    llm = ScriptedLLM(
        [
            _tool_call("bogus_tool", {"x": 1}),
            _tool_call("synthesize", {"question": "q", "chunks": [SURVEY_CHUNK]}),
            _text("ok [2501.09136]"),
        ]
    )
    result = Agent(llm, FakeEmbedder(), FakeStore([[]])).ask("q")

    assert result["steps"][0]["tool"] == "bogus_tool"
    assert "unknown tool" in result["steps"][0]["result_summary"]
    assert result["truncated"] is False
    assert result["answer"] == "ok [2501.09136]"


def test_tool_exception_is_surfaced_to_model_not_raised() -> None:
    def exploding_fetch(query: str, max_results: int) -> list[PaperMeta]:
        raise RuntimeError("arxiv is down")

    llm = ScriptedLLM(
        [
            _tool_call("arxiv_search", {"query": "q"}),
            _text("Cannot search right now."),
        ]
    )
    agent = Agent(llm, FakeEmbedder(), FakeStore([[]]), arxiv_fetch=exploding_fetch)

    result = agent.ask("q")

    assert "error: arxiv is down" in result["steps"][0]["result_summary"]
    # The error went back to the model as a tool_result and the loop went on.
    follow_up = llm.calls[1][0][-1]["content"][0]
    assert follow_up["type"] == "tool_result"
    assert "arxiv is down" in follow_up["content"]
    assert result["answer"] == "Cannot search right now."
