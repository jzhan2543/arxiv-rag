# Owner: Workstream D
"""Bounded tool-calling agent loop (max ~5 steps, 3 tools).

Hand-rolled rather than Pydantic AI: our frozen LLM protocol already returns
normalized {content, tool_calls}, so the loop is ~60 lines, while wiring that
protocol into Pydantic AI's Model abstraction would be more glue than loop.
WORKSTREAMS.md blesses this: "the contract is interfaces.py, not the
framework."

Message-shape note: the loop builds Anthropic-style content blocks (tool_use /
tool_result), because the AnthropicLLM adapter passes messages through to the
Messages API verbatim. A future non-Anthropic adapter must translate inside
the adapter — the loop stays unchanged.

Step accounting: one step = one LLM tool-decision round. On cap with gathered
sources, the loop force-calls synthesize (one extra LLM call inside the tool)
so the caller still gets a grounded answer, flagged truncated=true.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.interfaces import LLM, Embedder, Tool, VectorStore
from app.tools import ArxivSearchTool, RetrieveTool, SynthesizeTool

logger = logging.getLogger(__name__)

# Verbatim from the source plan §6.
SYSTEM_PROMPT = (
    "You answer questions about agentic-RAG research. Use `retrieve` first; "
    "use `arxiv_search` only if retrieval is insufficient; call `synthesize` "
    "to produce a final cited answer. You have at most 5 steps."
)

DEFAULT_MAX_STEPS = 5

NO_SOURCES_FALLBACK = (
    "I could not gather relevant sources within the step budget; no grounded "
    "answer is available."
)


class Agent:
    """Bounded loop over retrieve / arxiv_search / synthesize."""

    def __init__(
        self,
        llm: LLM,
        embedder: Embedder,
        store: VectorStore,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        retrieve_k: int = 5,
        arxiv_fetch: Any = None,
    ) -> None:
        self._llm = llm
        self._max_steps = max_steps
        self._synthesize = SynthesizeTool(llm)
        self._tools: list[Tool] = [
            RetrieveTool(embedder, store, default_k=retrieve_k),
            ArxivSearchTool(fetch=arxiv_fetch),
            self._synthesize,
        ]
        self._by_name: dict[str, Tool] = {t.name: t for t in self._tools}

    def ask(self, question: str) -> dict[str, Any]:
        """Run the loop. Returns the §3 response shape:
        {"answer", "citations", "steps", "truncated"}."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        specs = [t.spec() for t in self._tools]
        steps: list[dict[str, Any]] = []
        gathered: dict[str, dict[str, Any]] = {}  # arxiv_id/chunk id -> chunk-ish dict
        last_text: str | None = None

        for _ in range(self._max_steps):
            resp = self._llm.complete(messages, tools=specs)
            tool_calls = resp.get("tool_calls") or []
            if resp.get("content"):
                last_text = resp["content"]

            if not tool_calls:
                # Model answered in plain text without synthesize: accept it,
                # ungrounded (no citations) but within budget.
                return _result(last_text or "", [], steps, truncated=False)

            assistant_content: list[dict[str, Any]] = []
            if resp.get("content"):
                assistant_content.append({"type": "text", "text": resp["content"]})
            assistant_content.extend(
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                for tc in tool_calls
            )
            messages.append({"role": "assistant", "content": assistant_content})

            results_content: list[dict[str, Any]] = []
            for tc in tool_calls:
                name, args = tc["name"], dict(tc["arguments"])
                result, summary = self._execute(name, args)
                self._gather(name, result, gathered)
                steps.append({"tool": name, "args": args, "result_summary": summary})

                if name == self._synthesize.name and "answer" in result:
                    return _result(
                        result["answer"], result.get("citations", []), steps, truncated=False
                    )

                results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": results_content})

        # Step cap hit. Best effort: force-synthesize over whatever we gathered.
        if gathered:
            forced = self._synthesize(question=question, chunks=list(gathered.values()))
            steps.append(
                {
                    "tool": self._synthesize.name,
                    "args": {"forced_after_step_cap": True},
                    "result_summary": f"forced synthesize over {len(gathered)} gathered sources",
                }
            )
            return _result(forced["answer"], forced.get("citations", []), steps, truncated=True)
        return _result(last_text or NO_SOURCES_FALLBACK, [], steps, truncated=True)

    def _execute(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Run one tool call; never raises — errors go back to the model."""
        tool = self._by_name.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}, f"unknown tool: {name}"
        try:
            result = tool(**args)
        except Exception as e:  # noqa: BLE001 — surface to the model, keep looping
            logger.warning("tool %s failed: %s", name, e)
            return {"error": str(e)}, f"error: {e}"
        return result, _summarize(name, result)

    @staticmethod
    def _gather(name: str, result: dict[str, Any], gathered: dict[str, dict[str, Any]]) -> None:
        """Accumulate retrievable sources for the step-cap fallback synthesis."""
        if name == "retrieve":
            for c in result.get("chunks", []):
                gathered[str(c.get("id", c.get("arxiv_id", "")))] = c
        elif name == "arxiv_search":
            for p in result.get("papers", []):
                key = f"{p.get('arxiv_id', '')}#live"
                gathered[key] = {
                    "arxiv_id": p.get("arxiv_id", ""),
                    "title": p.get("title", ""),
                    "text": p.get("abstract", ""),
                }


def _summarize(name: str, result: dict[str, Any]) -> str:
    if name == "retrieve":
        chunks = result.get("chunks", [])
        ids = ", ".join(str(c.get("id", "?")) for c in chunks[:3])
        return f"{len(chunks)} chunks" + (f": {ids}" if ids else "")
    if name == "arxiv_search":
        return f"{len(result.get('papers', []))} papers"
    if name == "synthesize":
        return f"answer with {len(result.get('citations', []))} citations"
    return "ok"


def _result(
    answer: str,
    citations: list[dict[str, str]],
    steps: list[dict[str, Any]],
    truncated: bool,
) -> dict[str, Any]:
    return {"answer": answer, "citations": citations, "steps": steps, "truncated": truncated}
