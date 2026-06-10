# Owner: Workstream E
"""FastAPI app: POST /ask, GET /healthz.

This module is the composition root for the API runtime shape — the one place
(besides the eval/ingest CLIs) allowed to import concrete adapters (CLAUDE.md
§2 rule 2).

Cold-start posture (source plan §8): importing this module constructs nothing
heavy — adapters and the agent are built inside the lifespan, at startup, and
only when no agent was injected. Tests inject a fake agent via create_app().
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Protocol

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


class AskAgent(Protocol):
    def ask(self, question: str) -> dict[str, Any]: ...


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="natural-language research question")


class Citation(BaseModel):
    arxiv_id: str
    title: str


class Step(BaseModel):
    tool: str
    args: dict[str, Any]
    result_summary: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    steps: list[Step]
    truncated: bool


def _build_agent() -> AskAgent:
    """Construct the real adapter stack from env settings."""
    from app.agent import Agent
    from app.config import Settings
    from app.embed_api import VoyageEmbedder
    from app.llm_api import AnthropicLLM
    from app.store_sqlitevec import SqliteVecStore

    settings = Settings.from_env()
    llm = AnthropicLLM(api_key=settings.anthropic_api_key)
    embedder = VoyageEmbedder(api_key=settings.voyage_api_key, input_type="query")
    store = SqliteVecStore(settings.index_path, dim=embedder.dim)
    return Agent(llm, embedder, store)


def create_app(agent: AskAgent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        app_.state.agent = agent if agent is not None else _build_agent()
        yield

    app_ = FastAPI(title="arxiv-rag", lifespan=lifespan)

    @app_.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        # Sync endpoint on purpose: Agent.ask blocks on HTTP + sqlite, and
        # FastAPI runs sync endpoints in a worker thread off the event loop.
        result = request.app.state.agent.ask(req.question)
        return AskResponse.model_validate(result)

    @app_.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app_


app = create_app()
