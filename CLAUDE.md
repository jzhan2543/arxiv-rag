# CLAUDE.md — Coordination doc for `arxiv-rag`

This file is the contract between the six parallel workstreams. Read §2 (Golden rules) and §4 (your lane) before touching anything.

## 1. Project one-liner & non-goals

**Goal:** A bounded (max ~5 steps, 3 tools) agentic RAG service over a small arXiv "agentic RAG" corpus, served by FastAPI on Cloud Run. v0 doubles as a hands-on Docker + GitHub Actions + Cloud Run learning project.

**v0 NON-GOALS** (do not introduce, even if convenient):
- No GPU; no self-hosted models. LLM + embeddings are HTTP API calls only (Anthropic + Voyage).
- No planning, reflection, or multi-agent loops. One bounded tool-calling loop, three tools.
- No hosted vector DB (pgvector / Qdrant). `sqlite-vec` is the single embedded store for v0.
- No full-text PDF/HTML ingestion. Abstracts only.

## 2. Golden rules

1. **Never edit `src/app/interfaces.py`.** It is the frozen contract for all six workstreams. If something needs to change, stop and open an issue/PR for WS0.
2. **Program to the Protocols, not concretions.** Only `main.py` (composition root) and the eval CLI may import concrete adapters; everything else depends on the protocols.
3. **API-based LLM + embeddings only.** No `torch`, `transformers`, or CUDA. The container makes HTTP calls and stays lean.
4. **Keep the image lean.** Multi-stage Dockerfile, slim runtime, non-root user, no compilers in the final layer.
5. **Stay in your lane.** See §4 for owned files. If you need to touch something outside, stop and ask.
6. **No live network in unit tests.** Mock HTTP via `respx`/`httpx_mock`, use recorded fixtures, or use fakes.

## 3. Architecture

```
POST /ask ──▶ FastAPI service (Cloud Run, gen2)
              │
              ├── Bounded agent loop (max ~5 steps)
              │     tools: arxiv_search, retrieve, synthesize
              │
              ├── LLM (HTTP)      ── Anthropic Messages API
              ├── Embedder (HTTP) ── Voyage AI
              └── VectorStore     ── sqlite-vec file at INDEX_PATH
```

Two runtime shapes share one image:
- **API service** (Cloud Run service): serves `/ask`, reads the sqlite-vec index from `INDEX_PATH`.
- **Ingestion** (`python -m app.ingest`): in v0 runs on the GitHub Actions runner; graduates to a Cloud Run Job in v0.1.

### Request / response

`POST /ask` request body: `{"question": "..."}`.

Response:
```json
{
  "answer": "string",
  "citations": [{"arxiv_id": "2501.09136", "title": "..."}],
  "steps": [{"tool": "retrieve", "args": {}, "result_summary": "..."}],
  "truncated": false
}
```

`GET /healthz` returns `{"status": "ok"}` with 200.

## 4. Workstreams & owned files

| WS | Lane | Owned files | DoD |
|----|------|-------------|-----|
| 0 | Contracts | `src/app/interfaces.py`, `src/app/config.py`, `pyproject.toml`, `uv.lock`, `CLAUDE.md`, `README.md`, `.gitignore`, `.dockerignore` | mypy strict clean, smoke tests pass |
| A | Model adapters | `src/app/llm_api.py`, `src/app/embed_api.py`, `tests/test_llm_api.py`, `tests/test_embed_api.py` | unit tests pass offline; opt-in integration test passes with keys |
| B | Vector store | `src/app/store_sqlitevec.py`, `tests/test_store_sqlitevec.py`, `tests/fixtures/build_fixture_index.py`, `tests/fixtures/index.db` | upsert/search/empty tests pass; fixture index builds reproducibly |
| C | Ingestion | `src/app/ingest.py`, `tests/test_ingest.py` | runs end-to-end against fake Embedder+Store; opt-in live run populates index.db |
| D | Agent + tools | `src/app/agent.py`, `src/app/tools.py`, `tests/test_agent.py`, `tests/test_tools.py` | happy path, 0-retrieval fallback to arxiv_search, step-cap → `truncated=true` |
| E | API + eval | `src/app/main.py`, `src/app/eval.py`, `tests/golden/qa.jsonl`, `tests/test_api.py`, `tests/test_eval.py` | `/ask` returns the §3 schema; `eval --gate` exits 1 on planted regression |
| F | Infra | `Dockerfile`, `docker-compose.yml`, `.github/workflows/{ci,deploy,ingest}.yml`, `infra/README.md` (GCP setup runbook) | `docker compose up` serves `/healthz` 200; CI workflow green on draft PR |

## 5. Commands

```sh
uv sync                       # install deps from uv.lock
uv run ruff check .           # lint
uv run ruff format --check .  # formatting check
uv run mypy src               # type check (strict)
uv run pytest -q              # unit tests
uv run python -m app.eval     # eval against fixtures (WS-E will wire)
uv run python -m app.ingest   # full arXiv ingest (live, slow) (WS-C will wire)
docker compose up             # local end-to-end (WS-F will wire)
```

## 6. Definition of done (per workstream)

Universal: all unit tests in your area pass offline, `mypy --strict` is clean, no edits outside your lane, `interfaces.py` untouched. See §4 for the WS-specific gate.

## 7. Environment / secrets

| Var | Used by | Where set |
|-----|---------|-----------|
| `ANTHROPIC_API_KEY` | WS-A LLM adapter | local `.env`, GH repo secret |
| `VOYAGE_API_KEY` | WS-A embedder | local `.env`, GH repo secret |
| `INDEX_PATH` | WS-B store, WS-E main | local default `data/index.db` |
| `RUN_INTEGRATION` | WS-A integration tests | set `=1` for live tests |
| `WIF_PROVIDER` | Stage 3 deploy workflow | GH repo secret |
| `WIF_SERVICE_ACCOUNT` | Stage 3 deploy workflow | GH repo secret |

## 8. arXiv politeness rule (WS-C, load-bearing)

The arXiv API caps to **1 request every 3 seconds, single connection** (per their official Terms of Use for the legacy APIs). Use the `arxiv` Python package with `Client(page_size=100, delay_seconds=3.0, num_retries=5)` and a descriptive User-Agent string. On HTTP 503/429, back off exponentially. Never parallelize arXiv requests. Cron frequency is weekly — more is unnecessary and risks throttling.
