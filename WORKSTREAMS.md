# WORKSTREAMS

Launch briefs for the 6 parallel Claude Code sessions that build v0. Each session opens in its own git worktree, owns a small file set, and depends only on the Protocols in `src/app/interfaces.py` — **frozen, do not edit**.

**Before starting any workstream, read [CLAUDE.md](./CLAUDE.md) §2 (golden rules) and §4 (workstream table).** Shared context lives there; this doc has per-WS launch prompts only.

## Worktree map

| WS | Worktree dir                | Branch            |
|----|-----------------------------|-------------------|
| A  | `../docker-rag-ws-a`        | `ws-a-adapters`   |
| B  | `../docker-rag-ws-b`        | `ws-b-store`      |
| C  | `../docker-rag-ws-c`        | `ws-c-ingest`     |
| D  | `../docker-rag-ws-d`        | `ws-d-agent`      |
| E  | `../docker-rag-ws-e`        | `ws-e-api-eval`   |
| F  | `../docker-rag-ws-f`        | `ws-f-infra`      |

## Launch checklist (every session)

1. `cd` into your worktree dir.
2. **Create the venv on a Python with sqlite extension support** (sqlite-vec needs `enable_load_extension`; the pyenv-built Python on this machine lacks it): `uv venv --python /opt/homebrew/bin/python3.12`. Docker and CI Pythons are unaffected.
3. `uv run pytest -q` — confirm the smoke tests pass. uv lazy-installs the venv from the shared wheel cache; no explicit `uv sync` needed unless you're adding a new dep.
3. Read [CLAUDE.md](./CLAUDE.md) §2 + §4.
4. Read your WS prompt below. Implement.
5. Open a PR back to `main` when your DoD is hit.

> If you do run `uv sync` and it fails on a transient pypi timeout, `uv run <cmd>` works offline against the local cache. Use `uv sync --offline` to skip the index refresh deliberately.

Verified model availability (Anthropic + Voyage dry calls, 2026-06-09):
- LLM models in account: `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-5/4-6/4-7/4-8`.
- Voyage embedding model: `voyage-3.5`, dim **1024**.

---

## Workstream A — Model adapters

**Owned files:** `src/app/llm_api.py`, `src/app/embed_api.py`, `tests/test_llm_api.py`, `tests/test_embed_api.py`

Implement two Protocol adapters:

1. **`LLM` against the Anthropic Messages API.** Use the `anthropic` SDK. Tool use enabled — the WS-D agent loop drives multi-step tool calling, so `complete()` must surface `tool_calls` from the response. Default model `claude-haiku-4-5`; expose model as a constructor arg with `claude-sonnet-4-6` as a documented stronger alternative. Retry 429/5xx with exponential backoff.
2. **`Embedder` against Voyage AI.** Use `voyageai` SDK. Default model `voyage-3.5`, dim **1024**. Batch embeds (Voyage takes a list). `.dim` property returns 1024 without a network call.

**No live network in unit tests.** Mock via `respx` (already installed) or fakes. Add one opt-in integration test per adapter, gated by `os.environ.get("RUN_INTEGRATION") == "1"`, that calls the real API.

**DoD:** mypy strict clean against the protocols; offline unit tests pass; `RUN_INTEGRATION=1 uv run pytest tests/test_llm_api.py tests/test_embed_api.py` passes with `.env` keys.

---

## Workstream B — Vector store

**Owned files:** `src/app/store_sqlitevec.py`, `tests/test_store_sqlitevec.py`, `tests/fixtures/build_fixture_index.py`, `tests/fixtures/index.db`

Implement `VectorStore` on top of `sqlite-vec` (`sqlite_vec` Python pkg):

- Schema: one `vec0` virtual table for embeddings, a normal `chunks` table for metadata (`id`, `arxiv_id`, `title`, `text`), joined on `id`.
- `upsert(ids, vectors, metadata)`: delete-then-insert by id (idempotent). Persist only the metadata keys you've contracted on (`arxiv_id`, `title`, `text`); document this in a docstring.
- `search(vector, k)`: cosine similarity via sqlite-vec's `MATCH` syntax; return `list[Chunk]` sorted by score descending.

Build a tiny committed fixture index: `python tests/fixtures/build_fixture_index.py` writes `tests/fixtures/index.db` with ~10 deterministic chunks. Use a **fake embedder** that hashes text to a 1024-dim vector (match Voyage's dim so this index slots into prod-shaped tests). The `.db` file is whitelisted in `.gitignore` — commit it.

**DoD:** unit tests cover upsert, search, empty store, dim mismatch; fixture script is reproducible.

---

## Workstream C — Ingestion

**Owned files:** `src/app/ingest.py`, `tests/test_ingest.py`

arXiv → chunk → embed → upsert. Per CLAUDE.md §8 the arXiv API caps at **1 req / 3s, single connection**. Use the `arxiv` package with `Client(page_size=100, delay_seconds=3.0, num_retries=5)` and a descriptive User-Agent. Seed corpus with **arXiv:2501.09136** and its references; query terms within `cs.CL OR cs.IR OR cs.AI`; cap at ~300 papers.

For v0: **1 chunk per abstract** (chunk_id = `{arxiv_id}#0`). No full-text PDF/HTML.

`ingest.py` exposes `def run(embedder: Embedder, store: VectorStore, ...) -> int` taking the protocols by DI — no concrete-adapter imports. CLI: `python -m app.ingest` reads config, constructs concrete adapters, calls `run()`.

**DoD:** unit tests run end-to-end against fake `Embedder` + fake `VectorStore`; an opt-in marker (`pytest -m live`) actually hits arXiv and populates a real index file.

---

## Workstream D — Agent + tools

**Owned files:** `src/app/agent.py`, `src/app/tools.py`, `tests/test_agent.py`, `tests/test_tools.py`

Bounded tool-calling loop. **Pydantic AI is recommended** for the `UsageLimits` cap, but a ~40-line hand-rolled loop around `LLM.complete()` is fine — the contract is `interfaces.py`, not the framework.

Three tools:
- `arxiv_search(query) -> list[paper_meta]` — live arXiv search (respect the 3s delay).
- `retrieve(query, k=5) -> list[chunk]` — embed query via injected `Embedder`, call `VectorStore.search`.
- `synthesize(question, chunks) -> {answer, citations}` — terminal tool. Use the LLM to produce a cited answer; `citations` = `list[{arxiv_id, title}]` derived from the chunks passed in.

System prompt: *"You answer questions about agentic-RAG research. Use `retrieve` first; use `arxiv_search` only if retrieval is insufficient; call `synthesize` to produce a final cited answer. You have at most 5 steps."*

Hard cap: 5 steps. On cap → best-effort answer + `truncated: true`.

The agent accepts `LLM`, `VectorStore`, `Embedder` via constructor (DI). No concrete-adapter imports.

Output shape per CLAUDE.md §3: `{answer, citations, steps, truncated}` where `steps` is the tool-call trace.

**DoD:** unit tests use a fake LLM that scripts tool-call sequences. Cover (a) happy path: retrieve → synthesize; (b) 0-retrieval fallback: retrieve returns [] → arxiv_search → retrieve → synthesize; (c) cap hit → `truncated=true`.

---

## Workstream E — API + eval

**Owned files:** `src/app/main.py`, `src/app/eval.py`, `tests/golden/qa.jsonl`, `tests/test_api.py`, `tests/test_eval.py`

### `main.py` — FastAPI

- `POST /ask` body `{"question": "..."}`, response per CLAUDE.md §3.
- `GET /healthz` → `{"status": "ok"}`.
- Use FastAPI `lifespan` to wire DI. **Lazy-init heavy clients** (anthropic, voyage) — don't construct them at module import time. Keeps Cloud Run cold start low (source doc §8).

### `eval.py` — Hand-rolled + RAGAS

- Golden set at `tests/golden/qa.jsonl`, 20–30 hand-written items in shape `{q, expected_paper_ids, reference_answer, must_cite}`. Include the verbatim Singh et al. example from source doc §10:
  ```json
  {"q": "What four agentic design patterns does the Singh et al. survey identify?",
   "expected_paper_ids": ["2501.09136"],
   "reference_answer": "Reflection, planning, tool use, and multi-agent collaboration.",
   "must_cite": ["2501.09136"]}
  ```
- Metrics: `recall@k` (over `expected_paper_ids`), `citation_accuracy` (over `must_cite`), `faithfulness` (RAGAS, LLM-judge).
- CLI: `python -m app.eval` prints a table. `--gate --min-recall ... --min-citation-acc ... --min-faithfulness ...` exits non-zero on regression.
- Tests use the fixture index from WS-B and a mocked LLM (no live network in unit tests).

**DoD:** `/ask` returns the §3 schema with a mocked agent; `python -m app.eval --gate` exits 1 on a planted regression; offline tests green.

---

## Workstream F — Infra

**Owned files:** `Dockerfile`, `.dockerignore` (already exists; check it), `docker-compose.yml`, `.github/workflows/{ci,deploy,ingest}.yml`, `infra/README.md`

Implement per source doc §7 + §9:

- **`Dockerfile`** — multi-stage: `python:3.12-slim` builder with `uv` installed via `COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/`, two-stage `uv sync` for cached deps layer, `python:3.12-slim` runtime, non-root user, no compilers in final layer. `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]`.
- **`docker-compose.yml`** — `api` and `ingest` services sharing a `./data:/data` volume; both built from the same image; `ingest` overrides command to `python -m app.ingest`.
- **`.github/workflows/ci.yml`** — `on: pull_request, workflow_dispatch`. Two jobs: `lint-test` (ruff + mypy + pytest) → `eval-gate` (`needs: lint-test`) running `uv run python -m app.eval --index tests/fixtures/index.db --gate --min-recall 0.7 --min-citation-acc 0.8 --min-faithfulness 0.85` (the committed fixture index is semantically embedded — measured 1.000 on all three metrics, so those thresholds have margin). Both use `astral-sh/setup-uv@v5` with cache. Wire repo secrets `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`. Heads-up: a Voyage account without a payment method is capped at 3 RPM and the gate may flake until billing is unlocked.
- **`.github/workflows/deploy.yml`** — `on: push: branches: [main], workflow_dispatch`. Build → push to GHCR (`ghcr.io/jzhan2543/arxiv-rag:${{ github.sha }}`) → `google-github-actions/auth@v2` (WIF) → `google-github-actions/deploy-cloudrun@v2` flags `--min-instances=0 --cpu-boost --execution-environment=gen2`, service `arxiv-rag`, region `us-central1`. **Comment the WIF + deploy steps with a `TODO(Stage 3)` banner** — user has no GCP account yet. Leave the build+push wired so the workflow lints.
- **`.github/workflows/ingest.yml`** — `on: schedule: cron "0 6 * * 1", workflow_dispatch`. v0 runs `uv run python -m app.ingest` on the Actions runner. Note in a comment: "v0.1 graduates to a Cloud Run Job."
- **`infra/README.md`** — the GCP setup runbook from the plan file (`~/.claude/plans/users-jeffzhan-downloads-compass-artifa-snug-book.md`, Stage 3 prerequisite section). Project creation, billing + $1 budget alert, API enabling, SA with `roles/run.admin` + `roles/iam.serviceAccountUser`, WIF pool/provider with attribute condition `assertion.repository == 'jzhan2543/arxiv-rag'`, GH secrets `WIF_PROVIDER` / `WIF_SERVICE_ACCOUNT`.

**DoD:** `docker compose up` serves `/healthz` 200 locally; opening a draft PR with the WS-F branch runs `ci.yml` end-to-end and lands green; `infra/README.md` is followable by a human with no prior GCP context.
