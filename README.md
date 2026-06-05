# arxiv-rag

Bounded agentic RAG over arXiv "agentic-RAG" papers. v0 is a learning vehicle for Docker, GitHub Actions, and Cloud Run.

See [CLAUDE.md](./CLAUDE.md) for the workstream layout, file ownership, commands, and contracts.

## Quick start

```sh
uv sync
uv run pytest -q
uv run mypy src
```

Local end-to-end (after Workstream F lands):

```sh
docker compose up
```
