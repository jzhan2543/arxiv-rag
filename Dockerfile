# syntax=docker/dockerfile:1
# Multi-stage: uv resolves deps in the builder; the runtime stage is plain
# python:3.12-slim with only the venv — no uv, no compilers, non-root.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependency layer first: cached until pyproject/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Project layer: hatchling needs README.md (pyproject readme field) to build
# the wheel; --no-editable installs the app package into the venv so the
# runtime stage doesn't need /app/src on the path.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM python:3.12-slim AS runtime
RUN useradd -m -u 1001 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Stage 3 topology: the sqlite-vec index is baked into the image (source plan
# §8 option 2). The committed fixture index is a real 10-paper semantically
# embedded corpus; a full re-ingest rebuilds the image. v0.1 moves this to a
# GCS-mounted volume so re-indexing stops requiring a deploy.
COPY --chown=appuser:appuser tests/fixtures/index.db /app/data/index.db

ENV PATH="/app/.venv/bin:$PATH" \
    INDEX_PATH=/app/data/index.db \
    PYTHONUNBUFFERED=1
USER appuser
EXPOSE 8080
# Cloud Run injects $PORT (8080 by default); exec so uvicorn is PID 1 and
# receives SIGTERM directly for clean scale-to-zero shutdown.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
