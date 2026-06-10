# Owner: Workstream A
"""Voyage AI implementation of the Embedder protocol.

`input_type` controls Voyage's asymmetric embedding mode — "document" for the
ingest path, "query" for the retrieval path. The composition root constructs
one instance per role; the WS-C ingest pipeline keeps the default.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import voyageai

# Verified in account on 2026-06-09. voyage-3.5 returns 1024-dim vectors.
DEFAULT_MODEL = "voyage-3.5"
DEFAULT_DIM = 1024


class VoyageEmbedder:
    """Embedder Protocol adapter against Voyage AI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        input_type: str = "document",
        max_retries: int = 4,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key required when no pre-built client is provided")
            client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        self._client = client
        self._model = model
        self._dim = dim
        self._input_type = input_type
        self._max_retries = max_retries

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._call_with_retry(list(texts))
        return [list(v) for v in result.embeddings]

    def _call_with_retry(self, texts: list[str]) -> Any:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._client.embed(
                    texts,
                    model=self._model,
                    input_type=self._input_type,
                )
            except Exception as e:
                # voyageai's exception class hierarchy varies across versions, so
                # we fall back to string-matching on transient signals.
                msg = str(e).lower()
                transient = (
                    "429" in msg
                    or "rate" in msg
                    or "500" in msg
                    or "502" in msg
                    or "503" in msg
                    or "504" in msg
                    or "timeout" in msg
                    or "connection" in msg
                )
                if not transient:
                    raise
                last_exc = e
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
        assert last_exc is not None
        raise last_exc
