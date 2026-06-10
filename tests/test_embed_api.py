"""Unit tests for VoyageEmbedder. No live network — uses an injected fake client.

The opt-in integration test at the bottom calls the real API when
RUN_INTEGRATION=1 and VOYAGE_API_KEY are set.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from app.embed_api import DEFAULT_DIM, DEFAULT_MODEL, VoyageEmbedder


class _FakeVoyage:
    def __init__(
        self,
        results: list[Any] | None = None,
        raise_each: list[Exception | None] | None = None,
    ) -> None:
        self._results = list(results or [])
        self._raise_each = list(raise_each or [])
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def embed(self, texts: list[str], **kwargs: Any) -> Any:
        self.calls.append((texts, kwargs))
        if self._raise_each:
            exc = self._raise_each.pop(0)
            if exc is not None:
                raise exc
        return self._results.pop(0)


def _result(*vectors: list[float]) -> Any:
    return SimpleNamespace(embeddings=list(vectors), total_tokens=sum(len(v) for v in vectors))


# -- Construction --------------------------------------------------------------


def test_requires_key_when_no_client() -> None:
    with pytest.raises(ValueError, match="api_key required"):
        VoyageEmbedder()


def test_dim_property_no_network() -> None:
    client = _FakeVoyage(results=[])
    emb = VoyageEmbedder(client=client, dim=1024)
    assert emb.dim == 1024
    assert client.calls == []


def test_default_model_and_dim() -> None:
    client = _FakeVoyage(results=[_result([0.1] * 1024)])
    emb = VoyageEmbedder(client=client)
    assert emb.dim == DEFAULT_DIM == 1024
    emb.embed(["hi"])
    assert client.calls[0][1]["model"] == DEFAULT_MODEL == "voyage-3.5"


# -- embed() -------------------------------------------------------------------


def test_embed_returns_vectors_in_order() -> None:
    vecs = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    client = _FakeVoyage(results=[_result(*vecs)])
    emb = VoyageEmbedder(client=client)
    out = emb.embed(["a", "b", "c"])
    assert len(out) == 3
    assert [v[0] for v in out] == [0.1, 0.2, 0.3]
    sent_texts, sent_kwargs = client.calls[0]
    assert sent_texts == ["a", "b", "c"]
    assert sent_kwargs == {"model": "voyage-3.5", "input_type": "document"}


def test_embed_empty_short_circuits() -> None:
    client = _FakeVoyage()
    emb = VoyageEmbedder(client=client)
    assert emb.embed([]) == []
    assert client.calls == []


def test_input_type_is_configurable() -> None:
    client = _FakeVoyage(results=[_result([0.0] * 1024)])
    emb = VoyageEmbedder(client=client, input_type="query")
    emb.embed(["search me"])
    assert client.calls[0][1]["input_type"] == "query"


# -- Retry behavior ------------------------------------------------------------


def test_retries_on_rate_limit_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.embed_api.time.sleep", lambda _: None)
    client = _FakeVoyage(
        results=[_result([0.0] * 1024)],
        raise_each=[RuntimeError("HTTP 429 rate limit exceeded"), None],
    )
    emb = VoyageEmbedder(client=client, max_retries=3)
    out = emb.embed(["a"])
    assert out == [[0.0] * 1024]
    assert len(client.calls) == 2


def test_retries_on_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.embed_api.time.sleep", lambda _: None)
    client = _FakeVoyage(
        results=[_result([0.0] * 1024)],
        raise_each=[RuntimeError("HTTP 503 service unavailable"), None],
    )
    emb = VoyageEmbedder(client=client, max_retries=3)
    emb.embed(["a"])
    assert len(client.calls) == 2


def test_does_not_retry_on_non_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.embed_api.time.sleep", lambda _: None)
    client = _FakeVoyage(raise_each=[ValueError("invalid input: text too long")])
    emb = VoyageEmbedder(client=client, max_retries=3)
    with pytest.raises(ValueError, match="invalid input"):
        emb.embed(["a"])
    assert len(client.calls) == 1


def test_exhausts_retries_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.embed_api.time.sleep", lambda _: None)
    client = _FakeVoyage(
        raise_each=[
            RuntimeError("HTTP 429"),
            RuntimeError("HTTP 429"),
            RuntimeError("HTTP 429"),
        ],
    )
    emb = VoyageEmbedder(client=client, max_retries=3)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        emb.embed(["a"])
    assert len(client.calls) == 3


# -- Integration (opt-in) ------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1" or not os.environ.get("VOYAGE_API_KEY"),
    reason="RUN_INTEGRATION=1 and VOYAGE_API_KEY required",
)
def test_integration_embed() -> None:
    emb = VoyageEmbedder(api_key=os.environ["VOYAGE_API_KEY"])
    out = emb.embed(["hello world"])
    assert len(out) == 1
    assert len(out[0]) == DEFAULT_DIM == 1024
