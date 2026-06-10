"""Unit tests for SqliteVecStore. Each test uses a fresh tmp_path .db file."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.store_sqlitevec import SqliteVecStore

# Use a small dim for fast tests; production uses 1024 (matching Voyage).
TEST_DIM = 8


def _store(tmp_path: Path, dim: int = TEST_DIM) -> SqliteVecStore:
    return SqliteVecStore(tmp_path / "test.db", dim=dim)


def _vec(seed: float, dim: int = TEST_DIM) -> list[float]:
    """Deterministic seedable test vector. Simple linear progression, not normalized."""
    return [seed + 0.01 * i for i in range(dim)]


# -- Construction --------------------------------------------------------------


def test_creates_schema(tmp_path: Path) -> None:
    s = _store(tmp_path)
    # Should be able to search a freshly created empty store.
    assert s.search(_vec(0.1), k=5) == []


def test_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "dir"
    s = SqliteVecStore(nested / "test.db", dim=TEST_DIM)
    assert (nested / "test.db").exists()
    s.close()


def test_rejects_nonpositive_dim(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        SqliteVecStore(tmp_path / "x.db", dim=0)


def test_init_is_idempotent(tmp_path: Path) -> None:
    """Re-opening a store with the same path should not fail."""
    s1 = _store(tmp_path)
    s1.upsert(
        ["a"], [_vec(0.5)], [{"arxiv_id": "x", "title": "t", "text": "txt"}]
    )
    s1.close()
    s2 = _store(tmp_path)
    results = s2.search(_vec(0.5), k=5)
    assert len(results) == 1
    assert results[0]["id"] == "a"


# -- upsert + search round trip ------------------------------------------------


def test_upsert_then_search(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(
        ids=["a", "b", "c"],
        vectors=[_vec(0.1), _vec(0.5), _vec(0.9)],
        metadata=[
            {"arxiv_id": "1", "title": "T1", "text": "first chunk"},
            {"arxiv_id": "2", "title": "T2", "text": "second chunk"},
            {"arxiv_id": "3", "title": "T3", "text": "third chunk"},
        ],
    )
    results = s.search(_vec(0.1), k=2)
    assert len(results) == 2
    # Closest to query 0.1 is "a"; results sorted by ascending distance.
    assert results[0]["id"] == "a"
    assert results[0]["arxiv_id"] == "1"
    assert results[0]["title"] == "T1"
    assert results[0]["text"] == "first chunk"
    # Score is 1 - cosine_distance: higher means more similar.
    assert results[0]["score"] > results[1]["score"]


def test_search_orders_by_score_descending(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(
        ids=[f"chunk-{i}" for i in range(5)],
        vectors=[_vec(0.1 * i) for i in range(5)],
        metadata=[{"arxiv_id": f"{i}", "title": "", "text": ""} for i in range(5)],
    )
    results = s.search(_vec(0.1), k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_upsert_is_idempotent(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(["x"], [_vec(0.5)], [{"arxiv_id": "a", "title": "t1", "text": "v1"}])
    s.upsert(["x"], [_vec(0.5)], [{"arxiv_id": "a", "title": "t2", "text": "v2"}])
    results = s.search(_vec(0.5), k=10)
    assert len(results) == 1
    # The second upsert overwrote the metadata.
    assert results[0]["title"] == "t2"
    assert results[0]["text"] == "v2"


def test_upsert_empty_is_no_op(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert([], [], [])
    assert s.search(_vec(0.1), k=5) == []


def test_metadata_only_persists_known_keys(tmp_path: Path) -> None:
    """Extra metadata keys (not in the contract) are silently dropped."""
    s = _store(tmp_path)
    s.upsert(
        ["x"],
        [_vec(0.5)],
        [
            {
                "arxiv_id": "a",
                "title": "t",
                "text": "v",
                "extra_field": "should-be-dropped",
                "version": 7,
            }
        ],
    )
    results = s.search(_vec(0.5), k=1)
    assert len(results) == 1
    # Chunk TypedDict only has id/text/arxiv_id/title/score — no extras leak.
    assert set(results[0].keys()) == {"id", "text", "arxiv_id", "title", "score"}


def test_metadata_missing_fields_default_to_empty_string(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(["x"], [_vec(0.5)], [{"arxiv_id": "a"}])
    results = s.search(_vec(0.5), k=1)
    assert results[0]["arxiv_id"] == "a"
    assert results[0]["title"] == ""
    assert results[0]["text"] == ""


# -- Edge cases ----------------------------------------------------------------


def test_search_empty_store(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.search(_vec(0.5), k=10) == []


def test_search_k_zero_short_circuits(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(["a"], [_vec(0.5)], [{"arxiv_id": "x", "title": "", "text": ""}])
    assert s.search(_vec(0.5), k=0) == []


def test_search_k_negative_short_circuits(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.search(_vec(0.5), k=-1) == []


def test_search_k_larger_than_store_returns_all(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.upsert(
        ids=["a", "b"],
        vectors=[_vec(0.1), _vec(0.5)],
        metadata=[{"arxiv_id": "1", "title": "", "text": ""}] * 2,
    )
    results = s.search(_vec(0.1), k=100)
    assert len(results) == 2


# -- Dim mismatches ------------------------------------------------------------


def test_upsert_dim_mismatch_raises(tmp_path: Path) -> None:
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="dim"):
        s.upsert(["x"], [[0.1] * (TEST_DIM + 1)], [{"arxiv_id": "x"}])


def test_search_dim_mismatch_raises(tmp_path: Path) -> None:
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="dim"):
        s.search([0.1] * (TEST_DIM + 1), k=5)


def test_upsert_unequal_lengths_raises(tmp_path: Path) -> None:
    s = _store(tmp_path)
    with pytest.raises(ValueError, match="lengths must match"):
        s.upsert(["a", "b"], [_vec(0.1)], [{"arxiv_id": "x"}])


# -- Fixture index sanity ------------------------------------------------------


def test_fixture_index_exists_and_loads() -> None:
    """The committed fixture index from build_fixture_index.py should be loadable."""
    repo_root = Path(__file__).resolve().parent.parent
    fixture = repo_root / "tests" / "fixtures" / "index.db"
    if not fixture.exists():
        pytest.skip(
            f"fixture not built yet at {fixture}; run "
            f"tests/fixtures/build_fixture_index.py"
        )
    s = SqliteVecStore(fixture, dim=1024)
    try:
        # Query with a valid unit vector (an all-zeros vector has undefined
        # cosine distance) and confirm we get hits (10 chunks committed).
        query = [1.0] + [0.0] * 1023
        results = s.search(query, k=5)
        assert len(results) == 5
        # All chunks have arxiv_id populated.
        assert all(r["arxiv_id"] for r in results)
    finally:
        s.close()
