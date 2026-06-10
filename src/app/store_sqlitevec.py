# Owner: Workstream B
"""sqlite-vec implementation of the VectorStore protocol.

Schema:
    embeddings   vec0 virtual table with +id (auxiliary TEXT) + embedding (float[dim])
                 using cosine distance.
    chunks       normal table with id (PK), arxiv_id, title, text.
    Join on id.

Score convention: returned `score` is `1.0 - cosine_distance`. Higher is more
similar. Search results are ordered by ascending distance (= descending score).

Only the metadata keys `arxiv_id`, `title`, `text` are persisted from each
upsert metadata dict; extra keys are silently dropped (the protocol allows
arbitrary keys, so the contract is "we keep what we need").
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import sqlite_vec

from app.interfaces import Chunk

_VEC_TABLE = "embeddings"
_META_TABLE = "chunks"


class SqliteVecStore:
    """VectorStore backed by sqlite-vec."""

    def __init__(self, path: Path | str, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        # Ensure parent dir exists (sqlite3 won't create it).
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(p)
        self._dim = dim
        self._conn = self._connect()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _init_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {_VEC_TABLE} USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{self._dim}] distance_metric=cosine
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_META_TABLE} (
                id TEXT PRIMARY KEY,
                arxiv_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT ''
            )
            """
        )
        self._conn.commit()

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(vectors) == len(metadata)):
            raise ValueError(
                f"upsert lengths must match: "
                f"ids={len(ids)} vectors={len(vectors)} metadata={len(metadata)}"
            )
        if not ids:
            return
        for i, vec in enumerate(vectors):
            if len(vec) != self._dim:
                raise ValueError(
                    f"vector {i} has dim {len(vec)}, expected {self._dim}"
                )

        meta_rows = [
            (
                ids[i],
                str(metadata[i].get("arxiv_id", "")),
                str(metadata[i].get("title", "")),
                str(metadata[i].get("text", "")),
            )
            for i in range(len(ids))
        ]
        vec_rows = [
            (ids[i], _serialize(vectors[i])) for i in range(len(ids))
        ]
        placeholders = ",".join("?" * len(ids))

        with self._conn:
            # Delete-then-insert for idempotent upsert.
            self._conn.execute(
                f"DELETE FROM {_VEC_TABLE} WHERE id IN ({placeholders})", ids
            )
            self._conn.execute(
                f"DELETE FROM {_META_TABLE} WHERE id IN ({placeholders})", ids
            )
            self._conn.executemany(
                f"INSERT INTO {_META_TABLE} (id, arxiv_id, title, text) "
                f"VALUES (?, ?, ?, ?)",
                meta_rows,
            )
            self._conn.executemany(
                f"INSERT INTO {_VEC_TABLE} (id, embedding) VALUES (?, ?)",
                vec_rows,
            )

    def search(self, vector: list[float], k: int) -> list[Chunk]:
        if len(vector) != self._dim:
            raise ValueError(
                f"query vector has dim {len(vector)}, expected {self._dim}"
            )
        if k <= 0:
            return []

        rows: Iterable[tuple[Any, ...]] = self._conn.execute(
            f"""
            SELECT e.id, e.distance, c.arxiv_id, c.title, c.text
            FROM {_VEC_TABLE} e
            JOIN {_META_TABLE} c ON e.id = c.id
            WHERE e.embedding MATCH ? AND k = ?
            ORDER BY e.distance
            """,
            (_serialize(vector), k),
        ).fetchall()
        return [
            Chunk(
                id=row[0],
                text=row[4],
                arxiv_id=row[2],
                title=row[3],
                score=1.0 - float(row[1]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()


def _serialize(vec: list[float]) -> bytes:
    """Pack a Python list of floats into sqlite-vec's expected byte format."""
    return cast(bytes, sqlite_vec.serialize_float32(vec))
