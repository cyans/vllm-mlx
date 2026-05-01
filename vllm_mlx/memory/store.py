# SPDX-License-Identifier: Apache-2.0
"""SQLite store for the memory subsystem.

@CODE:MEMORY-01/store

Phase 1 only writes to ``vault_files``, ``vault_chunks``, ``fts_chunks``,
and ``meta``. The remaining tables (``chat_messages``,
``chat_summaries``, ``vault_themes``, ``vec_chunks``) are created up
front to avoid a schema migration when SPEC-MEMORY-02 lands. See
SPEC-MEMORY-01 §8 for column semantics.

Concurrency: WAL mode is enabled at connection time so multiple
readers can coexist with a single writer. ``sqlite3`` connections are
not thread-safe, so callers must keep the :class:`MemoryStore` confined
to one thread (or one event-loop thread). The integration into
``server.py`` performs all initial-scan work in a background asyncio
task, which already runs on a single thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Bumped manually when the schema changes. SPEC §8 mandates a hard error
# on schema drift; callers should detect a mismatch and either rebuild
# or refuse to start.
SCHEMA_VERSION = 1

# FTS5 default tokenizer ("unicode61") splits on whitespace and
# punctuation, which is sufficient for full-word Korean queries (the
# search target for SPEC §13). Trigram tokenization would enable
# substring matching but breaks BM25 ranking quality on English; we
# stick with the default for Phase 1 and revisit if Phase 4 evals show
# Korean recall problems. See module-level note in the indexer.
_FTS_TOKENIZER = "unicode61 remove_diacritics 2"


# ---------------------------------------------------------------------------
# Schema (forward-compat per SPEC-MEMORY-01 §8 / Path C)
# ---------------------------------------------------------------------------
_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vault_files (
    file_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    path      TEXT NOT NULL UNIQUE,
    mtime     REAL NOT NULL,
    sha256    TEXT NOT NULL,
    size      INTEGER NOT NULL,
    indexed_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_vault_files_path ON vault_files(path);

CREATE TABLE IF NOT EXISTS vault_chunks (
    chunk_id    TEXT PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES vault_files(file_id) ON DELETE CASCADE,
    header      TEXT NOT NULL DEFAULT '',
    text        TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    char_offset INTEGER NOT NULL,
    char_len    INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    tier        TEXT NOT NULL DEFAULT 'stm'
);

CREATE INDEX IF NOT EXISTS ix_vault_chunks_file ON vault_chunks(file_id);

-- Forward-compat: chat tables exist from day 1 but Phase 1 never writes here.
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id  TEXT PRIMARY KEY,
    request_id  TEXT NOT NULL,
    session_id  TEXT,
    role        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    tier        TEXT NOT NULL DEFAULT 'stm'
);

CREATE INDEX IF NOT EXISTS ix_chat_messages_session
    ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS ix_chat_messages_timestamp
    ON chat_messages(timestamp);

-- Forward-compat: SPEC-MEMORY-02 fills these via a daily consolidation job.
CREATE TABLE IF NOT EXISTS chat_summaries (
    session_id     TEXT PRIMARY KEY,
    summary_text   TEXT NOT NULL,
    period_start   TEXT NOT NULL,
    period_end     TEXT NOT NULL,
    source_count   INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS vault_themes (
    theme_id          TEXT PRIMARY KEY,
    summary_text      TEXT NOT NULL,
    period_start      TEXT NOT NULL,
    period_end        TEXT NOT NULL,
    source_chunk_ids  TEXT NOT NULL DEFAULT '[]',
    created_at        REAL NOT NULL
);

-- FTS5 BM25 index. ``content_rowid`` links to vault_chunks.rowid so we
-- can join FTS hits back to chunk metadata. Phase 1 only indexes vault
-- chunks here; chat rows are added in Phase 3.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_id UNINDEXED,
    source_type UNINDEXED,
    source_path UNINDEXED,
    text,
    tokenize='{_FTS_TOKENIZER}'
);
"""


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchHit:
    """One ranked search result, ready for the MCP tool envelope."""

    chunk_id: str
    source_type: str  # 'vault' | 'chat'
    source_path: str
    timestamp: str
    score: float
    excerpt: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def chunk_id_for(path: str, char_offset: int, char_len: int) -> str:
    """Deterministic chunk ID derived from path + offset + length.

    Same file at the same offset/length always yields the same id; this
    is what makes incremental re-indexing idempotent (REQ-E2 in Phase 4).
    """
    h = hashlib.sha256()
    h.update(path.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(char_offset).encode("ascii"))
    h.update(b"\x00")
    h.update(str(char_len).encode("ascii"))
    return h.hexdigest()[:32]


def file_sha256(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
class MemoryStore:
    """Thin wrapper around a SQLite connection with WAL + the SPEC schema."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        # Note: connection is opened lazily so an offline check (db_path
        # parent missing) does not raise at construction time.
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        """Open the connection and apply the schema (idempotent)."""
        if self._conn is not None:
            return

        # Ensure parent directory exists; SPEC §10 says volume disconnect
        # is a recoverable degraded state, so we surface OSError here and
        # let the caller decide whether to mark the subsystem unavailable.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # ``check_same_thread=False`` lets the FastAPI event-loop thread
        # share the conn with a background indexer task that runs on the
        # same loop. Higher-level callers must still serialize writes.
        self._conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,  # autocommit; we use explicit transactions
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

        # PRAGMAs first, then schema. WAL gives us safe one-writer / N-reader
        # concurrency without a separate lock manager (REQ-S3 / SPEC §10).
        self._conn.executescript(
            "PRAGMA journal_mode=WAL;\n"
            "PRAGMA synchronous=NORMAL;\n"
            "PRAGMA foreign_keys=ON;\n"
            "PRAGMA temp_store=MEMORY;\n"
        )
        self._conn.executescript(_SCHEMA_SQL)
        self._init_meta()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                logger.warning("[memory] error closing db: %s", exc)
            finally:
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryStore.open() must be called before use")
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Begin a write transaction with explicit BEGIN/COMMIT.

        Note: we set ``isolation_level=None`` on the connection so we
        manage transactions ourselves. This matches the WAL writer
        contract (one writer at a time; readers never block).
        """
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # -- meta ---------------------------------------------------------------

    def _init_meta(self) -> None:
        existing = self.get_meta("schema_version")
        if existing is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            self.set_meta("created_at", str(time.time()))
        elif int(existing) != SCHEMA_VERSION:
            # SPEC §8: "mismatch triggers a hard error with a documented
            # `vllm-mlx memory rebuild` recovery path". Phase 1 only has
            # version 1, so this branch is for forward-compat.
            raise RuntimeError(
                f"memory schema version mismatch: db={existing}, "
                f"code={SCHEMA_VERSION}. Run `vllm-mlx memory rebuild` to "
                f"recreate the database (CLI lands in Phase 3)."
            )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # -- vault writes -------------------------------------------------------

    def upsert_vault_file(
        self,
        *,
        path: str,
        mtime: float,
        sha256: str,
        size: int,
    ) -> int:
        """Insert or update a vault file row, returning the file_id.

        Idempotent: the same (path, mtime, sha256) triple yields the same
        file_id and skips chunk re-insertion at the caller (the indexer
        compares ``sha256`` before chunking).
        """
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO vault_files(path, mtime, sha256, size, indexed_at) "
            "VALUES(?, ?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET "
            "  mtime = excluded.mtime, "
            "  sha256 = excluded.sha256, "
            "  size = excluded.size, "
            "  indexed_at = excluded.indexed_at",
            (path, mtime, sha256, size, now),
        )
        # ``cur.lastrowid`` is unreliable on UPDATE; re-query to be safe.
        if cur.lastrowid:
            return cur.lastrowid
        row = self.conn.execute(
            "SELECT file_id FROM vault_files WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"upsert_vault_file lost row for path={path!r}"
            )
        return int(row["file_id"])

    def get_vault_file_sha(self, path: str) -> str | None:
        row = self.conn.execute(
            "SELECT sha256 FROM vault_files WHERE path = ?", (path,)
        ).fetchone()
        return row["sha256"] if row is not None else None

    def replace_chunks_for_file(
        self,
        *,
        file_id: int,
        source_path: str,
        chunks: Iterable[dict[str, Any]],
    ) -> int:
        """Replace all chunks for a file (delete + insert).

        Phase 1 always replaces because the indexer only rebuilds when
        sha256 differs; by then any previous chunks are stale. Returns
        the count of inserted rows.
        """
        # Remove old chunks (and their FTS rows).
        old = self.conn.execute(
            "SELECT chunk_id FROM vault_chunks WHERE file_id = ?",
            (file_id,),
        ).fetchall()
        for row in old:
            self.conn.execute(
                "DELETE FROM fts_chunks WHERE chunk_id = ?",
                (row["chunk_id"],),
            )
        self.conn.execute(
            "DELETE FROM vault_chunks WHERE file_id = ?", (file_id,)
        )

        count = 0
        for ch in chunks:
            self.conn.execute(
                "INSERT INTO vault_chunks(chunk_id, file_id, header, text, "
                "  chunk_index, char_offset, char_len, timestamp, tier) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'stm')",
                (
                    ch["chunk_id"],
                    file_id,
                    ch.get("header", ""),
                    ch["text"],
                    ch["chunk_index"],
                    ch["char_offset"],
                    ch["char_len"],
                    ch["timestamp"],
                ),
            )
            self.conn.execute(
                "INSERT INTO fts_chunks(chunk_id, source_type, source_path, text) "
                "VALUES(?, 'vault', ?, ?)",
                (ch["chunk_id"], source_path, ch["text"]),
            )
            count += 1
        return count

    # -- read paths ---------------------------------------------------------

    def count_vault_files(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM vault_files"
        ).fetchone()
        return int(row["c"])

    def count_vault_chunks(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM vault_chunks"
        ).fetchone()
        return int(row["c"])

    def search_bm25(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[SearchHit]:
        """Run a BM25 search over ``fts_chunks`` and return ranked hits.

        ``source_filter`` is one of ``"vault"``, ``"chat"``, ``"both"``
        or ``None`` (treated as both). FTS5 ``bm25()`` returns a negative
        relevance number — closer to zero = better — which we negate and
        then squash into ``[0, 1]`` via ``1 / (1 + |bm25|)`` so the score
        contract in REQ-U3 holds (positive float in [0,1]).
        """
        query = (query or "").strip()
        if not query:
            return []

        # Sanitize: drop FTS5 special characters that would otherwise
        # raise a syntax error for free-form natural-language queries.
        # We keep ``-`` so phrase exclusion still works if the model
        # passes it, and we wrap each remaining whitespace-separated
        # token with double quotes so MATCH treats them as literals.
        # This protects us from queries like ``foo:bar`` or ``"`` in
        # raw user input.
        match_expr = _normalize_fts5_query(query)
        if not match_expr:
            return []

        sql = (
            "SELECT chunk_id, source_type, source_path, text, "
            "       bm25(fts_chunks) AS bm25_score, "
            "       snippet(fts_chunks, 3, '', '', '...', 24) AS excerpt "
            "FROM fts_chunks "
            "WHERE fts_chunks MATCH ? "
        )
        params: list[Any] = [match_expr]
        if source_filter and source_filter != "both":
            sql += "AND source_type = ? "
            params.append(source_filter)
        sql += "ORDER BY bm25(fts_chunks) LIMIT ?"
        params.append(int(top_k))

        rows = self.conn.execute(sql, params).fetchall()

        hits: list[SearchHit] = []
        for r in rows:
            chunk_meta = self._chunk_meta(r["chunk_id"], r["source_type"])
            timestamp = chunk_meta.get("timestamp", "")
            score = _bm25_to_unit(r["bm25_score"])
            excerpt_text = (r["excerpt"] or r["text"] or "")[:500]
            hits.append(
                SearchHit(
                    chunk_id=r["chunk_id"],
                    source_type=r["source_type"],
                    source_path=r["source_path"],
                    timestamp=timestamp,
                    score=score,
                    excerpt=excerpt_text,
                )
            )
        return hits

    def _chunk_meta(self, chunk_id: str, source_type: str) -> dict[str, Any]:
        """Fetch timestamp/header for a hit so we can populate REQ-U3 fields."""
        if source_type == "vault":
            row = self.conn.execute(
                "SELECT timestamp, header FROM vault_chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                return {}
            return {"timestamp": row["timestamp"], "header": row["header"]}
        # chat rows are not written in Phase 1; fallback if we ever see one.
        row = self.conn.execute(
            "SELECT timestamp FROM chat_messages WHERE message_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return {}
        return {"timestamp": row["timestamp"]}


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------
def _normalize_fts5_query(query: str) -> str:
    """Make a free-form query safe for FTS5 ``MATCH``.

    Strategy: split on whitespace, drop tokens consisting only of FTS5
    operator chars, double-quote everything else (FTS5 treats quoted
    tokens as literals so ``foo:bar`` and ``"`` no longer trip the
    parser). This is intentionally conservative — Phase 1 prioritizes
    "never raise" over "feature-rich query language".
    """
    # FTS5 reserved chars that should not appear inside a quoted token.
    # We only need to strip embedded double quotes from each token.
    tokens = []
    for raw in query.replace("\x00", " ").split():
        cleaned = raw.replace('"', "")
        if not cleaned:
            continue
        # Drop pure punctuation tokens.
        if not any(c.isalnum() for c in cleaned):
            continue
        tokens.append(f'"{cleaned}"')
    return " ".join(tokens)


def _bm25_to_unit(bm25: float | None) -> float:
    """Map FTS5 BM25 to a positive ``[0, 1]`` score (better=higher).

    REQ-U3 mandates ``score`` in ``[0, 1]``. FTS5's ``bm25()`` returns
    negative numbers where *more* negative = better match. We use the
    smooth squash ``magnitude / (1 + magnitude)`` so:

    - bm25 = 0 (degenerate)        → score = 0.0
    - bm25 ≈ -1.0 (typical match)   → score ≈ 0.50
    - bm25 ≈ -10  (very strong)     → score ≈ 0.91
    - bm25 ≈ -100 (huge magnitude)  → score ≈ 0.99

    The exact curve is not important; what matters is monotonic-better
    and bounded in ``[0, 1]``.
    """
    if bm25 is None:
        return 0.0
    magnitude = abs(float(bm25))
    return magnitude / (1.0 + magnitude)


# ---------------------------------------------------------------------------
# Convenience for tests / one-shot scripts
# ---------------------------------------------------------------------------
def open_store(db_path: Path | str) -> MemoryStore:
    """Open and initialize a store at ``db_path``."""
    store = MemoryStore(db_path)
    store.open()
    return store


def dump_meta(store: MemoryStore) -> dict[str, str]:
    """Return all rows of the ``meta`` table as a plain dict."""
    rows = store.conn.execute("SELECT key, value FROM meta").fetchall()
    return {r["key"]: r["value"] for r in rows}


# Re-export json for callers that want to (de)serialize chat payloads.
__all__ = [
    "MemoryStore",
    "SCHEMA_VERSION",
    "SearchHit",
    "chunk_id_for",
    "dump_meta",
    "file_sha256",
    "open_store",
]


# Silence lints about ``json`` being unused — it is part of the public
# surface for callers that (de)serialize chat payloads in Phase 3.
_ = json
