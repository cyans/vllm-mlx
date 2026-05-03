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

# Phase 2 — sqlite-vec virtual table SQL is templated because the embedding
# dimension is config-driven (default 1024 for bge-m3). The CREATE is
# emitted only if the extension successfully loads on this connection.
# We use ``chunk_id`` as a TEXT primary key so the same identifier
# threads through ``vault_chunks``, ``fts_chunks``, and ``vec_chunks``;
# this is the same scheme the FTS5 table uses today.
_VEC_TABLE_SQL_TEMPLATE = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
    "chunk_id TEXT PRIMARY KEY, embedding float[{dim}], +source_type TEXT)"
)


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

    def __init__(
        self,
        db_path: Path | str,
        *,
        embed_dim: int = 1024,
    ):
        self.db_path = Path(db_path)
        self.embed_dim = int(embed_dim)
        # Note: connection is opened lazily so an offline check (db_path
        # parent missing) does not raise at construction time.
        self._conn: sqlite3.Connection | None = None
        # ``True`` after sqlite-vec successfully attached to this
        # connection AND the ``vec_chunks`` virtual table was created
        # (or already existed). Methods that touch vec_chunks first
        # check this flag and degrade gracefully when False (REQ-O3).
        self._vec_loaded: bool = False

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
        # Try to load sqlite-vec. Failure is degraded-but-functional:
        # BM25 still works (REQ-O3), the chat path is unaffected.
        self._try_load_sqlite_vec()
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

    # -- sqlite-vec --------------------------------------------------------
    def _try_load_sqlite_vec(self) -> None:
        """Attempt to load the sqlite-vec extension and create vec_chunks.

        Sets :attr:`_vec_loaded` to True on success. On failure the flag
        stays False and the caller's dense path will be skipped (REQ-O3
        fallback to BM25). Both the import and the load are wrapped so
        a missing dylib, a mismatched SQLite, or a permission failure
        all result in the same "no vector path" degradation.
        """
        if self._conn is None:
            return
        try:
            import sqlite_vec  # noqa: PLC0415 - intentional lazy import
        except Exception:  # noqa: BLE001 - REQ-O3
            logger.warning(
                "[memory] sqlite-vec is not installed; dense search disabled "
                "(BM25 still works). Install with: pip install sqlite-vec"
            )
            return
        try:
            self._conn.enable_load_extension(True)
            try:
                sqlite_vec.load(self._conn)
            finally:
                # Always re-disable extension loading after the call so
                # an attacker who later compromises the DB cannot pivot
                # via load_extension(). REQ-N4 belt-and-braces.
                self._conn.enable_load_extension(False)
            # Create the vec0 virtual table at our configured dim.
            self._conn.executescript(
                _VEC_TABLE_SQL_TEMPLATE.format(dim=self.embed_dim)
            )
            self._vec_loaded = True
            logger.info(
                "[memory] sqlite-vec loaded; vec_chunks ready (dim=%d)",
                self.embed_dim,
            )
        except Exception:  # noqa: BLE001 - REQ-O3
            logger.exception(
                "[memory] failed to enable sqlite-vec on this connection; "
                "dense search disabled (BM25 still works)"
            )
            self._vec_loaded = False

    @property
    def vec_loaded(self) -> bool:
        """True iff dense search via sqlite-vec is available on this conn."""
        return self._vec_loaded

    def assert_embed_compat(
        self, *, model: str, dim: int
    ) -> tuple[bool, str | None]:
        """Verify the embedding model+dim match what's already in the DB.

        REQ-U4: "Mixing models in one DB is forbidden." On the *first*
        embedding write the model + dim are recorded in ``meta``; on
        every subsequent open/write the caller must check compatibility.

        Returns ``(ok, reason)``. If ``ok`` is False the caller should
        disable the embedder for this DB. ``reason`` carries a human-
        readable hint for log messages.
        """
        recorded_model = self.get_meta("embedding_model")
        recorded_dim = self.get_meta("embedding_dim")

        if recorded_model is None and recorded_dim is None:
            # First-write case: caller will set these via
            # ``record_embedding_identity`` after the first INSERT.
            return True, None

        if recorded_model != model:
            return False, (
                f"DB was indexed with model={recorded_model!r} but "
                f"current MEMORY_EMBED_MODEL={model!r}. Run "
                f"`python -m vllm_mlx.memory.backfill --force-rebuild` "
                f"after wiping vec_chunks to switch models."
            )
        if recorded_dim is not None and int(recorded_dim) != int(dim):
            return False, (
                f"DB was indexed at dim={recorded_dim} but current "
                f"MEMORY_EMBED_DIM={dim}. Backfill --force-rebuild to "
                f"recreate vec_chunks at the new dim."
            )
        return True, None

    def record_embedding_identity(self, *, model: str, dim: int) -> None:
        """Persist the embedding model + dim if not already recorded.

        Idempotent on subsequent calls. The store does not auto-update
        a previously recorded value — REQ-U4 forbids mixing models so a
        change requires explicit operator action (drop the table or
        rebuild).
        """
        if self.get_meta("embedding_model") is None:
            self.set_meta("embedding_model", str(model))
        if self.get_meta("embedding_dim") is None:
            self.set_meta("embedding_dim", str(int(dim)))

    # -- vector writes -----------------------------------------------------

    def insert_vector(
        self,
        *,
        chunk_id: str,
        embedding: bytes,
        source_type: str = "vault",
    ) -> bool:
        """Insert (or replace) a single embedding for ``chunk_id``.

        Returns True on success, False if the dense path is unavailable
        on this connection. Errors during the actual INSERT are not
        swallowed — they propagate so the caller's transaction rolls
        back, mirroring the behaviour of ``replace_chunks_for_file``.

        ``vec0`` does not support ``INSERT OR REPLACE`` directly (it
        rejects the SQLite ``REPLACE`` conflict resolution path), so
        we DELETE-then-INSERT to make the operation idempotent.
        """
        if not self._vec_loaded:
            return False
        self.conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
        )
        self.conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding, source_type) "
            "VALUES(?, ?, ?)",
            (chunk_id, embedding, source_type),
        )
        return True

    def insert_vectors_batch(
        self,
        *,
        rows: Iterable[tuple[str, bytes]],
        source_type: str = "vault",
    ) -> int:
        """Bulk-insert ``(chunk_id, embedding_bytes)`` pairs. Returns count.

        See :meth:`insert_vector` for why we DELETE-then-INSERT instead
        of using ``INSERT OR REPLACE``.
        """
        if not self._vec_loaded:
            return 0
        n = 0
        for chunk_id, blob in rows:
            self.conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
            )
            self.conn.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding, source_type) "
                "VALUES(?, ?, ?)",
                (chunk_id, blob, source_type),
            )
            n += 1
        return n

    def delete_vectors_for_chunks(self, chunk_ids: Iterable[str]) -> int:
        """Remove vector rows whose chunk_id is in the supplied iterable.

        Returns the number of rows actually deleted. No-op when the
        vec extension failed to load.
        """
        if not self._vec_loaded:
            return 0
        n = 0
        for cid in chunk_ids:
            cur = self.conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,)
            )
            # ``rowcount`` is 1 iff the row existed.
            n += max(0, cur.rowcount or 0)
        return n

    # -- vector reads ------------------------------------------------------

    def search_dense(
        self,
        query_embedding: bytes,
        *,
        top_k: int = 5,
        source_filter: str | None = None,
        chat_retention_cutoff_iso: str | None = None,
    ) -> list[SearchHit]:
        """Run a dense vector search against ``vec_chunks``.

        ``query_embedding`` is the float32-packed bytes blob produced
        by :func:`vllm_mlx.memory.embedder.pack_float32`. Returns a
        list of :class:`SearchHit` ranked by ascending L2 distance
        (closer = better). For normalized embeddings (which bge-m3
        produces) L2² = 2(1 - cos_sim), so this ranking is equivalent
        to cosine similarity ranking.

        We map the (unbounded) L2 distance back into a positive
        ``[0, 1]`` score for REQ-U3 via ``1 / (1 + distance)``.

        ``sqlite-vec`` rejects WHERE constraints on auxiliary columns
        inside a KNN query, so when a ``source_filter`` is supplied
        we over-fetch by 4× and post-filter in Python. This still
        returns ``top_k`` items in the typical case where the vault
        dominates the index.

        Phase 3: chat hits (``source_type='chat'``) are looked up via
        :meth:`_chat_meta_for_chunk` since vault joins return NULL for
        them. ``chat_retention_cutoff_iso`` (REQ-N5) is applied here so
        rows older than ``MEMORY_CHAT_RETENTION_DAYS`` never escape the
        store, even if their embeddings still exist in vec_chunks.
        """
        if not self._vec_loaded:
            return []
        if not query_embedding:
            return []

        top_k = max(1, int(top_k))
        # vec0 requires the ``k = ?`` constraint OR ``LIMIT`` directly
        # on the virtual table; ``k = ?`` is the documented form and
        # works regardless of JOINs above it.
        # Over-fetch when filtering OR when chat rows can be present so
        # the post-filter retention cutoff cannot silently truncate.
        needs_overfetch = (
            (source_filter and source_filter != "both")
            or chat_retention_cutoff_iso is not None
        )
        knn_k = top_k * 4 if needs_overfetch else top_k

        # ``vec_chunks`` is the source-of-truth for "which embeddings
        # exist"; we LEFT JOIN ``vault_chunks`` so a vector with no
        # surviving metadata row (should not happen, but defensive) is
        # still emitted with a synthetic excerpt instead of crashing.
        sql = (
            "SELECT v.chunk_id, v.source_type, v.distance, "
            "       vc.text       AS vault_text, "
            "       vc.timestamp  AS vault_ts, "
            "       vf.path       AS vault_path "
            "FROM vec_chunks v "
            "LEFT JOIN vault_chunks vc ON vc.chunk_id = v.chunk_id "
            "LEFT JOIN vault_files vf ON vf.file_id = vc.file_id "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance"
        )
        rows = self.conn.execute(sql, (query_embedding, int(knn_k))).fetchall()

        wanted: str | None = None
        if source_filter and source_filter != "both":
            wanted = source_filter

        hits: list[SearchHit] = []
        for r in rows:
            stype = r["source_type"] or "vault"
            if wanted is not None and stype != wanted:
                continue
            if stype == "chat":
                meta = self._chat_meta_for_chunk(r["chunk_id"])
                if meta is None:
                    # Vector exists but the chat row was deleted (e.g. by
                    # SPEC-MEMORY-02 retention sweeper). Skip silently.
                    continue
                ts = meta["timestamp"]
                if (
                    chat_retention_cutoff_iso is not None
                    and ts < chat_retention_cutoff_iso
                ):
                    continue
                hits.append(
                    SearchHit(
                        chunk_id=r["chunk_id"],
                        source_type="chat",
                        source_path=meta["source_path"],
                        timestamp=ts,
                        score=_distance_to_unit(r["distance"]),
                        excerpt=meta["excerpt"],
                    )
                )
            else:
                text = r["vault_text"] or ""
                excerpt = text[:500]
                hits.append(
                    SearchHit(
                        chunk_id=r["chunk_id"],
                        source_type=stype,
                        source_path=r["vault_path"] or "",
                        timestamp=r["vault_ts"] or "",
                        score=_distance_to_unit(r["distance"]),
                        excerpt=excerpt,
                    )
                )
            if len(hits) >= top_k:
                break
        return hits

    def count_vectors(self) -> int:
        """Return the number of rows in ``vec_chunks`` (0 if disabled)."""
        if not self._vec_loaded:
            return 0
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM vec_chunks"
        ).fetchone()
        return int(row["c"])

    def count_chunks_missing_vectors(self) -> int:
        """How many ``vault_chunks`` rows still need an embedding.

        Used at server startup to print a one-line backfill hint when
        Phase-1 data is loaded into a Phase-2 binary.
        """
        if not self._vec_loaded:
            # Without sqlite-vec we cannot meaningfully report missing
            # vectors; the answer is "all of them, but you cannot
            # backfill anyway".
            return 0
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM vault_chunks "
            "WHERE chunk_id NOT IN (SELECT chunk_id FROM vec_chunks)"
        ).fetchone()
        return int(row["c"])

    def iter_chunks_missing_vectors(
        self, *, batch_size: int = 100
    ) -> Iterator[list[tuple[str, str]]]:
        """Yield batches of (chunk_id, text) pairs that lack an embedding.

        Used by the backfill job. The cursor is paged so a long-running
        backfill can be killed and resumed without rescanning. Pagination
        uses ``chunk_id`` ordering to be deterministic.

        When sqlite-vec is unavailable this generator is empty so the
        backfill becomes a no-op.
        """
        if not self._vec_loaded:
            return
        last_id = ""
        while True:
            rows = self.conn.execute(
                "SELECT chunk_id, text FROM vault_chunks "
                "WHERE chunk_id > ? "
                "  AND chunk_id NOT IN (SELECT chunk_id FROM vec_chunks) "
                "ORDER BY chunk_id "
                "LIMIT ?",
                (last_id, int(batch_size)),
            ).fetchall()
            if not rows:
                return
            yield [(r["chunk_id"], r["text"]) for r in rows]
            last_id = rows[-1]["chunk_id"]


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
        chat_retention_cutoff_iso: str | None = None,
    ) -> list[SearchHit]:
        """Run a BM25 search over ``fts_chunks`` and return ranked hits.

        ``source_filter`` is one of ``"vault"``, ``"chat"``, ``"both"``
        or ``None`` (treated as both). FTS5 ``bm25()`` returns a negative
        relevance number — closer to zero = better — which we negate and
        then squash into ``[0, 1]`` via ``1 / (1 + |bm25|)`` so the score
        contract in REQ-U3 holds (positive float in [0,1]).

        Phase 3: ``chat_retention_cutoff_iso`` (REQ-N5) hides chat rows
        whose timestamp is older than the cutoff. Vault rows are not
        retention-filtered (only chat history has a TTL per SPEC §11).
        We over-fetch when retention is active so the post-filter cannot
        truncate us below ``top_k``.
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

        # When chat retention is active we fetch a wider candidate pool
        # so post-filter retention does not silently truncate the result.
        fetch_top_k = int(top_k)
        if chat_retention_cutoff_iso is not None and (
            source_filter in (None, "both", "chat")
        ):
            fetch_top_k = max(fetch_top_k, int(top_k) * 4)

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
        params.append(fetch_top_k)

        rows = self.conn.execute(sql, params).fetchall()

        hits: list[SearchHit] = []
        for r in rows:
            chunk_meta = self._chunk_meta(r["chunk_id"], r["source_type"])
            timestamp = chunk_meta.get("timestamp", "")
            if (
                r["source_type"] == "chat"
                and chat_retention_cutoff_iso is not None
                and timestamp
                and timestamp < chat_retention_cutoff_iso
            ):
                continue
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
            if len(hits) >= int(top_k):
                break
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
        # Phase 3: chat rows now have rows in this table. Look them up via
        # message_id which matches chunk_id by design (see chatlog.py).
        row = self.conn.execute(
            "SELECT timestamp FROM chat_messages WHERE message_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            return {}
        return {"timestamp": row["timestamp"]}

    def _chat_meta_for_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Look up timestamp + source_path + excerpt for a chat chunk.

        Used by :meth:`search_dense` to attach per-result metadata for
        chat hits (the vault JOINs in the dense SQL return NULL for them).
        Returns ``None`` when no row exists, which lets the caller skip
        orphan vec_chunks entries silently.

        Excerpt is taken from the FTS5 row (which already holds the
        searchable, redacted concatenation of last-user + assistant text)
        when available; otherwise reconstructed from the payload.
        """
        cm = self.conn.execute(
            "SELECT request_id, session_id, timestamp, payload "
            "FROM chat_messages WHERE message_id = ?",
            (chunk_id,),
        ).fetchone()
        if cm is None:
            return None

        # Excerpt: FTS row holds the search text we already redacted.
        fts = self.conn.execute(
            "SELECT source_path, text FROM fts_chunks "
            "WHERE chunk_id = ? AND source_type = 'chat' LIMIT 1",
            (chunk_id,),
        ).fetchone()
        if fts is not None:
            excerpt = (fts["text"] or "")[:500]
            source_path = fts["source_path"] or (
                f"chat/{cm['session_id'] or ''}/{cm['request_id']}"
            )
        else:
            # Fallback: reconstruct from payload if the FTS row was
            # evicted (defensive — should not happen in MVP).
            from .chatlog import _searchable_from_payload  # noqa: PLC0415

            excerpt = _searchable_from_payload(cm["payload"] or "{}")[:500]
            source_path = (
                f"chat/{cm['session_id'] or ''}/{cm['request_id']}"
            )

        return {
            "timestamp": cm["timestamp"],
            "source_path": source_path,
            "excerpt": excerpt,
        }

    # -- chat introspection (Phase 3) --------------------------------------

    def count_chat_messages(self) -> int:
        """Total number of chat rows persisted (any tier)."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM chat_messages"
        ).fetchone()
        return int(row["c"])

    def count_chat_messages_missing_vectors(self) -> int:
        """Chat rows that still need an embedding (always 0 if vec disabled).

        Used by the background poller diagnostic log line and by tests
        to verify the embed loop processes the queue.
        """
        if not self._vec_loaded:
            return 0
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM chat_messages "
            "WHERE message_id NOT IN (SELECT chunk_id FROM vec_chunks)"
        ).fetchone()
        return int(row["c"])


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


def _distance_to_unit(distance: float | None) -> float:
    """Map sqlite-vec L2 distance into REQ-U3's positive ``[0, 1]`` score.

    bge-m3's ``text_embeds`` are L2-normalized; for two unit vectors
    ``L2_distance² = 2 * (1 - cos_sim)`` so the L2 distance ranges in
    ``[0, 2]`` for "perfect match" → "antipodal". We squash via
    ``1 / (1 + distance)`` so:

    - distance 0.0 (perfect match)   → 1.00
    - distance 0.5 (close)           → 0.67
    - distance ~1.0 (orthogonal)     → 0.50
    - distance ~1.41 (cos -0.5)      → 0.41
    - distance 2.0 (opposite)        → 0.33

    The ``[0, 1]`` shape matters for the envelope contract; the exact
    curve does not — RRF re-ranks by *position*, not score, so the
    scores serve only as a human-readable hint downstream.
    """
    if distance is None:
        return 0.0
    d = max(0.0, float(distance))
    return 1.0 / (1.0 + d)


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

# Wire ``open_store`` to forward the new ``embed_dim`` kwarg too.
def _patched_open_store(db_path: Path | str, *, embed_dim: int = 1024) -> MemoryStore:
    """Open and initialize a store at ``db_path`` (with vec dim).

    Re-defines ``open_store`` to accept the Phase-2 ``embed_dim`` kwarg
    while staying backwards-compatible with Phase-1 callers that pass
    only ``db_path``.
    """
    store = MemoryStore(db_path, embed_dim=embed_dim)
    store.open()
    return store


# Replace the simple Phase-1 helper with the dim-aware version.
open_store = _patched_open_store  # noqa: F811 - intentional override


# Silence lints about ``json`` being unused — it is part of the public
# surface for callers that (de)serialize chat payloads in Phase 3.
_ = json
