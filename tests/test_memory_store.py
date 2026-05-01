# SPDX-License-Identifier: Apache-2.0
"""Tests for the memory subsystem SQLite store.

@TEST:MEMORY-01/store

Phase 1 scope: schema creation, idempotent upserts, BM25 search and the
score-normalization invariant from REQ-U3 (positive float in ``[0, 1]``).
"""

from __future__ import annotations

import sqlite3

import pytest

from vllm_mlx.memory.config import resolve_memory_config
from vllm_mlx.memory.store import (
    SCHEMA_VERSION,
    MemoryStore,
    SearchHit,
    chunk_id_for,
    dump_meta,
    file_sha256,
    open_store,
)

# ---------------------------------------------------------------------------
# Schema and lifecycle
# ---------------------------------------------------------------------------


class TestStoreLifecycle:
    def test_open_creates_schema_and_meta(self, tmp_path):
        db = tmp_path / "memory.db"
        store = open_store(db)
        try:
            assert db.exists()
            meta = dump_meta(store)
            assert meta["schema_version"] == str(SCHEMA_VERSION)
            assert "created_at" in meta

            # All forward-compat tables must exist (Path C: no migration in P2/P3).
            tables = {
                row["name"]
                for row in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            for required in (
                "meta",
                "vault_files",
                "vault_chunks",
                "chat_messages",
                "chat_summaries",
                "vault_themes",
            ):
                assert required in tables, f"missing table: {required}"
            # FTS5 table is registered with type='table' in sqlite_master.
            fts_kind_rows = list(
                store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'fts_chunks'"
                )
            )
            assert fts_kind_rows
        finally:
            store.close()

    def test_double_open_is_idempotent(self, tmp_path):
        db = tmp_path / "memory.db"
        store = open_store(db)
        store.open()  # second call is a no-op
        store.close()

    def test_wal_mode_enabled(self, tmp_path):
        db = tmp_path / "memory.db"
        store = open_store(db)
        try:
            mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            store.close()

    def test_use_before_open_raises(self, tmp_path):
        store = MemoryStore(tmp_path / "x.db")
        with pytest.raises(RuntimeError):
            _ = store.conn

    def test_schema_version_drift_is_a_hard_error(self, tmp_path):
        db = tmp_path / "memory.db"
        # Create an "old version" db that mimics a future-version drift.
        with sqlite3.connect(str(db)) as raw:
            raw.execute(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            raw.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', '999')"
            )
        with pytest.raises(RuntimeError, match="schema version mismatch"):
            open_store(db)


# ---------------------------------------------------------------------------
# Vault writes (idempotency)
# ---------------------------------------------------------------------------


class TestVaultWrites:
    @pytest.fixture
    def store(self, tmp_path):
        s = open_store(tmp_path / "memory.db")
        yield s
        s.close()

    def test_upsert_vault_file_returns_stable_id(self, store):
        with store.transaction():
            fid_a = store.upsert_vault_file(
                path="notes/a.md", mtime=1.0, sha256="aa", size=10
            )
        with store.transaction():
            fid_b = store.upsert_vault_file(
                path="notes/a.md", mtime=2.0, sha256="bb", size=20
            )
        assert fid_a == fid_b
        # The row should reflect the latest sha.
        assert store.get_vault_file_sha("notes/a.md") == "bb"

    def test_replace_chunks_for_file_inserts_and_indexes(self, store):
        with store.transaction():
            fid = store.upsert_vault_file(
                path="notes/x.md", mtime=1.0, sha256="cafe", size=42
            )
            count = store.replace_chunks_for_file(
                file_id=fid,
                source_path="notes/x.md",
                chunks=[
                    {
                        "chunk_id": chunk_id_for("notes/x.md", 0, 12),
                        "text": "Hello distillation world",
                        "header": "Greeting",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": 12,
                        "timestamp": "2025-01-01T00:00:00Z",
                    }
                ],
            )
        assert count == 1
        assert store.count_vault_files() == 1
        assert store.count_vault_chunks() == 1

    def test_replace_is_destructive(self, store):
        path = "notes/y.md"
        with store.transaction():
            fid = store.upsert_vault_file(
                path=path, mtime=1.0, sha256="aa", size=10
            )
            store.replace_chunks_for_file(
                file_id=fid,
                source_path=path,
                chunks=[
                    {
                        "chunk_id": chunk_id_for(path, 0, 5),
                        "text": "first",
                        "header": "",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": 5,
                        "timestamp": "2025-01-01T00:00:00Z",
                    },
                    {
                        "chunk_id": chunk_id_for(path, 5, 6),
                        "text": "second",
                        "header": "",
                        "chunk_index": 1,
                        "char_offset": 5,
                        "char_len": 6,
                        "timestamp": "2025-01-01T00:00:00Z",
                    },
                ],
            )

        # Replace with a single chunk; previous two must vanish.
        with store.transaction():
            store.replace_chunks_for_file(
                file_id=fid,
                source_path=path,
                chunks=[
                    {
                        "chunk_id": chunk_id_for(path, 0, 7),
                        "text": "rewrote",
                        "header": "",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": 7,
                        "timestamp": "2025-01-01T00:00:00Z",
                    }
                ],
            )
        assert store.count_vault_chunks() == 1


# ---------------------------------------------------------------------------
# BM25 search (REQ-U3 score contract)
# ---------------------------------------------------------------------------


class TestBm25Search:
    @pytest.fixture
    def store(self, tmp_path):
        s = open_store(tmp_path / "memory.db")
        with s.transaction():
            fid = s.upsert_vault_file(
                path="notes/distillation.md",
                mtime=1.0,
                sha256="abc",
                size=100,
            )
            s.replace_chunks_for_file(
                file_id=fid,
                source_path="notes/distillation.md",
                chunks=[
                    {
                        "chunk_id": chunk_id_for("notes/distillation.md", 0, 50),
                        "text": (
                            "I think distillation only works when the teacher "
                            "carries information."
                        ),
                        "header": "Distillation note",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": 50,
                        "timestamp": "2025-09-13T00:00:00Z",
                    }
                ],
            )
            fid2 = s.upsert_vault_file(
                path="notes/korean.md", mtime=1.0, sha256="def", size=80
            )
            s.replace_chunks_for_file(
                file_id=fid2,
                source_path="notes/korean.md",
                chunks=[
                    {
                        "chunk_id": chunk_id_for("notes/korean.md", 0, 60),
                        "text": "지난 달에 distillation 에 대해 어떻게 생각했지",
                        "header": "한국어 노트",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": 60,
                        "timestamp": "2025-10-01T00:00:00Z",
                    }
                ],
            )
            fid3 = s.upsert_vault_file(
                path="notes/unrelated.md", mtime=1.0, sha256="ghi", size=20
            )
            s.replace_chunks_for_file(
                file_id=fid3,
                source_path="notes/unrelated.md",
                chunks=[
                    {
                        "chunk_id": chunk_id_for("notes/unrelated.md", 0, 30),
                        "text": "completely unrelated banana muffin recipe",
                        "header": "",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": 30,
                        "timestamp": "2025-08-01T00:00:00Z",
                    }
                ],
            )
        yield s
        s.close()

    def test_bm25_finds_english_query(self, store):
        hits = store.search_bm25("distillation", top_k=5)
        assert hits, "expected at least one hit"
        assert any(
            h.source_path == "notes/distillation.md" for h in hits
        ), f"got hits: {hits}"

    def test_bm25_finds_korean_query(self, store):
        # Full-token Korean queries work with FTS5 unicode61.
        hits = store.search_bm25("지난 달에", top_k=5)
        assert hits
        assert any(h.source_path == "notes/korean.md" for h in hits)

    def test_bm25_empty_query_returns_empty(self, store):
        assert store.search_bm25("", top_k=5) == []
        assert store.search_bm25("   ", top_k=5) == []

    def test_bm25_score_in_unit_interval(self, store):
        hits = store.search_bm25("distillation", top_k=5)
        for h in hits:
            assert isinstance(h, SearchHit)
            assert 0.0 <= h.score <= 1.0, f"score out of range: {h}"

    def test_bm25_excerpt_truncated_to_500(self, store):
        # Add a giant chunk to confirm 500-char excerpt cap (REQ-U3).
        with store.transaction():
            fid = store.upsert_vault_file(
                path="notes/huge.md", mtime=1.0, sha256="huge", size=10_000
            )
            big_text = "marigold " + ("padding " * 200) + "marigold"
            store.replace_chunks_for_file(
                file_id=fid,
                source_path="notes/huge.md",
                chunks=[
                    {
                        "chunk_id": chunk_id_for(
                            "notes/huge.md", 0, len(big_text)
                        ),
                        "text": big_text,
                        "header": "",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": len(big_text),
                        "timestamp": "2025-01-01T00:00:00Z",
                    }
                ],
            )
        hits = store.search_bm25("marigold", top_k=1)
        assert hits
        assert len(hits[0].excerpt) <= 500

    def test_bm25_source_filter_vault_only(self, store):
        hits = store.search_bm25(
            "distillation", top_k=5, source_filter="vault"
        )
        assert hits
        assert all(h.source_type == "vault" for h in hits)

    def test_bm25_source_filter_chat_returns_empty_in_phase1(self, store):
        # Phase 1 doesn't write chat rows, so chat-only must return [].
        hits = store.search_bm25(
            "distillation", top_k=5, source_filter="chat"
        )
        assert hits == []

    def test_bm25_handles_query_with_special_chars(self, store):
        # The normalizer must strip FTS5 syntax that would otherwise
        # raise (e.g. embedded quotes, colons).
        hits = store.search_bm25('foo:bar "weird" distillation', top_k=5)
        # Either we find the english doc or none; the important thing
        # is no exception.
        assert isinstance(hits, list)


# ---------------------------------------------------------------------------
# Chunk ID determinism (REQ for incremental update path in Phase 4)
# ---------------------------------------------------------------------------


class TestChunkIdDeterminism:
    def test_same_inputs_same_id(self):
        assert chunk_id_for("a.md", 0, 100) == chunk_id_for("a.md", 0, 100)

    def test_different_path_different_id(self):
        assert chunk_id_for("a.md", 0, 100) != chunk_id_for("b.md", 0, 100)

    def test_different_offset_different_id(self):
        assert chunk_id_for("a.md", 0, 100) != chunk_id_for("a.md", 1, 100)

    def test_different_length_different_id(self):
        assert chunk_id_for("a.md", 0, 99) != chunk_id_for("a.md", 0, 100)

    def test_id_is_short_hex(self):
        cid = chunk_id_for("a.md", 0, 100)
        assert len(cid) == 32
        assert all(c in "0123456789abcdef" for c in cid)


# ---------------------------------------------------------------------------
# Sha helper
# ---------------------------------------------------------------------------


class TestFileSha256:
    def test_matches_known_sha(self, tmp_path):
        p = tmp_path / "x.md"
        p.write_bytes(b"hello\n")
        # Pre-computed sha256 of b"hello\n".
        assert (
            file_sha256(p)
            == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        )

    def test_streams_large_file(self, tmp_path):
        p = tmp_path / "big.bin"
        p.write_bytes(b"x" * (256 * 1024))
        # No assertion on the actual hash; just confirm no MemoryError.
        assert len(file_sha256(p)) == 64


# ---------------------------------------------------------------------------
# Cross-module config integration smoke
# ---------------------------------------------------------------------------


class TestStoreConfigIntegration:
    def test_resolve_then_open_uses_supplied_paths(self, tmp_path):
        db = tmp_path / "deep" / "memory.db"
        cfg = resolve_memory_config(
            {"MEMORY_ENABLED": "1", "MEMORY_DB_PATH": str(db)}
        )
        assert cfg.enabled
        assert cfg.db_path == db
        store = open_store(cfg.db_path)
        try:
            assert db.exists()
        finally:
            store.close()
