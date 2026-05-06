# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-2 sqlite-vec dense store path.

@TEST:MEMORY-01/store-dense

These tests exercise:

* sqlite-vec extension loading (skipped if the wheel is unavailable).
* ``vec_chunks`` virtual table creation.
* ``insert_vector`` / ``insert_vectors_batch`` round-trip.
* ``search_dense`` ranking by ascending distance.
* REQ-U4 model identity tracking + mismatch detection.
* ``count_chunks_missing_vectors`` / ``iter_chunks_missing_vectors``
  (used by the backfill module).
"""

from __future__ import annotations


import pytest

from vllm_mlx.memory.embedder import pack_float32
from vllm_mlx.memory.store import MemoryStore, chunk_id_for, open_store

# All tests in this module require sqlite-vec to be importable. Skip
# the whole file otherwise so a developer without the extension can
# still run the rest of the memory test suite.
pytest.importorskip("sqlite_vec")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path):
    s = open_store(tmp_path / "memory.db", embed_dim=8)
    yield s
    s.close()


def _vec(values):
    """Helper: pack a list of floats into the sqlite-vec wire format."""
    return pack_float32(values)


def _populate_vault(store: MemoryStore, rows: list[tuple[str, str]]):
    """Insert (path, text) rows so vec_chunks rows have a JOIN target."""
    out: list[tuple[str, str]] = []  # (chunk_id, text)
    with store.transaction():
        for path, text in rows:
            fid = store.upsert_vault_file(
                path=path, mtime=1.0, sha256=path, size=len(text)
            )
            cid = chunk_id_for(path, 0, len(text))
            store.replace_chunks_for_file(
                file_id=fid,
                source_path=path,
                chunks=[
                    {
                        "chunk_id": cid,
                        "text": text,
                        "header": "",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": len(text),
                        "timestamp": "2025-01-01T00:00:00Z",
                    }
                ],
            )
            out.append((cid, text))
    return out


# ---------------------------------------------------------------------------
# Extension load + table creation
# ---------------------------------------------------------------------------
class TestVecLoaded:
    def test_open_loads_vec_extension(self, store):
        assert store.vec_loaded is True

    def test_vec_chunks_table_exists(self, store):
        # sqlite_master lists virtual tables under type='table' with
        # the "USING vec0" string in their CREATE statement.
        rows = list(
            store.conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'vec_chunks'"
            )
        )
        assert rows, "vec_chunks must be created when sqlite-vec is loaded"

    def test_count_vectors_starts_at_zero(self, store):
        assert store.count_vectors() == 0


# ---------------------------------------------------------------------------
# Inserts + reads
# ---------------------------------------------------------------------------
class TestInsertAndRead:
    def test_single_insert_and_count(self, store):
        chunks = _populate_vault(store, [("note.md", "alpha")])
        cid = chunks[0][0]
        with store.transaction():
            ok = store.insert_vector(
                chunk_id=cid,
                embedding=_vec([0.1] * 8),
                source_type="vault",
            )
        assert ok
        assert store.count_vectors() == 1

    def test_batch_insert(self, store):
        chunks = _populate_vault(
            store, [(f"note{i}.md", f"text-{i}") for i in range(4)]
        )
        with store.transaction():
            store.insert_vectors_batch(
                rows=[(cid, _vec([float(i) / 10] * 8)) for i, (cid, _) in enumerate(chunks)],
                source_type="vault",
            )
        assert store.count_vectors() == 4

    def test_insert_or_replace_is_idempotent(self, store):
        chunks = _populate_vault(store, [("note.md", "alpha")])
        cid = chunks[0][0]
        with store.transaction():
            store.insert_vector(chunk_id=cid, embedding=_vec([0.1] * 8))
            store.insert_vector(chunk_id=cid, embedding=_vec([0.2] * 8))
        # Second insert replaces, count stays 1.
        assert store.count_vectors() == 1

    def test_delete_vectors(self, store):
        chunks = _populate_vault(store, [("a.md", "a"), ("b.md", "b")])
        with store.transaction():
            for cid, _ in chunks:
                store.insert_vector(chunk_id=cid, embedding=_vec([0.1] * 8))
        assert store.count_vectors() == 2

        n = store.delete_vectors_for_chunks([chunks[0][0]])
        assert n == 1
        assert store.count_vectors() == 1


# ---------------------------------------------------------------------------
# search_dense ranking + JOIN behaviour
# ---------------------------------------------------------------------------
class TestSearchDense:
    @pytest.fixture
    def populated(self, store):
        # Three known chunks at increasing distance from the query.
        chunks = _populate_vault(
            store,
            [
                ("close.md", "near the query"),
                ("middle.md", "somewhat related"),
                ("far.md", "completely unrelated"),
            ],
        )
        with store.transaction():
            store.insert_vector(
                chunk_id=chunks[0][0], embedding=_vec([1.0] + [0.0] * 7)
            )
            store.insert_vector(
                chunk_id=chunks[1][0], embedding=_vec([0.5, 0.5] + [0.0] * 6)
            )
            store.insert_vector(
                chunk_id=chunks[2][0], embedding=_vec([0.0] * 7 + [1.0])
            )
        return store, chunks

    def test_ranking_orders_by_distance(self, populated):
        s, chunks = populated
        q = _vec([1.0] + [0.0] * 7)
        hits = s.search_dense(q, top_k=3)
        assert [h.source_path for h in hits] == [
            "close.md",
            "middle.md",
            "far.md",
        ]
        # Scores must be in unit interval and monotonically decreasing.
        assert all(0.0 <= h.score <= 1.0 for h in hits)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self, populated):
        s, _ = populated
        hits = s.search_dense(_vec([1.0] + [0.0] * 7), top_k=2)
        assert len(hits) == 2

    def test_empty_query_returns_empty(self, populated):
        s, _ = populated
        assert s.search_dense(b"", top_k=5) == []

    def test_source_filter_vault_only(self, populated):
        s, _ = populated
        hits = s.search_dense(
            _vec([1.0] + [0.0] * 7), top_k=3, source_filter="vault"
        )
        assert hits
        assert all(h.source_type == "vault" for h in hits)

    def test_source_filter_chat_returns_empty(self, populated):
        # Phase 2 only writes vault rows; chat-only filter is empty.
        s, _ = populated
        hits = s.search_dense(
            _vec([1.0] + [0.0] * 7), top_k=3, source_filter="chat"
        )
        assert hits == []

    def test_excerpt_truncation(self, store):
        path = "huge.md"
        big = "marigold " + ("padding " * 200) + "marigold"
        chunks = _populate_vault(store, [(path, big)])
        with store.transaction():
            store.insert_vector(
                chunk_id=chunks[0][0], embedding=_vec([1.0] + [0.0] * 7)
            )
        hits = store.search_dense(_vec([1.0] + [0.0] * 7), top_k=1)
        assert hits
        # REQ-U3 envelope cap: excerpt ≤ 500 chars.
        assert len(hits[0].excerpt) <= 500


# ---------------------------------------------------------------------------
# REQ-U4 — model identity tracking
# ---------------------------------------------------------------------------
class TestModelIdentity:
    def test_first_open_has_no_recorded_model(self, store):
        assert store.get_meta("embedding_model") is None
        assert store.get_meta("embedding_dim") is None

    def test_assert_compat_passes_on_empty_db(self, store):
        ok, reason = store.assert_embed_compat(model="BAAI/bge-m3", dim=8)
        assert ok
        assert reason is None

    def test_record_then_assert_matches(self, store):
        store.record_embedding_identity(model="BAAI/bge-m3", dim=8)
        ok, reason = store.assert_embed_compat(model="BAAI/bge-m3", dim=8)
        assert ok
        assert reason is None

    def test_assert_compat_rejects_model_mismatch(self, store):
        store.record_embedding_identity(model="BAAI/bge-m3", dim=8)
        ok, reason = store.assert_embed_compat(model="other/model", dim=8)
        assert not ok
        assert reason is not None
        assert "force-rebuild" in reason

    def test_assert_compat_rejects_dim_mismatch(self, store):
        store.record_embedding_identity(model="BAAI/bge-m3", dim=8)
        ok, reason = store.assert_embed_compat(model="BAAI/bge-m3", dim=1024)
        assert not ok
        assert reason is not None

    def test_record_is_one_shot(self, store):
        store.record_embedding_identity(model="m1", dim=8)
        # A second call with a different value must NOT overwrite —
        # REQ-U4 mandates explicit operator action to switch models.
        store.record_embedding_identity(model="m2", dim=16)
        assert store.get_meta("embedding_model") == "m1"
        assert store.get_meta("embedding_dim") == "8"


# ---------------------------------------------------------------------------
# count_chunks_missing_vectors / iter_chunks_missing_vectors
# ---------------------------------------------------------------------------
class TestMissingVectorsAccounting:
    def test_count_before_any_vector(self, store):
        chunks = _populate_vault(
            store, [(f"n{i}.md", f"text-{i}") for i in range(3)]
        )
        assert len(chunks) == 3
        assert store.count_chunks_missing_vectors() == 3

    def test_count_after_partial_backfill(self, store):
        chunks = _populate_vault(
            store, [(f"n{i}.md", f"text-{i}") for i in range(3)]
        )
        with store.transaction():
            store.insert_vector(chunk_id=chunks[0][0], embedding=_vec([0.1] * 8))
        assert store.count_chunks_missing_vectors() == 2

    def test_iter_chunks_missing_vectors_pages(self, store):
        # 5 rows, batch size 2 → 3 pages (2, 2, 1).
        chunks = _populate_vault(
            store, [(f"n{i}.md", f"text-{i}") for i in range(5)]
        )
        pages = list(store.iter_chunks_missing_vectors(batch_size=2))
        sizes = [len(p) for p in pages]
        assert sum(sizes) == 5
        assert sizes[0] == 2 and sizes[1] == 2 and sizes[-1] == 1

    def test_iter_skips_already_embedded(self, store):
        chunks = _populate_vault(
            store, [(f"n{i}.md", f"text-{i}") for i in range(4)]
        )
        with store.transaction():
            for cid, _ in chunks[:2]:
                store.insert_vector(chunk_id=cid, embedding=_vec([0.1] * 8))
        pages = list(store.iter_chunks_missing_vectors(batch_size=10))
        assert len(pages) == 1
        ids = {row[0] for row in pages[0]}
        # Only the last 2 chunks should be in the missing list.
        assert ids == {chunks[2][0], chunks[3][0]}


# ---------------------------------------------------------------------------
# Schema forward-compat: opening a Phase-1 DB on a Phase-2 binary
# ---------------------------------------------------------------------------
class TestForwardCompatibility:
    def test_phase1_db_gains_vec_chunks_on_open(self, tmp_path):
        # Step 1: simulate a Phase-1 DB by creating a store, populating
        # vault data, and closing it. The Phase-2 schema is the same
        # superset — vec_chunks comes from sqlite-vec and is created
        # via IF NOT EXISTS so the open must be safe.
        db = tmp_path / "memory.db"
        s1 = open_store(db, embed_dim=8)
        try:
            _populate_vault(s1, [("a.md", "first phase content")])
        finally:
            s1.close()

        # Step 2: re-open. count_chunks_missing_vectors > 0 because no
        # embeddings exist yet for the pre-existing chunk.
        s2 = open_store(db, embed_dim=8)
        try:
            assert s2.vec_loaded
            assert s2.count_chunks_missing_vectors() == 1
        finally:
            s2.close()


# ---------------------------------------------------------------------------
# Hardening: graceful no-op when sqlite-vec did not load
# ---------------------------------------------------------------------------
class TestVecUnloaded:
    @pytest.fixture
    def unloaded_store(self, store):
        # Force the failure mode by clearing the flag. This is
        # equivalent to the production path where sqlite-vec was not
        # installed or could not load on this connection.
        store._vec_loaded = False  # noqa: SLF001 - intentional test poke
        return store

    def test_insert_vector_returns_false(self, unloaded_store):
        assert unloaded_store.insert_vector(
            chunk_id="abc", embedding=_vec([0.1] * 8)
        ) is False

    def test_search_dense_returns_empty(self, unloaded_store):
        assert unloaded_store.search_dense(_vec([0.1] * 8), top_k=5) == []

    def test_count_vectors_returns_zero(self, unloaded_store):
        assert unloaded_store.count_vectors() == 0

    def test_count_missing_returns_zero(self, unloaded_store):
        assert unloaded_store.count_chunks_missing_vectors() == 0

    def test_iter_missing_yields_nothing(self, unloaded_store):
        assert list(unloaded_store.iter_chunks_missing_vectors()) == []
