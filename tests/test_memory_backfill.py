# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-2 backfill module.

@TEST:MEMORY-01/backfill

We do NOT load real bge-m3 weights. Instead, every test uses a
deterministic fake embedder that returns one-byte-per-axis blobs of
the correct width. The backfill itself is the unit under test;
``Embedder`` is exercised separately in
:mod:`tests.test_memory_embedder`.
"""

from __future__ import annotations

import pytest

from vllm_mlx.memory.backfill import BackfillStats, backfill
from vllm_mlx.memory.embedder import pack_float32
from vllm_mlx.memory.store import chunk_id_for, open_store

pytest.importorskip("sqlite_vec")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _OkEmbedder:
    """Returns a single-axis vector per text. Always succeeds."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode_batch(self, texts: list[str]) -> list[bytes]:
        self.calls.append(list(texts))
        return [pack_float32([float(len(t) % 7) / 10.0] * self.dim) for t in texts]

    def disabled(self) -> bool:
        return False


class _FailingEmbedder:
    """Always returns ``[]`` and reports itself as permanently disabled."""

    def encode_batch(self, texts: list[str]) -> list[bytes]:
        return []

    def disabled(self) -> bool:
        return True


class _PartialEmbedder:
    """Returns fewer vectors than requested — backfill must skip the batch."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode_batch(self, texts: list[str]) -> list[bytes]:
        if not texts:
            return []
        # Drop the last text deliberately.
        return [pack_float32([0.5] * self.dim) for _ in texts[:-1]]

    def disabled(self) -> bool:
        return False


class _RaisingEmbedder:
    """Raises on every encode call. Backfill catches and proceeds."""

    def encode_batch(self, texts: list[str]) -> list[bytes]:
        raise RuntimeError("synthetic encoder failure")

    def disabled(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store_with_chunks(tmp_path):
    """Open a store and seed it with N vault chunks (no vectors yet)."""
    s = open_store(tmp_path / "memory.db", embed_dim=8)
    with s.transaction():
        for i in range(7):
            path = f"notes/n{i}.md"
            text = f"chunk text content {i}"
            fid = s.upsert_vault_file(
                path=path, mtime=1.0, sha256=path, size=len(text)
            )
            s.replace_chunks_for_file(
                file_id=fid,
                source_path=path,
                chunks=[
                    {
                        "chunk_id": chunk_id_for(path, 0, len(text)),
                        "text": text,
                        "header": "",
                        "chunk_index": 0,
                        "char_offset": 0,
                        "char_len": len(text),
                        "timestamp": "2025-01-01T00:00:00Z",
                    }
                ],
            )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestBackfillHappyPath:
    def test_backfills_all_missing(self, store_with_chunks):
        store = store_with_chunks
        embedder = _OkEmbedder(dim=8)

        stats = backfill(store, embedder, batch=2, log_every=10)

        assert stats.total == 7
        assert stats.succeeded == 7
        assert stats.failed == 0
        assert store.count_vectors() == 7
        assert store.count_chunks_missing_vectors() == 0

    def test_returns_zero_when_already_done(self, store_with_chunks):
        store = store_with_chunks
        # First pass populates everything.
        backfill(store, _OkEmbedder(dim=8), batch=4)
        # Second pass is a no-op.
        stats = backfill(store, _OkEmbedder(dim=8), batch=4)
        assert stats.total == 0
        assert stats.succeeded == 0
        assert stats.failed == 0

    def test_resumable_after_partial_run(self, store_with_chunks):
        store = store_with_chunks
        # Manually pre-embed one chunk to simulate a previously
        # interrupted run that completed N rows.
        first_id, first_text = next(
            iter(store.iter_chunks_missing_vectors(batch_size=1))
        )[0]
        with store.transaction():
            store.insert_vector(
                chunk_id=first_id, embedding=pack_float32([0.0] * 8)
            )
        assert store.count_chunks_missing_vectors() == 6

        stats = backfill(store, _OkEmbedder(dim=8), batch=3)
        # Backfill only sees the remaining 6 rows.
        assert stats.total == 6
        assert stats.succeeded == 6
        assert store.count_vectors() == 7

    def test_batch_size_drives_invocation_count(self, store_with_chunks):
        store = store_with_chunks
        embedder = _OkEmbedder(dim=8)
        backfill(store, embedder, batch=2)
        # 7 rows at batch=2 → 4 calls (2,2,2,1).
        assert [len(c) for c in embedder.calls] == [2, 2, 2, 1]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------
class TestBackfillFailures:
    def test_aborts_on_disabled_embedder(self, store_with_chunks):
        store = store_with_chunks
        stats = backfill(store, _FailingEmbedder(), batch=4)
        assert stats.aborted_reason == "embedder_disabled"
        assert stats.succeeded == 0
        assert store.count_vectors() == 0

    def test_partial_batch_is_skipped(self, store_with_chunks):
        store = store_with_chunks
        # Each batch is missing exactly one vector → every batch is
        # treated as failed and skipped, so nothing is written.
        stats = backfill(store, _PartialEmbedder(dim=8), batch=3)
        # Backfill processes 3 + 3 + 1 chunks but writes none of them.
        # ``failed`` accumulates over partial pages; not all pages
        # are guaranteed to be visited because iter loops while there
        # are still missing rows. We only assert no writes happened.
        assert stats.succeeded == 0
        assert store.count_vectors() == 0

    def test_raising_encoder_does_not_propagate(self, store_with_chunks):
        store = store_with_chunks
        # Raising encoder should be caught by backfill and treated as
        # an "empty result" — no rows written, no exception escaping.
        stats = backfill(store, _RaisingEmbedder(), batch=2)
        assert stats.failed > 0
        assert stats.succeeded == 0
        assert store.count_vectors() == 0

    def test_skipped_when_extension_missing(self, tmp_path):
        # Construct a store, then forcibly mark vec as not loaded to
        # simulate the no-extension case.
        store = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            store._vec_loaded = False  # noqa: SLF001 - intentional poke
            stats = backfill(store, _OkEmbedder(dim=8), batch=2)
            assert isinstance(stats, BackfillStats)
            assert stats.skipped_extension_missing is True
            assert stats.total == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Stats / log formatting
# ---------------------------------------------------------------------------
class TestBackfillStatsLog:
    def test_log_contains_counts(self, store_with_chunks):
        stats = backfill(store_with_chunks, _OkEmbedder(dim=8), batch=4)
        line = stats.as_log()
        assert "succeeded=7" in line
        assert "failed=0" in line
        assert "total=7" in line

    def test_log_contains_aborted_reason(self):
        s = BackfillStats(
            total=10, succeeded=4, failed=6, duration_s=0.5,
            aborted_reason="embedder_disabled",
        )
        assert "aborted" in s.as_log()


# ---------------------------------------------------------------------------
# CLI (``main()``) — the heavy MLX path is bypassed by monkeypatching
# the embedder factory so no real model load happens.
# ---------------------------------------------------------------------------
class TestBackfillCli:
    def test_main_returns_2_when_disabled(self, monkeypatch):
        monkeypatch.delenv("MEMORY_ENABLED", raising=False)
        from vllm_mlx.memory import backfill as bf

        rc = bf.main([])
        assert rc == 2

    def test_main_dry_run_succeeds(self, tmp_path, monkeypatch, capsys):
        # Seed a populated store so the CLI has something to count.
        s = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            with s.transaction():
                fid = s.upsert_vault_file(
                    path="a.md", mtime=1.0, sha256="aa", size=10
                )
                s.replace_chunks_for_file(
                    file_id=fid,
                    source_path="a.md",
                    chunks=[
                        {
                            "chunk_id": chunk_id_for("a.md", 0, 5),
                            "text": "alpha",
                            "header": "",
                            "chunk_index": 0,
                            "char_offset": 0,
                            "char_len": 5,
                            "timestamp": "2025-01-01T00:00:00Z",
                        }
                    ],
                )
        finally:
            s.close()
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        monkeypatch.setenv("MEMORY_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
        monkeypatch.setenv("MEMORY_EMBED_DIM", "8")

        from vllm_mlx.memory import backfill as bf

        rc = bf.main(["--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 chunks need embedding" in out

    def test_main_returns_3_on_model_mismatch(
        self, tmp_path, monkeypatch
    ):
        # Pre-record a different model in the store, then try to run
        # backfill with another model id — REQ-U4 protection kicks in.
        s = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            s.record_embedding_identity(model="someone/else", dim=8)
        finally:
            s.close()
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        monkeypatch.setenv("MEMORY_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
        monkeypatch.setenv("MEMORY_EMBED_MODEL", "BAAI/bge-m3")
        monkeypatch.setenv("MEMORY_EMBED_DIM", "8")

        from vllm_mlx.memory import backfill as bf

        rc = bf.main([])
        assert rc == 3

    def test_main_returns_4_when_embedder_load_fails(
        self, tmp_path, monkeypatch
    ):
        # Ensure there is at least one chunk so we get past dry-run logic.
        s = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            with s.transaction():
                fid = s.upsert_vault_file(
                    path="a.md", mtime=1.0, sha256="aa", size=10
                )
                s.replace_chunks_for_file(
                    file_id=fid,
                    source_path="a.md",
                    chunks=[
                        {
                            "chunk_id": chunk_id_for("a.md", 0, 5),
                            "text": "alpha",
                            "header": "",
                            "chunk_index": 0,
                            "char_offset": 0,
                            "char_len": 5,
                            "timestamp": "2025-01-01T00:00:00Z",
                        }
                    ],
                )
        finally:
            s.close()
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        monkeypatch.setenv("MEMORY_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
        monkeypatch.setenv("MEMORY_EMBED_DIM", "8")

        from vllm_mlx.memory import backfill as bf

        # Replace Embedder.from_config with a stub whose .load() returns False.
        class _BrokenEmbedder:
            def load(self) -> bool:
                return False

        monkeypatch.setattr(
            bf.Embedder,
            "from_config",
            classmethod(lambda cls, cfg: _BrokenEmbedder()),
        )
        rc = bf.main([])
        assert rc == 4

    def test_force_rebuild_flag_drops_vec_table(
        self, tmp_path, monkeypatch
    ):
        # Populate a chunk + vector pair, then run --force-rebuild
        # with a fake embedder so the table is dropped + recreated.
        s = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            with s.transaction():
                fid = s.upsert_vault_file(
                    path="a.md", mtime=1.0, sha256="aa", size=10
                )
                cid = chunk_id_for("a.md", 0, 5)
                s.replace_chunks_for_file(
                    file_id=fid,
                    source_path="a.md",
                    chunks=[
                        {
                            "chunk_id": cid,
                            "text": "alpha",
                            "header": "",
                            "chunk_index": 0,
                            "char_offset": 0,
                            "char_len": 5,
                            "timestamp": "2025-01-01T00:00:00Z",
                        }
                    ],
                )
                s.insert_vector(chunk_id=cid, embedding=pack_float32([0.1] * 8))
            assert s.count_vectors() == 1
        finally:
            s.close()
        monkeypatch.setenv("MEMORY_ENABLED", "1")
        monkeypatch.setenv("MEMORY_VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))
        monkeypatch.setenv("MEMORY_EMBED_DIM", "8")

        from vllm_mlx.memory import backfill as bf

        # Stub Embedder so we don't need MLX.
        class _RebuildEmbedder:
            def load(self):
                return True

            def encode_batch(self, texts):
                return [pack_float32([0.5] * 8) for _ in texts]

            def disabled(self):
                return False

        monkeypatch.setattr(
            bf.Embedder,
            "from_config",
            classmethod(lambda cls, cfg: _RebuildEmbedder()),
        )

        rc = bf.main(["--force-rebuild"])
        assert rc == 0
        # The vec table is recreated and re-populated by the backfill.
        s2 = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            assert s2.count_vectors() == 1
        finally:
            s2.close()
