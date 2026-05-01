# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-2 hybrid search dispatcher (``search_memory``).

@TEST:MEMORY-01/server-hybrid

Covers:

* Pure BM25 fallback when no embedder is supplied.
* Hybrid fusion via reciprocal rank fusion (RRF).
* ``degraded: true`` flag plumbed through every dense-failure path
  required by REQ-O3 / REQ-N4.
* The off-switch: ``MEMORY_EMBED_DISABLED=1`` returns BM25 results
  WITHOUT setting ``degraded`` (the operator opted out).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from vllm_mlx.memory.config import MEMORY_DEFAULT_DENYLIST, resolve_memory_config
from vllm_mlx.memory.embedder import pack_float32
from vllm_mlx.memory.indexer import VaultIndexer
from vllm_mlx.memory.server import (
    _rrf_fuse,
    _try_build_embedder,
    _try_open_store,
    search_memory,
)
from vllm_mlx.memory.store import SearchHit, chunk_id_for, open_store

# All tests in this module require sqlite-vec (the dense leg).
pytest.importorskip("sqlite_vec")

FIXTURES_VAULT = Path(__file__).parent / "fixtures" / "vault_small"


# ---------------------------------------------------------------------------
# Test doubles for the embedder
# ---------------------------------------------------------------------------
class _DeterministicEmbedder:
    """Returns a fixed 8-dim vector based on a content keyword hint.

    The fake embedder lets tests place specific docs near or far from
    a query vector without loading any real model weights.
    """

    def __init__(self, dim: int = 8, keyword: str = ""):
        self.dim = dim
        # ``keyword`` toggles which axis carries the query weight.
        # Tests use the ``query_axis`` arg directly through encode_one.
        self._keyword = keyword
        self._encoded: list[str] = []
        self._loaded = True

    # Required protocol surface
    def available(self) -> bool:
        return self._loaded

    def disabled(self) -> bool:
        return False

    def encode_one(self, text: str) -> bytes | None:
        self._encoded.append(text)
        # Map the input to a one-hot-ish vector based on a shared
        # axis index keyed on text length modulo dim. The fixture
        # rigs vector inserts to match these axes.
        axis = len(text) % self.dim
        v = [0.0] * self.dim
        v[axis] = 1.0
        return pack_float32(v)

    def encode_batch(self, texts: list[str]) -> list[bytes]:
        # Used by VaultIndexer._embed_chunks_safely.
        out: list[bytes] = []
        for t in texts:
            blob = self.encode_one(t)
            if blob is None:
                return []
            out.append(blob)
        return out


class _FailingEmbedder:
    """Embedder that pretends to load OK but every encode raises."""

    def __init__(self):
        self._encoded = 0

    def available(self) -> bool:
        return True

    def disabled(self) -> bool:
        return False

    def encode_one(self, text: str) -> bytes | None:
        self._encoded += 1
        raise RuntimeError("simulated metal failure")


class _DisabledEmbedder:
    """Embedder that has already entered the permanent failed state."""

    def available(self) -> bool:
        return False

    def disabled(self) -> bool:
        return True

    def encode_one(self, text: str) -> bytes | None:
        # Defensive: if anybody actually called us we'd return None.
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vec(values):
    return pack_float32(values)


@pytest.fixture
def populated_store(tmp_path):
    """A small store with three vault chunks AND matching vectors."""
    s = open_store(tmp_path / "memory.db", embed_dim=8)
    rows = [
        ("notes/distillation.md", "I think distillation only works when teacher logits carry information"),
        ("notes/korean.md", "지난 달에 distillation 에 대해 어떻게 생각했지"),
        ("notes/banana.md", "completely unrelated banana muffin recipe"),
    ]
    chunk_ids: list[str] = []
    with s.transaction():
        for path, text in rows:
            fid = s.upsert_vault_file(
                path=path, mtime=1.0, sha256=path, size=len(text)
            )
            cid = chunk_id_for(path, 0, len(text))
            s.replace_chunks_for_file(
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
            chunk_ids.append(cid)
        # The fake embedder maps texts to one-hot axes by len(text)%8.
        # We embed each row with the same axis the embedder will
        # produce so a query for that text snaps right back to it.
        for cid, (_, text) in zip(chunk_ids, rows):
            axis = len(text) % 8
            v = [0.0] * 8
            v[axis] = 1.0
            s.insert_vector(chunk_id=cid, embedding=_vec(v))
    yield s
    s.close()


@pytest.fixture
def enabled_config(tmp_path):
    return resolve_memory_config(
        {
            "MEMORY_ENABLED": "1",
            "MEMORY_VAULT_PATH": str(FIXTURES_VAULT),
            "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
            "MEMORY_EMBED_DIM": "8",
        }
    )


# ---------------------------------------------------------------------------
# RRF fusion (pure function)
# ---------------------------------------------------------------------------
class TestRrfFusion:
    def _hit(self, cid: str, text: str = "x") -> SearchHit:
        return SearchHit(
            chunk_id=cid,
            source_type="vault",
            source_path=f"{cid}.md",
            timestamp="2025-01-01T00:00:00Z",
            score=0.5,
            excerpt=text,
        )

    def test_empty_streams_return_empty(self):
        assert _rrf_fuse([], [], k=60, top_k=5) == []

    def test_only_bm25_preserves_order(self):
        bm = [self._hit("a"), self._hit("b"), self._hit("c")]
        out = _rrf_fuse(bm, [], k=60, top_k=5)
        assert [h.chunk_id for h in out] == ["a", "b", "c"]

    def test_only_dense_preserves_order(self):
        de = [self._hit("a"), self._hit("b"), self._hit("c")]
        out = _rrf_fuse([], de, k=60, top_k=5)
        assert [h.chunk_id for h in out] == ["a", "b", "c"]

    def test_intersect_promotes_shared_chunks(self):
        # ``b`` appears top of both streams → highest fused rank.
        bm = [self._hit("a"), self._hit("b"), self._hit("c")]
        de = [self._hit("b"), self._hit("d"), self._hit("a")]
        out = _rrf_fuse(bm, de, k=60, top_k=4)
        assert out[0].chunk_id == "b"
        # ``a`` appears in both at decent ranks → second.
        assert out[1].chunk_id == "a"

    def test_top_k_truncation(self):
        bm = [self._hit(f"x{i}") for i in range(10)]
        out = _rrf_fuse(bm, [], k=60, top_k=3)
        assert len(out) == 3

    def test_excerpt_prefers_bm25_when_both_streams_hit(self):
        bm_hit = self._hit("shared", text="bm25-excerpt")
        de_hit = self._hit("shared", text="dense-excerpt")
        out = _rrf_fuse([bm_hit], [de_hit], k=60, top_k=1)
        # BM25 carries the FTS5 snippet so it wins as the representative.
        assert out[0].excerpt == "bm25-excerpt"

    def test_k_minimum_clamped(self):
        # k=0 is degenerate; helper clamps to 1.
        out = _rrf_fuse([self._hit("a")], [], k=0, top_k=1)
        assert [h.chunk_id for h in out] == ["a"]


# ---------------------------------------------------------------------------
# search_memory hybrid path
# ---------------------------------------------------------------------------
class TestSearchMemoryHybrid:
    def test_hybrid_fuses_bm25_and_dense(self, populated_store, enabled_config):
        embedder = _DeterministicEmbedder(dim=8)
        # Use the english doc text so BM25 finds it AND the fake
        # embedder maps the query to the same axis.
        query = "I think distillation only works when teacher logits carry information"
        envelope = search_memory(
            store=populated_store,
            query=query,
            top_k=2,
            source_filter="both",
            config=enabled_config,
            embedder=embedder,
        )
        assert envelope["status"] == "ok"
        assert envelope["degraded"] is False
        # Best hit is the english distillation note.
        assert any(
            r["source_path"] == "notes/distillation.md"
            for r in envelope["results"]
        )

    def test_dense_failure_falls_back_to_bm25_with_degraded(
        self, populated_store, enabled_config
    ):
        envelope = search_memory(
            store=populated_store,
            query="distillation",
            top_k=3,
            source_filter="both",
            config=enabled_config,
            embedder=_FailingEmbedder(),
        )
        # We still got results from BM25 even though dense raised.
        assert envelope["status"] == "ok"
        assert envelope["degraded"] is True
        assert envelope["results"], "BM25 must still produce hits"

    def test_disabled_embedder_uses_bm25_only(self, populated_store, enabled_config):
        envelope = search_memory(
            store=populated_store,
            query="distillation",
            top_k=3,
            source_filter="both",
            config=enabled_config,
            embedder=_DisabledEmbedder(),
        )
        assert envelope["status"] == "ok"
        # Permanently-disabled embedder is NOT a runtime degradation:
        # it indicates we never even attempted dense, so degraded=False.
        assert envelope["degraded"] is False

    def test_no_embedder_uses_bm25_only(self, populated_store, enabled_config):
        envelope = search_memory(
            store=populated_store,
            query="distillation",
            top_k=3,
            source_filter="both",
            config=enabled_config,
            embedder=None,
        )
        assert envelope["status"] == "ok"
        assert envelope["degraded"] is False
        assert envelope["results"]

    def test_embed_disabled_env_uses_bm25_only(self, populated_store, tmp_path):
        cfg = resolve_memory_config(
            {
                "MEMORY_ENABLED": "1",
                "MEMORY_VAULT_PATH": str(FIXTURES_VAULT),
                "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
                "MEMORY_EMBED_DIM": "8",
                "MEMORY_EMBED_DISABLED": "1",
            }
        )
        # Even with a healthy embedder available, the operator opt-out
        # keeps the dense leg silent and the result is non-degraded.
        envelope = search_memory(
            store=populated_store,
            query="distillation",
            top_k=3,
            source_filter="both",
            config=cfg,
            embedder=_DeterministicEmbedder(dim=8),
        )
        assert envelope["status"] == "ok"
        assert envelope["degraded"] is False

    def test_envelope_carries_required_fields(
        self, populated_store, enabled_config
    ):
        envelope = search_memory(
            store=populated_store,
            query="distillation",
            top_k=2,
            source_filter="both",
            config=enabled_config,
            embedder=_DeterministicEmbedder(dim=8),
        )
        # REQ-U3: every result must carry these contract fields.
        for r in envelope["results"]:
            assert set(r.keys()) >= {
                "source_type",
                "source_path",
                "timestamp",
                "score",
                "excerpt",
            }
            assert 0.0 <= float(r["score"]) <= 1.0
            assert isinstance(r["excerpt"], str)
            assert len(r["excerpt"]) <= 500

    def test_dense_query_failure_logs_with_exception(
        self, populated_store, enabled_config, caplog
    ):
        embedder = _FailingEmbedder()
        with caplog.at_level(logging.ERROR, logger="vllm_mlx.memory.server"):
            envelope = search_memory(
                store=populated_store,
                query="distillation",
                top_k=3,
                source_filter="both",
                config=enabled_config,
                embedder=embedder,
            )
        assert envelope["degraded"] is True
        # The exception path uses logger.exception; pytest captures it.
        assert any(
            "dense search failed" in rec.message for rec in caplog.records
        )
        # Confirm the embedder really did get called once.
        assert embedder._encoded == 1  # noqa: SLF001 - probing test double

    def test_disabled_config_returns_unavailable_envelope(
        self, populated_store, tmp_path
    ):
        # MEMORY_ENABLED defaults to disabled when unset.
        cfg = resolve_memory_config({})
        envelope = search_memory(
            store=populated_store,
            query="distillation",
            top_k=3,
            source_filter="both",
            config=cfg,
            embedder=_DeterministicEmbedder(dim=8),
        )
        assert envelope["status"] == "unavailable"
        assert envelope["results"] == []
        assert envelope["degraded"] is True


# ---------------------------------------------------------------------------
# End-to-end: indexer with embedder writes both BM25 + dense rows
# ---------------------------------------------------------------------------
class TestIndexerWithEmbedder:
    def test_initial_scan_writes_vectors(self, tmp_path):
        store = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            indexer = VaultIndexer(
                store,
                vault_root=FIXTURES_VAULT,
                denylist=MEMORY_DEFAULT_DENYLIST,
                embedder=_DeterministicEmbedder(dim=8),
                embed_disabled=False,
            )
            stats = indexer.initial_scan()
            assert stats.chunks_written > 0
            # All chunks in this fixture are short → all should now
            # have a vector counterpart.
            assert store.count_vectors() == stats.chunks_written
            assert store.count_chunks_missing_vectors() == 0
        finally:
            store.close()

    def test_initial_scan_skips_when_embed_disabled(self, tmp_path):
        store = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            indexer = VaultIndexer(
                store,
                vault_root=FIXTURES_VAULT,
                denylist=MEMORY_DEFAULT_DENYLIST,
                embedder=_DeterministicEmbedder(dim=8),
                embed_disabled=True,
            )
            stats = indexer.initial_scan()
            assert stats.chunks_written > 0
            assert store.count_vectors() == 0
        finally:
            store.close()

    def test_initial_scan_skips_when_embedder_none(self, tmp_path):
        store = open_store(tmp_path / "memory.db", embed_dim=8)
        try:
            indexer = VaultIndexer(
                store,
                vault_root=FIXTURES_VAULT,
                denylist=MEMORY_DEFAULT_DENYLIST,
                embedder=None,
            )
            stats = indexer.initial_scan()
            assert stats.chunks_written > 0
            assert store.count_vectors() == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# _try_open_store / _try_build_embedder lifecycle helpers
# ---------------------------------------------------------------------------
class TestTryOpenStore:
    def test_returns_none_when_disabled(self):
        cfg = resolve_memory_config({})
        assert _try_open_store(cfg) is None

    def test_returns_store_when_enabled(self, tmp_path):
        cfg = resolve_memory_config(
            {
                "MEMORY_ENABLED": "1",
                "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
                "MEMORY_VAULT_PATH": str(tmp_path),
                "MEMORY_EMBED_DIM": "8",
            }
        )
        s = _try_open_store(cfg)
        try:
            assert s is not None
            assert s.embed_dim == 8
            assert s.vec_loaded is True
        finally:
            if s is not None:
                s.close()


class TestTryBuildEmbedder:
    def test_returns_none_when_store_is_none(self, enabled_config):
        assert _try_build_embedder(None, enabled_config) is None

    def test_returns_none_when_disabled_config(self, populated_store):
        cfg = resolve_memory_config({})
        assert _try_build_embedder(populated_store, cfg) is None

    def test_returns_none_when_embed_disabled(self, populated_store, tmp_path):
        cfg = resolve_memory_config(
            {
                "MEMORY_ENABLED": "1",
                "MEMORY_VAULT_PATH": str(tmp_path),
                "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
                "MEMORY_EMBED_DIM": "8",
                "MEMORY_EMBED_DISABLED": "1",
            }
        )
        assert _try_build_embedder(populated_store, cfg) is None

    def test_returns_none_when_vec_unloaded(self, populated_store, enabled_config):
        # Force the unloaded flag so we test the early return path.
        populated_store._vec_loaded = False  # noqa: SLF001 - intentional poke
        try:
            assert _try_build_embedder(populated_store, enabled_config) is None
        finally:
            populated_store._vec_loaded = True  # restore

    def test_records_identity_on_success(self, populated_store, enabled_config):
        embedder = _try_build_embedder(populated_store, enabled_config)
        assert embedder is not None
        # The store must now have the model + dim recorded so a
        # subsequent backfill or restart picks up the same identity.
        assert populated_store.get_meta("embedding_model") == enabled_config.embed_model
        assert int(populated_store.get_meta("embedding_dim")) == enabled_config.embed_dim

    def test_returns_none_on_recorded_model_mismatch(
        self, populated_store, tmp_path
    ):
        # Pre-record a different model to trigger REQ-U4 protection.
        populated_store.record_embedding_identity(model="other/model", dim=8)
        cfg = resolve_memory_config(
            {
                "MEMORY_ENABLED": "1",
                "MEMORY_VAULT_PATH": str(tmp_path),
                "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
                "MEMORY_EMBED_MODEL": "BAAI/bge-m3",
                "MEMORY_EMBED_DIM": "8",
            }
        )
        assert _try_build_embedder(populated_store, cfg) is None
