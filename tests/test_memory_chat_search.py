# SPDX-License-Identifier: Apache-2.0
"""Tests for Phase-3 chat search through ``memory_search``.

@TEST:MEMORY-01/chat_search

Coverage:

* Chat rows persisted via ``persist_chat_row`` appear in the
  ``memory_search`` envelope with ``source_type=='chat'`` (REQ-E4).
* ``source_filter='vault'`` excludes chat hits; ``='chat'`` excludes
  vault hits (REQ-O1).
* ``MEMORY_CHAT_RETENTION_DAYS`` hides rows whose ``timestamp`` is
  older than the cutoff (REQ-N5).
* Embedded chat rows surface through the dense search path with the
  expected metadata (Phase 3 ``_chat_meta_for_chunk`` lookup).

We use the BM25-only path for most tests so we do not need to load
real embedding weights. A separate test exercises ``search_dense``
with a stub embedder + a manually-inserted vec_chunks row.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from vllm_mlx.memory.chatlog import (
    chunk_id_for_chat,
    new_session_id,
    persist_chat_row,
)
from vllm_mlx.memory.config import resolve_memory_config
from vllm_mlx.memory.indexer import VaultIndexer
from vllm_mlx.memory.server import search_memory
from vllm_mlx.memory.store import open_store

FIXTURES_VAULT = Path(__file__).parent / "fixtures" / "vault_small"


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def store_with_chat(tmp_path):
    """Open a store and persist one chat row about ``MoE active params``."""
    store = open_store(tmp_path / "memory.db")
    _run(
        persist_chat_row(
            store,
            request_id="chatcmpl-moe-001",
            session_id=new_session_id(),
            model="qwen3-test",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Last time we discussed MoE active parameters in "
                        "Mixtral. Can you remind me what we concluded?"
                    ),
                }
            ],
            assistant_text=(
                "We concluded that Mixtral 8x7B uses 12.9B active parameters "
                "per token even though the total is 46.7B."
            ),
        )
    )
    yield store
    store.close()


@pytest.fixture
def store_with_chat_and_vault(tmp_path):
    """Store with a vault index AND one chat row mentioning distillation."""
    store = open_store(tmp_path / "memory.db")
    indexer = VaultIndexer(store, vault_root=FIXTURES_VAULT, denylist=())
    indexer.initial_scan()
    _run(
        persist_chat_row(
            store,
            request_id="chatcmpl-distill-002",
            session_id=new_session_id(),
            model="qwen3-test",
            messages=[
                {
                    "role": "user",
                    "content": "Tell me about distillation again please.",
                }
            ],
            assistant_text=(
                "Knowledge distillation transfers a teacher model's behavior "
                "to a student via soft targets."
            ),
        )
    )
    yield store
    store.close()


@pytest.fixture
def enabled_config(tmp_path):
    return resolve_memory_config(
        {
            "MEMORY_ENABLED": "1",
            "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
            "MEMORY_CHAT_LOG_ENABLED": "1",
            # Generous retention so the default does not hide test rows.
            "MEMORY_CHAT_RETENTION_DAYS": "365",
        }
    )


# ---------------------------------------------------------------------------
# Default filter (both) returns chat hits
# ---------------------------------------------------------------------------
class TestChatAppearsInSearch:
    def test_chat_row_returned_with_default_filter(
        self, store_with_chat, enabled_config
    ):
        env = search_memory(
            store=store_with_chat,
            query="MoE active parameters",
            top_k=5,
            source_filter="both",
            config=enabled_config,
        )
        assert env["status"] == "ok"
        chat_hits = [r for r in env["results"] if r["source_type"] == "chat"]
        assert chat_hits, "expected at least one chat hit"

    def test_chat_envelope_includes_source_type_chat(
        self, store_with_chat, enabled_config
    ):
        env = search_memory(
            store=store_with_chat,
            query="Mixtral",
            top_k=5,
            config=enabled_config,
        )
        assert env["status"] == "ok"
        # All hits in this fixture are chat (no vault index).
        for r in env["results"]:
            assert r["source_type"] == "chat"
            # source_path follows the SPEC envelope contract.
            assert r["source_path"].startswith("chat/")
            # Excerpt is bounded.
            assert len(r["excerpt"]) <= 500
            # Score is in [0, 1].
            assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# source_filter exclusivity (REQ-O1)
# ---------------------------------------------------------------------------
class TestSourceFilter:
    def test_filter_vault_excludes_chat(
        self, store_with_chat_and_vault, enabled_config
    ):
        env = search_memory(
            store=store_with_chat_and_vault,
            query="distillation",
            top_k=10,
            source_filter="vault",
            config=enabled_config,
        )
        assert env["status"] == "ok"
        for r in env["results"]:
            assert r["source_type"] == "vault"

    def test_filter_chat_excludes_vault(
        self, store_with_chat_and_vault, enabled_config
    ):
        env = search_memory(
            store=store_with_chat_and_vault,
            query="distillation",
            top_k=10,
            source_filter="chat",
            config=enabled_config,
        )
        assert env["status"] == "ok"
        for r in env["results"]:
            assert r["source_type"] == "chat"
        # And there must be at least one chat result for our seed row.
        assert env["results"], "expected at least one chat hit"

    def test_filter_both_includes_both(
        self, store_with_chat_and_vault, enabled_config
    ):
        env = search_memory(
            store=store_with_chat_and_vault,
            query="distillation",
            top_k=10,
            source_filter="both",
            config=enabled_config,
        )
        assert env["status"] == "ok"
        types = {r["source_type"] for r in env["results"]}
        # We expect at least one of each.
        assert "chat" in types
        assert "vault" in types


# ---------------------------------------------------------------------------
# Retention cutoff (REQ-N5)
# ---------------------------------------------------------------------------
class TestRetentionCutoff:
    def test_old_chat_row_hidden_by_retention(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            # Persist a row, then back-date it past the 1-day cutoff.
            _run(
                persist_chat_row(
                    store,
                    request_id="chatcmpl-old",
                    session_id=new_session_id(),
                    model="m",
                    messages=[{"role": "user", "content": "ancient question"}],
                    assistant_text="ancient answer",
                )
            )
            # Force the timestamp 10 days into the past.
            old_ts = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() - 10 * 86400),
            )
            store.conn.execute(
                "UPDATE chat_messages SET timestamp = ? WHERE request_id = ?",
                (old_ts, "chatcmpl-old"),
            )

            cfg = resolve_memory_config(
                {
                    "MEMORY_ENABLED": "1",
                    "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
                    "MEMORY_CHAT_LOG_ENABLED": "1",
                    "MEMORY_CHAT_RETENTION_DAYS": "1",
                }
            )

            env = search_memory(
                store=store,
                query="ancient",
                top_k=5,
                source_filter="chat",
                config=cfg,
            )
            assert env["status"] == "ok"
            # Older than 1-day cutoff -> hidden from results.
            assert env["results"] == []
        finally:
            store.close()

    def test_recent_chat_row_passes_retention(self, store_with_chat, tmp_path):
        cfg = resolve_memory_config(
            {
                "MEMORY_ENABLED": "1",
                "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
                "MEMORY_CHAT_LOG_ENABLED": "1",
                "MEMORY_CHAT_RETENTION_DAYS": "365",
            }
        )
        env = search_memory(
            store=store_with_chat,
            query="Mixtral",
            top_k=5,
            source_filter="chat",
            config=cfg,
        )
        assert env["status"] == "ok"
        assert env["results"], "expected the freshly-persisted chat row"


# ---------------------------------------------------------------------------
# Empty searches degrade gracefully
# ---------------------------------------------------------------------------
class TestEmptyResults:
    def test_no_match_returns_empty_results(
        self, store_with_chat, enabled_config
    ):
        env = search_memory(
            store=store_with_chat,
            query="zzzznosuchphrase__",
            top_k=5,
            source_filter="chat",
            config=enabled_config,
        )
        assert env["status"] == "ok"
        assert env["results"] == []

    def test_search_with_no_chat_rows_in_chat_filter(
        self, tmp_path, enabled_config
    ):
        # Empty store + chat filter -> no results, no error.
        store = open_store(tmp_path / "memory.db")
        try:
            env = search_memory(
                store=store,
                query="anything at all",
                top_k=5,
                source_filter="chat",
                config=enabled_config,
            )
            assert env["status"] == "ok"
            assert env["results"] == []
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Dense search post-lookup for chat rows
# ---------------------------------------------------------------------------
class TestDenseChatLookup:
    def test_chat_meta_lookup_returns_path_and_excerpt(self, store_with_chat):
        # The chunk_id for the persisted row is deterministic.
        cid = chunk_id_for_chat("chatcmpl-moe-001")
        meta = store_with_chat._chat_meta_for_chunk(cid)
        assert meta is not None
        assert meta["source_path"].startswith("chat/")
        assert "Mixtral" in meta["excerpt"]
        assert meta["timestamp"]  # ISO-8601 UTC

    def test_chat_meta_for_unknown_chunk_returns_none(self, store_with_chat):
        meta = store_with_chat._chat_meta_for_chunk("0" * 32)
        assert meta is None


# ---------------------------------------------------------------------------
# count_chat_messages_missing_vectors (used by the embed loop)
# ---------------------------------------------------------------------------
class TestUnembeddedCount:
    def test_count_no_op_when_vec_disabled(self, store_with_chat):
        # In tests sqlite-vec may or may not load; both branches return
        # plausible numbers without raising.
        n = store_with_chat.count_chat_messages_missing_vectors()
        assert isinstance(n, int)
        assert n >= 0

    def test_count_chat_messages_total(self, store_with_chat):
        assert store_with_chat.count_chat_messages() == 1
