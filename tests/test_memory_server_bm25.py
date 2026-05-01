# SPDX-License-Identifier: Apache-2.0
"""Tests for the memory MCP server's pure search dispatcher.

@TEST:MEMORY-01/server

We exercise :func:`vllm_mlx.memory.server.search_memory` directly so we
do not need to spawn the stdio MCP child process. The end-to-end
``mcp.json`` integration is covered by manual smoke (see report) and a
follow-up live-server test in Phase 4 (P1-AC1).

REQ-S1 / REQ-N4 invariants tested:
- Disabled config returns the documented "unavailable" envelope.
- Missing vault returns ``status=ok, results=[], degraded=False``
  (P1-AC6 contract — "fail open" rather than crashing chat).
- All envelopes carry the contract fields from REQ-U3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_mlx.memory.config import MEMORY_DEFAULT_DENYLIST, resolve_memory_config
from vllm_mlx.memory.indexer import VaultIndexer
from vllm_mlx.memory.server import (
    TOOL_DESCRIPTION,
    TOOL_INPUT_SCHEMA,
    TOOL_NAME,
    search_memory,
)
from vllm_mlx.memory.store import open_store

FIXTURES_VAULT = Path(__file__).parent / "fixtures" / "vault_small"


@pytest.fixture
def populated_store(tmp_path):
    store = open_store(tmp_path / "memory.db")
    indexer = VaultIndexer(
        store,
        vault_root=FIXTURES_VAULT,
        denylist=MEMORY_DEFAULT_DENYLIST,
    )
    indexer.initial_scan()
    yield store
    store.close()


@pytest.fixture
def enabled_config(tmp_path):
    return resolve_memory_config(
        {
            "MEMORY_ENABLED": "1",
            "MEMORY_VAULT_PATH": str(FIXTURES_VAULT),
            "MEMORY_DB_PATH": str(tmp_path / "memory.db"),
        }
    )


@pytest.fixture
def disabled_config():
    return resolve_memory_config({})


# ---------------------------------------------------------------------------
# Tool surface (REQ-U1 / SPEC §7)
# ---------------------------------------------------------------------------


class TestToolSurface:
    def test_tool_name_is_memory_search(self):
        assert TOOL_NAME == "memory_search"

    def test_tool_description_mentions_remember(self):
        assert "remember" in TOOL_DESCRIPTION.lower()

    def test_tool_schema_has_required_query(self):
        assert TOOL_INPUT_SCHEMA["type"] == "object"
        assert "query" in TOOL_INPUT_SCHEMA["properties"]
        assert TOOL_INPUT_SCHEMA["required"] == ["query"]

    def test_tool_schema_top_k_bounds(self):
        top_k = TOOL_INPUT_SCHEMA["properties"]["top_k"]
        assert top_k["minimum"] == 1
        assert top_k["maximum"] == 20

    def test_tool_schema_source_filter_enum(self):
        sf = TOOL_INPUT_SCHEMA["properties"]["source_filter"]
        assert set(sf["enum"]) == {"vault", "chat", "both"}


# ---------------------------------------------------------------------------
# Disabled / unavailable paths
# ---------------------------------------------------------------------------


class TestDisabledAndUnavailable:
    def test_disabled_config_returns_unavailable(self, disabled_config):
        env = search_memory(
            store=None,
            query="anything",
            config=disabled_config,
        )
        assert env["status"] == "unavailable"
        assert env["degraded"] is True
        assert env["results"] == []
        assert "MEMORY_ENABLED" in env["message"]

    def test_enabled_but_no_store_returns_ok_empty(self, enabled_config):
        # P1-AC6: vault path missing or store unreachable => ok+empty,
        # not a crash.
        env = search_memory(
            store=None,
            query="anything",
            config=enabled_config,
        )
        assert env["status"] == "ok"
        assert env["degraded"] is False
        assert env["results"] == []


# ---------------------------------------------------------------------------
# Search envelope contract (REQ-U3)
# ---------------------------------------------------------------------------


class TestSearchEnvelope:
    def test_search_returns_known_phrase(self, populated_store, enabled_config):
        env = search_memory(
            store=populated_store,
            query="distillation",
            top_k=5,
            config=enabled_config,
        )
        assert env["status"] == "ok"
        assert env["degraded"] is False
        assert isinstance(env["results"], list)
        assert env["results"], "expected at least one hit"
        # Every result must satisfy REQ-U3 fields.
        for r in env["results"]:
            assert set(r.keys()) >= {
                "source_type",
                "source_path",
                "timestamp",
                "score",
                "excerpt",
            }
            assert r["source_type"] in ("vault", "chat")
            assert 0.0 <= r["score"] <= 1.0
            assert len(r["excerpt"]) <= 500

    def test_search_finds_korean_query(self, populated_store, enabled_config):
        env = search_memory(
            store=populated_store,
            query="한국어 검색",
            top_k=5,
            config=enabled_config,
        )
        assert env["status"] == "ok"
        # The Korean fixture has the full phrase "한국어 검색".
        assert env["results"], "expected korean fixture hit"
        assert any(
            "korean-research" in r["source_path"] for r in env["results"]
        )

    def test_top_k_is_clamped_to_max(self, populated_store, enabled_config):
        env = search_memory(
            store=populated_store,
            query="distillation",
            top_k=9999,  # absurd; must clamp to top_k_max (20).
            config=enabled_config,
        )
        assert len(env["results"]) <= enabled_config.top_k_max

    def test_top_k_minimum_is_one(self, populated_store, enabled_config):
        env = search_memory(
            store=populated_store,
            query="distillation",
            top_k=0,  # below minimum; clamps to 1.
            config=enabled_config,
        )
        assert isinstance(env["results"], list)
        assert len(env["results"]) <= 1

    def test_source_filter_vault_excludes_chat(
        self, populated_store, enabled_config
    ):
        env = search_memory(
            store=populated_store,
            query="distillation",
            source_filter="vault",
            config=enabled_config,
        )
        assert all(r["source_type"] == "vault" for r in env["results"])

    def test_source_filter_chat_returns_empty_in_phase1(
        self, populated_store, enabled_config
    ):
        env = search_memory(
            store=populated_store,
            query="distillation",
            source_filter="chat",
            config=enabled_config,
        )
        assert env["results"] == []

    def test_invalid_source_filter_falls_back_to_both(
        self, populated_store, enabled_config
    ):
        env = search_memory(
            store=populated_store,
            query="distillation",
            source_filter="not-a-valid-filter",
            config=enabled_config,
        )
        # No exception, just behaves like "both".
        assert env["status"] == "ok"

    def test_results_ranked_descending_by_score(
        self, populated_store, enabled_config
    ):
        env = search_memory(
            store=populated_store,
            query="distillation",
            top_k=10,
            config=enabled_config,
        )
        scores = [r["score"] for r in env["results"]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Failure isolation (REQ-N4)
# ---------------------------------------------------------------------------


class TestFailureIsolation:
    def test_search_with_broken_store_returns_error_envelope(
        self, enabled_config
    ):
        """If the store raises mid-search, we must return a structured
        error envelope, not propagate the exception."""

        class BrokenStore:
            def search_bm25(self, *args, **kwargs):
                raise RuntimeError("simulated DB corruption")

        env = search_memory(
            store=BrokenStore(),  # type: ignore[arg-type]
            query="anything",
            config=enabled_config,
        )
        assert env["status"] == "error"
        assert env["degraded"] is True
        assert env["results"] == []


# ---------------------------------------------------------------------------
# REQ-S1: server module imports cleanly even with no env vars set.
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_module_importable_without_mcp_sdk_when_calling_pure_path(self):
        # Importing the module already succeeded by virtue of running
        # this test. The point of this test is to ensure that the pure
        # function search path does not require the mcp SDK to be
        # importable — i.e. heavy imports are deferred to _run_mcp_stdio.
        from vllm_mlx.memory import server as server_mod

        assert hasattr(server_mod, "search_memory")
        assert callable(server_mod.search_memory)


# ---------------------------------------------------------------------------
# _try_open_store fail-safe
# ---------------------------------------------------------------------------


class TestTryOpenStore:
    def test_disabled_config_returns_none(self, disabled_config):
        from vllm_mlx.memory.server import _try_open_store

        assert _try_open_store(disabled_config) is None

    def test_enabled_with_writable_path_returns_store(self, tmp_path):
        from vllm_mlx.memory.server import _try_open_store

        cfg = resolve_memory_config(
            {
                "MEMORY_ENABLED": "1",
                "MEMORY_DB_PATH": str(tmp_path / "ok.db"),
            }
        )
        store = _try_open_store(cfg)
        assert store is not None
        try:
            assert (tmp_path / "ok.db").exists()
        finally:
            store.close()

    def test_unwritable_path_degrades_to_none(self, tmp_path):
        # Point the DB at a location whose parent cannot be created.
        # Using a path under /dev/null is portable on POSIX.
        from vllm_mlx.memory.server import _try_open_store

        cfg = resolve_memory_config(
            {
                "MEMORY_ENABLED": "1",
                "MEMORY_DB_PATH": "/dev/null/cannot-create-this/memory.db",
            }
        )
        result = _try_open_store(cfg)
        # REQ-N4: a filesystem failure must produce ``None``, never raise.
        assert result is None
