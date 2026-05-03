# SPDX-License-Identifier: Apache-2.0
"""MCP stdio server that exposes ``memory_search`` to Qwen3.6.

@CODE:MEMORY-01/server

Phase 1: BM25-only over SQLite FTS5 (no embeddings yet). Spawned by
the existing ``MCPClientManager`` as a stdio child via the entry in
``mcp.json`` / ``mcp.example.json``.

The server reads its configuration from environment variables (see
``vllm_mlx/memory/config.py``). When ``MEMORY_ENABLED`` is not truthy
the server still starts (the parent MCP manager spawns it eagerly) but
exposes the tool with a "memory disabled" notice and returns
``{status: "unavailable", results: []}``.

REQ-N4: every tool invocation is wrapped in a try/except so that an
internal failure cannot poison the chat path. The MCP protocol layer
itself enforces structured error responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

from .config import MemoryRuntimeConfig, resolve_memory_config
from .store import MemoryStore, SearchHit

logger = logging.getLogger("vllm_mlx.memory.server")


# Phase 2 — keep the embedder ref loose so MCP server tests do not
# need to import mlx-embeddings. The runtime check is a duck-typed
# ``encode_one`` call.
class _SupportsEmbedOne:  # pragma: no cover - protocol-only sentinel
    def encode_one(self, text: str) -> bytes | None: ...
    def available(self) -> bool: ...
    def disabled(self) -> bool: ...


# ---------------------------------------------------------------------------
# Tool schema (REQ-U1 / SPEC §7)
# ---------------------------------------------------------------------------
TOOL_NAME = "memory_search"
TOOL_DESCRIPTION = (
    "Search the user's long-term memory (Obsidian vault notes and prior "
    "conversations). Returns ranked snippets with citations. Use when the "
    "user references past notes, prior conversations, or asks 'do you "
    "remember...'."
)

TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language query. May be in any language.",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "default": 5,
        },
        "source_filter": {
            "type": "string",
            "enum": ["vault", "chat", "both"],
            "default": "both",
        },
    },
    "required": ["query"],
}


# ---------------------------------------------------------------------------
# Pure search dispatcher (testable without any MCP transport)
# ---------------------------------------------------------------------------
def search_memory(
    *,
    store: MemoryStore | None,
    query: str,
    top_k: int = 5,
    source_filter: str = "both",
    config: MemoryRuntimeConfig,
    embedder: _SupportsEmbedOne | None = None,
) -> dict[str, Any]:
    """Return the JSON envelope for one ``memory_search`` invocation.

    This function never raises — every failure path produces a
    ``{status, degraded, results}`` envelope so the model's tool-use
    loop always sees structured data.

    Phase 2: hybrid BM25 + dense vector search with reciprocal-rank-
    fusion (RRF). When ``embedder`` is None, ``config.embed_disabled``
    is True, or sqlite-vec is not loaded on the store, falls back to
    BM25-only. When the dense path was *attempted* but failed
    (REQ-O3), the envelope carries ``degraded: true`` so the model
    knows the result quality is reduced.
    """
    if not config.enabled:
        return {
            "status": "unavailable",
            "degraded": True,
            "results": [],
            "message": (
                "Memory subsystem is disabled (MEMORY_ENABLED!=1). Set "
                "MEMORY_ENABLED=1 and MEMORY_VAULT_PATH on the server to "
                "enable."
            ),
        }

    if store is None:
        # Vault path missing or DB unreachable. P1-AC6: still return ok+empty.
        logger.warning(
            "[memory] search called but store is unavailable (vault missing?)"
        )
        return {
            "status": "ok",
            "degraded": False,
            "results": [],
            "message": "vault not indexed (path missing or unreadable)",
        }

    # Sanitize args. REQ-E5: top_k between 1 and top_k_max (default 5, max 20).
    # Use an explicit ``None`` check rather than ``or`` so an explicit
    # ``top_k=0`` is rejected (clamped to 1) instead of falling back to
    # the configured default.
    if top_k is None:
        top_k = config.top_k_default
    top_k = max(1, min(int(top_k), config.top_k_max))
    source_filter = (source_filter or "both").lower()
    if source_filter not in ("vault", "chat", "both"):
        source_filter = "both"

    # We over-fetch on each leg so RRF has more candidates to fuse.
    # 2x is a common heuristic that balances recall vs. cost.
    leg_k = top_k * 2

    # REQ-N5 — chat retention is enforced at QUERY time. Eviction
    # (deleting old rows) is SPEC-MEMORY-02 work; here we just hide
    # them so a stale row cannot leak into the model's context window.
    chat_cutoff_iso: str | None = None
    if config.chat_retention_days and config.chat_retention_days > 0:
        cutoff_epoch = time.time() - (
            float(config.chat_retention_days) * 86400.0
        )
        chat_cutoff_iso = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(cutoff_epoch)
        )

    # ----- BM25 leg -------------------------------------------------------
    bm25_hits: list[SearchHit] = []
    bm25_failed = False
    try:
        bm25_hits = store.search_bm25(
            query,
            top_k=leg_k,
            source_filter=source_filter,
            chat_retention_cutoff_iso=chat_cutoff_iso,
        )
    except TypeError:
        # Backwards-compat for older Store signatures (Phase 1+2 tests).
        try:
            bm25_hits = store.search_bm25(
                query, top_k=leg_k, source_filter=source_filter
            )
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception("[memory] BM25 search failed for query=%r", query)
            bm25_failed = True
    except Exception:  # noqa: BLE001 - REQ-N4
        logger.exception("[memory] BM25 search failed for query=%r", query)
        bm25_failed = True

    # ----- Dense leg (Phase 2) -------------------------------------------
    dense_hits: list[SearchHit] = []
    dense_attempted = False
    dense_failed = False
    if (
        embedder is not None
        and not config.embed_disabled
        and getattr(store, "vec_loaded", False)
        and not embedder.disabled()
    ):
        dense_attempted = True
        try:
            qvec = embedder.encode_one(query)
            if qvec is None:
                dense_failed = True
            else:
                try:
                    dense_hits = store.search_dense(
                        qvec,
                        top_k=leg_k,
                        source_filter=source_filter,
                        chat_retention_cutoff_iso=chat_cutoff_iso,
                    )
                except TypeError:
                    dense_hits = store.search_dense(
                        qvec, top_k=leg_k, source_filter=source_filter
                    )
        except Exception:  # noqa: BLE001 - REQ-N4 / REQ-O3
            logger.exception(
                "[memory] dense search failed for query=%r — falling back "
                "to BM25-only with degraded:true",
                query,
            )
            dense_failed = True

    # ----- Fusion ---------------------------------------------------------
    # Both legs failed → no results possible; surface as error.
    if bm25_failed and (dense_failed or not dense_attempted):
        return {
            "status": "error",
            "degraded": True,
            "results": [],
            "message": "internal search error (see server logs)",
        }

    fused = _rrf_fuse(
        bm25_hits, dense_hits, k=int(config.hybrid_rrf_k), top_k=top_k
    )

    # ``degraded`` is True iff a dense result was supposed to happen
    # and didn't. A user who explicitly set MEMORY_EMBED_DISABLED=1
    # gets degraded=False because BM25 is the contracted result.
    degraded = bm25_failed or dense_failed

    return {
        "status": "ok",
        "degraded": degraded,
        "results": [
            {
                "source_type": h.source_type,
                "source_path": h.source_path,
                "timestamp": h.timestamp,
                "score": round(h.score, 4),
                "excerpt": h.excerpt,
            }
            for h in fused
        ],
    }


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (Cormack et al. 2009)
# ---------------------------------------------------------------------------
def _rrf_fuse(
    bm25_hits: list[SearchHit],
    dense_hits: list[SearchHit],
    *,
    k: int,
    top_k: int,
) -> list[SearchHit]:
    """Merge two ranked lists with reciprocal-rank fusion.

    For each chunk we compute ``score = sum(1 / (k + rank))`` over the
    streams it appears in (ranks are 1-indexed). The merged list is
    sorted by descending RRF score. When dense_hits is empty this
    reduces to BM25 ordering; the SearchHit objects themselves are
    taken from the BM25 stream (which carries the snippet excerpt
    from FTS5) when both streams hit the same chunk_id, falling back
    to the dense entry otherwise.

    The original :class:`SearchHit.score` (a per-leg unit score) is
    preserved on the returned object — it remains meaningful as a
    "how good was the best signal for this hit" hint for the model.
    """
    if k < 1:
        k = 1
    if not bm25_hits and not dense_hits:
        return []

    # rank_score keeps the running RRF sum per chunk_id.
    rank_score: dict[str, float] = {}
    # representative SearchHit per chunk_id; BM25 entries win because
    # they carry the FTS5-extracted excerpt with hit highlights.
    rep: dict[str, SearchHit] = {}

    for rank, h in enumerate(bm25_hits, start=1):
        rank_score[h.chunk_id] = rank_score.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
        rep.setdefault(h.chunk_id, h)
    for rank, h in enumerate(dense_hits, start=1):
        rank_score[h.chunk_id] = rank_score.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
        # Dense hits usually carry the same chunk_id as a BM25 hit;
        # only adopt them as the representative when BM25 missed.
        rep.setdefault(h.chunk_id, h)

    ordered = sorted(rep.values(), key=lambda h: rank_score[h.chunk_id], reverse=True)
    return ordered[: max(1, int(top_k))]


# ---------------------------------------------------------------------------
# MCP wiring
# ---------------------------------------------------------------------------
async def _run_mcp_stdio() -> None:
    """Run the MCP stdio server (spawned by the parent MCPClientManager).

    Heavy imports (``mcp.server``) are deferred to call time so unit
    tests for ``search_memory`` do not pull in the SDK.
    """
    try:
        from mcp import types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError as exc:  # pragma: no cover - guarded by pyproject dep
        logger.error("[memory] MCP SDK not available: %s", exc)
        raise

    config = resolve_memory_config(os.environ)
    logger.info(
        "[memory] starting MCP stdio server: enabled=%s vault=%s db=%s",
        config.enabled,
        config.vault_path,
        config.db_path,
    )

    # Open the store eagerly so search latency does not include init.
    store = _try_open_store(config)
    embedder = _try_build_embedder(store, config)

    # Run the indexer if a real vault is reachable. Failures degrade to
    # "no vault" but still keep the MCP tool registered.
    if store is not None and config.enabled and config.vault_path.is_dir():
        try:
            from .indexer import VaultIndexer

            indexer = VaultIndexer(
                store,
                vault_root=config.vault_path,
                denylist=config.denylist,
                allowlist=config.allowlist,
                embedder=embedder,
                embed_disabled=config.embed_disabled,
            )
            stats = indexer.initial_scan()
            logger.info(
                "[memory] initial scan complete: %d chunks across %d files",
                stats.chunks_written,
                stats.files_indexed,
            )
        except Exception:  # noqa: BLE001 - never crash on indexer failure
            logger.exception("[memory] initial scan failed")

    # If we have data from a previous run that lacks vectors, hint the
    # operator. Phase 2 deliberately does NOT auto-backfill on startup
    # because embedding 36k chunks can take many minutes; we want this
    # to be an explicit, observable operation.
    if store is not None and store.vec_loaded:
        missing = store.count_chunks_missing_vectors()
        if missing > 0:
            logger.warning(
                "[memory] %d chunks lack embeddings; run "
                "`python -m vllm_mlx.memory.backfill` to enable dense "
                "search over them (BM25 still works in the meantime)",
                missing,
            )

    # Phase 3: kick off the background chat embed loop. The loop polls
    # ``chat_messages`` for rows that have no matching ``vec_chunks``
    # entry, embeds them in batches, and writes them back. Cost is one
    # background asyncio task that wakes every ``chat_embed_interval``
    # seconds (default 10s — REQ-E4 budget). The loop is robust to
    # embedder failure (returns 0 silently) so it costs ~nothing when
    # MEMORY_EMBED_DISABLED=1 or the model is unavailable.
    if store is not None and config.enabled and config.chat_log_enabled:
        from .chatlog import chat_embed_loop  # noqa: PLC0415

        asyncio.create_task(
            chat_embed_loop(
                store,
                embedder,
                interval_seconds=config.chat_embed_interval,
                batch_size=max(1, int(config.embed_batch)),
            )
        )
        logger.info(
            "[memory] chat embed loop scheduled (interval=%.1fs)",
            float(config.chat_embed_interval),
        )

    # Phase 4: vault watcher. Runs alongside the chat embed loop and
    # the retention sweeper. Gated on MEMORY_INDEXER=watchdog (the
    # default). Failures inside the watcher loop are isolated per
    # REQ-N4, so a flaky FSEvents subscription cannot poison the chat
    # path.
    if (
        store is not None
        and config.enabled
        and config.vault_path.is_dir()
        and config.indexer == "watchdog"
    ):
        try:
            from .watcher import VaultWatcher  # noqa: PLC0415

            watcher = VaultWatcher(
                store,
                vault_root=config.vault_path,
                denylist=config.denylist,
                allowlist=config.allowlist,
                embedder=embedder,
                embed_disabled=config.embed_disabled,
                debounce_ms=int(config.watcher_debounce_ms),
            )
            asyncio.create_task(watcher.run())
            logger.info(
                "[memory] vault watcher scheduled (debounce=%dms)",
                int(config.watcher_debounce_ms),
            )
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] failed to schedule vault watcher; "
                "incremental updates disabled"
            )

    # Phase 4: retention sweeper. Gated on a positive
    # MEMORY_CHAT_RETENTION_DAYS — when zero or negative the sweep
    # would no-op forever, so we just skip scheduling. Mode defaults
    # to ``delete`` per SPEC §9.
    if (
        store is not None
        and config.enabled
        and config.chat_log_enabled
        and config.chat_retention_days > 0
    ):
        try:
            from .sweeper import RetentionSweeper  # noqa: PLC0415

            sweeper = RetentionSweeper(
                store,
                retention_days=int(config.chat_retention_days),
                mode=config.chat_retention_mode,
                sweep_interval_seconds=float(
                    config.retention_sweep_interval_seconds
                ),
            )
            asyncio.create_task(sweeper.run())
            logger.info(
                "[memory] retention sweeper scheduled "
                "(days=%d mode=%s interval=%ds)",
                int(config.chat_retention_days),
                config.chat_retention_mode,
                int(config.retention_sweep_interval_seconds),
            )
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] failed to schedule retention sweeper; "
                "old chat rows will not be evicted automatically"
            )

    server = Server("memory")

    @server.list_tools()
    async def list_tools() -> list[Any]:  # noqa: D401 - MCP decorator
        return [
            types.Tool(
                name=TOOL_NAME,
                description=TOOL_DESCRIPTION,
                inputSchema=TOOL_INPUT_SCHEMA,
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
        if name != TOOL_NAME:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"status": "error", "message": f"unknown tool: {name}"}
                    ),
                )
            ]
        envelope = search_memory(
            store=store,
            query=str(arguments.get("query") or ""),
            top_k=int(arguments.get("top_k") or config.top_k_default),
            source_filter=str(arguments.get("source_filter") or "both"),
            config=config,
            embedder=embedder,
        )
        return [
            types.TextContent(
                type="text",
                text=json.dumps(envelope, ensure_ascii=False),
            )
        ]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def _try_open_store(config: MemoryRuntimeConfig) -> MemoryStore | None:
    """Best-effort store open; returns None on any I/O failure.

    REQ-S1 / REQ-N4: every code path that could raise a filesystem
    error (volume not mounted, permission denied) is degraded to
    "no store" so the MCP server still serves a tool list and returns
    ``{status: ok, results: []}`` per P1-AC6.
    """
    if not config.enabled:
        return None
    try:
        store = MemoryStore(config.db_path, embed_dim=config.embed_dim)
        store.open()
        return store
    except Exception:  # noqa: BLE001 - REQ-N4
        logger.exception("[memory] failed to open store at %s", config.db_path)
        return None


def _try_build_embedder(
    store: MemoryStore | None, config: MemoryRuntimeConfig
) -> Any | None:
    """Construct an :class:`Embedder` if the dense path makes sense.

    Returns None when:

    * memory is disabled, or
    * the store failed to open, or
    * sqlite-vec did not load on the connection, or
    * MEMORY_EMBED_DISABLED=1, or
    * the DB has a recorded model that does not match the configured
      one (REQ-U4 model-mismatch protection).

    On REQ-U4 mismatch we log an actionable error and return None so
    BM25 keeps working. The operator must run
    ``python -m vllm_mlx.memory.backfill --force-rebuild`` to swap
    models cleanly.

    Note: this does NOT call ``Embedder.load()`` — model weights are
    only loaded on the first encode. Failures during load are
    swallowed by the embedder itself (REQ-O3).
    """
    if store is None or not config.enabled:
        return None
    if config.embed_disabled:
        logger.info(
            "[memory] MEMORY_EMBED_DISABLED=1; dense search skipped (BM25 only)"
        )
        return None
    if not store.vec_loaded:
        logger.warning(
            "[memory] sqlite-vec unavailable on this connection; dense "
            "search disabled. Install with: pip install sqlite-vec"
        )
        return None

    ok, reason = store.assert_embed_compat(
        model=config.embed_model, dim=config.embed_dim
    )
    if not ok:
        logger.error("[memory] embedder disabled: %s", reason)
        return None

    try:
        from .embedder import Embedder

        embedder = Embedder.from_config(config)
        # Record the (model, dim) we're about to use *before* the first
        # encode call so a concurrent backfill sees the same identity.
        store.record_embedding_identity(
            model=config.embed_model, dim=config.embed_dim
        )
        return embedder
    except Exception:  # noqa: BLE001 - REQ-N4
        logger.exception("[memory] failed to construct embedder; BM25 only")
        return None


def main() -> None:
    """Entry point used by ``python -m vllm_mlx.memory.server``."""
    # Send our logs to stderr so they do not corrupt the stdio MCP
    # framing on stdout.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run_mcp_stdio())


if __name__ == "__main__":
    main()
