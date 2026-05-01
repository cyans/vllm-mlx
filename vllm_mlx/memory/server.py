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
from typing import Any

from .config import MemoryRuntimeConfig, resolve_memory_config
from .store import MemoryStore, SearchHit

logger = logging.getLogger("vllm_mlx.memory.server")


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
) -> dict[str, Any]:
    """Return the JSON envelope for one ``memory_search`` invocation.

    This function never raises — every failure path produces a
    ``{status, degraded, results}`` envelope so the model's tool-use
    loop always sees structured data.

    Phase 1: only BM25 over the vault is wired up. Phase 2 will graft
    dense + RRF on top of this same envelope shape.
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

    try:
        hits: list[SearchHit] = store.search_bm25(
            query, top_k=top_k, source_filter=source_filter
        )
    except Exception:  # noqa: BLE001 - REQ-N4: never crash the chat path
        logger.exception("[memory] search failed for query=%r", query)
        return {
            "status": "error",
            "degraded": True,
            "results": [],
            "message": "internal search error (see server logs)",
        }

    return {
        "status": "ok",
        "degraded": False,
        "results": [
            {
                "source_type": h.source_type,
                "source_path": h.source_path,
                "timestamp": h.timestamp,
                "score": round(h.score, 4),
                "excerpt": h.excerpt,
            }
            for h in hits
        ],
    }


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
            )
            stats = indexer.initial_scan()
            logger.info(
                "[memory] initial scan complete: %d chunks across %d files",
                stats.chunks_written,
                stats.files_indexed,
            )
        except Exception:  # noqa: BLE001 - never crash on indexer failure
            logger.exception("[memory] initial scan failed")

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
        store = MemoryStore(config.db_path)
        store.open()
        return store
    except Exception:  # noqa: BLE001 - REQ-N4
        logger.exception("[memory] failed to open store at %s", config.db_path)
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
