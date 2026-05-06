# SPDX-License-Identifier: Apache-2.0
"""Long-term semantic memory subsystem for vllm-mlx.

@CODE:MEMORY-01 — Phase 1 (BM25-only PoC).

This package implements a local, retrieval-only memory MCP server that
indexes the user's Obsidian vault and exposes a single ``memory_search``
tool. Phase 1 is intentionally narrow: BM25 over SQLite FTS5 only,
read-only over the vault. Embeddings, chat persistence, watchdog
incremental updates, and retention sweepers all land in later phases.

REQ-S1 / REQ-N4: every public entry point is a no-op when memory is
disabled, and every callable is wrapped so a memory failure cannot
propagate into ``/v1/chat/completions``.
"""

from __future__ import annotations

from .config import (
    MEMORY_DEFAULT_DB_PATH,
    MEMORY_DEFAULT_EMBED_DIM,
    MEMORY_DEFAULT_EMBED_MODEL,
    MEMORY_DEFAULT_VAULT_PATH,
    MemoryRuntimeConfig,
    resolve_memory_config,
)
from .embedder import Embedder, pack_float32, unpack_float32

__all__ = [
    "MEMORY_DEFAULT_DB_PATH",
    "MEMORY_DEFAULT_EMBED_DIM",
    "MEMORY_DEFAULT_EMBED_MODEL",
    "MEMORY_DEFAULT_VAULT_PATH",
    "Embedder",
    "MemoryRuntimeConfig",
    "pack_float32",
    "resolve_memory_config",
    "unpack_float32",
]
