# SPDX-License-Identifier: Apache-2.0
"""Environment-variable resolution for the memory subsystem.

@CODE:MEMORY-01/config

Mirrors the resolver style used by
``vllm_mlx.config.models.resolve_legacy_think_tags``: every function takes
an explicit ``env`` mapping (default ``None`` = never read process env)
so unit tests can drive it without monkeypatching :data:`os.environ`.

Phase 1 only reads a small subset of the env vars committed in
SPEC-MEMORY-01 §9 — the ones that gate behavior (master switch, vault
path, db path, denylist, allowlist, top_k bounds). The remaining vars
(chat retention, embed model, redact patterns) are recognized but
ignored until Phase 2/3 lands.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env var names (single source of truth — quoted by tests for stability)
# ---------------------------------------------------------------------------
ENV_MEMORY_ENABLED = "MEMORY_ENABLED"
ENV_MEMORY_VAULT_PATH = "MEMORY_VAULT_PATH"
ENV_MEMORY_DB_PATH = "MEMORY_DB_PATH"
ENV_MEMORY_VAULT_DENYLIST = "MEMORY_VAULT_DENYLIST"
ENV_MEMORY_VAULT_ALLOWLIST = "MEMORY_VAULT_ALLOWLIST"
ENV_MEMORY_TOP_K_DEFAULT = "MEMORY_TOP_K_DEFAULT"
ENV_MEMORY_TOP_K_MAX = "MEMORY_TOP_K_MAX"
# Phase 2 — embeddings + hybrid search (SPEC-MEMORY-01 §9)
ENV_MEMORY_EMBED_MODEL = "MEMORY_EMBED_MODEL"
ENV_MEMORY_EMBED_BATCH = "MEMORY_EMBED_BATCH"
ENV_MEMORY_EMBED_DIM = "MEMORY_EMBED_DIM"
ENV_MEMORY_EMBED_DISABLED = "MEMORY_EMBED_DISABLED"
ENV_MEMORY_HYBRID_RRF_K = "MEMORY_HYBRID_RRF_K"
# Phase 3 — chat persistence + indexing (SPEC-MEMORY-01 §9 / REQ-E3..E4)
ENV_MEMORY_CHAT_LOG_ENABLED = "MEMORY_CHAT_LOG_ENABLED"
ENV_MEMORY_REDACT_PATTERNS = "MEMORY_REDACT_PATTERNS"
ENV_MEMORY_CHAT_RETENTION_DAYS = "MEMORY_CHAT_RETENTION_DAYS"
ENV_MEMORY_CHAT_EMBED_INTERVAL = "MEMORY_CHAT_EMBED_INTERVAL"


# ---------------------------------------------------------------------------
# Defaults from SPEC-MEMORY-01 §9
# ---------------------------------------------------------------------------
MEMORY_DEFAULT_VAULT_PATH = "/Volumes/data/Obsidian/obsi"
MEMORY_DEFAULT_DB_PATH = "/Volumes/data/vllm-mlx-memory/memory.db"
MEMORY_DEFAULT_DENYLIST = (".obsidian/**", "**/.trash/**", "**/Templates/**")
MEMORY_DEFAULT_TOP_K = 5
MEMORY_DEFAULT_TOP_K_MAX = 20
# Phase 2 — bge-m3 multilingual XLM-RoBERTa, 1024-dim dense.
# Default uses the upstream HF id to match SPEC §9; the server logs an
# explicit hint to swap to ``mlx-community/bge-m3-mlx-fp16`` for the
# pre-converted MLX weights when load fails (REQ-O3 fallback).
MEMORY_DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
MEMORY_DEFAULT_EMBED_BATCH = 16
MEMORY_DEFAULT_EMBED_DIM = 1024
# Reciprocal-rank-fusion constant (Cormack et al. 2009). 60 is the
# canonical value from the original RRF paper; tuned higher to soften
# the contribution of low-rank items.
MEMORY_DEFAULT_HYBRID_RRF_K = 60
# Phase 3 — defaults from SPEC-MEMORY-01 §9.
# Chat persistence is OFF by default so REQ-S1 invariance holds even
# when MEMORY_ENABLED=1 (vault-only mode).
MEMORY_DEFAULT_CHAT_RETENTION_DAYS = 365
# REQ-E4 mandates "queryable within 10 seconds" — the embed loop polls
# every 10s by default; operators can lower this for tests.
MEMORY_DEFAULT_CHAT_EMBED_INTERVAL = 10.0

_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class MemoryRuntimeConfig:
    """Resolved configuration for the memory subsystem.

    All fields are populated by :func:`resolve_memory_config`. ``enabled``
    is the master gate — when False the rest of the values are still
    populated (useful for diagnostic logging) but the subsystem must
    behave as if absent (REQ-S1).
    """

    enabled: bool
    vault_path: Path
    db_path: Path
    denylist: tuple[str, ...] = MEMORY_DEFAULT_DENYLIST
    allowlist: tuple[str, ...] = ()
    top_k_default: int = MEMORY_DEFAULT_TOP_K
    top_k_max: int = MEMORY_DEFAULT_TOP_K_MAX
    # Phase 2 — embeddings + hybrid search.
    embed_model: str = MEMORY_DEFAULT_EMBED_MODEL
    embed_batch: int = MEMORY_DEFAULT_EMBED_BATCH
    embed_dim: int = MEMORY_DEFAULT_EMBED_DIM
    embed_disabled: bool = False
    hybrid_rrf_k: int = MEMORY_DEFAULT_HYBRID_RRF_K
    # Phase 3 — chat persistence + indexing knobs.
    chat_log_enabled: bool = False
    redact_patterns: tuple[str, ...] = ()
    chat_retention_days: int = MEMORY_DEFAULT_CHAT_RETENTION_DAYS
    chat_embed_interval: float = MEMORY_DEFAULT_CHAT_EMBED_INTERVAL
    # Raw env snapshot retained for diagnostic logging only — never used
    # to drive logic.
    raw_env: Mapping[str, str] = field(default_factory=dict)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def _split_globs(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    parts = [p.strip() for p in value.split(",")]
    return tuple(p for p in parts if p)


def _int_or_default(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "[memory] invalid integer for env var, using default=%s (got %r)",
            default,
            value,
        )
        return default


def _float_or_default(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "[memory] invalid float for env var, using default=%s (got %r)",
            default,
            value,
        )
        return default


def _split_patterns(value: str | None) -> tuple[str, ...]:
    """Parse a semicolon-separated regex list into a tuple.

    SPEC §9 documents the env-var format as semicolon-separated so the
    pattern strings can themselves contain commas (which is common in
    regex character classes). Empty entries are dropped.
    """
    if not value:
        return ()
    return tuple(p.strip() for p in value.split(";") if p.strip())


def resolve_memory_enabled(env: Mapping[str, str] | None) -> bool:
    """Return True iff ``MEMORY_ENABLED`` is set to a truthy value.

    REQ-S1: default is False so the rest of the server is bit-for-bit
    identical to a build without this SPEC.
    """
    if env is None:
        return False
    return _truthy(env.get(ENV_MEMORY_ENABLED))


def resolve_memory_config(
    env: Mapping[str, str] | None = None,
) -> MemoryRuntimeConfig:
    """Resolve the full memory runtime config from an env mapping.

    Returns a :class:`MemoryRuntimeConfig` even when memory is disabled
    so callers can log the resolved paths for operator visibility.
    Passing ``env=None`` (the default) returns a fully default-disabled
    config without consulting :data:`os.environ`.
    """
    if env is None:
        env = {}

    enabled = _truthy(env.get(ENV_MEMORY_ENABLED))
    vault_path = Path(
        env.get(ENV_MEMORY_VAULT_PATH) or MEMORY_DEFAULT_VAULT_PATH
    ).expanduser()
    db_path = Path(
        env.get(ENV_MEMORY_DB_PATH) or MEMORY_DEFAULT_DB_PATH
    ).expanduser()

    deny = _split_globs(env.get(ENV_MEMORY_VAULT_DENYLIST))
    if not deny:
        deny = MEMORY_DEFAULT_DENYLIST
    allow = _split_globs(env.get(ENV_MEMORY_VAULT_ALLOWLIST))

    top_k_default = _int_or_default(
        env.get(ENV_MEMORY_TOP_K_DEFAULT), MEMORY_DEFAULT_TOP_K
    )
    top_k_max = _int_or_default(
        env.get(ENV_MEMORY_TOP_K_MAX), MEMORY_DEFAULT_TOP_K_MAX
    )
    # Clamp to spec bounds (REQ-E5: max 20, default 5).
    top_k_max = max(1, min(top_k_max, MEMORY_DEFAULT_TOP_K_MAX))
    top_k_default = max(1, min(top_k_default, top_k_max))

    # Phase 2 knobs. Each keeps a sane default when the env var is missing
    # or malformed; embed_disabled defaults to False so the dense path is
    # attempted whenever memory itself is enabled.
    embed_model = (env.get(ENV_MEMORY_EMBED_MODEL) or "").strip() \
        or MEMORY_DEFAULT_EMBED_MODEL
    embed_batch = max(
        1, _int_or_default(env.get(ENV_MEMORY_EMBED_BATCH), MEMORY_DEFAULT_EMBED_BATCH)
    )
    embed_dim = max(
        1, _int_or_default(env.get(ENV_MEMORY_EMBED_DIM), MEMORY_DEFAULT_EMBED_DIM)
    )
    embed_disabled = _truthy(env.get(ENV_MEMORY_EMBED_DISABLED))
    hybrid_rrf_k = max(
        1,
        _int_or_default(
            env.get(ENV_MEMORY_HYBRID_RRF_K), MEMORY_DEFAULT_HYBRID_RRF_K
        ),
    )

    # Phase 3 — chat persistence + indexing.
    chat_log_enabled = _truthy(env.get(ENV_MEMORY_CHAT_LOG_ENABLED))
    redact_patterns = _split_patterns(env.get(ENV_MEMORY_REDACT_PATTERNS))
    chat_retention_days = max(
        1,
        _int_or_default(
            env.get(ENV_MEMORY_CHAT_RETENTION_DAYS),
            MEMORY_DEFAULT_CHAT_RETENTION_DAYS,
        ),
    )
    chat_embed_interval = _float_or_default(
        env.get(ENV_MEMORY_CHAT_EMBED_INTERVAL),
        MEMORY_DEFAULT_CHAT_EMBED_INTERVAL,
    )
    # Clamp to >=1s so a typo cannot pin the CPU at 100%.
    chat_embed_interval = max(1.0, float(chat_embed_interval))

    return MemoryRuntimeConfig(
        enabled=enabled,
        vault_path=vault_path,
        db_path=db_path,
        denylist=deny,
        allowlist=allow,
        top_k_default=top_k_default,
        top_k_max=top_k_max,
        embed_model=embed_model,
        embed_batch=embed_batch,
        embed_dim=embed_dim,
        embed_disabled=embed_disabled,
        hybrid_rrf_k=hybrid_rrf_k,
        chat_log_enabled=chat_log_enabled,
        redact_patterns=redact_patterns,
        chat_retention_days=chat_retention_days,
        chat_embed_interval=chat_embed_interval,
        raw_env={k: v for k, v in env.items() if k.startswith("MEMORY_")},
    )


def resolve_memory_config_from_os() -> MemoryRuntimeConfig:
    """Convenience helper that reads :data:`os.environ`.

    Use the explicit-mapping :func:`resolve_memory_config` from tests so
    they never need to monkeypatch the process env.
    """
    return resolve_memory_config(os.environ)
