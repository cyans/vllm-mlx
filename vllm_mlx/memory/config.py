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


# ---------------------------------------------------------------------------
# Defaults from SPEC-MEMORY-01 §9
# ---------------------------------------------------------------------------
MEMORY_DEFAULT_VAULT_PATH = "/Volumes/data/Obsidian/obsi"
MEMORY_DEFAULT_DB_PATH = "/Volumes/data/vllm-mlx-memory/memory.db"
MEMORY_DEFAULT_DENYLIST = (".obsidian/**", "**/.trash/**", "**/Templates/**")
MEMORY_DEFAULT_TOP_K = 5
MEMORY_DEFAULT_TOP_K_MAX = 20

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

    return MemoryRuntimeConfig(
        enabled=enabled,
        vault_path=vault_path,
        db_path=db_path,
        denylist=deny,
        allowlist=allow,
        top_k_default=top_k_default,
        top_k_max=top_k_max,
        raw_env={k: v for k, v in env.items() if k.startswith("MEMORY_")},
    )


def resolve_memory_config_from_os() -> MemoryRuntimeConfig:
    """Convenience helper that reads :data:`os.environ`.

    Use the explicit-mapping :func:`resolve_memory_config` from tests so
    they never need to monkeypatch the process env.
    """
    return resolve_memory_config(os.environ)
