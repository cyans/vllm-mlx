# SPDX-License-Identifier: Apache-2.0
"""Runtime memory-pressure guardrails for the vllm-mlx server.

Note on namespace: this module lives under ``vllm_mlx.memory`` for proximity
to the rest of the memory subsystem, but it is *runtime* memory budgeting
(free RAM headroom on the host) — distinct from the SPEC-MEMORY-01/02
conversational memory store, indexer, scheduler, etc. Nothing in here loads
an LLM, opens the SQLite memory store, or imports ``mlx_lm``.

What this module does
---------------------
1. Exposes :class:`MemoryBudget`, a tiny dependency-injected wrapper around
   ``psutil.virtual_memory()`` (or any callable returning free bytes).
2. Provides :meth:`MemoryBudget.check_headroom` for one-shot post-load
   sanity checks and a periodic async task
   :meth:`MemoryBudget.start_periodic_check` that warns when free RAM drops
   below the configured headroom. It does *not* panic or kill the server —
   operators decide.
3. Provides :func:`resolve_memory_headroom_gb`, a config helper following
   the same pattern as ``vllm_mlx.config.models.resolve_reasoning_parser``:
   explicit override > env (``MEMORY_HEADROOM_GB``) > module default.

Why this exists: on a 64 GB Apple Silicon Mac running a 35B-A3B 4-bit
model with continuous batching at 40% KV cache memory, a heavy GUI app in
the background can push the machine into swap and trigger watchdog-timeout
kernel panics. This subsystem catches that condition early at startup and
keeps an eye on it during the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

DEFAULT_HEADROOM_GB: float = 6.0
"""Default minimum free-RAM headroom in GiB.

Tuned for a 64 GB unified-memory Mac running a 35B-A3B 4-bit Qwen quant
with continuous batching at ``--cache-memory-percent 0.4``. Below this
threshold the OS starts to compress aggressively and swap, which on the
M-series watchdog is the most common precursor to a kernel panic.
"""

DEFAULT_INTERVAL_SECONDS: float = 30.0
"""Default interval between periodic headroom checks (seconds).

30 s is well under the 60-90 s window between memory pressure becoming
visible in ``vm_stat`` and the kernel watchdog tripping in our
observations, while keeping the per-check cost (a single ``psutil`` call)
negligible.
"""

ENV_HEADROOM_GB: str = "MEMORY_HEADROOM_GB"
"""Environment variable name for operator overrides of the headroom target."""

_BYTES_PER_GIB: int = 1024 * 1024 * 1024


# ---------------------------------------------------------------------------
# Free-bytes reader
# ---------------------------------------------------------------------------


def _default_read_free_bytes() -> int:
    """Return current free (available) RAM in bytes via psutil.

    ``psutil.virtual_memory().available`` is the right metric here — on
    macOS it accounts for inactive + free + speculative + purgeable pages,
    which matches what the kernel can actually hand out before it starts
    compressing or swapping. On Linux it maps to ``MemAvailable``.

    Imported lazily so unit tests that inject their own reader never touch
    psutil.
    """
    import psutil  # noqa: PLC0415 — lazy import for test isolation

    return int(psutil.virtual_memory().available)


def _default_read_memory_snapshot() -> dict[str, int]:
    """Return a coarse memory snapshot (best-effort, never raises).

    Used by ``/memory/budget`` to surface a richer view than just ``free``.
    Missing fields (e.g. ``wired`` on Linux) are simply omitted rather than
    fabricated. Failures collapse to an empty dict and the endpoint will
    still report the free/total numbers from ``read_free_bytes``.
    """
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        snap: dict[str, int] = {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "used_bytes": int(getattr(vm, "used", 0) or 0),
            "active_bytes": int(getattr(vm, "active", 0) or 0),
            "inactive_bytes": int(getattr(vm, "inactive", 0) or 0),
            "wired_bytes": int(getattr(vm, "wired", 0) or 0),
        }
        # macOS exposes a compressed-page count via psutil on newer versions;
        # treat its absence as 0 rather than failing the snapshot.
        compressed = getattr(vm, "compressed", None)
        if compressed is not None:
            snap["compressed_bytes"] = int(compressed)
        return snap
    except Exception:  # noqa: BLE001 — telemetry must never crash callers
        return {}


# ---------------------------------------------------------------------------
# Budget container
# ---------------------------------------------------------------------------


@dataclass
class MemoryBudget:
    """Lightweight memory-pressure guard for the FastAPI lifespan.

    Parameters
    ----------
    read_free_bytes:
        Callable returning free RAM in bytes. Defaults to a psutil-backed
        reader; tests inject a stub so they never hit the OS.
    read_snapshot:
        Optional callable returning a richer memory snapshot dict. Used by
        the ``/memory/budget`` admin endpoint. Defaults to a psutil-backed
        reader that returns ``{}`` on any failure.
    """

    read_free_bytes: Callable[[], int] = _default_read_free_bytes
    read_snapshot: Callable[[], dict[str, int]] = _default_read_memory_snapshot

    # ---------------------------------------------------------------------
    # Synchronous primitives — cheap (≤ 1 ms with default psutil reader).
    # ---------------------------------------------------------------------

    def free_bytes(self) -> int:
        """Return current free RAM in bytes via the injected reader."""
        return int(self.read_free_bytes())

    def check_headroom(self, target_gb: float) -> tuple[bool, dict[str, Any]]:
        """Compare free RAM against ``target_gb``.

        Returns a tuple ``(ok, details)`` where ``ok`` is ``True`` iff
        ``free_bytes >= target_gb * 1 GiB``. ``details`` is a JSON-friendly
        dict suitable for logging and for the ``/memory/budget`` endpoint.

        Negative ``margin_bytes`` means the system is already below the
        configured headroom — a structured warning is the appropriate
        response, not an exception.
        """
        target_bytes = int(max(0.0, float(target_gb)) * _BYTES_PER_GIB)
        free = self.free_bytes()
        margin = free - target_bytes
        ok = margin >= 0

        details: dict[str, Any] = {
            "headroom_target_gb": float(target_gb),
            "headroom_target_bytes": target_bytes,
            "free_bytes": free,
            "headroom_actual_bytes": margin,
            "headroom_actual_gb": round(margin / _BYTES_PER_GIB, 3),
            "headroom_ok": ok,
        }
        # Best-effort enrichment from the snapshot reader. Snapshot failure
        # never invalidates the headroom decision.
        snap = self.read_snapshot() or {}
        for key in (
            "total_bytes",
            "used_bytes",
            "active_bytes",
            "inactive_bytes",
            "wired_bytes",
            "compressed_bytes",
            "available_bytes",
        ):
            if key in snap:
                details[key] = int(snap[key])
        return ok, details

    # ---------------------------------------------------------------------
    # Async periodic warner.
    # ---------------------------------------------------------------------

    def log_headroom(self, target_gb: float, *, force_info: bool = False) -> bool:
        """Emit a structured log line for the current headroom state.

        Returns the ``ok`` flag from :meth:`check_headroom`. When ``ok`` is
        ``False`` we always log at WARNING; when ``ok`` is ``True`` we log
        at INFO only on the post-load one-shot (``force_info=True``) and
        DEBUG otherwise so periodic checks stay quiet on healthy systems.
        """
        ok, details = self.check_headroom(target_gb)
        if not ok:
            logger.warning(
                "[memory-budget] headroom BELOW target: "
                "free=%.2f GiB, target=%.2f GiB, margin=%.2f GiB",
                details["free_bytes"] / _BYTES_PER_GIB,
                details["headroom_target_gb"],
                details["headroom_actual_gb"],
                extra={"memory_budget": details},
            )
        elif force_info:
            logger.info(
                "[memory-budget] headroom OK: "
                "free=%.2f GiB, target=%.2f GiB, margin=%.2f GiB",
                details["free_bytes"] / _BYTES_PER_GIB,
                details["headroom_target_gb"],
                details["headroom_actual_gb"],
                extra={"memory_budget": details},
            )
        else:
            logger.debug(
                "[memory-budget] headroom OK: "
                "free=%.2f GiB, target=%.2f GiB, margin=%.2f GiB",
                details["free_bytes"] / _BYTES_PER_GIB,
                details["headroom_target_gb"],
                details["headroom_actual_gb"],
            )
        return ok

    async def _periodic_loop(
        self,
        *,
        target_gb: float,
        interval_s: float,
    ) -> None:
        """Internal loop body. Cancellation is the normal exit path."""
        # Sleep first so the lifespan startup logs are not racing the
        # periodic INFO/WARNING line; the post-load one-shot check is the
        # caller's responsibility.
        try:
            while True:
                await asyncio.sleep(max(1.0, float(interval_s)))
                try:
                    self.log_headroom(target_gb)
                except Exception:  # noqa: BLE001 — never kill the loop
                    logger.exception(
                        "[memory-budget] periodic check raised; continuing"
                    )
        except asyncio.CancelledError:
            # Normal shutdown path. Re-raising keeps asyncio's cancel
            # bookkeeping happy on Python 3.10+.
            raise

    def start_periodic_check(
        self,
        app: Any,
        *,
        interval_s: float = DEFAULT_INTERVAL_SECONDS,
        headroom_gb: float = DEFAULT_HEADROOM_GB,
    ) -> asyncio.Task[None]:
        """Spawn the periodic warner and stash it on ``app.state``.

        Returns the asyncio task so the lifespan teardown can cancel it.
        ``app`` is typed loosely (``Any``) because this module deliberately
        does not import FastAPI — it is unit-tested without it.
        """
        coro = self._periodic_loop(
            target_gb=headroom_gb,
            interval_s=interval_s,
        )
        task = asyncio.create_task(coro, name="memory-budget-periodic")
        # Best-effort attachment so the server can find it on shutdown
        # without us having to thread a global through. ``app`` may be a
        # stub object in unit tests; that is fine.
        with contextlib.suppress(Exception):
            app.state.memory_budget_task = task  # type: ignore[attr-defined]
        logger.info(
            "[memory-budget] periodic check started: "
            "interval=%.1fs, target=%.2f GiB",
            float(interval_s),
            float(headroom_gb),
        )
        return task


# ---------------------------------------------------------------------------
# Config helper — mirrors vllm_mlx.config.models.resolve_reasoning_parser.
# ---------------------------------------------------------------------------


def resolve_memory_headroom_gb(
    cli_value: float | None = None,
    env: Mapping[str, str] | None = None,
    default: float = DEFAULT_HEADROOM_GB,
) -> float:
    """Resolve the effective memory-headroom target in GiB.

    Precedence (highest first):

    1. ``cli_value`` — typically the parsed ``--memory-headroom-gb`` value.
       ``None`` is treated as "not provided".
    2. ``MEMORY_HEADROOM_GB`` environment variable, when ``env`` is given
       and the value parses as a non-negative float.
    3. ``default`` (defaults to :data:`DEFAULT_HEADROOM_GB`).

    Negative or unparseable values fall through to the next precedence
    step rather than raising — this is a soft guardrail, not a validation
    layer.
    """
    if cli_value is not None:
        try:
            v = float(cli_value)
        except (TypeError, ValueError):
            v = -1.0
        if v >= 0.0:
            return v
    if env is not None:
        raw = env.get(ENV_HEADROOM_GB, "")
        if raw:
            try:
                v = float(raw)
            except ValueError:
                v = -1.0
            if v >= 0.0:
                return v
    return float(default)


def resolve_memory_headroom_gb_from_os(
    cli_value: float | None = None,
    default: float = DEFAULT_HEADROOM_GB,
) -> float:
    """Convenience wrapper that consults :data:`os.environ`."""
    return resolve_memory_headroom_gb(cli_value=cli_value, env=os.environ, default=default)


__all__ = [
    "DEFAULT_HEADROOM_GB",
    "DEFAULT_INTERVAL_SECONDS",
    "ENV_HEADROOM_GB",
    "MemoryBudget",
    "resolve_memory_headroom_gb",
    "resolve_memory_headroom_gb_from_os",
]
