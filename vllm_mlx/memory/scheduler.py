# SPDX-License-Identifier: Apache-2.0
"""Hand-rolled asyncio cron for the memory consolidator.

@CODE:MEMORY-02/scheduler

Phase 1 ships the daily chat-consolidation tick. The vault thematic
digest (Phase 2) and tier-aware ranking (Phase 3) remain out of scope
for this commit — the scheduler only fires the daily callback for now.

Design principles (see SPEC-MEMORY-02 §8):

* Hand-rolled — zero new pip deps, < 100 LOC.
* Fully unit-testable: pass an explicit ``clock`` and ``sleep`` so
  tests run synchronously without ``freezegun``.
* REQ-N4 isolation: every callback invocation is wrapped so a single
  bad tick cannot kill the loop.
* TZ-aware: ``compute_next_tick`` uses :class:`datetime.datetime`
  naively against the wall clock so DST transitions are handled by the
  OS, matching MEMORY-01's existing local-time conventions.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# Type aliases for clarity in test signatures.
ClockFn = Callable[[], datetime]
SleepFn = Callable[[float], Awaitable[None]]
TickCallback = Callable[[], Awaitable[None]]


def compute_next_tick(now: datetime, hour: int) -> datetime:
    """Return the next wall-clock instant at ``hour:00:00``.

    Semantics:

    * If ``now`` is *before* today's ``hour`` boundary, returns today.
    * If ``now`` is *at or after* today's ``hour`` boundary, returns
      tomorrow.

    The returned datetime carries the same ``tzinfo`` as ``now`` (None
    when ``now`` is naive). This lets callers stay in their preferred
    timezone — local for production (matches SPEC §10), explicit UTC
    for tests.

    DST: a naive ``now`` in a DST-aware OS will produce the right
    physical wall-clock instant, because the next-day arithmetic is
    purely calendar (``timedelta(days=1)``) and the seconds-until
    computation done by the caller against the OS clock absorbs any
    DST shift on the actual sleep.
    """
    hour = int(hour) % 24
    target_today = now.replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    if now < target_today:
        return target_today
    return target_today + timedelta(days=1)


async def _default_sleep(seconds: float) -> None:
    """Default :func:`asyncio.sleep` wrapper (override in tests)."""
    await asyncio.sleep(seconds)


def _default_clock() -> datetime:
    """Default wall-clock reader (override in tests)."""
    return datetime.now()


async def run_scheduler_loop(
    callback: TickCallback,
    *,
    hour: int,
    enabled_check: Callable[[], bool] | None = None,
    clock: ClockFn = _default_clock,
    sleep: SleepFn = _default_sleep,
    max_iterations: int | None = None,
) -> None:
    """Long-running daily-at-hour scheduler.

    Args:
        callback: Coroutine fired once per tick. Must be safe to call
            many times. Failures are caught and logged (REQ-N4); the
            loop survives.
        hour: Hour of day (0-23) at which to fire. Wrapped into range.
        enabled_check: Called before *each* tick; when it returns False
            the callback is skipped but the loop keeps sleeping. This
            lets an operator flip ``MEMORY_CONSOLIDATOR_ENABLED=0`` at
            runtime without restarting the server.
        clock: Returns the current wall-clock time. Default is
            :func:`datetime.now`. Tests pass a stub.
        sleep: Async sleeper used between ticks. Default is
            :func:`asyncio.sleep`. Tests pass a stub.
        max_iterations: Cap on the number of tick iterations. ``None``
            (the default) runs forever; tests pass a positive int.

    The loop never raises out — only ``asyncio.CancelledError`` from a
    task shutdown propagates so the lifespan can clean up.
    """
    iteration = 0
    while True:
        if max_iterations is not None and iteration >= max_iterations:
            return
        iteration += 1

        try:
            now = clock()
            next_tick = compute_next_tick(now, hour)
            wait_seconds = max(0.0, (next_tick - now).total_seconds())
        except Exception:  # noqa: BLE001 - REQ-N4
            # ``clock()`` failure is exotic but we still must not crash
            # the supervising task; sleep a minute and try again.
            logger.exception(
                "[memory-02] scheduler clock read failed; "
                "sleeping 60s before retry"
            )
            await sleep(60.0)
            continue

        try:
            await sleep(wait_seconds)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            logger.info("[memory-02] scheduler cancelled during sleep")
            raise

        # Re-read enabled flag *after* sleeping so an operator who
        # disables the consolidator overnight doesn't see a tick fire
        # at startup.
        try:
            if enabled_check is not None and not enabled_check():
                logger.debug(
                    "[memory-02] consolidator disabled; skipping tick"
                )
                continue
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory-02] enabled_check raised; skipping tick"
            )
            continue

        try:
            await callback()
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory-02] consolidator tick failed; "
                "loop continues to next tick"
            )


__all__ = [
    "ClockFn",
    "SleepFn",
    "TickCallback",
    "compute_next_tick",
    "run_scheduler_loop",
]
