# SPDX-License-Identifier: Apache-2.0
"""Tests for the SPEC-MEMORY-02 Phase 1 hand-rolled asyncio scheduler.

@TEST:MEMORY-02/scheduler

The scheduler is intentionally tiny and fully unit-testable: it accepts
explicit ``clock`` and ``sleep`` callables so we can drive it without
``freezegun`` or a real wall clock. Every test here uses a fake clock
that returns pre-baked datetimes and a fake sleep that records its
argument and yields control back immediately.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from vllm_mlx.memory.scheduler import (
    compute_next_tick,
    run_scheduler_loop,
)


# ---------------------------------------------------------------------------
# compute_next_tick
# ---------------------------------------------------------------------------
class TestComputeNextTick:
    def test_before_target_hour_returns_today(self):
        now = datetime(2026, 4, 30, 1, 30, 0, tzinfo=timezone.utc)
        nxt = compute_next_tick(now, hour=3)
        assert nxt == datetime(2026, 4, 30, 3, 0, 0, tzinfo=timezone.utc)

    def test_after_target_hour_returns_tomorrow(self):
        now = datetime(2026, 4, 30, 5, 15, 7, tzinfo=timezone.utc)
        nxt = compute_next_tick(now, hour=3)
        assert nxt == datetime(2026, 5, 1, 3, 0, 0, tzinfo=timezone.utc)

    def test_exactly_at_target_hour_returns_tomorrow(self):
        # SPEC §8: a tick that lands exactly on the boundary should not
        # fire immediately again — that would create back-to-back ticks
        # if a previous tick took zero seconds.
        now = datetime(2026, 4, 30, 3, 0, 0, tzinfo=timezone.utc)
        nxt = compute_next_tick(now, hour=3)
        assert nxt == datetime(2026, 5, 1, 3, 0, 0, tzinfo=timezone.utc)

    def test_crosses_midnight_correctly(self):
        # 23:30 with a 03:00 target → next tick is the next day's 03:00.
        now = datetime(2026, 4, 30, 23, 30, 0, tzinfo=timezone.utc)
        nxt = compute_next_tick(now, hour=3)
        assert nxt == datetime(2026, 5, 1, 3, 0, 0, tzinfo=timezone.utc)

    def test_hour_wraps_modulo_24(self):
        # An out-of-range hour (e.g. typo of 25) is clamped via modulo.
        now = datetime(2026, 4, 30, 0, 30, 0, tzinfo=timezone.utc)
        nxt = compute_next_tick(now, hour=25)  # 25 % 24 == 1
        assert nxt == datetime(2026, 4, 30, 1, 0, 0, tzinfo=timezone.utc)

    def test_naive_datetime_supported(self):
        # Production runs in local time without an explicit tz; ensure
        # the computation also works with naive datetimes.
        now = datetime(2026, 4, 30, 14, 0, 0)
        nxt = compute_next_tick(now, hour=3)
        assert nxt == datetime(2026, 5, 1, 3, 0, 0)
        assert nxt.tzinfo is None


# ---------------------------------------------------------------------------
# run_scheduler_loop
# ---------------------------------------------------------------------------
class _FakeClock:
    """Returns a sequence of pre-baked datetimes; cycles last value."""

    def __init__(self, sequence):
        self._seq = list(sequence)
        self._idx = 0

    def __call__(self) -> datetime:
        if self._idx >= len(self._seq):
            return self._seq[-1]
        v = self._seq[self._idx]
        self._idx += 1
        return v


class _RecordingSleep:
    """Records every sleep duration; returns control immediately."""

    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))


@pytest.mark.asyncio
async def test_loop_runs_callback_after_sleep():
    """Scheduler computes wait, sleeps, then invokes the callback."""
    # 02:00 → next tick at 03:00 → 1 hour = 3600 seconds.
    clock = _FakeClock(
        [
            datetime(2026, 4, 30, 2, 0, 0, tzinfo=timezone.utc),
        ]
    )
    sleep = _RecordingSleep()
    fired = 0

    async def cb():
        nonlocal fired
        fired += 1

    await run_scheduler_loop(
        cb,
        hour=3,
        clock=clock,
        sleep=sleep,
        max_iterations=1,
    )

    assert fired == 1
    assert sleep.calls == [3600.0]


@pytest.mark.asyncio
async def test_loop_survives_callback_exception():
    """One bad tick must not kill the loop (REQ-N4)."""
    clock = _FakeClock(
        [
            datetime(2026, 4, 30, 2, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 30, 3, 0, 0, tzinfo=timezone.utc),
        ]
    )
    sleep = _RecordingSleep()
    fired = []

    async def cb():
        fired.append(len(fired))
        if len(fired) == 1:
            raise RuntimeError("simulated bad tick")

    # Two iterations: first raises, second succeeds. The loop should
    # complete both.
    await run_scheduler_loop(
        cb,
        hour=3,
        clock=clock,
        sleep=sleep,
        max_iterations=2,
    )

    assert len(fired) == 2


@pytest.mark.asyncio
async def test_disabled_check_skips_callback_but_keeps_looping():
    """An ``enabled_check`` returning False blocks the callback."""
    clock = _FakeClock(
        [
            datetime(2026, 4, 30, 2, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 30, 3, 0, 0, tzinfo=timezone.utc),
        ]
    )
    sleep = _RecordingSleep()
    fired = 0

    async def cb():
        nonlocal fired
        fired += 1

    enabled_states = iter([False, True])

    def enabled() -> bool:
        return next(enabled_states)

    await run_scheduler_loop(
        cb,
        hour=3,
        clock=clock,
        sleep=sleep,
        enabled_check=enabled,
        max_iterations=2,
    )

    # First iteration skipped due to disabled flag; second fires.
    assert fired == 1


@pytest.mark.asyncio
async def test_max_iterations_zero_returns_immediately():
    """A 0-iteration call is a no-op (used by tests)."""
    clock = _FakeClock([datetime(2026, 4, 30, 2, 0, 0, tzinfo=timezone.utc)])
    sleep = _RecordingSleep()

    async def cb():
        pytest.fail("callback should not fire when max_iterations=0")

    await run_scheduler_loop(
        cb,
        hour=3,
        clock=clock,
        sleep=sleep,
        max_iterations=0,
    )
    assert sleep.calls == []


@pytest.mark.asyncio
async def test_clock_failure_is_caught_and_loop_continues():
    """A clock that raises should not kill the supervising task."""
    sleep = _RecordingSleep()
    states = iter(
        [
            RuntimeError("clock failure"),
            datetime(2026, 4, 30, 2, 0, 0, tzinfo=timezone.utc),
        ]
    )

    def clock() -> datetime:
        v = next(states)
        if isinstance(v, Exception):
            raise v
        return v

    fired = 0

    async def cb():
        nonlocal fired
        fired += 1

    await run_scheduler_loop(
        cb,
        hour=3,
        clock=clock,
        sleep=sleep,
        max_iterations=2,
    )
    # Iteration 1: clock raises → 60s recovery sleep, no callback.
    # Iteration 2: clock returns valid time → normal sleep + callback.
    assert fired == 1
    assert 60.0 in sleep.calls
