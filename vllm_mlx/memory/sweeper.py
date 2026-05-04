# SPDX-License-Identifier: Apache-2.0
"""Retention sweeper for chat-message rows.

@CODE:MEMORY-01/sweeper

Phase 4 implementation of REQ-N5: chat conversations older than
``MEMORY_CHAT_RETENTION_DAYS`` are either evicted (mode ``delete``,
default) or kept-with-redacted-payload (mode ``redact``). The sweeper
runs as an asyncio task spawned by the MCP child alongside the chat
embed loop and the vault watcher.

Design properties:

* Per-tick batches (default 1000 rows) so a single sweep never holds
  the WAL writer for longer than ``O(batch_size)`` SQL statements.
* Failure isolated (REQ-N4): every tick body is wrapped, failures log
  and the loop sleeps until the next interval.
* Idempotent: rows already past the cutoff but missing FTS / vec rows
  are still processed cleanly (the per-row helpers no-op on empty
  cascades).
* ``MEMORY_CHAT_RETENTION_DAYS=0`` (or ``<=0``) is treated as "no
  retention enforced" and the sweeper exits early.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from .store import MemoryStore

logger = logging.getLogger(__name__)


# Conservative batch size: 1000 rows × 3 statements (FTS delete +
# optional vec delete + chat update/delete) ≈ 3k SQL ops per
# transaction. Comfortable for SQLite WAL on Apple Silicon SSD.
DEFAULT_SWEEP_BATCH_SIZE = 1000

# Maximum number of batches a single sweep will attempt before yielding
# back to the event loop. Prevents pathological vaults (e.g. 10M chat
# rows) from monopolizing one tick. Subsequent ticks pick up where this
# one left off.
MAX_BATCHES_PER_SWEEP = 64

# Payload written to redacted rows. JSON shape mirrors what the rest of
# the memory subsystem expects (see :func:`vllm_mlx.memory.chatlog.serialize_chat_payload`)
# so a forensic dump of ``chat_messages`` still parses cleanly.
REDACTED_PAYLOAD = json.dumps(
    {
        "messages": [],
        "assistant": "[redacted: older than retention period]",
        "redacted": True,
    },
    ensure_ascii=False,
)


@dataclass(frozen=True)
class SweepStats:
    """One sweep tick summary, returned for tests + diagnostic logs."""

    rows_evaluated: int = 0
    rows_changed: int = 0
    batches: int = 0
    cutoff_iso: str = ""
    mode: str = "delete"


class RetentionSweeper:
    """Periodically purge or redact chat rows older than the cutoff.

    Construction does no I/O. Call :meth:`run` from an asyncio task to
    activate the periodic loop, OR call :meth:`sweep_once` directly
    from a test or one-off script. The store argument MUST already be
    open; the sweeper does not own the connection lifecycle.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        retention_days: int,
        mode: str,
        sweep_interval_seconds: float,
        batch_size: int = DEFAULT_SWEEP_BATCH_SIZE,
        summarize_before_delete: bool = False,
    ):
        self.store = store
        self.retention_days = int(retention_days)
        # Validate mode here so a bad value cannot land on disk via
        # ``meta.last_sweep_mode``.
        mode = (mode or "").lower()
        if mode not in ("delete", "redact"):
            logger.warning(
                "[memory] sweeper: unknown retention mode %r; "
                "falling back to 'delete'",
                mode,
            )
            mode = "delete"
        self.mode = mode
        self.sweep_interval = max(1.0, float(sweep_interval_seconds))
        self.batch_size = max(1, int(batch_size))
        # SPEC-MEMORY-02 Phase 1 integration: when True, the sweeper
        # treats the consolidator as the owner of un-summarized rows
        # and only touches rows whose ``session_id`` already has a
        # ``chat_summaries`` entry. When False (the original Phase 4
        # behavior) the sweeper deletes/redacts every row past the
        # retention boundary regardless of summarization state.
        self.summarize_before_delete = bool(summarize_before_delete)

    # --------------------------------------------------------- run loop

    async def run(self) -> None:
        """Long-lived asyncio task. Sleeps then sweeps, forever.

        The very first sweep runs immediately on entry so an operator
        who restarts the server after extending the retention window
        sees the change applied without waiting a full interval. After
        that we sleep before each subsequent tick.

        REQ-N4: every iteration is wrapped so a single failure (DB
        lock, schema drift, encoding error) cannot kill the loop.
        """
        if self.retention_days <= 0:
            logger.info(
                "[memory] retention sweeper disabled "
                "(MEMORY_CHAT_RETENTION_DAYS<=0); task exiting"
            )
            return

        logger.info(
            "[memory] retention sweeper starting: days=%d mode=%s interval=%.0fs",
            self.retention_days,
            self.mode,
            self.sweep_interval,
        )

        first = True
        while True:
            if not first:
                try:
                    await asyncio.sleep(self.sweep_interval)
                except asyncio.CancelledError:  # pragma: no cover - shutdown
                    logger.info("[memory] retention sweeper cancelled")
                    raise
            first = False

            try:
                stats = self.sweep_once()
                if stats.rows_changed > 0:
                    logger.info(
                        "[memory] sweep done: mode=%s changed=%d batches=%d "
                        "cutoff=%s",
                        stats.mode,
                        stats.rows_changed,
                        stats.batches,
                        stats.cutoff_iso,
                    )
            except Exception:  # noqa: BLE001 - REQ-N4
                logger.exception(
                    "[memory] retention sweep failed; will retry next interval"
                )

    # --------------------------------------------------------- one tick

    def sweep_once(self) -> SweepStats:
        """Run a single retention pass, batched.

        Returns a :class:`SweepStats` describing what changed. Safe to
        call from synchronous code (tests use it that way).
        """
        if self.retention_days <= 0:
            return SweepStats(mode=self.mode)

        cutoff_iso = _cutoff_iso_for(retention_days=self.retention_days)
        total_changed = 0
        total_evaluated = 0
        batches = 0

        for _ in range(MAX_BATCHES_PER_SWEEP):
            ids = self.store.fetch_chat_message_ids_older_than(
                cutoff_iso=cutoff_iso,
                limit=self.batch_size,
            )
            if not ids:
                break
            total_evaluated += len(ids)

            # SPEC-MEMORY-02 Phase 1 integration (REQ-N4): when
            # ``summarize_before_delete`` is on, narrow the candidate
            # set to message_ids whose session has been consolidated.
            # The consolidator owns the rest. This ensures the sweeper
            # never races the consolidator into deleting raw rows
            # without a surviving summary.
            if self.summarize_before_delete:
                ids = self._filter_to_consolidated(ids)

            if not ids:
                # All candidates in this batch belong to sessions still
                # waiting for the consolidator. Skip the write step but
                # check the next batch — the cutoff cursor advances.
                if self.batch_size > 0:
                    # Advance past these message_ids by re-querying
                    # excluding the unsummarized ones is too expensive;
                    # just bail out — next sweep tick re-checks once
                    # the consolidator has caught up.
                    break
                continue

            with self.store.transaction():
                if self.mode == "delete":
                    changed = self.store.delete_chat_messages(ids)
                else:
                    changed = self.store.redact_chat_messages(
                        ids, placeholder_payload=REDACTED_PAYLOAD
                    )
            total_changed += changed
            batches += 1

            # If we evicted FEWER rows than the batch size requested,
            # we have caught up with the cutoff.
            if len(ids) < self.batch_size:
                break

        # Always stamp the meta timestamp so operators can verify the
        # sweeper ran even when the cutoff did not match any rows.
        try:
            self.store.set_meta("last_sweep_at", _now_iso())
            self.store.set_meta("last_sweep_count", str(int(total_changed)))
            self.store.set_meta("last_sweep_mode", self.mode)
        except Exception:  # noqa: BLE001 - meta failures are diagnostic only
            logger.exception("[memory] failed to write last_sweep_at meta")

        return SweepStats(
            rows_evaluated=total_evaluated,
            rows_changed=total_changed,
            batches=batches,
            cutoff_iso=cutoff_iso,
            mode=self.mode,
        )

    # ---------------------------------------------- consolidator helper
    def _filter_to_consolidated(
        self, message_ids: list[str]
    ) -> list[str]:
        """Restrict ``message_ids`` to rows whose session is summarized.

        SPEC-MEMORY-02 Phase 1 / REQ-N4: when the operator opts into
        ``MEMORY_SUMMARIZE_BEFORE_DELETE=1`` (the default in the new
        config), the sweeper must skip rows whose ``session_id`` has
        no entry in ``chat_summaries``. Those rows belong to the
        consolidator and will be evicted in the same transaction that
        writes the summary.

        Returns the (possibly empty) subset of ``message_ids`` that
        are safe for the sweeper to delete or redact this tick.
        """
        if not message_ids:
            return []
        # Look up each message's session_id; rows with NULL session_id
        # are pre-Phase-3 data that the consolidator cannot summarize,
        # so we let the sweeper handle them as before.
        placeholders = ",".join("?" * len(message_ids))
        rows = self.store.conn.execute(
            f"SELECT message_id, session_id FROM chat_messages "
            f"WHERE message_id IN ({placeholders})",
            list(message_ids),
        ).fetchall()
        if not rows:
            return []
        sid_for: dict[str, str | None] = {
            str(r["message_id"]): (
                str(r["session_id"]) if r["session_id"] is not None else None
            )
            for r in rows
        }
        # Sessions present in chat_summaries are safe to evict.
        non_null_sids = {s for s in sid_for.values() if s}
        consolidated_sids: set[str] = (
            self.store.fetch_consolidated_session_ids(non_null_sids)
            if non_null_sids
            else set()
        )
        out: list[str] = []
        for mid in message_ids:
            sid = sid_for.get(str(mid))
            if sid is None:
                # NULL session_id — pre-Phase-3 row, sweep as before.
                out.append(str(mid))
            elif sid in consolidated_sids:
                out.append(str(mid))
            # else: skip — consolidator owns it.
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cutoff_iso_for(*, retention_days: int) -> str:
    """ISO-8601 timestamp ``retention_days`` in the past, UTC.

    Mirrors the format produced by ``vllm_mlx.memory.chatlog.now_iso_utc``
    so string comparisons against ``chat_messages.timestamp`` are
    well-defined.
    """
    epoch = time.time() - (float(retention_days) * 86400.0)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _now_iso() -> str:
    """Current UTC time in the same ISO-8601 format used elsewhere."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = [
    "DEFAULT_SWEEP_BATCH_SIZE",
    "MAX_BATCHES_PER_SWEEP",
    "REDACTED_PAYLOAD",
    "RetentionSweeper",
    "SweepStats",
]
