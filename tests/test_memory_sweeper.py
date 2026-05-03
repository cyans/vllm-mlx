# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-4 retention sweeper (REQ-N5).

@TEST:MEMORY-01/sweeper

The sweeper deletes (or redacts) chat rows older than
``MEMORY_CHAT_RETENTION_DAYS``. These tests drive
:class:`RetentionSweeper.sweep_once` synchronously so we never need a
running event loop, and they back-date rows by overwriting the
``timestamp`` column to simulate aging without sleeping.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from vllm_mlx.memory.chatlog import (
    new_session_id,
    persist_chat_row,
)
from vllm_mlx.memory.store import open_store
from vllm_mlx.memory.sweeper import (
    REDACTED_PAYLOAD,
    RetentionSweeper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _persist(store, *, request_id: str, user: str, assistant: str) -> None:
    """Drive the public chatlog persistence path synchronously."""
    asyncio.run(
        persist_chat_row(
            store,
            request_id=request_id,
            session_id=new_session_id(),
            model="m",
            messages=[{"role": "user", "content": user}],
            assistant_text=assistant,
        )
    )


def _backdate(store, *, request_id: str, days_ago: int) -> None:
    """Overwrite a chat row's timestamp to ``days_ago`` days in the past."""
    iso = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() - days_ago * 86400),
    )
    store.conn.execute(
        "UPDATE chat_messages SET timestamp = ? WHERE request_id = ?",
        (iso, request_id),
    )


def _row_count(store, *, request_id: str) -> int:
    return int(
        store.conn.execute(
            "SELECT COUNT(*) AS c FROM chat_messages WHERE request_id = ?",
            (request_id,),
        ).fetchone()["c"]
    )


def _fts_count(store, *, message_id: str) -> int:
    return int(
        store.conn.execute(
            "SELECT COUNT(*) AS c FROM fts_chunks WHERE chunk_id = ?",
            (message_id,),
        ).fetchone()["c"]
    )


def _message_id_for(store, request_id: str) -> str | None:
    row = store.conn.execute(
        "SELECT message_id FROM chat_messages WHERE request_id = ?",
        (request_id,),
    ).fetchone()
    return row["message_id"] if row is not None else None


# ---------------------------------------------------------------------------
# Delete mode (default)
# ---------------------------------------------------------------------------
class TestDeleteMode:
    def test_old_row_is_deleted(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _persist(
                store,
                request_id="req-old",
                user="ancient question",
                assistant="ancient answer",
            )
            _backdate(store, request_id="req-old", days_ago=400)
            mid = _message_id_for(store, "req-old")
            assert mid is not None

            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 1
            assert _row_count(store, request_id="req-old") == 0
            assert _fts_count(store, message_id=mid) == 0
        finally:
            store.close()

    def test_recent_row_is_preserved(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _persist(
                store,
                request_id="req-fresh",
                user="recent question",
                assistant="recent answer",
            )
            _backdate(store, request_id="req-fresh", days_ago=100)
            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 0
            assert _row_count(store, request_id="req-fresh") == 1
        finally:
            store.close()

    def test_delete_batches_respect_size_limit(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            for i in range(5):
                _persist(
                    store,
                    request_id=f"req-old-{i}",
                    user=f"q{i}",
                    assistant=f"a{i}",
                )
                _backdate(store, request_id=f"req-old-{i}", days_ago=400)

            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
                batch_size=2,
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 5
            # 3 batches: 2 + 2 + 1
            assert stats.batches == 3
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Redact mode
# ---------------------------------------------------------------------------
class TestRedactMode:
    def test_old_row_payload_is_replaced(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _persist(
                store,
                request_id="req-r",
                user="sensitive question",
                assistant="sensitive answer",
            )
            _backdate(store, request_id="req-r", days_ago=400)
            mid = _message_id_for(store, "req-r")
            assert mid is not None

            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="redact",
                sweep_interval_seconds=60,
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 1
            # Row preserved...
            assert _row_count(store, request_id="req-r") == 1
            # ...but payload is the redaction placeholder.
            row = store.conn.execute(
                "SELECT payload FROM chat_messages WHERE request_id = ?",
                ("req-r",),
            ).fetchone()
            assert row["payload"] == REDACTED_PAYLOAD
            parsed = json.loads(row["payload"])
            assert parsed.get("redacted") is True
            # FTS row removed so the redacted text cannot be searched.
            assert _fts_count(store, message_id=mid) == 0
        finally:
            store.close()

    def test_redacted_row_is_not_returned_by_bm25(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _persist(
                store,
                request_id="req-redact",
                user="needle-in-haystack",
                assistant="haystack-needle",
            )
            _backdate(store, request_id="req-redact", days_ago=400)

            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="redact",
                sweep_interval_seconds=60,
            )
            sweeper.sweep_once()

            assert (
                store.search_bm25(
                    "needle-in-haystack",
                    top_k=5,
                    source_filter="chat",
                )
                == []
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_zero_retention_is_no_op(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _persist(
                store,
                request_id="req-keep",
                user="q",
                assistant="a",
            )
            _backdate(store, request_id="req-keep", days_ago=400)

            # The constructor floors retention_days at 1 via int(),
            # so we drive sweep_once() directly with 0 to exercise the
            # explicit guard inside the method.
            sweeper = RetentionSweeper(
                store,
                retention_days=0,
                mode="delete",
                sweep_interval_seconds=60,
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 0
            assert _row_count(store, request_id="req-keep") == 1
        finally:
            store.close()

    def test_unknown_mode_falls_back_to_delete(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sweeper = RetentionSweeper(
                store,
                retention_days=1,
                mode="banana",  # garbage
                sweep_interval_seconds=60,
            )
            assert sweeper.mode == "delete"
        finally:
            store.close()

    def test_empty_store_is_safe(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 0
        finally:
            store.close()

    def test_sweep_records_meta_last_sweep_at(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _persist(
                store,
                request_id="req-old",
                user="q",
                assistant="a",
            )
            _backdate(store, request_id="req-old", days_ago=400)

            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
            )
            sweeper.sweep_once()
            assert store.get_meta("last_sweep_at") is not None
            assert store.get_meta("last_sweep_count") == "1"
            assert store.get_meta("last_sweep_mode") == "delete"

            stats = store.get_memory_stats()
            assert stats["last_sweep_count"] == 1
            assert stats["last_sweep_mode"] == "delete"
            assert stats["last_sweep_at"] is not None
        finally:
            store.close()

    def test_sweep_records_meta_even_when_no_rows_match(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _persist(
                store,
                request_id="req-fresh",
                user="q",
                assistant="a",
            )
            _backdate(store, request_id="req-fresh", days_ago=10)

            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
            )
            sweeper.sweep_once()
            assert store.get_meta("last_sweep_at") is not None
            assert store.get_meta("last_sweep_count") == "0"
        finally:
            store.close()


# ---------------------------------------------------------------------------
# REQ-N4 — failure isolation in the run loop
# ---------------------------------------------------------------------------
class TestRunLoopIsolation:
    def test_run_loop_continues_after_failure(self, tmp_path, monkeypatch):
        store = open_store(tmp_path / "memory.db")
        try:
            sweeper = RetentionSweeper(
                store,
                retention_days=1,
                mode="delete",
                sweep_interval_seconds=0.05,
            )

            calls = {"n": 0}

            def flaky_sweep_once():
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("simulated DB lock")
                # Stop the loop on the second call by raising
                # CancelledError, which is the same thing asyncio uses
                # for shutdown.
                raise asyncio.CancelledError

            monkeypatch.setattr(sweeper, "sweep_once", flaky_sweep_once)

            with pytest.raises(asyncio.CancelledError):
                asyncio.run(sweeper.run())

            # Both calls happened — first failed, second cancelled.
            assert calls["n"] == 2
        finally:
            store.close()

    def test_run_exits_immediately_when_retention_disabled(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sweeper = RetentionSweeper(
                store,
                retention_days=0,
                mode="delete",
                sweep_interval_seconds=60,
            )
            # ``run`` should return without scheduling any work.
            asyncio.run(sweeper.run())
        finally:
            store.close()
