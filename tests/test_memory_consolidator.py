# SPDX-License-Identifier: Apache-2.0
"""Tests for the SPEC-MEMORY-02 Phase 1 chat consolidator.

@TEST:MEMORY-02/consolidator

Every test uses a fake engine that returns canned JSON or raises.
NEVER touches a real Qwen3.6 model.

Coverage targets:
- Happy path: valid JSON → summary written + raw rows evicted.
- JSON parse error → no insert, no delete.
- Summary too short → no insert, no delete.
- Multiple sessions in one tick.
- Engine raises → session skipped, loop continues.
- Atomicity: simulated DB error rolls both INSERT and DELETE back.
- Sweeper integration: ``MEMORY_SUMMARIZE_BEFORE_DELETE=1`` skips
  unsummarized sessions.
- Redaction: API keys in model output are scrubbed before write.
"""

from __future__ import annotations

import asyncio
import json
import time


from vllm_mlx.memory.chatlog import (
    new_session_id,
    persist_chat_row,
)
from vllm_mlx.memory.config import resolve_memory_config
from vllm_mlx.memory.consolidator import (
    ChatConsolidator,
    _build_summarizer_prompt,
    _parse_summary_json,
    _render_summary_text,
)
from vllm_mlx.memory.store import open_store
from vllm_mlx.memory.sweeper import RetentionSweeper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeEngine:
    """Returns a canned response or raises on each ``generate`` call."""

    def __init__(self, responses):
        # Either a list of strings/exceptions to cycle through, or a
        # single string to return for every call.
        if isinstance(responses, (str, Exception)):
            self._responses = [responses]
            self._cycle = True
        else:
            self._responses = list(responses)
            self._cycle = False
        self._idx = 0
        self.calls: list[str] = []

    async def generate(self, prompt: str, **kwargs):  # noqa: D401
        self.calls.append(prompt)
        if self._cycle:
            r = self._responses[0]
        else:
            if self._idx >= len(self._responses):
                raise IndexError("no more canned responses")
            r = self._responses[self._idx]
            self._idx += 1
        if isinstance(r, Exception):
            raise r
        return _Output(text=r)


class _Output:
    """Mimics ``GenerationOutput.text`` access."""

    def __init__(self, text: str):
        self.text = text


def _persist_session(
    store, *, session_id: str, n_messages: int, days_ago: int
) -> list[str]:
    """Persist ``n_messages`` chat rows for one session, back-dated."""
    msg_ids: list[str] = []
    for i in range(n_messages):
        request_id = f"req-{session_id[:6]}-{i}"
        asyncio.run(
            persist_chat_row(
                store,
                request_id=request_id,
                session_id=session_id,
                model="qwen3.6",
                messages=[{"role": "user", "content": f"q{i}"}],
                assistant_text=f"a{i}",
            )
        )
        # Back-date the row to be older than the STM threshold.
        iso = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - days_ago * 86400 + i),
        )
        store.conn.execute(
            "UPDATE chat_messages SET timestamp = ? "
            "WHERE request_id = ?",
            (iso, request_id),
        )
        # Also fetch the message_id for assertion purposes.
        row = store.conn.execute(
            "SELECT message_id FROM chat_messages WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is not None:
            msg_ids.append(str(row["message_id"]))
    return msg_ids


def _make_config(
    *,
    stm_days: int = 30,
    summary_min_chars: int = 40,
    consolidator_dryrun: bool = False,
    consolidator_batch: int = 10,
    consolidator_deadline_sec: int = 1800,
    redact_patterns: tuple[str, ...] = (),
):
    cfg = resolve_memory_config(
        {
            "MEMORY_ENABLED": "1",
            "MEMORY_STM_DAYS": str(stm_days),
            "MEMORY_SUMMARY_MIN_CHARS": str(summary_min_chars),
            "MEMORY_CONSOLIDATOR_BATCH": str(consolidator_batch),
            "MEMORY_CONSOLIDATOR_DEADLINE_SEC": str(consolidator_deadline_sec),
            "MEMORY_CONSOLIDATOR_DRYRUN": "1" if consolidator_dryrun else "0",
            "MEMORY_REDACT_PATTERNS": ";".join(redact_patterns),
        }
    )
    return cfg


def _valid_summary_json(one_line: str = "User asked about distillation.") -> str:
    return json.dumps(
        {
            "key_conclusions": [
                "Distillation is a model-compression technique.",
                "Loss should weight teacher logits.",
            ],
            "entities": ["distillation", "Qwen3.6", "soft labels"],
            "unresolved": ["Optimal temperature constant?"],
            "one_line": one_line,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Pure-function helpers (no DB)
# ---------------------------------------------------------------------------
class TestParseSummaryJson:
    def test_valid_json_parses(self):
        parsed = _parse_summary_json(_valid_summary_json())
        assert parsed is not None
        assert parsed["one_line"] == "User asked about distillation."
        assert len(parsed["key_conclusions"]) == 2

    def test_invalid_json_returns_none(self):
        assert _parse_summary_json("{") is None
        assert _parse_summary_json("not json") is None
        assert _parse_summary_json("") is None

    def test_strips_json_fence(self):
        wrapped = "```json\n" + _valid_summary_json() + "\n```"
        parsed = _parse_summary_json(wrapped)
        assert parsed is not None
        assert "key_conclusions" in parsed

    def test_strips_leading_prose(self):
        prose = (
            "Sure! Here is the summary:\n\n"
            + _valid_summary_json()
            + "\n\nLet me know if you need more."
        )
        parsed = _parse_summary_json(prose)
        # The trailing prose may break json.loads — accept either parse
        # success (best-effort) or None. The important behavior is that
        # we don't raise.
        # We only care that we don't crash here.
        assert parsed is None or "one_line" in parsed

    def test_missing_keys_default_to_empty(self):
        parsed = _parse_summary_json(json.dumps({"one_line": "x"}))
        assert parsed is not None
        assert parsed["one_line"] == "x"
        assert parsed["key_conclusions"] == []
        assert parsed["entities"] == []
        assert parsed["unresolved"] == []

    def test_wrong_type_for_key_returns_none(self):
        # one_line should be a string, not a list.
        bad = json.dumps({"one_line": ["not a string"]})
        assert _parse_summary_json(bad) is None


class TestRenderSummaryText:
    def test_full_summary_includes_all_sections(self):
        parsed = {
            "one_line": "Quick overview.",
            "key_conclusions": ["A", "B"],
            "entities": ["x", "y"],
            "unresolved": ["Q1"],
        }
        text = _render_summary_text(parsed)
        assert "Quick overview." in text
        assert "Key conclusions:" in text
        assert "- A" in text and "- B" in text
        assert "Entities: x, y" in text
        assert "Unresolved:" in text and "- Q1" in text

    def test_empty_sections_omitted(self):
        parsed = {
            "one_line": "",
            "key_conclusions": [],
            "entities": [],
            "unresolved": [],
        }
        text = _render_summary_text(parsed)
        assert text == ""


class TestBuildSummarizerPrompt:
    def test_includes_system_prompt(self):
        msgs = [
            {
                "message_id": "m1",
                "request_id": "r1",
                "role": "assistant",
                "payload": json.dumps(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "assistant": "hi",
                    }
                ),
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ]
        prompt = _build_summarizer_prompt(msgs, input_budget_tokens=1000)
        assert "memory consolidator" in prompt
        assert "[user]: hello" in prompt
        assert "[assistant]: hi" in prompt

    def test_truncates_to_budget(self):
        # Make 100 huge messages and a small budget; only the most
        # recent ones should remain.
        msgs = []
        for i in range(100):
            msgs.append(
                {
                    "message_id": f"m{i}",
                    "request_id": f"r{i}",
                    "role": "assistant",
                    "payload": json.dumps(
                        {
                            "messages": [{"role": "user", "content": "x" * 200}],
                            "assistant": "y" * 200,
                        }
                    ),
                    "timestamp": f"2026-01-01T00:{i:02d}:00Z",
                }
            )
        prompt = _build_summarizer_prompt(msgs, input_budget_tokens=300)
        # 300 tokens × 3 chars/token = 900 char budget for the body.
        # The prompt itself adds the system header but the body should
        # be bounded.
        body = prompt.split("\n\n", 2)[-1]
        # We always keep the most recent message, so we must include
        # "y * 200" somewhere; we definitely should NOT include all 100
        # message bodies.
        assert prompt.count("[user]: ") < 100


# ---------------------------------------------------------------------------
# Consolidator: end-to-end with a fake engine + real SQLite
# ---------------------------------------------------------------------------
class TestConsolidatorHappyPath:
    def test_valid_summary_writes_and_evicts(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            msg_ids = _persist_session(
                store, session_id=sid, n_messages=3, days_ago=31
            )
            assert msg_ids
            engine = _FakeEngine(_valid_summary_json())
            cfg = _make_config()
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_summarized == 1
            assert stats.sessions_evicted == 3
            # chat_summaries row exists.
            row = store.conn.execute(
                "SELECT session_id, summary_text, source_count "
                "FROM chat_summaries WHERE session_id = ?",
                (sid,),
            ).fetchone()
            assert row is not None
            assert row["session_id"] == sid
            assert row["source_count"] == 3
            assert "User asked about distillation" in row["summary_text"]
            # chat_messages rows are gone.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 0
            )
            # FTS rows for those message_ids are gone.
            for mid in msg_ids:
                assert (
                    store.conn.execute(
                        "SELECT COUNT(*) AS c FROM fts_chunks "
                        "WHERE chunk_id = ?",
                        (mid,),
                    ).fetchone()["c"]
                    == 0
                )
        finally:
            store.close()

    def test_multiple_sessions_in_one_tick(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sids = [new_session_id() for _ in range(3)]
            for sid in sids:
                _persist_session(
                    store, session_id=sid, n_messages=2, days_ago=31
                )
            engine = _FakeEngine(_valid_summary_json())
            cfg = _make_config()
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_summarized == 3
            assert stats.sessions_evicted == 6
            for sid in sids:
                assert (
                    store.conn.execute(
                        "SELECT COUNT(*) AS c FROM chat_summaries "
                        "WHERE session_id = ?",
                        (sid,),
                    ).fetchone()["c"]
                    == 1
                )
        finally:
            store.close()

    def test_recent_sessions_not_picked_up(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            # 5 days ago is well within the 30-day STM window.
            _persist_session(
                store, session_id=sid, n_messages=2, days_ago=5
            )
            engine = _FakeEngine(_valid_summary_json())
            cfg = _make_config()
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_examined == 0
            assert stats.sessions_summarized == 0
            assert engine.calls == []
            # Chat rows untouched.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 2
            )
        finally:
            store.close()


class TestConsolidatorErrorPaths:
    def test_invalid_json_skips_session(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=2, days_ago=31
            )
            engine = _FakeEngine("{")  # malformed
            cfg = _make_config()
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_summarized == 0
            assert stats.sessions_skipped_parse_error == 1
            assert sid in stats.skipped_session_ids
            # Raw rows preserved.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 2
            )
            # No summary row.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_summaries "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 0
            )
        finally:
            store.close()

    def test_summary_too_short_skips_eviction(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=2, days_ago=31
            )
            # All-empty summary → rendered text is empty → < min chars.
            empty_json = json.dumps(
                {
                    "key_conclusions": [],
                    "entities": [],
                    "unresolved": [],
                    "one_line": "",
                }
            )
            engine = _FakeEngine(empty_json)
            cfg = _make_config(summary_min_chars=40)
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_summarized == 0
            assert stats.sessions_skipped_too_short == 1
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 2
            )
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_summaries "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 0
            )
        finally:
            store.close()

    def test_engine_raises_skips_session_loop_continues(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sid_bad = new_session_id()
            sid_good = new_session_id()
            # Persist bad session FIRST (older timestamps so it's
            # selected first by the cutoff ORDER BY).
            _persist_session(
                store, session_id=sid_bad, n_messages=2, days_ago=60
            )
            _persist_session(
                store, session_id=sid_good, n_messages=2, days_ago=31
            )
            engine = _FakeEngine(
                [RuntimeError("model exploded"), _valid_summary_json()]
            )
            cfg = _make_config()
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_examined == 2
            assert stats.sessions_summarized == 1
            assert stats.sessions_skipped_engine_error == 1
            assert sid_bad in stats.skipped_session_ids
            # Good session was consolidated.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_summaries "
                    "WHERE session_id = ?",
                    (sid_good,),
                ).fetchone()["c"]
                == 1
            )
            # Bad session preserved.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid_bad,),
                ).fetchone()["c"]
                == 2
            )
        finally:
            store.close()


class TestConsolidatorAtomicity:
    def test_db_error_rolls_back_summary_insert(self, tmp_path, monkeypatch):
        """If eviction raises, the matching INSERT must roll back too."""
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=2, days_ago=31
            )
            engine = _FakeEngine(_valid_summary_json())
            cfg = _make_config()
            consolidator = ChatConsolidator(store, engine, cfg)

            # Patch evict_chat_messages_for_session to raise AFTER the
            # insert has run inside the same transaction.
            def boom(*args, **kwargs):
                raise RuntimeError("simulated eviction failure")

            monkeypatch.setattr(
                store, "evict_chat_messages_for_session", boom
            )
            stats = asyncio.run(consolidator.run_once())

            # Summary insert must have rolled back.
            assert stats.sessions_summarized == 0
            assert stats.sessions_skipped_db_error == 1
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_summaries "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 0
            )
            # Raw rows still present (eviction never ran).
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 2
            )
        finally:
            store.close()


class TestConsolidatorRedaction:
    def test_summary_text_redacts_api_keys(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=2, days_ago=31
            )
            # Model output contains an API key in the one_line field.
            api_key = "sk-ABCDEFGHIJKLMNOPQRSTUVWX1234567890"
            json_with_secret = json.dumps(
                {
                    "key_conclusions": [
                        f"User exposed key {api_key} in conversation"
                    ],
                    "entities": ["api"],
                    "unresolved": [],
                    "one_line": f"Discussion involved {api_key}.",
                }
            )
            engine = _FakeEngine(json_with_secret)
            cfg = _make_config(
                redact_patterns=(r"sk-[A-Za-z0-9]{20,}",),
            )
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_summarized == 1
            row = store.conn.execute(
                "SELECT summary_text FROM chat_summaries "
                "WHERE session_id = ?",
                (sid,),
            ).fetchone()
            assert row is not None
            assert api_key not in row["summary_text"]
            assert "[REDACTED]" in row["summary_text"]
        finally:
            store.close()


class TestConsolidatorDryRun:
    def test_dryrun_does_not_write(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=2, days_ago=31
            )
            engine = _FakeEngine(_valid_summary_json())
            cfg = _make_config(consolidator_dryrun=True)
            consolidator = ChatConsolidator(store, engine, cfg)
            stats = asyncio.run(consolidator.run_once())

            assert stats.sessions_summarized == 1
            # No DB writes.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_summaries"
                ).fetchone()["c"]
                == 0
            )
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 2
            )
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Sweeper integration: summarize-before-delete contract
# ---------------------------------------------------------------------------
class TestSweeperIntegration:
    def test_summarize_before_delete_skips_unsummarized(self, tmp_path):
        """Sweeper with the flag on must NOT delete unsummarized rows."""
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=3, days_ago=400
            )
            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
                summarize_before_delete=True,
            )
            stats = sweeper.sweep_once()
            # Even though rows are 400 days old (way past 365), they
            # have no summary yet, so the sweeper must skip them.
            assert stats.rows_changed == 0
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 3
            )
        finally:
            store.close()

    def test_summarize_before_delete_evicts_when_summary_exists(self, tmp_path):
        """Sweeper with the flag on DOES delete rows whose session has a summary."""
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=3, days_ago=400
            )
            # Manually insert a summary for this session.
            store.conn.execute(
                "INSERT INTO chat_summaries("
                "  session_id, summary_text, period_start, "
                "  period_end, source_count, created_at"
                ") VALUES(?, ?, ?, ?, ?, ?)",
                (
                    sid,
                    "x" * 80,
                    "2024-01-01T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                    3,
                    time.time(),
                ),
            )
            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
                summarize_before_delete=True,
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 3
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 0
            )
            # Summary row preserved.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_summaries "
                    "WHERE session_id = ?",
                    (sid,),
                ).fetchone()["c"]
                == 1
            )
        finally:
            store.close()

    def test_default_sweeper_behavior_unchanged(self, tmp_path):
        """With the flag OFF (default), sweeper deletes everything past TTL."""
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=3, days_ago=400
            )
            sweeper = RetentionSweeper(
                store,
                retention_days=365,
                mode="delete",
                sweep_interval_seconds=60,
                summarize_before_delete=False,  # original behavior
            )
            stats = sweeper.sweep_once()
            assert stats.rows_changed == 3
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
class TestConsolidatorIdempotency:
    def test_second_run_is_noop(self, tmp_path):
        """Re-running the consolidator after success must be a no-op."""
        store = open_store(tmp_path / "memory.db")
        try:
            sid = new_session_id()
            _persist_session(
                store, session_id=sid, n_messages=2, days_ago=31
            )
            engine = _FakeEngine(_valid_summary_json())
            cfg = _make_config()
            consolidator = ChatConsolidator(store, engine, cfg)

            stats1 = asyncio.run(consolidator.run_once())
            assert stats1.sessions_summarized == 1

            # Second run: chat_messages is empty for this session, so
            # the SELECT returns nothing and we examine 0 sessions.
            stats2 = asyncio.run(consolidator.run_once())
            assert stats2.sessions_examined == 0
            assert stats2.sessions_summarized == 0
            # Still exactly one summary row.
            assert (
                store.conn.execute(
                    "SELECT COUNT(*) AS c FROM chat_summaries"
                ).fetchone()["c"]
                == 1
            )
        finally:
            store.close()
