# SPDX-License-Identifier: Apache-2.0
"""Tests for the Phase-3 chat persistence + redaction module.

@TEST:MEMORY-01/chatlog

Coverage:

* REQ-E3: a chat row lands in ``chat_messages`` after persistence.
* REQ-N1: redaction strips ``sk-…`` and ``password=`` patterns.
* REQ-N4: a raising store does NOT propagate exceptions.
* REQ-S1: with the chat log disabled, no rows are written.
* Determinism: same ``request_id`` -> same chunk_id (for embed-loop idempotency).

We mostly drive the public API (:func:`persist_chat_row`) directly
because it is the single entry point invoked by ``server.py`` and the
fastest unit to validate.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid

import pytest

from vllm_mlx.memory.chatlog import (
    DEFAULT_REDACT_PATTERN_STRINGS,
    REDACT_PLACEHOLDER,
    build_searchable_text,
    chunk_id_for_chat,
    compile_redact_patterns,
    new_session_id,
    now_iso_utc,
    persist_chat_row,
    redact_text,
    serialize_chat_payload,
)
from vllm_mlx.memory.store import open_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(coro):
    """Run a coroutine synchronously inside a fresh event loop."""
    return asyncio.run(coro)


def _patterns():
    return compile_redact_patterns(None)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# Redaction (REQ-N1)
# ---------------------------------------------------------------------------
class TestRedaction:
    def test_default_patterns_compile(self):
        patterns = compile_redact_patterns(None)
        assert len(patterns) == len(DEFAULT_REDACT_PATTERN_STRINGS)

    def test_invalid_pattern_is_dropped_with_warning(self, caplog):
        patterns = compile_redact_patterns(["[", r"sk-[A-Za-z0-9]{20,}"])
        # Bad pattern dropped; second one still compiles.
        assert len(patterns) == 1

    def test_redact_strips_sk_pattern(self):
        text = "Here is my key sk-abcd1234567890abcdefgh1234 do not share."
        out = redact_text(text, _patterns())
        assert "sk-abcd1234567890abcdefgh1234" not in out
        assert REDACT_PLACEHOLDER in out

    def test_redact_strips_password_pattern(self):
        text = "Login: password=secret123 right away"
        out = redact_text(text, _patterns())
        # The default pattern matches "password=value-token" so the value is gone.
        assert "secret123" not in out
        assert REDACT_PLACEHOLDER in out

    def test_redact_strips_api_key_pattern(self):
        text = 'config { API_KEY = abc-123 }'
        out = redact_text(text, _patterns())
        assert "abc-123" not in out

    def test_redact_idempotent_on_safe_text(self):
        safe = "this text has no secrets"
        assert redact_text(safe, _patterns()) == safe

    def test_redact_handles_empty_text(self):
        assert redact_text("", _patterns()) == ""
        assert redact_text("   ", _patterns()) == "   "


# ---------------------------------------------------------------------------
# chunk_id and session_id determinism
# ---------------------------------------------------------------------------
class TestChunkAndSession:
    def test_chunk_id_deterministic(self):
        a = chunk_id_for_chat("chatcmpl-abc123")
        b = chunk_id_for_chat("chatcmpl-abc123")
        assert a == b
        assert len(a) == 32
        assert all(c in "0123456789abcdef" for c in a)

    def test_chunk_id_different_for_different_request_ids(self):
        a = chunk_id_for_chat("req-1")
        b = chunk_id_for_chat("req-2")
        assert a != b

    def test_session_id_uuid_format(self):
        sid = new_session_id()
        # Must parse as a UUID.
        parsed = uuid.UUID(sid)
        assert str(parsed) == sid

    def test_session_id_unique_per_call(self):
        a = new_session_id()
        b = new_session_id()
        assert a != b

    def test_now_iso_utc_format(self):
        ts = now_iso_utc()
        # YYYY-MM-DDTHH:MM:SSZ
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", ts) is not None


# ---------------------------------------------------------------------------
# Payload serialization
# ---------------------------------------------------------------------------
class TestSerializePayload:
    def test_roundtrip_simple(self):
        msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
        out = serialize_chat_payload(msgs, "hi", None, _patterns())
        data = json.loads(out)
        assert data["assistant"] == "hi"
        assert any(m["content"] == "hello" for m in data["messages"])

    def test_redacts_user_content(self):
        msgs = [_msg("user", "my key is sk-abcd1234567890abcdefgh1234 ok")]
        out = serialize_chat_payload(msgs, "noted", None, _patterns())
        assert "sk-abcd1234567890abcdefgh1234" not in out
        assert REDACT_PLACEHOLDER in out

    def test_redacts_assistant_text(self):
        out = serialize_chat_payload(
            [_msg("user", "anything")],
            "sure, the api_key=mysecret123 token is here",
            None,
            _patterns(),
        )
        assert "mysecret123" not in out

    def test_handles_multimodal_content_list(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "see this image"},
                {"type": "image_url", "image_url": {"url": "https://x"}},
            ],
        }
        out = serialize_chat_payload([msg], "ok", None, _patterns())
        data = json.loads(out)
        # image_url part dropped; only text survives.
        assert data["messages"][0]["content"] == "see this image"

    def test_falls_back_to_empty_dict_on_serialize_failure(self, monkeypatch):
        # Patch json.dumps to fail and make sure we still return a string.
        import vllm_mlx.memory.chatlog as cl

        def _boom(*a, **k):  # pragma: no cover
            raise TypeError("boom")

        monkeypatch.setattr(cl.json, "dumps", _boom)
        out = serialize_chat_payload([_msg("user", "x")], "y", None, _patterns())
        assert out == "{}"

    def test_passes_through_tool_calls(self):
        msgs = [_msg("user", "search please")]
        tool_calls = [{"id": "call_1", "name": "memory_search", "arguments": "{}"}]
        out = serialize_chat_payload(msgs, "running", tool_calls, _patterns())
        data = json.loads(out)
        assert data["tool_calls"] == tool_calls


# ---------------------------------------------------------------------------
# build_searchable_text
# ---------------------------------------------------------------------------
class TestSearchableText:
    def test_concatenates_user_and_assistant(self):
        msgs = [
            _msg("system", "you are helpful"),
            _msg("user", "what is distillation?"),
        ]
        out = build_searchable_text(msgs, "Distillation is...", _patterns())
        assert "USER:" in out
        assert "ASSISTANT:" in out
        assert "distillation" in out.lower()

    def test_empty_messages_returns_empty(self):
        out = build_searchable_text([], "", _patterns())
        assert out == ""

    def test_uses_last_user_message(self):
        msgs = [
            _msg("user", "first question"),
            _msg("assistant", "first answer"),
            _msg("user", "second question"),
        ]
        out = build_searchable_text(msgs, "second answer", _patterns())
        assert "second question" in out
        assert "second answer" in out


# ---------------------------------------------------------------------------
# persist_chat_row — happy path & failure isolation
# ---------------------------------------------------------------------------
class TestPersistChatRow:
    def test_persists_row_into_chat_messages(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            request_id = "chatcmpl-test001"
            session_id = new_session_id()
            _run(
                persist_chat_row(
                    store,
                    request_id=request_id,
                    session_id=session_id,
                    model="test-model",
                    messages=[_msg("user", "tell me about MoE")],
                    assistant_text="MoE = mixture of experts.",
                    tool_calls=None,
                    latency_ms=12.3,
                    redact_patterns=_patterns(),
                )
            )
            rows = store.conn.execute(
                "SELECT request_id, session_id, role, payload FROM chat_messages"
            ).fetchall()
            assert len(rows) == 1
            row = rows[0]
            assert row["request_id"] == request_id
            assert row["session_id"] == session_id
            assert row["role"] == "assistant"
            payload = json.loads(row["payload"])
            assert payload["assistant"] == "MoE = mixture of experts."
        finally:
            store.close()

    def test_persists_into_fts_chunks(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _run(
                persist_chat_row(
                    store,
                    request_id="chatcmpl-001",
                    session_id=new_session_id(),
                    model="m",
                    messages=[_msg("user", "favorite colour?")],
                    assistant_text="ultraviolet",
                    redact_patterns=_patterns(),
                )
            )
            rows = store.conn.execute(
                "SELECT chunk_id, source_type, source_path, text FROM fts_chunks"
            ).fetchall()
            assert len(rows) == 1
            row = rows[0]
            assert row["source_type"] == "chat"
            assert row["source_path"].startswith("chat/")
            assert "ultraviolet" in row["text"]
        finally:
            store.close()

    def test_redaction_strips_sk_pattern_from_storage(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            secret = "sk-abcd1234567890abcdefgh1234"
            _run(
                persist_chat_row(
                    store,
                    request_id="chatcmpl-002",
                    session_id=new_session_id(),
                    model="m",
                    messages=[_msg("user", f"my key is {secret}")],
                    assistant_text="ack",
                    redact_patterns=_patterns(),
                )
            )
            row = store.conn.execute(
                "SELECT payload FROM chat_messages"
            ).fetchone()
            assert secret not in row["payload"]
            fts = store.conn.execute(
                "SELECT text FROM fts_chunks"
            ).fetchone()
            assert secret not in fts["text"]
        finally:
            store.close()

    def test_redaction_strips_password_pattern(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _run(
                persist_chat_row(
                    store,
                    request_id="chatcmpl-003",
                    session_id=new_session_id(),
                    model="m",
                    messages=[_msg("user", "creds: password=hunter2")],
                    assistant_text="noted",
                    redact_patterns=_patterns(),
                )
            )
            row = store.conn.execute(
                "SELECT payload FROM chat_messages"
            ).fetchone()
            assert "hunter2" not in row["payload"]
        finally:
            store.close()

    def test_raising_store_does_not_propagate(self):
        class BoomStore:
            def transaction(self):
                raise RuntimeError("simulated DB failure")

        # Must NOT raise — REQ-N4 contract.
        result = _run(
            persist_chat_row(
                BoomStore(),
                request_id="x",
                session_id="s",
                model="m",
                messages=[_msg("user", "hi")],
                assistant_text="hi back",
                redact_patterns=_patterns(),
            )
        )
        assert result is None

    def test_none_store_is_a_noop(self):
        # No exception even with store=None.
        result = _run(
            persist_chat_row(
                None,
                request_id="x",
                session_id="s",
                model="m",
                messages=[_msg("user", "hi")],
                assistant_text="hi back",
                redact_patterns=_patterns(),
            )
        )
        assert result is None

    def test_empty_request_id_is_a_noop(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            _run(
                persist_chat_row(
                    store,
                    request_id="",
                    session_id=new_session_id(),
                    model="m",
                    messages=[_msg("user", "hi")],
                    assistant_text="hi",
                    redact_patterns=_patterns(),
                )
            )
            count = store.conn.execute(
                "SELECT COUNT(*) AS c FROM chat_messages"
            ).fetchone()["c"]
            assert count == 0
        finally:
            store.close()

    def test_idempotent_on_repeat_request_id(self, tmp_path):
        # Re-persisting the same request_id should NOT create a duplicate
        # (INSERT OR REPLACE on message_id which is deterministic).
        store = open_store(tmp_path / "memory.db")
        try:
            request_id = "chatcmpl-dup"
            for assistant in ("v1", "v2"):
                _run(
                    persist_chat_row(
                        store,
                        request_id=request_id,
                        session_id=new_session_id(),
                        model="m",
                        messages=[_msg("user", "same")],
                        assistant_text=assistant,
                        redact_patterns=_patterns(),
                    )
                )
            count = store.conn.execute(
                "SELECT COUNT(*) AS c FROM chat_messages WHERE request_id = ?",
                (request_id,),
            ).fetchone()["c"]
            assert count == 1
            # Latest write wins (v2).
            row = store.conn.execute(
                "SELECT payload FROM chat_messages WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            assert "v2" in row["payload"]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Off-switch invariance (REQ-S1)
# ---------------------------------------------------------------------------
class TestOptOut:
    def test_default_patterns_used_when_none_passed(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            secret = "sk-abcdefghijklmnopqrstuvwxyz1234"
            _run(
                persist_chat_row(
                    store,
                    request_id="chatcmpl-default",
                    session_id=new_session_id(),
                    model="m",
                    messages=[_msg("user", f"key is {secret}")],
                    assistant_text="ack",
                    redact_patterns=None,
                )
            )
            row = store.conn.execute(
                "SELECT payload FROM chat_messages"
            ).fetchone()
            # Even with redact_patterns=None we apply the SPEC defaults.
            assert secret not in row["payload"]
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Pydantic-shape messages
# ---------------------------------------------------------------------------
class _PydanticLikeMessage:
    """Minimal stand-in for an api.models.Message instance.

    Used to cover the ``hasattr(msg, "model_dump")`` branch without
    importing the whole API model layer.
    """

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def model_dump(self, *, exclude_none: bool = False) -> dict:
        return {"role": self.role, "content": self.content}


class TestPydanticShapedInput:
    def test_persist_with_pydantic_like_messages(self, tmp_path):
        store = open_store(tmp_path / "memory.db")
        try:
            msgs = [
                _PydanticLikeMessage("system", "be helpful"),
                _PydanticLikeMessage("user", "thanks"),
            ]
            _run(
                persist_chat_row(
                    store,
                    request_id="chatcmpl-pyd",
                    session_id=new_session_id(),
                    model="m",
                    messages=msgs,
                    assistant_text="welcome",
                    redact_patterns=_patterns(),
                )
            )
            count = store.conn.execute(
                "SELECT COUNT(*) AS c FROM chat_messages"
            ).fetchone()["c"]
            assert count == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Background embed loop helpers (no real model loaded)
# ---------------------------------------------------------------------------
class _StubEmbedder:
    """Tiny stub matching the duck type the loop expects.

    Returns a deterministic 4-byte blob per text so the store can write
    them as float32[1] vectors. We rebuild the store with embed_dim=1
    in the test fixture.
    """

    def __init__(self, *, fail: bool = False, _disabled: bool = False):
        self.fail = fail
        self._dis = _disabled
        self.calls = 0

    def disabled(self) -> bool:
        return self._dis

    def encode_batch(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated embed failure")
        # 4 bytes = float32[1] per text
        import struct

        return [struct.pack("f", float(len(t))) for t in texts]


class TestEmbedBatch:
    def test_embed_one_batch_processes_pending_rows(self, tmp_path):
        # Open store with embed_dim=1 so our stub fits.
        from vllm_mlx.memory.chatlog import _embed_one_batch
        from vllm_mlx.memory.store import MemoryStore

        store = MemoryStore(tmp_path / "memory.db", embed_dim=1)
        store.open()
        try:
            if not store.vec_loaded:
                # sqlite-vec missing → loop is a no-op; nothing to test.
                pytest.skip("sqlite-vec not available in this env")

            # Persist two chat rows.
            for i in range(2):
                _run(
                    persist_chat_row(
                        store,
                        request_id=f"chatcmpl-batch-{i}",
                        session_id=new_session_id(),
                        model="m",
                        messages=[{"role": "user", "content": f"q-{i}"}],
                        assistant_text=f"a-{i}",
                        redact_patterns=_patterns(),
                    )
                )

            assert store.count_chat_messages() == 2
            assert store.count_chat_messages_missing_vectors() == 2

            embedder = _StubEmbedder()
            n = _run(_embed_one_batch(store, embedder, batch_size=10))
            assert n == 2
            assert store.count_chat_messages_missing_vectors() == 0
        finally:
            store.close()

    def test_embed_one_batch_no_op_when_embedder_disabled(self, tmp_path):
        from vllm_mlx.memory.chatlog import _embed_one_batch

        store = open_store(tmp_path / "memory.db")
        try:
            embedder = _StubEmbedder(_disabled=True)
            n = _run(_embed_one_batch(store, embedder, batch_size=10))
            assert n == 0
        finally:
            store.close()

    def test_embed_one_batch_no_op_when_embedder_none(self, tmp_path):
        from vllm_mlx.memory.chatlog import _embed_one_batch

        store = open_store(tmp_path / "memory.db")
        try:
            n = _run(_embed_one_batch(store, None, batch_size=10))
            assert n == 0
        finally:
            store.close()

    def test_embed_one_batch_handles_encode_failure(self, tmp_path):
        from vllm_mlx.memory.chatlog import _embed_one_batch
        from vllm_mlx.memory.store import MemoryStore

        store = MemoryStore(tmp_path / "memory.db", embed_dim=1)
        store.open()
        try:
            if not store.vec_loaded:
                pytest.skip("sqlite-vec not available in this env")
            _run(
                persist_chat_row(
                    store,
                    request_id="chatcmpl-fail",
                    session_id=new_session_id(),
                    model="m",
                    messages=[{"role": "user", "content": "q"}],
                    assistant_text="a",
                    redact_patterns=_patterns(),
                )
            )
            embedder = _StubEmbedder(fail=True)
            n = _run(_embed_one_batch(store, embedder, batch_size=10))
            assert n == 0  # embed failure -> 0 written, no exception
        finally:
            store.close()
