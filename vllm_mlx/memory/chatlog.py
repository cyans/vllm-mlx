# SPDX-License-Identifier: Apache-2.0
"""Chat log persistence for the memory subsystem.

@CODE:MEMORY-01/chatlog

Phase 3: persist a structured row per ``/v1/chat/completions`` request
into ``chat_messages`` + ``fts_chunks`` so the memory MCP child can
later embed and serve them via ``memory_search`` (REQ-E3 / REQ-E4 /
REQ-N1 / REQ-N4).

Failure isolation (REQ-N4): every public coroutine wraps its body in a
single try/except and logs via :func:`logger.exception`. The chat path
NEVER awaits anything that could raise out of this module.

Redaction (REQ-N1): :func:`redact_text` applies the configured regex
list to user messages AND to the assistant text BEFORE they are
serialized to JSON. Tests in ``tests/test_memory_chatlog.py`` cover
the canonical patterns from SPEC §9.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redaction (REQ-N1)
# ---------------------------------------------------------------------------
# Default patterns from SPEC-MEMORY-01 §9. Compiled once and reused.
DEFAULT_REDACT_PATTERN_STRINGS: tuple[str, ...] = (
    r"sk-[A-Za-z0-9]{20,}",
    r"(?i)password\s*[:=]\s*\S+",
    r"(?i)api[_-]?key\s*[:=]\s*\S+",
)

REDACT_PLACEHOLDER = "[REDACTED]"


def compile_redact_patterns(
    raw_patterns: Sequence[str] | None,
) -> list[re.Pattern[str]]:
    """Compile a sequence of regex strings into ``re.Pattern`` objects.

    Invalid patterns are dropped with a single warning so a typo in the
    user-supplied env var cannot crash the chat path. The returned list
    falls back to the SPEC defaults when ``raw_patterns`` is empty.
    """
    sources: Sequence[str]
    if raw_patterns is None or len(raw_patterns) == 0:
        sources = DEFAULT_REDACT_PATTERN_STRINGS
    else:
        sources = raw_patterns

    compiled: list[re.Pattern[str]] = []
    for src in sources:
        if not src:
            continue
        try:
            compiled.append(re.compile(src))
        except re.error as exc:  # noqa: PERF203 - cold path
            logger.warning(
                "[memory] dropping invalid redaction pattern %r: %s",
                src,
                exc,
            )
    return compiled


def redact_text(text: str, patterns: Sequence[re.Pattern[str]]) -> str:
    """Replace every match of every pattern with :data:`REDACT_PLACEHOLDER`.

    Order of patterns is preserved; later patterns operate on the
    already-redacted text so overlapping rules are idempotent.
    """
    if not text:
        return text
    out = text
    for pat in patterns:
        try:
            out = pat.sub(REDACT_PLACEHOLDER, out)
        except re.error:  # pragma: no cover - sub() is forgiving
            continue
    return out


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------
def chunk_id_for_chat(request_id: str) -> str:
    """Deterministic chunk ID for a chat row, derived from ``request_id``.

    The same ``request_id`` always yields the same id so the background
    embedder is idempotent and a re-run cannot duplicate vec_chunks rows.
    Length matches :func:`vllm_mlx.memory.store.chunk_id_for` (32 hex
    chars) so downstream tooling does not need a separate column-width
    contract.
    """
    h = hashlib.sha256()
    h.update(b"chat\x00")
    h.update(str(request_id).encode("utf-8", errors="replace"))
    return h.hexdigest()[:32]


def new_session_id() -> str:
    """Return a fresh UUID4 string for a one-request conversation.

    MVP strategy: every ``/v1/chat/completions`` request gets its own
    session_id. SPEC-MEMORY-02 will replace this with a header-based or
    heuristic-based correlation so multi-request conversations cluster
    in ``chat_summaries``.
    """
    return str(uuid.uuid4())


def now_iso_utc() -> str:
    """ISO-8601 UTC timestamp suitable for ``chat_messages.timestamp``."""
    # ``time.gmtime()`` + strftime keeps stdlib-only and avoids tz quirks.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Payload serialization
# ---------------------------------------------------------------------------
def _stringify_message_content(content: Any) -> str:
    """Convert a message ``content`` field into a single string.

    OpenAI multimodal messages allow ``content`` to be either a string
    or a list of content parts. We flatten to text-only because the
    memory subsystem only indexes text — image parts are dropped.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Only consider text parts; skip image_url/video_url etc.
                t = item.get("type")
                if t in (None, "text"):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
        return "\n".join(p for p in parts if p)
    # Fallback: best-effort str() so persistence never crashes the chat path.
    try:
        return str(content)
    except Exception:  # noqa: BLE001 - last-resort guard
        return ""


def _normalize_messages_for_payload(
    messages: Iterable[Any],
    patterns: Sequence[re.Pattern[str]],
) -> list[dict[str, Any]]:
    """Serialize the request messages list into a redacted JSON-able list.

    Accepts either pydantic models (with ``model_dump``) or plain dicts
    so this helper works for both the streaming and non-streaming paths
    in ``vllm_mlx/server.py`` without coupling to the API model layer.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            try:
                d = msg.model_dump(exclude_none=True)
            except Exception:  # noqa: BLE001 - defensive
                d = {}
        elif isinstance(msg, dict):
            d = dict(msg)
        else:
            # Unknown shape; capture role + str(content) as a best effort.
            d = {
                "role": getattr(msg, "role", "unknown"),
                "content": getattr(msg, "content", ""),
            }

        role = d.get("role", "unknown")
        content_text = _stringify_message_content(d.get("content"))
        redacted = redact_text(content_text, patterns)
        entry: dict[str, Any] = {"role": role, "content": redacted}

        # Preserve tool-call metadata (no content; safe to pass through).
        tc = d.get("tool_calls")
        if tc:
            entry["tool_calls"] = tc
        if d.get("name") is not None:
            entry["name"] = d["name"]
        if d.get("tool_call_id") is not None:
            entry["tool_call_id"] = d["tool_call_id"]

        out.append(entry)
    return out


def serialize_chat_payload(
    messages: Iterable[Any],
    assistant_text: str,
    tool_calls: Any | None,
    patterns: Sequence[re.Pattern[str]],
) -> str:
    """Build the JSON-encoded ``chat_messages.payload`` value.

    The payload contains:
      - ``messages``: redacted request messages (role + content per turn)
      - ``assistant``: redacted assistant_text
      - ``tool_calls``: pass-through of any tool_calls the assistant emitted

    Returns a single JSON string ready for the SQL INSERT. Never raises
    — falls back to ``"{}"`` on serialization failure.
    """
    redacted_assistant = redact_text(assistant_text or "", patterns)
    msg_list = _normalize_messages_for_payload(messages, patterns)

    payload: dict[str, Any] = {
        "messages": msg_list,
        "assistant": redacted_assistant,
    }
    if tool_calls:
        try:
            # Pydantic models -> dump; dicts -> as-is. The values are
            # tool-call structs (no user content) so we do NOT redact.
            tc_serialized: list[Any] = []
            for tc in tool_calls:
                if hasattr(tc, "model_dump"):
                    tc_serialized.append(tc.model_dump(exclude_none=True))
                else:
                    tc_serialized.append(tc)
            payload["tool_calls"] = tc_serialized
        except Exception:  # noqa: BLE001
            # Tool-call serialization is best-effort; never block persist.
            logger.warning("[memory] failed to serialize tool_calls; dropping")

    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        logger.exception("[memory] failed to encode chat payload to JSON")
        return "{}"


def build_searchable_text(
    messages: Iterable[Any],
    assistant_text: str,
    patterns: Sequence[re.Pattern[str]],
) -> str:
    """Produce the BM25/embedding text for a chat row.

    We index the LAST user message + the assistant reply. Earlier turns
    are persisted in the payload for completeness but excluded from the
    search text to keep the BM25 signal-to-noise ratio high (REQ-E4).
    Both halves are redacted before they are joined.
    """
    last_user_text = ""
    for msg in messages:
        role = (
            getattr(msg, "role", None)
            if not isinstance(msg, dict)
            else msg.get("role")
        )
        if role == "user":
            content = (
                getattr(msg, "content", None)
                if not isinstance(msg, dict)
                else msg.get("content")
            )
            last_user_text = _stringify_message_content(content)
            # Keep iterating so we end with the FINAL user turn.

    redacted_user = redact_text(last_user_text, patterns)
    redacted_assistant = redact_text(assistant_text or "", patterns)

    parts: list[str] = []
    if redacted_user:
        parts.append(f"USER: {redacted_user}")
    if redacted_assistant:
        parts.append(f"ASSISTANT: {redacted_assistant}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Persist coroutine (fire-and-forget)
# ---------------------------------------------------------------------------
async def persist_chat_row(
    store: Any,
    *,
    request_id: str,
    session_id: str,
    model: str,
    messages: Iterable[Any],
    assistant_text: str,
    tool_calls: Any | None = None,
    latency_ms: float | None = None,
    redact_patterns: Sequence[re.Pattern[str]] | None = None,
) -> None:
    """Persist one chat-completion turn into the memory store.

    Fire-and-forget contract (REQ-N4):

    * Returns ``None`` on success and ``None`` on every failure.
    * Never raises — all exceptions are caught and logged.
    * Safe to call from ``asyncio.create_task(...)``; no result is
      ever awaited by the caller.
    """
    try:
        if store is None:
            return
        if not request_id:
            logger.debug("[memory] persist_chat_row: empty request_id, skip")
            return

        patterns = (
            list(redact_patterns)
            if redact_patterns is not None
            else compile_redact_patterns(None)
        )

        msg_id = chunk_id_for_chat(request_id)
        ts = now_iso_utc()
        payload_json = serialize_chat_payload(
            messages, assistant_text, tool_calls, patterns
        )
        search_text = build_searchable_text(
            messages, assistant_text, patterns
        )

        # If the search text is empty (e.g. an empty assistant reply with
        # no user content), skip BM25/vector indexing but still record
        # the row in chat_messages so a later forensic dump can find it.
        source_path = f"chat/{session_id}/{request_id}"

        with store.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chat_messages("
                "  message_id, request_id, session_id, role, payload, "
                "  timestamp, tier"
                ") VALUES(?, ?, ?, ?, ?, ?, 'stm')",
                (
                    msg_id,
                    request_id,
                    session_id,
                    "assistant",
                    payload_json,
                    ts,
                ),
            )
            if search_text:
                # Replace any prior FTS row for this chunk_id (idempotent
                # on retries even though message_id is unique today).
                conn.execute(
                    "DELETE FROM fts_chunks WHERE chunk_id = ?",
                    (msg_id,),
                )
                conn.execute(
                    "INSERT INTO fts_chunks("
                    "  chunk_id, source_type, source_path, text"
                    ") VALUES(?, 'chat', ?, ?)",
                    (msg_id, source_path, search_text),
                )

        # Diagnostic log line — operators can grep ``[memory] chat`` to
        # confirm persistence is live without enabling DEBUG.
        logger.info(
            "[memory] chat persisted: request_id=%s session=%s model=%s "
            "latency_ms=%s msg_id=%s",
            request_id,
            session_id[:8] + "..." if len(session_id) > 8 else session_id,
            model,
            f"{latency_ms:.1f}" if isinstance(latency_ms, (int, float)) else "n/a",
            msg_id,
        )
    except Exception:  # noqa: BLE001 - REQ-N4
        logger.exception(
            "[memory] persist_chat_row failed for request_id=%s "
            "(chat path unaffected)",
            request_id,
        )


# ---------------------------------------------------------------------------
# Background embedder loop (runs inside the MCP child process)
# ---------------------------------------------------------------------------
async def chat_embed_loop(
    store: Any,
    embedder: Any | None,
    *,
    interval_seconds: float = 10.0,
    batch_size: int = 32,
) -> None:
    """Periodically embed chat rows whose vectors are missing.

    This is a long-running asyncio task spawned by the MCP child server
    (``vllm_mlx.memory.server._run_mcp_stdio``) BEFORE the stdio loop
    starts. It runs forever and never raises — every batch is wrapped
    so a single bad row cannot kill the loop.

    REQ-E4: with ``interval_seconds=10`` (the default) a chat row is
    embedded and queryable within at most 10 seconds of persistence.

    No-ops when:
      * ``store`` is None,
      * sqlite-vec is unavailable on the connection,
      * ``embedder`` is None or permanently disabled.

    The loop still sleeps in those cases so the caller can swap an
    embedder in later (Phase 4 follow-up) without restarting.
    """
    if store is None:
        return
    interval = max(1.0, float(interval_seconds))
    bs = max(1, int(batch_size))
    logger.info(
        "[memory] chat embed loop starting: interval=%.1fs batch=%d",
        interval,
        bs,
    )

    while True:
        try:
            await _embed_one_batch(store, embedder, batch_size=bs)
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory] chat embed loop tick failed (will retry)"
            )
        try:
            import asyncio

            await asyncio.sleep(interval)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            logger.info("[memory] chat embed loop cancelled")
            raise


async def _embed_one_batch(
    store: Any,
    embedder: Any | None,
    *,
    batch_size: int,
) -> int:
    """Embed up to ``batch_size`` chat rows that lack a vec_chunks entry.

    Returns the number of rows embedded so callers (tests + the loop)
    can decide whether to keep going. When the dense path is unavailable
    the function returns 0 silently so the loop keeps ticking.
    """
    if store is None:
        return 0
    if not getattr(store, "vec_loaded", False):
        return 0
    if embedder is None or embedder.disabled():
        return 0

    rows = _fetch_unembedded_chat_rows(store, limit=batch_size)
    if not rows:
        return 0

    texts = [text for (_chunk_id, text) in rows]
    try:
        blobs = embedder.encode_batch(texts)
    except Exception:  # noqa: BLE001 - REQ-N4
        logger.exception("[memory] chat batch encode failed")
        return 0

    if not blobs or len(blobs) != len(rows):
        # encode_batch returns [] on failure (REQ-O3). Skip this tick.
        return 0

    try:
        with store.transaction():
            store.insert_vectors_batch(
                rows=(
                    (chunk_id, blob)
                    for (chunk_id, _text), blob in zip(rows, blobs, strict=True)
                ),
                source_type="chat",
            )
    except Exception:  # noqa: BLE001 - REQ-N4
        logger.exception(
            "[memory] failed to insert chat vectors batch (size=%d)",
            len(rows),
        )
        return 0

    logger.info(
        "[memory] embedded %d chat row(s) in background tick", len(rows)
    )
    return len(rows)


def _fetch_unembedded_chat_rows(
    store: Any, *, limit: int
) -> list[tuple[str, str]]:
    """Return ``[(chunk_id, search_text)]`` for chat rows missing a vector.

    The search_text is reconstructed from the persisted payload so the
    embedder sees the same input the BM25 path indexed. Rows whose
    payload cannot be parsed are skipped (they remain in chat_messages
    for forensic recall but stay out of the dense index).
    """
    sql = (
        "SELECT cm.message_id, cm.payload "
        "FROM chat_messages cm "
        "WHERE cm.message_id NOT IN (SELECT chunk_id FROM vec_chunks) "
        "ORDER BY cm.timestamp "
        "LIMIT ?"
    )
    out: list[tuple[str, str]] = []
    try:
        rows = store.conn.execute(sql, (int(limit),)).fetchall()
    except Exception:  # noqa: BLE001
        logger.exception("[memory] failed to query unembedded chat rows")
        return []

    for r in rows:
        chunk_id = r["message_id"]
        payload_text = r["payload"] or "{}"
        text = _searchable_from_payload(payload_text)
        if not text:
            # Empty searchable text -> skip dense embedding, but the row
            # stays in chat_messages so a forensic dump can still find it.
            continue
        out.append((chunk_id, text))
    return out


def _searchable_from_payload(payload_text: str) -> str:
    """Reconstruct ``USER: ...\\nASSISTANT: ...`` text from a stored payload.

    The payload was redacted before write so the reconstructed text is
    safe to embed and surface verbatim in tool results.
    """
    try:
        data = json.loads(payload_text)
    except (TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""

    user_text = ""
    for entry in data.get("messages", []) or []:
        if isinstance(entry, dict) and entry.get("role") == "user":
            content = entry.get("content")
            if isinstance(content, str) and content:
                user_text = content  # last user wins, mirroring builder

    assistant_text = data.get("assistant") or ""
    parts: list[str] = []
    if user_text:
        parts.append(f"USER: {user_text}")
    if assistant_text:
        parts.append(f"ASSISTANT: {assistant_text}")
    return "\n\n".join(parts)


__all__ = [
    "DEFAULT_REDACT_PATTERN_STRINGS",
    "REDACT_PLACEHOLDER",
    "build_searchable_text",
    "chat_embed_loop",
    "chunk_id_for_chat",
    "compile_redact_patterns",
    "new_session_id",
    "now_iso_utc",
    "persist_chat_row",
    "redact_text",
    "serialize_chat_payload",
]
