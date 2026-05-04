# SPDX-License-Identifier: Apache-2.0
"""Daily chat-session consolidator (STM → LTM).

@CODE:MEMORY-02/consolidator

Phase 1 of SPEC-MEMORY-02: each scheduler tick selects every chat
session whose newest message is older than ``MEMORY_STM_DAYS``, calls
the live engine to produce a JSON summary, writes it to
``chat_summaries``, and atomically evicts the source rows from
``chat_messages`` (plus their FTS / vec entries).

Architectural notes (from SPEC-MEMORY-02 §8 and the implementation
brief):

* Lives in the FastAPI parent process so it can call the engine
  directly via ``await engine.generate(...)``. The MCP child still
  owns the embed loop, the watcher, and the retention sweeper.
* ``summarizer_slot = asyncio.Semaphore(1)`` is acquired by the
  consolidator and released between sessions so the chat path always
  wins under contention. The chat path NEVER acquires this slot.
* Per-tick wall-clock cap (``MEMORY_CONSOLIDATOR_DEADLINE_SEC``) bounds
  any one tick. Sessions not processed roll over to next tick.
* REQ-N4: per-session ``INSERT chat_summaries`` + eviction is one
  transaction. If summary text < ``MEMORY_SUMMARY_MIN_CHARS`` or the
  model output fails JSON parse, the eviction is SKIPPED — the raw
  rows wait for a subsequent retry.
* No outbound network calls (REQ-N6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import MemoryRuntimeConfig
from .store import MemoryStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine protocol (duck-typed so tests can pass a fake)
# ---------------------------------------------------------------------------
class _EngineProto(Protocol):  # pragma: no cover - protocol-only sentinel
    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = ...,
        temperature: float = ...,
        top_p: float = ...,
        **kwargs: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Summarization protocol (SPEC-MEMORY-02 §6)
# ---------------------------------------------------------------------------
SUMMARIZER_SYSTEM_PROMPT = (
    "You are a memory consolidator. Summarize the following conversation "
    "between a user and an assistant into a compact long-term memory "
    "record. Preserve the original language of the user's messages — if "
    "the user wrote in Korean, your summary stays in Korean.\n\n"
    "Output a single JSON object with EXACTLY these keys:\n"
    '  "key_conclusions": array of 1-5 short strings (≤ 200 chars each)\n'
    '  "entities":        array of 0-10 short strings (people, projects, '
    "concepts)\n"
    '  "unresolved":      array of 0-5 short strings (questions left open)\n'
    '  "one_line":        single string ≤ 240 chars\n\n'
    "Do not include any prose outside the JSON. Do not invent facts not "
    "present in the source. If the conversation is empty or trivial, return "
    "all empty arrays and an empty one_line."
)

# Per-call generation parameters from the implementation brief / SPEC §6.
_SUMMARY_MAX_TOKENS = 600
_SUMMARY_TEMPERATURE = 0.3
_SUMMARY_TOP_P = 0.95

# Approximation: ~3 chars per token for mixed Korean+English text. Used to
# truncate the prompt to the configured input budget without pulling in a
# full tokenizer (the consolidator runs in the parent which already owns
# the model — we keep imports minimal and let the engine truncate further
# if needed). The factor is intentionally pessimistic so we never over-
# fill the input budget.
_CHARS_PER_TOKEN = 3

# Per-session retry cap inside one tick. After this many summarization
# attempts the consolidator gives up on the session for the current tick;
# the next tick re-queries and tries again.
_MAX_RETRIES_PER_SESSION = 5


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
@dataclass
class TickStats:
    """One consolidator tick summary, returned for tests + diagnostic logs."""

    sessions_examined: int = 0
    sessions_summarized: int = 0
    sessions_evicted: int = 0
    sessions_skipped_too_short: int = 0
    sessions_skipped_parse_error: int = 0
    sessions_skipped_engine_error: int = 0
    sessions_skipped_db_error: int = 0
    deadline_reached: bool = False
    skipped_session_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Consolidator
# ---------------------------------------------------------------------------
class ChatConsolidator:
    """Owns one consolidation tick — STM→LTM migration of chat sessions.

    Construction does no I/O; pass an open :class:`MemoryStore`, the
    live FastAPI engine reference, and the resolved
    :class:`MemoryRuntimeConfig`. Call :meth:`run_once` from the
    scheduler callback — the consolidator handles batching, the
    summarizer slot, and the per-tick deadline internally.
    """

    def __init__(
        self,
        store: MemoryStore,
        engine: _EngineProto | None,
        config: MemoryRuntimeConfig,
        *,
        summarizer_slot: asyncio.Semaphore | None = None,
        clock: Any | None = None,
        redact_patterns: Sequence[re.Pattern[str]] | None = None,
    ):
        self.store = store
        self.engine = engine
        self.config = config
        # Single global slot enforces "one summarization at a time" so
        # even concurrent ticks (in tests) cannot stack two model calls.
        # Production wiring passes a process-global Semaphore(1).
        self._slot = summarizer_slot or asyncio.Semaphore(1)
        self._clock = clock or time.monotonic
        # Redaction patterns: caller may pass an explicit list (e.g. from
        # ``compile_redact_patterns(config.redact_patterns)``); when None
        # the default patterns from chatlog.py are used.
        self._redact_patterns = list(redact_patterns) if redact_patterns is not None else None

    # ----------------------------------------------------------------- API
    async def run_once(self) -> TickStats:
        """Run one consolidation tick.

        Returns a :class:`TickStats` describing what happened. Always
        returns — never raises out (REQ-N4) so the scheduler loop
        survives any single failure.
        """
        stats = TickStats()
        if self.config.consolidator_dryrun:
            logger.info("[memory-02] DRYRUN mode: no DB writes will occur")

        cutoff = _stm_cutoff_iso(stm_days=self.config.stm_days)
        deadline = self._clock() + float(self.config.consolidator_deadline_sec)

        try:
            attempted: set[str] = set()
            while True:
                if self._clock() >= deadline:
                    stats.deadline_reached = True
                    logger.info(
                        "[memory-02] deadline reached; rolling over to next "
                        "tick (examined=%d summarized=%d)",
                        stats.sessions_examined,
                        stats.sessions_summarized,
                    )
                    return stats

                try:
                    session_ids = self.store.select_sessions_due_for_consolidation(
                        cutoff_iso=cutoff,
                        limit=int(self.config.consolidator_batch),
                    )
                except Exception:  # noqa: BLE001 - REQ-N4
                    logger.exception(
                        "[memory-02] failed to select due sessions; "
                        "ending tick"
                    )
                    return stats

                if not session_ids:
                    return stats

                # Filter out sessions already attempted in this tick to
                # prevent re-selecting the same failed sessions.
                fresh = [sid for sid in session_ids if sid not in attempted]
                if not fresh:
                    logger.info(
                        "[memory-02] no fresh sessions in batch; ending tick"
                    )
                    return stats

                # Process each fresh session in this batch. Drop out of the
                # batch the moment we cross the deadline so we don't
                # leave a half-processed batch behind.
                for sid in fresh:
                    if self._clock() >= deadline:
                        stats.deadline_reached = True
                        logger.info(
                            "[memory-02] deadline reached mid-batch"
                        )
                        return stats
                    attempted.add(sid)
                    stats.sessions_examined += 1
                    await self._consolidate_one_session(sid, stats)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory-02] unexpected consolidator failure; ending tick"
            )
            return stats

    # --------------------------------------------------------- per-session
    async def _consolidate_one_session(
        self, session_id: str, stats: TickStats
    ) -> bool:
        """Summarize + evict one session. Returns True iff anything changed.

        Returns False when the session was skipped (engine error, parse
        error, summary too short, or DB error). The caller uses this
        signal to break out of a non-progressing batch.
        """
        try:
            messages = self.store.fetch_chat_session_messages(session_id)
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory-02] failed to fetch messages for session=%s; skipping",
                session_id,
            )
            stats.sessions_skipped_db_error += 1
            stats.skipped_session_ids.append(session_id)
            return False

        if not messages:
            # Session vanished between the SELECT and now. Treat as
            # progress so the loop continues.
            return True

        # Build prompt + period metadata.
        period_start = messages[0]["timestamp"]
        period_end = messages[-1]["timestamp"]
        source_count = len(messages)

        prompt = _build_summarizer_prompt(
            messages,
            input_budget_tokens=int(self.config.summary_input_budget),
        )

        # Engine call gated by the global summarizer slot. We take the
        # slot, run the call, release between sessions so the chat path
        # always wins under contention (REQ-S2).
        summary_text: str | None = None
        for attempt in range(_MAX_RETRIES_PER_SESSION):
            try:
                async with self._slot:
                    raw_output = await self._call_engine(prompt)
            except asyncio.CancelledError:  # pragma: no cover - shutdown
                raise
            except Exception:  # noqa: BLE001 - REQ-N4
                logger.exception(
                    "[memory-02] engine call failed for session=%s "
                    "(attempt %d/%d); skipping",
                    session_id,
                    attempt + 1,
                    _MAX_RETRIES_PER_SESSION,
                )
                stats.sessions_skipped_engine_error += 1
                stats.skipped_session_ids.append(session_id)
                return False

            parsed = _parse_summary_json(raw_output)
            if parsed is None:
                logger.warning(
                    "[memory-02] summary JSON parse failed for "
                    "session=%s (attempt %d/%d); will retry next tick",
                    session_id,
                    attempt + 1,
                    _MAX_RETRIES_PER_SESSION,
                )
                # JSON parse errors are deterministic for a given prompt;
                # retrying inside the same tick is wasteful. Skip and let
                # the next tick re-queue.
                stats.sessions_skipped_parse_error += 1
                stats.skipped_session_ids.append(session_id)
                return False

            summary_text = _render_summary_text(parsed)
            # Apply redaction patterns to the rendered summary text so a
            # model that absorbed an API key into its output cannot leak
            # it (REQ-N5 of SPEC-MEMORY-02).
            summary_text = self._apply_redaction(summary_text)
            break

        if summary_text is None:
            stats.sessions_skipped_engine_error += 1
            stats.skipped_session_ids.append(session_id)
            return False

        if len(summary_text) < int(self.config.summary_min_chars):
            logger.warning(
                "[memory-02] summary too short for session=%s "
                "(len=%d, min=%d); raw rows preserved",
                session_id,
                len(summary_text),
                int(self.config.summary_min_chars),
            )
            stats.sessions_skipped_too_short += 1
            stats.skipped_session_ids.append(session_id)
            return False

        if self.config.consolidator_dryrun:
            logger.info(
                "[memory-02] DRYRUN session=%s would write summary "
                "(len=%d) and evict %d rows; period=[%s, %s]",
                session_id,
                len(summary_text),
                source_count,
                period_start,
                period_end,
            )
            stats.sessions_summarized += 1
            return True

        # Atomic write: INSERT chat_summaries + DELETE chat_messages
        # for this session_id in one transaction. On any error the
        # transaction rolls back, leaving both tables intact for the
        # next tick to retry (REQ-N4).
        try:
            with self.store.transaction():
                self.store.insert_chat_summary(
                    session_id=session_id,
                    summary_text=summary_text,
                    period_start=period_start,
                    period_end=period_end,
                    source_count=source_count,
                )
                evicted = self.store.evict_chat_messages_for_session(
                    session_id
                )
        except Exception:  # noqa: BLE001 - REQ-N4
            logger.exception(
                "[memory-02] atomic summary+evict failed for session=%s; "
                "transaction rolled back, will retry next tick",
                session_id,
            )
            stats.sessions_skipped_db_error += 1
            stats.skipped_session_ids.append(session_id)
            return False

        stats.sessions_summarized += 1
        stats.sessions_evicted += evicted
        logger.info(
            "[memory-02] consolidated session=%s: summary_len=%d "
            "evicted=%d period=[%s, %s]",
            session_id,
            len(summary_text),
            evicted,
            period_start,
            period_end,
        )
        return True

    # ----------------------------------------------------------- engine
    async def _call_engine(self, prompt: str) -> str:
        """Call the live engine and return the raw output text.

        Resolution rules:

        * ``MEMORY_SUMMARIZER_MODEL=self`` (the default) uses the live
          engine reference passed at construction. When ``self.engine``
          is None, raises RuntimeError so the caller's try/except can
          mark the session skipped.
        * Any other value is reserved for the lazy-load path in a
          future phase; for now it raises NotImplementedError so an
          operator typo cannot silently disable consolidation.
        """
        model_choice = (self.config.summarizer_model or "self").strip().lower()
        if model_choice != "self":
            raise NotImplementedError(
                f"MEMORY_SUMMARIZER_MODEL={model_choice!r} requires the "
                f"Phase 4 lazy-load path; only 'self' is supported in "
                f"SPEC-MEMORY-02 Phase 1"
            )
        if self.engine is None:
            raise RuntimeError(
                "consolidator has no engine reference; cannot summarize"
            )

        out = await self.engine.generate(
            prompt,
            max_tokens=_SUMMARY_MAX_TOKENS,
            temperature=_SUMMARY_TEMPERATURE,
            top_p=_SUMMARY_TOP_P,
        )
        # The engine returns either a ``GenerationOutput`` with ``.text``
        # or a plain string (test fakes). Both are handled here.
        text = getattr(out, "text", None)
        if text is None:
            text = str(out) if out is not None else ""
        return text

    # ----------------------------------------------------------- redact
    def _apply_redaction(self, text: str) -> str:
        """Run the configured redact patterns over ``text`` (REQ-N5)."""
        if self._redact_patterns is None:
            from .chatlog import compile_redact_patterns  # noqa: PLC0415

            self._redact_patterns = compile_redact_patterns(
                self.config.redact_patterns or None
            )
        if not self._redact_patterns:
            return text
        out = text
        for pat in self._redact_patterns:
            out = pat.sub("[REDACTED]", out)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stm_cutoff_iso(*, stm_days: int) -> str:
    """ISO-8601 UTC string ``stm_days`` days in the past.

    Matches the format produced by ``vllm_mlx.memory.chatlog.now_iso_utc``
    so string comparisons against ``chat_messages.timestamp`` are
    well-defined.
    """
    epoch = time.time() - (float(max(0, int(stm_days))) * 86400.0)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _build_summarizer_prompt(
    messages: Sequence[dict[str, Any]],
    *,
    input_budget_tokens: int,
) -> str:
    """Concatenate session messages into one summarizer prompt.

    The format is:

    .. code-block:: text

        <system prompt>

        [user]: ...
        [assistant]: ...

    Conservatively truncated to ``input_budget_tokens * 3`` characters.
    Truncation drops the OLDEST messages first so the "newest summary"
    captures the most recent state of the conversation.
    """
    rendered_lines: list[str] = []
    # Walk in reverse so we can stop appending once we've consumed the
    # budget, then reverse back to chronological order for the prompt.
    char_budget = max(1024, int(input_budget_tokens) * _CHARS_PER_TOKEN)
    used = 0
    kept: list[str] = []
    for entry in reversed(messages):
        line = _render_one_message_line(entry)
        # Always include at least the very last message even if oversized.
        if kept and used + len(line) > char_budget:
            break
        kept.append(line)
        used += len(line)
    rendered_lines = list(reversed(kept))
    body = "\n\n---\n\n".join(rendered_lines)
    return SUMMARIZER_SYSTEM_PROMPT + "\n\n" + body


def _render_one_message_line(entry: dict[str, Any]) -> str:
    """Render one chat row into a ``[role]: content`` line.

    The chatlog stores the assistant role with a JSON payload that
    contains both the request messages and the assistant text. We
    flatten the payload back into a series of lines so the summarizer
    sees the actual conversation rather than serialized JSON.
    """
    payload_raw = entry.get("payload") or "{}"
    try:
        payload = json.loads(payload_raw)
    except Exception:  # noqa: BLE001 - tolerate corrupt payloads
        payload = {}

    parts: list[str] = []
    msgs = payload.get("messages") or []
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "user")
            content = m.get("content")
            if isinstance(content, list):
                # Multimodal content list — flatten text segments only.
                text_parts = []
                for seg in content:
                    if isinstance(seg, dict) and seg.get("type") == "text":
                        text_parts.append(str(seg.get("text") or ""))
                content = "\n".join(text_parts)
            content_str = str(content or "").strip()
            if content_str:
                parts.append(f"[{role}]: {content_str}")

    assistant_text = str(payload.get("assistant") or "").strip()
    if assistant_text:
        parts.append(f"[assistant]: {assistant_text}")

    if not parts:
        # Fallback: serialize the raw payload so the summarizer can at
        # least see something non-empty.
        parts.append(f"[{entry.get('role', 'system')}]: {payload_raw[:500]}")

    return "\n".join(parts)


def _parse_summary_json(raw: str) -> dict[str, Any] | None:
    """Parse the model's JSON-only output, returning None on any error.

    Strategy:

    1. Strip leading / trailing whitespace.
    2. If the model wrapped the JSON in a ```json fence``` block, strip
       the fence.
    3. ``json.loads`` the remainder.
    4. Validate that it is a dict with the four expected keys.

    Returns the parsed dict on success, ``None`` on any failure. The
    caller logs and skips the session — no exceptions escape.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    # Strip ```json ... ``` fences if present.
    if cleaned.startswith("```"):
        # Remove the opening fence (with optional language tag).
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Some models emit text BEFORE the JSON body; extract the first
    # balanced object as a best-effort.
    if not cleaned.startswith("{"):
        idx = cleaned.find("{")
        if idx == -1:
            return None
        cleaned = cleaned[idx:]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    # Coerce missing keys to safe defaults so downstream rendering can
    # rely on the four-key shape. Type errors (wrong type for a key)
    # are reported as parse failures so the consolidator does not
    # silently drop information.
    expected = {
        "key_conclusions": list,
        "entities": list,
        "unresolved": list,
        "one_line": str,
    }
    out: dict[str, Any] = {}
    for key, kind in expected.items():
        val = parsed.get(key)
        if val is None:
            out[key] = "" if kind is str else []
            continue
        if not isinstance(val, kind):
            return None
        out[key] = val
    return out


def _render_summary_text(parsed: dict[str, Any]) -> str:
    """Flatten the parsed JSON summary into one searchable text blob.

    Format:

    .. code-block:: text

        <one_line>

        Key conclusions:
        - ...
        - ...

        Entities: a, b, c

        Unresolved:
        - ...

    Empty sections are omitted. The result is the value stored in
    ``chat_summaries.summary_text`` and is what the future
    ``memory_search`` will return as the excerpt.
    """
    parts: list[str] = []
    one_line = str(parsed.get("one_line") or "").strip()
    if one_line:
        parts.append(one_line)

    conclusions = parsed.get("key_conclusions") or []
    if isinstance(conclusions, list) and conclusions:
        bullet_lines = [
            f"- {str(item).strip()}"
            for item in conclusions
            if str(item).strip()
        ]
        if bullet_lines:
            parts.append("Key conclusions:\n" + "\n".join(bullet_lines))

    entities = parsed.get("entities") or []
    if isinstance(entities, list):
        ents = [str(e).strip() for e in entities if str(e).strip()]
        if ents:
            parts.append("Entities: " + ", ".join(ents))

    unresolved = parsed.get("unresolved") or []
    if isinstance(unresolved, list) and unresolved:
        bullet_lines = [
            f"- {str(item).strip()}"
            for item in unresolved
            if str(item).strip()
        ]
        if bullet_lines:
            parts.append("Unresolved:\n" + "\n".join(bullet_lines))

    return "\n\n".join(parts)


__all__ = [
    "ChatConsolidator",
    "SUMMARIZER_SYSTEM_PROMPT",
    "TickStats",
]
