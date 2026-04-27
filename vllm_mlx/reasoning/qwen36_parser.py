# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Qwen3.6 models.

Qwen3.6 ships with a chat template that auto-prefills ``<think>\n`` for
the assistant turn, so the model's streaming output begins *inside*
the thinking section without ever emitting an opening ``<think>`` tag
itself. The shared
:class:`~vllm_mlx.reasoning.think_parser.BaseThinkingReasoningParser`
implicit-think handler models this correctly only when the closing
``</think>`` already lives in the *current chunk* — when the closing
tag arrives in a later chunk, every preceding chunk has already been
emitted to the wrong channel. The bug surfaced in production as
reasoning content leaking into the user-visible answer (and, in the
boundary case, fragments of ``</think>`` itself leaking through).

This parser tracks a small amount of streaming state (``mode`` plus a
``_pending_tail`` buffer holding at most ``len('</think>')-1`` bytes)
so that:

* deltas arriving before the close tag are routed to ``reasoning``;
* the chunk containing ``</think>`` is split — pre-tag → reasoning,
  post-tag → content — exactly as the non-streaming
  :meth:`extract_reasoning` would have done;
* a ``</think>`` fragmented across chunk boundaries is held back until
  it can be classified, so neither half of the partial tag escapes
  into either channel.

@CODE:FIX-QWEN36-RUNTIME/parser
"""

from __future__ import annotations

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class Qwen36ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Qwen3.6 models.

    Streaming contract: until ``</think>`` is observed, deltas are
    treated as the reasoning section the chat template prefilled
    ``<think>\\n`` for. Once ``</think>`` is seen, the parser switches
    to content mode for the remainder of the request.

    The non-streaming :meth:`extract_reasoning` keeps the
    legacy short-circuit ("no ``</think>`` ⇒ pure content") because
    the full output is visible in one shot and a Qwen3.6 reply that
    never produces a close tag is by construction a non-thinking
    reply. The streaming path cannot make that determination ahead of
    time, so it must default to reasoning to avoid the leak.
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        # Per-request streaming state. ``_mode`` is "implicit_think"
        # while we are still inside the prefilled <think> section;
        # flips to "content" once </think> is observed. ``_pending_tail``
        # buffers a suffix that *might* be the start of "</think>" so
        # we never emit a partial close tag.
        self._mode: str = "implicit_think"
        self._pending_tail: str = ""

    def reset_state(self) -> None:
        """Reset per-request streaming state.

        Called by the server before each new streaming request.
        """
        self._mode = "implicit_think"
        self._pending_tail = ""

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from complete Qwen3.6 output.

        Identical to the Qwen3.5 parser's short-circuit: if ``</think>``
        never appears, the full output is content and there is no
        reasoning to extract. Parity with
        :meth:`vllm_mlx.reasoning.qwen3_parser.Qwen3ReasoningParser.extract_reasoning`.
        """
        if self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Extract reasoning from a streaming delta.

        Three behaviour paths:

        * Explicit ``<think>`` somewhere in ``current_text`` →
          delegate to the base class so the standard
          ``<think>...</think>`` handler runs (forward-compat with any
          future Qwen3.6 firmware that emits real opening tags).
        * Already in ``content`` mode (``</think>`` was observed in a
          previous chunk) → pass the delta through to ``content``.
        * Otherwise → implicit-think mode: route to ``reasoning``,
          watching for an embedded or split ``</think>`` to trigger
          the transition into content mode.
        """
        # Swallow tag tokens that arrive as their own deltas, matching
        # the base class behaviour.
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            return None
        if stripped_delta == self.end_token:
            # Tag-only delta acts as the transition point itself; the
            # caller's accumulated text already contains </think>, so
            # any subsequent chunk will route to content correctly.
            self._mode = "content"
            self._pending_tail = ""
            return None

        # Forward-compat: explicit <think> tag observed. Delegate to
        # the shared explicit-think handler so any future refinement
        # of think-tag semantics stays in one place. (Qwen3.6 itself
        # is not expected to take this branch — its chat template
        # prefills the opening tag and never emits one in the stream.)
        if self.start_token in current_text:
            start_in_prev = self.start_token in previous_text
            end_in_prev = self.end_token in previous_text
            end_in_delta = self.end_token in delta_text
            return self._handle_explicit_think(
                previous_text,
                delta_text,
                start_in_prev,
                end_in_prev,
                end_in_delta,
            )

        # Already past the close tag — straightforward passthrough.
        if self._mode == "content":
            return DeltaMessage(content=delta_text) if delta_text else None

        # Implicit-think mode: the Qwen3.6 chat template prefilled
        # <think>\n, so we are inside the reasoning section by default
        # until </think> appears.
        return self._handle_implicit_stream(delta_text)

    def _handle_implicit_stream(
        self, delta_text: str,
    ) -> DeltaMessage | None:
        """Process a delta while implicitly inside the prefilled <think>.

        Maintains ``self._pending_tail`` so a ``</think>`` straddling
        two chunks is never emitted as fragments.
        """
        # Anything held back from the previous chunk belongs at the
        # head of this one for tag-detection purposes.
        buffered = self._pending_tail + delta_text
        self._pending_tail = ""

        end = self.end_token
        idx = buffered.find(end)
        if idx != -1:
            # Transition chunk: split around </think>.
            reasoning_part = buffered[:idx]
            content_part = buffered[idx + len(end):]
            self._mode = "content"
            return DeltaMessage(
                reasoning=reasoning_part if reasoning_part else None,
                content=content_part if content_part else None,
            )

        # No complete </think> yet, but the tail of `buffered` might
        # be a prefix of </think> that completes in the next chunk.
        # The longest prefix to suspect is len(end)-1 == 7 bytes.
        suspect_max = len(end) - 1
        hold = 0
        for k in range(min(suspect_max, len(buffered)), 0, -1):
            if end.startswith(buffered[-k:]):
                hold = k
                break

        if hold:
            self._pending_tail = buffered[-hold:]
            safe = buffered[:-hold]
        else:
            safe = buffered

        return DeltaMessage(reasoning=safe) if safe else None
