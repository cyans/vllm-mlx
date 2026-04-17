# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Qwen3.6 models.

Qwen3.6 is an instruct-tuned model that normally does NOT emit
``<think>`` / ``</think>`` tags around its output. The shared
:class:`~vllm_mlx.reasoning.think_parser.BaseThinkingReasoningParser`
defaults to routing any streaming delta that arrives *before* a
``</think>`` is seen to the ``reasoning`` channel — which for Qwen3.6
means every token is invisible to non-reasoning-aware clients (the bug
documented in ``.moai/reports/qwen36-validation-2026-04/report.md``).

This parser flips that default: in the absence of observed think tags,
streaming deltas are routed to ``content``. When tags *are* present
(forward-compat protection plus the OpenCode-style prompt-injected
``<think>`` case) the parser delegates to the base class so behaviour
matches the Qwen3.5 parser.

@CODE:FIX-QWEN36-RUNTIME/parser
"""

from __future__ import annotations

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class Qwen36ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Qwen3.6 models.

    Differs from :class:`~vllm_mlx.reasoning.qwen3_parser.Qwen3ReasoningParser`
    in one critical way: when no think tags are observed yet during
    streaming, output is treated as **content** rather than **reasoning**.

    Rationale: Qwen3.6 instruct-tuned output carries its answer directly,
    without wrapping it in ``<think>...</think>``. Treating such output as
    reasoning (the base class's Case 3 default) caused the entire response
    to be suppressed for tool-calling clients in production.

    Compatibility with the base parser is preserved for the three explicit
    tag scenarios:

    1. Both tags present: ``<think>reasoning</think>content`` → split.
    2. Only closing tag (``<think>`` injected in prompt): everything before
       ``</think>`` is reasoning, everything after is content.
    3. Neither tag present: **entire output is content** (this method's
       behaviour change relative to the base class).
    """

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

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
        """Extract reasoning from streaming delta.

        Behaviour matrix:

        * ``<think>`` seen in accumulated text → delegate to base class's
          explicit-think handler (content inside tags becomes reasoning,
          content after ``</think>`` becomes content).
        * Only ``</think>`` seen (no ``<think>``) → base class's implicit
          handler (mirrors Qwen3.5's OpenCode compat).
        * Neither tag seen → route delta to ``content`` (this is the bug
          fix relative to :class:`BaseThinkingReasoningParser`).
        """
        # Swallow the tag tokens themselves, matching the base class.
        stripped_delta = delta_text.strip()
        if stripped_delta == self.start_token:
            return None
        if stripped_delta == self.end_token:
            return None

        start_in_prev = self.start_token in previous_text
        start_in_current = self.start_token in current_text
        end_in_prev = self.end_token in previous_text
        end_in_delta = self.end_token in delta_text

        # Case 1: explicit <think> — delegate to the shared handler so
        # any future refinement to think-tag semantics stays in one place.
        if start_in_current:
            return self._handle_explicit_think(
                previous_text,
                delta_text,
                start_in_prev,
                end_in_prev,
                end_in_delta,
            )

        # Case 2: implicit mode (<think> was in the prompt, only </think>
        # appears in the output). Same semantics as the Qwen3.5 parser.
        if self.end_token in current_text:
            return self._handle_implicit_think(
                delta_text, end_in_prev, end_in_delta
            )

        # Case 3: neither tag seen. This is the Qwen3.6 default path — the
        # response is plain content, not reasoning. This is the
        # intentional divergence from the BaseThinkingReasoningParser
        # Case 3 default (which routes to reasoning).
        return DeltaMessage(content=delta_text)
