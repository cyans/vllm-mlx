# SPDX-License-Identifier: Apache-2.0
"""Regression guard: the existing ``qwen3`` parser must be unchanged.

SPEC-FIX-QWEN36-RUNTIME Phase 1 introduces a NEW ``qwen36`` parser class.
It must NOT modify the behaviour of the existing ``qwen3`` parser, which
still powers the live Qwen3.5 server. This suite locks down the streaming
and non-streaming output of the ``qwen3`` parser against representative
Qwen3.5-style inputs so any accidental edit to ``think_parser.py`` or
``qwen3_parser.py`` is caught immediately.

@TEST:FIX-QWEN36-RUNTIME/regression
"""

from __future__ import annotations


class TestQwen3StreamingUnchangedForQwen35:
    """Freeze the ``qwen3`` parser's streaming behaviour for Qwen3.5 input."""

    def test_streaming_think_tag_flow(self) -> None:
        # Known-good Qwen3.5 stream: <think> ... </think> ... content.
        # The qwen3 parser must still split these correctly.
        from vllm_mlx.reasoning import get_parser

        parser = get_parser("qwen3")()
        parser.reset_state()

        deltas = ["<think>", "Let", " me", " analyze", "</think>", "Answer: 42"]
        accumulated = ""
        results = []
        for delta in deltas:
            prev = accumulated
            accumulated += delta
            msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
            if msg is not None:
                results.append(msg)

        reasoning_parts = [msg.reasoning for msg in results if msg.reasoning]
        content_parts = [msg.content for msg in results if msg.content]

        assert "".join(reasoning_parts) == "Let me analyze"
        assert "".join(content_parts) == "Answer: 42"

    def test_streaming_implicit_think_mode(self) -> None:
        # OpenCode-style: <think> was injected in the prompt so only
        # </think> appears in the output. Everything before </think> is
        # reasoning, everything after is content.
        from vllm_mlx.reasoning import get_parser

        parser = get_parser("qwen3")()
        parser.reset_state()

        # First chunk arrives before the </think> — under the existing
        # qwen3 parser's behaviour this goes to reasoning (the implicit
        # mode fallback). This is the exact behaviour Qwen3.6 must NOT
        # inherit, but Qwen3.5 still depends on.
        msg = parser.extract_reasoning_streaming("", "reasoning", "reasoning")
        assert msg is not None
        assert msg.reasoning == "reasoning"
        assert msg.content is None

        # Transition chunk: </think> plus content.
        msg2 = parser.extract_reasoning_streaming(
            "reasoning", "reasoning</think>answer", "</think>answer"
        )
        assert msg2 is not None
        assert msg2.content == "answer"

    def test_nonstream_both_tags(self) -> None:
        # Non-streaming parity with the known-good extract_with_both_tags.
        from vllm_mlx.reasoning import get_parser

        parser = get_parser("qwen3")()
        reasoning, content = parser.extract_reasoning(
            "<think>Let me analyze this problem</think>The answer is 42."
        )
        assert reasoning == "Let me analyze this problem"
        assert content == "The answer is 42."

    def test_nonstream_no_tags_is_pure_content(self) -> None:
        # Qwen3.5 non-streaming: no </think> => everything is content.
        # (qwen3_parser.py:60-61 short-circuit.)
        from vllm_mlx.reasoning import get_parser

        parser = get_parser("qwen3")()
        reasoning, content = parser.extract_reasoning("plain response, no tags")
        assert reasoning is None
        assert content == "plain response, no tags"
