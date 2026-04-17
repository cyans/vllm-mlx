# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.6 reasoning parser.

Qwen3.6 is an instruct-tuned model that does NOT emit ``<think>`` / ``</think>``
tags around its output by default. The parser must therefore default to routing
streaming tokens to ``content`` unless thinking tags are explicitly observed.

Regression target: ``think_parser.py:140`` (Case 3 fallback) used to
unconditionally route every pre-``</think>`` streaming token to ``reasoning``,
which meant Qwen3.6 output was entirely invisible to tool-calling clients.

@TEST:FIX-QWEN36-RUNTIME/parser
"""

from __future__ import annotations


class TestQwen36ParserRegistered:
    """The ``qwen36`` parser must be importable and registered."""

    def test_qwen36_parser_exists(self) -> None:
        # REQ-U1: a dedicated Qwen3.6 parser class must exist and be
        # importable via the public reasoning package.
        from vllm_mlx.reasoning import Qwen36ReasoningParser  # noqa: F401

    def test_qwen36_parser_registered(self) -> None:
        # REQ-U2: the parser must be registered under the "qwen36" key so
        # that ``--reasoning-parser qwen36`` succeeds at server boot.
        from vllm_mlx.reasoning import get_parser, list_parsers

        assert "qwen36" in list_parsers()
        parser_cls = get_parser("qwen36")
        instance = parser_cls()
        # Sanity: the registered class is the public one.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        assert isinstance(instance, Qwen36ReasoningParser)


class TestQwen36StreamingDefaultsToContent:
    """Primary bug-fix assertion: plain-text streaming routes to content."""

    def test_streaming_defaults_to_content(self) -> None:
        # REQ-U3 (core fix): when Qwen3.6 streams plain text tokens with no
        # ``<think>`` / ``</think>`` tags present, each DeltaMessage must
        # carry ``content`` (not ``reasoning``). Prior to the fix the
        # BaseThinkingReasoningParser Case 3 fallback routed everything to
        # reasoning, hiding the output from non-reasoning-aware clients.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        parser.reset_state()

        deltas = ["Here's", " a", " thinking", " process", ": 2+2=4"]
        accumulated = ""
        results = []
        for delta in deltas:
            prev = accumulated
            accumulated += delta
            msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
            assert msg is not None, f"parser dropped delta {delta!r}"
            results.append(msg)

        # Every chunk must be content, never reasoning.
        for msg, delta in zip(results, deltas, strict=True):
            assert msg.content == delta, (
                f"expected content={delta!r}, got content={msg.content!r}"
            )
            assert msg.reasoning is None, (
                f"Qwen3.6 plain text must not be routed to reasoning; "
                f"got reasoning={msg.reasoning!r} for delta={delta!r}"
            )

        # Full reconstruction must match input.
        reconstructed = "".join(msg.content or "" for msg in results)
        assert reconstructed == "".join(deltas)


class TestQwen36StreamingRespectsThinkTags:
    """Forward-compat guard: if Qwen3.6 ever emits explicit think tags,
    the parser must still route content between tags to reasoning.
    """

    def test_streaming_with_think_tags(self) -> None:
        # REQ-U4: defensive protection. Even though Qwen3.6 is not expected
        # to emit ``<think>`` tags, if they ever appear the parser must
        # honour them so a future firmware update cannot silently regress
        # into a "no reasoning extraction at all" state.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        parser.reset_state()

        deltas = ["<think>", "analyze", "</think>", "answer"]
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

        assert "".join(reasoning_parts) == "analyze"
        assert "".join(content_parts) == "answer"


class TestQwen36NonStreaming:
    """Non-streaming extract_reasoning behaviour."""

    def test_nonstream_plain_text_all_content(self) -> None:
        # REQ-U5: when there are no think tags at all, ``extract_reasoning``
        # must return ``(None, full_text)`` — reasoning stays empty, the
        # whole output is content. Mirrors the Qwen3.5 parser's
        # ``end_token not in output`` short-circuit at qwen3_parser.py:60.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        text = "Here's a thinking process: 2+2=4"
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_nonstream_with_think_tags(self) -> None:
        # REQ-U4 mirror for non-streaming: explicit tags must still split.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        reasoning, content = parser.extract_reasoning("<think>a</think>b")
        assert reasoning == "a"
        assert content == "b"

    def test_nonstream_only_closing_tag_implicit_mode(self) -> None:
        # Forward-compat with OpenCode-style ``<think>`` injection in the
        # prompt: only ``</think>`` appears in the output. Everything
        # before it is reasoning. Parity with Qwen3.5 parser.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        reasoning, content = parser.extract_reasoning("reasoning</think>answer")
        assert reasoning == "reasoning"
        assert content == "answer"


class TestQwen36StreamingImplicitMode:
    """Forward-compat: streaming with ``<think>`` injected in the prompt
    so only ``</think>`` appears in the stream.
    """

    def test_streaming_implicit_think_mode(self) -> None:
        # Matches the Qwen3.5 OpenCode-compat behaviour: once ``</think>``
        # appears in ``current_text``, content after it routes to the
        # content channel. The chunk BEFORE ``</think>`` still routes to
        # content for Qwen3.6 (our Case 3 divergence) — then the
        # transition chunk splits correctly via the implicit handler.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        parser.reset_state()

        # First chunk: no tags yet — goes to content (Qwen3.6 default).
        msg1 = parser.extract_reasoning_streaming("", "reasoning", "reasoning")
        assert msg1 is not None
        assert msg1.content == "reasoning"
        assert msg1.reasoning is None

        # Second chunk: contains </think> + content. The implicit handler
        # routes the pre-</think> text to reasoning and post-</think> to
        # content, matching the Qwen3.5 parser behaviour.
        msg2 = parser.extract_reasoning_streaming(
            "reasoning", "reasoning</think>answer", "</think>answer"
        )
        assert msg2 is not None
        assert msg2.content == "answer"
