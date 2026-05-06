# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3.6 reasoning parser.

The Qwen3 chat template auto-prefills ``<think>\\n`` for the assistant
turn, so the model's streaming output begins *inside* the thinking
section without ever emitting an opening ``<think>`` tag itself. The
parser must therefore default to routing streaming tokens to
``reasoning`` (implicit-think mode) until ``</think>`` is observed; the
chunk containing ``</think>`` splits — pre-tag to reasoning, post-tag
to content — and any subsequent deltas pass through to content.

The non-streaming :meth:`extract_reasoning` keeps the legacy
short-circuit ("no ``</think>`` ⇒ pure content") because the full
output is visible in one shot and a Qwen3.6 reply that never produces
a close tag is by construction a non-thinking reply.

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


class TestQwen36StreamingDefaultsToReasoningInImplicitThinkMode:
    """Primary contract: with no ``</think>`` in the stream, every delta
    routes to ``reasoning`` (implicit-think mode)."""

    def test_streaming_defaults_to_reasoning_in_implicit_think_mode(self) -> None:
        # REQ-U3: the Qwen3 chat template prefills ``<think>\n``, so the
        # streaming output begins inside the thinking section. Until
        # ``</think>`` is observed each DeltaMessage must carry
        # ``reasoning`` (not ``content``). If ``</think>`` never arrives,
        # everything stays in reasoning — the streaming path cannot
        # decide retroactively that the reply was non-thinking.
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

        # Every chunk must be reasoning, never content.
        for msg, delta in zip(results, deltas, strict=True):
            assert msg.reasoning == delta, (
                f"expected reasoning={delta!r}, got reasoning={msg.reasoning!r}"
            )
            assert msg.content in (None, ""), (
                f"Qwen3.6 implicit-think output must not be routed to "
                f"content; got content={msg.content!r} for delta={delta!r}"
            )

        # Full reconstruction must match input.
        reconstructed = "".join(msg.reasoning or "" for msg in results)
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
        # The Qwen3 chat template prefilled ``<think>\n``, so until
        # ``</think>`` is observed deltas route to reasoning. The
        # transition chunk that contains ``</think>`` splits — pre-tag
        # text to reasoning, post-tag text to content — matching the
        # non-streaming :meth:`extract_reasoning` behaviour.
        from vllm_mlx.reasoning import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        parser.reset_state()

        # First chunk: no </think> yet — goes to reasoning (implicit
        # think mode).
        msg1 = parser.extract_reasoning_streaming("", "reasoning", "reasoning")
        assert msg1 is not None
        assert msg1.reasoning == "reasoning"
        assert msg1.content in (None, "")

        # Second chunk: contains </think> + content. The implicit handler
        # routes the pre-</think> text to reasoning and post-</think> to
        # content.
        msg2 = parser.extract_reasoning_streaming(
            "reasoning", "reasoning</think>answer", "</think>answer"
        )
        assert msg2 is not None
        assert msg2.content == "answer"
