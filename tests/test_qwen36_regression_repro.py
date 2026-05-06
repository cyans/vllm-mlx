# SPDX-License-Identifier: Apache-2.0
"""Reproduction tests for two diagnosed regressions.

This file is intentionally test-only and verification-only. It does NOT
attempt to fix anything. Each test is constructed to FAIL if the
hypothesis from the prior expert-debug analysis is correct, so that we
have unambiguous evidence to drive a subsequent fix.

Bug 1 - Reasoning leak (streaming asymmetry):
    File: vllm_mlx/reasoning/qwen36_parser.py:97-124
    When streaming and ``</think>`` arrives in a *later* chunk, the
    thinking content emitted in *earlier* chunks has already been routed
    to the ``content`` channel. The parser does not retro-classify
    previously-emitted content as ``reasoning`` once the close tag
    arrives. This leaks parts of <think>...</think> text into final
    answer content. The non-streaming path (``extract_reasoning``) is
    correct because it sees the full text at once.

    Note: The Qwen3 chat template auto-prefills ``<think>\\n``, so model
    output starts WITHOUT an opening ``<think>`` tag. This is the
    "implicit think" path that triggers the bug.

Bug 2 - Auto-tool-choice corrupts plain-text streaming:
    Files: start-server.sh:45-48, vllm_mlx/tool_parsers/hermes_tool_parser.py
    The launcher unconditionally enables ``--enable-auto-tool-choice``,
    which chains the tool parser into the streaming path even for
    conversations with no tool-call intent. The hypothesis is that
    ``extract_tool_calls_streaming`` may suppress or alter regular text
    deltas, causing the user-visible answer to be truncated or
    incoherent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _stream_through_reasoning_parser(parser, deltas):
    """Drive a reasoning parser with a list of streaming deltas.

    Returns a list of (delta_input, DeltaMessage_or_None) pairs and the
    concatenated reasoning / content channels (skipping None outputs).
    """
    accumulated = ""
    pairs = []
    reasoning_parts = []
    content_parts = []
    for delta in deltas:
        prev = accumulated
        accumulated += delta
        msg = parser.extract_reasoning_streaming(prev, accumulated, delta)
        pairs.append((delta, msg))
        if msg is None:
            continue
        if msg.reasoning:
            reasoning_parts.append(msg.reasoning)
        if msg.content:
            content_parts.append(msg.content)
    return pairs, "".join(reasoning_parts), "".join(content_parts)


def _stream_through_tool_parser(parser, deltas):
    """Drive a tool parser with a list of streaming deltas.

    Returns a list of (delta_input, parser_output_dict_or_None) pairs and
    the concatenated content output across all non-None chunks.
    """
    accumulated = ""
    pairs = []
    content_parts = []
    tool_calls_seen = []
    for delta in deltas:
        prev = accumulated
        accumulated += delta
        out = parser.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=accumulated,
            delta_text=delta,
        )
        pairs.append((delta, out))
        if out is None:
            continue
        if "tool_calls" in out and out["tool_calls"]:
            tool_calls_seen.extend(out["tool_calls"])
        if "content" in out and out["content"] is not None:
            content_parts.append(out["content"])
    return pairs, "".join(content_parts), tool_calls_seen


# ---------------------------------------------------------------------------
# Bug 1: Reasoning leak (streaming asymmetry)
# ---------------------------------------------------------------------------


class TestBug1ReasoningLeakOnLateCloseTag:
    """The streaming parser must not leak thinking content into ``content``
    when the closing ``</think>`` tag arrives in a later chunk than the
    thinking text itself.
    """

    def test_reasoning_leak_when_close_tag_arrives_late(self) -> None:
        """Hypothesis: chunks A, B (thinking text) leak into content.

        Streaming order mimics what the Qwen3.6 model produces under the
        chat template that auto-prefills ``<think>\\n``: there is NO
        opening tag in the output, the thinking text streams first, then
        ``</think>``, then the final answer. With the current parser,
        chunks A and B hit Case 3 (no tags seen) and are routed to
        ``content``. Once ``</think>`` lands in chunk C, the implicit
        handler kicks in but it cannot retroactively reclassify A and B.

        Expected (correct) behaviour:
            reasoning channel == "The answer is 42."
            content   channel == "\\nFinal: 42"  (or similar, no leak)

        Observed (buggy) behaviour:
            reasoning channel == "" (nothing routed there before close tag)
            content   channel contains "The answer is 42." leaked from
            the thinking portion.
        """
        from vllm_mlx.reasoning.qwen36_parser import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        parser.reset_state()

        # No opening <think> because the chat template prefilled it.
        # Thinking text spans chunks A and B; close tag lands in C; final
        # answer in D.
        deltas = [
            "The answer ",     # A — thinking
            "is 42.",          # B — thinking
            "</think>\n",      # C — transition
            "\nFinal: 42",     # D — content
        ]

        _, reasoning_out, content_out = _stream_through_reasoning_parser(
            parser, deltas
        )

        # Sanity check: full text should be reconstructable from inputs.
        assert "".join(deltas) == (
            "The answer is 42.</think>\n\nFinal: 42"
        )

        # The thinking portion must end up in the reasoning channel...
        assert "The answer is 42." in reasoning_out, (
            "EXPECTED FAIL (Bug 1 confirmed) — pre-</think> chunks were "
            "not routed to reasoning.\n"
            f"  reasoning channel: {reasoning_out!r}\n"
            f"  content channel:   {content_out!r}\n"
        )

        # ...and must NOT leak into the content channel.
        assert "The answer" not in content_out, (
            "EXPECTED FAIL (Bug 1 confirmed) — thinking text 'The answer'"
            " leaked into content channel.\n"
            f"  content channel:   {content_out!r}\n"
            f"  reasoning channel: {reasoning_out!r}\n"
        )
        assert "42." not in content_out.replace("Final: 42", ""), (
            "EXPECTED FAIL (Bug 1 confirmed) — thinking text '42.' leaked "
            "into content channel (after stripping the legitimate "
            "'Final: 42').\n"
            f"  content channel: {content_out!r}\n"
        )

        # The legitimate final answer must survive in content.
        assert "Final: 42" in content_out, (
            "Final answer 'Final: 42' missing from content channel.\n"
            f"  content channel: {content_out!r}\n"
        )

    def test_reasoning_no_leak_in_non_streaming_path(self) -> None:
        """Asymmetry proof: extract_reasoning (full-text) is correct.

        This test is EXPECTED TO PASS, demonstrating that the bug is
        specific to the streaming code path. Same input as the streaming
        test, but fed in one shot.
        """
        from vllm_mlx.reasoning.qwen36_parser import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()

        full_text = "The answer is 42.</think>\n\nFinal: 42"
        reasoning, content = parser.extract_reasoning(full_text)

        assert reasoning == "The answer is 42.", (
            f"non-streaming reasoning extraction wrong: {reasoning!r}"
        )
        assert content == "Final: 42", (
            f"non-streaming content extraction wrong: {content!r}"
        )

    def test_streaming_thinking_content_does_not_leak_when_close_tag_at_chunk_boundary(
        self,
    ) -> None:
        """Edge case: ``</think>`` itself is split across two chunks.

        Even more fragile: when the close tag is fragmented (``</thi`` +
        ``nk>``), the parser has no chance to detect the close until the
        FULL tag has accumulated, so all preceding chunks (including the
        ``</thi`` fragment) are routed to content under Case 3. This
        test is EXPECTED TO FAIL.
        """
        from vllm_mlx.reasoning.qwen36_parser import Qwen36ReasoningParser

        parser = Qwen36ReasoningParser()
        parser.reset_state()

        deltas = [
            "Reason text.",   # A — thinking
            "</thi",          # B — partial close tag
            "nk>",            # C — completes close tag
            "Visible.",       # D — content
        ]

        _, reasoning_out, content_out = _stream_through_reasoning_parser(
            parser, deltas
        )

        # Reasoning must contain the thinking text; content must not.
        assert "Reason text." in reasoning_out, (
            "EXPECTED FAIL (tag-boundary fragility) — thinking text "
            "did not reach reasoning channel.\n"
            f"  reasoning: {reasoning_out!r}\n"
            f"  content:   {content_out!r}\n"
        )
        assert "Reason text." not in content_out, (
            "EXPECTED FAIL (tag-boundary fragility) — thinking text "
            "leaked into content channel.\n"
            f"  content: {content_out!r}\n"
        )
        # And the partial tag fragment </thi must not appear in content.
        assert "</thi" not in content_out, (
            "EXPECTED FAIL — partial close-tag fragment leaked into "
            f"content: {content_out!r}"
        )


# ---------------------------------------------------------------------------
# Bug 2: Auto-tool-choice in plain-text conversations
# ---------------------------------------------------------------------------


class TestBug2ToolParserPlainTextPassthrough:
    """The tool parser is chained into the stream even for conversations
    with no tool-call intent (because --enable-auto-tool-choice is on
    unconditionally). It must therefore pass plain text through verbatim.
    """

    def _make_hermes_parser(self):
        # The launcher resolves to "qwen3_coder" via
        # vllm_mlx.config.models.resolve_tool_parser, which is registered
        # to the HermesToolParser class (see hermes_tool_parser.py:52).
        from vllm_mlx.tool_parsers import ToolParserManager

        # Both names should map to HermesToolParser; prefer "qwen3_coder"
        # which matches the launcher default.
        registered = ToolParserManager.list_registered()
        assert "qwen3_coder" in registered or "hermes" in registered, (
            f"neither qwen3_coder nor hermes parser registered; got "
            f"{registered}"
        )
        name = "qwen3_coder" if "qwen3_coder" in registered else "hermes"
        parser_cls = ToolParserManager.get_tool_parser(name)
        return parser_cls(tokenizer=None)

    def test_tool_parser_streaming_passthrough_for_plain_text(self) -> None:
        """Plain-text streaming with NO tool-call markers must pass
        through verbatim with zero tool calls extracted.

        Per the diagnosed hypothesis, ``--enable-auto-tool-choice``
        chains this parser into every streaming response. If the parser
        suppresses or alters non-tool text, the user-visible answer gets
        corrupted.
        """
        parser = self._make_hermes_parser()

        deltas = [
            "Hello, ",
            "the weather ",
            "today ",
            "is sunny.",
        ]

        pairs, content_out, tool_calls_seen = _stream_through_tool_parser(
            parser, deltas
        )

        # No tool calls should be extracted from plain text.
        assert tool_calls_seen == [], (
            f"unexpected tool calls extracted from plain text: "
            f"{tool_calls_seen!r}"
        )

        # Each chunk must produce a passthrough content delta, and the
        # concatenation must equal the original input verbatim.
        full_input = "".join(deltas)
        assert content_out == full_input, (
            "EXPECTED FAIL if Bug 2 is real — tool parser altered or "
            "suppressed plain text.\n"
            f"  input chunks:   {deltas!r}\n"
            f"  output content: {content_out!r}\n"
            f"  per-chunk:      "
            f"{[(d, out) for d, out in pairs]!r}\n"
        )

        # Stronger assertion: every non-empty input chunk must yield a
        # non-None output that is exactly that chunk's content.
        for delta, out in pairs:
            assert out is not None, (
                f"parser returned None for non-tool delta {delta!r} — "
                "this would suppress the chunk in the SSE stream and "
                "corrupt the user-visible answer."
            )
            assert "tool_calls" not in out or not out.get("tool_calls"), (
                f"parser unexpectedly emitted tool_calls for plain "
                f"delta {delta!r}: {out!r}"
            )
            assert out.get("content") == delta, (
                f"plain-text delta {delta!r} altered by tool parser; "
                f"got {out!r}"
            )


# ---------------------------------------------------------------------------
# Bug 2 (configuration evidence)
# ---------------------------------------------------------------------------


class TestLauncherConfiguration:
    """Document the launcher's unconditional (modulo opt-out) enabling
    of ``--enable-auto-tool-choice``. This test is EXPECTED TO PASS and
    serves as the configuration claim that Bug 2's hypothesis depends on.
    """

    def test_auto_tool_choice_default_in_launcher(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        script = repo_root / "start-server.sh"
        assert script.exists(), f"launcher script not found at {script}"

        text = script.read_text(encoding="utf-8")

        # The launcher must reference --enable-auto-tool-choice.
        assert "--enable-auto-tool-choice" in text, (
            "start-server.sh no longer references "
            "--enable-auto-tool-choice; the configuration claim behind "
            "Bug 2 needs revisiting."
        )

        # And it must be enabled by default — the only gate is the
        # VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE opt-out env var.
        gate = re.search(
            r"VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE.*?!=\s*\"1\"",
            text,
            re.DOTALL,
        )
        assert gate is not None, (
            "could not find the VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE != \"1\" "
            "gate in start-server.sh — launcher behaviour may have "
            "changed."
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
