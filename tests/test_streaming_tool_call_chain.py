# SPDX-License-Identifier: Apache-2.0
"""Streaming reasoning-parser + tool-parser chain tests.

@TEST:FIX-QWEN36-TOOL-CALL-STREAMING/chain

These tests verify that when both a reasoning parser and a tool parser are
active in the server's SSE stream loop, ``<tool_call>`` XML emitted by the
model is routed to ``delta.tool_calls`` instead of leaking into
``delta.content``.

The bug (pre-fix state, documented in SPEC-FIX-QWEN36-TOOL-CALL-STREAMING
§Pre-diagnosed facts): the reasoning-parser branch in
``vllm_mlx.server.stream_chat_completion`` emits ``delta_msg.content``
directly, never consulting the tool parser. As a result, clients see the
raw ``<tool_call>...</tool_call>`` XML in ``delta.content`` and never
receive a ``delta.tool_calls`` structured chunk while streaming.

The helper ``chain_reasoning_and_tool_parsers`` encapsulates the
chaining decision as a pure function so it can be exercised without
spinning up the full FastAPI + engine stack. The server code then calls
this helper from both the reasoning-parser branch and the no-reasoning
branch, which keeps production behaviour and unit-test coverage in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_mlx.reasoning.qwen36_parser import Qwen36ReasoningParser
from vllm_mlx.server import (
    _ToolChainState,
    chain_reasoning_and_tool_parsers,
)
from vllm_mlx.tool_parsers.hermes_tool_parser import HermesToolParser

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass
class _StreamOutcome:
    """Aggregated view of a simulated SSE stream.

    Mirrors the three types of chunks the server yields for a single
    delta: tool-call chunk, content chunk, or skip (None).
    """

    content_chunks: list[str] = field(default_factory=list)
    tool_calls_chunks: list[list[dict[str, Any]]] = field(default_factory=list)
    reasoning_chunks: list[str] = field(default_factory=list)
    skipped: int = 0

    @property
    def concatenated_content(self) -> str:
        return "".join(self.content_chunks)

    @property
    def emitted_tool_calls(self) -> bool:
        return bool(self.tool_calls_chunks)


def _drive_stream(
    deltas: list[str],
    *,
    reasoning_parser: Qwen36ReasoningParser | None,
    tool_parser: HermesToolParser | None,
) -> _StreamOutcome:
    """Replay ``deltas`` through the chain helper and aggregate results.

    This mirrors exactly the decision logic the server uses in its main
    SSE loop: for each delta, first route through the reasoning parser
    (if any), then route the resulting ``content`` through the tool
    parser (if any), then decide whether to emit content, tool_calls,
    or suppress the chunk.
    """
    outcome = _StreamOutcome()
    state = _ToolChainState()
    accumulated_reasoning_input = ""

    if reasoning_parser is not None:
        reasoning_parser.reset_state()
    if tool_parser is not None:
        tool_parser.reset()

    for delta in deltas:
        previous_reasoning_input = accumulated_reasoning_input
        accumulated_reasoning_input += delta

        result = chain_reasoning_and_tool_parsers(
            previous_text=previous_reasoning_input,
            current_text=accumulated_reasoning_input,
            delta_text=delta,
            reasoning_parser=reasoning_parser,
            tool_parser=tool_parser,
            state=state,
        )

        if result is None:
            outcome.skipped += 1
            continue

        if result.tool_calls:
            outcome.tool_calls_chunks.append(result.tool_calls)
        if result.content:
            outcome.content_chunks.append(result.content)
        if result.reasoning:
            outcome.reasoning_chunks.append(result.reasoning)

    return outcome


# ---------------------------------------------------------------------------
# Deltas used by multiple tests
# ---------------------------------------------------------------------------


_TOOL_CALL_BLOCK_DELTAS = [
    "<tool_call>\n",
    '<function=web_search>\n',
    "<parameter=query>",
    "Qwen 3.6 release notes",
    "</parameter>\n",
    "</function>\n",
    "</tool_call>",
]


_PLAIN_CONTENT_DELTAS = [
    "Hello",
    ", ",
    "how can I ",
    "help today?",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pure_content_flows_through_as_content():
    """Plain content without tool_call XML must reach the client unchanged.

    This is the baseline: the chain helper must not mangle normal
    streaming output when the model produces no tool calls.
    """
    reasoning = Qwen36ReasoningParser()
    tool = HermesToolParser(tokenizer=None)

    outcome = _drive_stream(
        _PLAIN_CONTENT_DELTAS, reasoning_parser=reasoning, tool_parser=tool
    )

    assert outcome.concatenated_content == "Hello, how can I help today?"
    assert not outcome.emitted_tool_calls
    assert outcome.skipped == 0


def test_tool_call_xml_becomes_delta_tool_calls():
    """A full <tool_call> XML block must produce delta.tool_calls, not content.

    The concatenated content must not contain any ``<tool_call>``
    substring. The final emitted tool_calls list must carry the
    function name.
    """
    reasoning = Qwen36ReasoningParser()
    tool = HermesToolParser(tokenizer=None)

    outcome = _drive_stream(
        _TOOL_CALL_BLOCK_DELTAS, reasoning_parser=reasoning, tool_parser=tool
    )

    assert outcome.emitted_tool_calls, (
        "tool_calls must be emitted when <tool_call> XML is fully streamed"
    )
    assert "<tool_call>" not in outcome.concatenated_content
    assert "</tool_call>" not in outcome.concatenated_content

    emitted = outcome.tool_calls_chunks[-1]
    assert emitted, "tool_calls chunk must not be empty"
    first = emitted[0]
    # The hermes parser emits either the OpenAI-style or the internal
    # shape; check the function name in whichever field holds it.
    name = None
    if isinstance(first, dict):
        if "function" in first and isinstance(first["function"], dict):
            name = first["function"].get("name")
        elif "name" in first:
            name = first["name"]
    assert name == "web_search", f"unexpected tool call payload: {first!r}"


def test_mixed_content_then_tool_call():
    """Plain content followed by a tool_call XML must split cleanly.

    Expected sequence: content deltas for the pre-text, then a
    tool_calls delta for the XML block. No ``<tool_call>`` substring
    may leak into any content chunk.
    """
    reasoning = Qwen36ReasoningParser()
    tool = HermesToolParser(tokenizer=None)

    pre_text = ["I'll ", "search the web. "]
    deltas = pre_text + _TOOL_CALL_BLOCK_DELTAS

    outcome = _drive_stream(
        deltas, reasoning_parser=reasoning, tool_parser=tool
    )

    assert outcome.concatenated_content.startswith("I'll search the web.")
    assert "<tool_call>" not in outcome.concatenated_content
    assert outcome.emitted_tool_calls


def test_reasoning_parser_chain_preserves_content_split():
    """Reasoning content and tool_call content must be routed to the
    correct channels."""
    reasoning = Qwen36ReasoningParser()
    tool = HermesToolParser(tokenizer=None)

    deltas = [
        "<think>",
        "Deciding to use the search tool.",
        "</think>",
    ] + _TOOL_CALL_BLOCK_DELTAS

    outcome = _drive_stream(
        deltas, reasoning_parser=reasoning, tool_parser=tool
    )

    # Reasoning content must be populated.
    reasoning_joined = "".join(outcome.reasoning_chunks)
    assert "Deciding to use the search tool." in reasoning_joined

    # tool_calls must still be emitted after the reasoning block.
    assert outcome.emitted_tool_calls

    # No <tool_call> / <think> leakage into visible content.
    assert "<tool_call>" not in outcome.concatenated_content
    assert "<think>" not in outcome.concatenated_content


def test_non_stream_parser_unchanged():
    """Regression guard: the non-streaming extract_tool_calls behaviour
    is unaffected by the streaming chain logic."""
    tool = HermesToolParser(tokenizer=None)

    full_text = "".join(_TOOL_CALL_BLOCK_DELTAS)
    result = tool.extract_tool_calls(full_text)

    assert result.tools_called
    assert result.tool_calls
    assert result.tool_calls[0]["name"] == "web_search"
    # Arguments must round-trip through json decode to the expected query.
    import json as _json

    args = _json.loads(result.tool_calls[0]["arguments"])
    assert args == {"query": "Qwen 3.6 release notes"}
