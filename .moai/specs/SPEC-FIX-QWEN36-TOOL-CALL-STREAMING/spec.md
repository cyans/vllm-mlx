---
id: SPEC-FIX-QWEN36-TOOL-CALL-STREAMING
title: Fix Qwen3.6 tool_call XML leaking into streaming delta.content
status: planned
priority: high
mode: ddd
lifecycle: spec-first
tags:
  - "@SPEC:FIX-QWEN36-TOOL-CALL-STREAMING"
predecessors:
  - SPEC-MIGRATE-QWEN36
  - SPEC-FIX-QWEN36-RUNTIME
  - SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE
---

# SPEC-FIX-QWEN36-TOOL-CALL-STREAMING

## 1. Problem

On the `feature/spec-migrate-qwen36` branch the non-streaming path correctly returns
`message.tool_calls=[{...}]` with `finish_reason=tool_calls`, but the streaming SSE
path emits the raw `<tool_call>...<function=...><parameter=...>...</function></tool_call>`
XML as `delta.content` text and finishes with `finish_reason=stop`. OpenAI-compatible
clients never see structured tool_calls, so they cannot send a `tool` result message,
and the model re-attempts the same call on the next turn — causing an infinite loop.

## 2. Evidence / Current State (code audit)

- Parser registration: `vllm_mlx/tool_parsers/hermes_tool_parser.py:52`
  `@ToolParserManager.register_module(["hermes", "nous", "qwen3_coder"])`
- Default parser name: `vllm_mlx/config/models.py:67` `TOOL_PARSER_PREFERRED = "qwen3_coder"`
- Non-streaming `extract_tool_calls`: `hermes_tool_parser.py:92` — handles JSON, Nemotron
  XML (`<function=...><parameter=...>`), bare function, lenient, and raw JSON. Works today.
- Streaming `extract_tool_calls_streaming`: `hermes_tool_parser.py:267` — exists, uses
  `<tool_call>`/`</tool_call>` counting and returns `{"tool_calls": [...]}` when closed.
- SSE loop: `vllm_mlx/server.py:1989` `stream_chat_completion(...)`
  - Tool parser init block: `server.py:2037-2051`
  - **Bug site**: `server.py:2065-2090` — when `_reasoning_parser` is active (it IS for
    Qwen3.6 via `Qwen36ReasoningParser` registered at `server.py:2425-2429`), the branch
    emits `delta.content` directly from the reasoning-parser output and **never calls
    `tool_parser.extract_tool_calls_streaming`**. The tool-parser codepath at
    `server.py:2105-2147` only runs inside the `else` branch (no reasoning parser).
  - Fallback at `server.py:2170-2201` checks `tool_accumulated_text`, which is only
    populated in the non-reasoning branch, so it never triggers for Qwen3.6 either.

## 3. Root Cause

The streaming SSE loop treats "reasoning parser enabled" and "tool parser enabled" as
mutually exclusive branches. For Qwen3.6 both must run in series: reasoning parser
strips `<think>...</think>`, then tool parser inspects the remaining visible tokens
for `<tool_call>` XML. Today only the reasoning branch runs, so XML leaks through.

## 4. Scope

### In scope
- Route `delta_msg.content` from the reasoning parser through
  `tool_parser.extract_tool_calls_streaming` before emitting the SSE chunk.
- Share a single `tool_accumulated_text` buffer across both branches so the fallback
  at end-of-stream works uniformly.
- Ensure the final chunk carries `finish_reason="tool_calls"` whenever any tool call
  was emitted during the stream.

### Out of scope
- Reasoning parser behavior changes (SPEC-FIX-QWEN36-RUNTIME Phase 1 is authoritative).
- New CLI flags — reuse `--tool-call-parser qwen3_coder`.
- `chat_template.jinja` changes; Qwen-Agent integration.

## 5. EARS Requirements

- **REQ-U1** (Ubiquitous): The streaming SSE path SHALL parse Qwen3.6 `<tool_call>` XML
  blocks into `delta.tool_calls`, matching non-streaming behavior.
- **REQ-U2** (Ubiquitous): The final streaming chunk for a tool-call response SHALL set
  `finish_reason="tool_calls"` (not `"stop"`).
- **REQ-N1** (Unwanted): Tokens inside a recognized `<tool_call>...</tool_call>` span
  SHALL NOT appear in any `delta.content` field.
- **REQ-N2** (Unwanted): The non-streaming response path SHALL remain behaviorally
  identical (regression guard via existing unit tests).
- **REQ-S1** (State-driven): WHILE a tool_call XML span is open (between `<tool_call>`
  and `</tool_call>`), the streaming parser SHALL buffer/suppress tokens rather than
  emit them as content.
- **REQ-O1** (Optional): WHERE the client passes `tool_choice="none"`, the server SHALL
  skip tool-call XML parsing and treat any emitted XML as plain content.

## 6. Traceability

- `@SPEC:FIX-QWEN36-TOOL-CALL-STREAMING` / `@PLAN:...` / `@ACCEPT:...` — this dir
- `@CODE:FIX-QWEN36-TOOL-CALL-STREAMING/parser` — `vllm_mlx/tool_parsers/hermes_tool_parser.py`
- `@CODE:FIX-QWEN36-TOOL-CALL-STREAMING/server` — `vllm_mlx/server.py::stream_chat_completion`
- `@TEST:FIX-QWEN36-TOOL-CALL-STREAMING` — `tests/test_tool_parsers.py` + new streaming tests
