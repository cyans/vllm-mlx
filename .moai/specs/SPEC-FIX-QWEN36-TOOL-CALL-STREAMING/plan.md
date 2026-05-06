---
id: SPEC-FIX-QWEN36-TOOL-CALL-STREAMING
tags:
  - "@PLAN:FIX-QWEN36-TOOL-CALL-STREAMING"
---

# Plan — FIX-QWEN36-TOOL-CALL-STREAMING

## Primary Goal: SSE tool-call parity with non-streaming path

Route reasoning-parser content through the tool parser before emitting, so
`<tool_call>` XML is converted to `delta.tool_calls` structured frames.

## Technical Approach

### Change 1 — `vllm_mlx/server.py::stream_chat_completion`
- Lift `tool_accumulated_text`, `tool_calls_detected`, `tool_markup_possible`
  to cover both the reasoning branch and the non-reasoning branch.
- Extract a local helper `_apply_tool_parser(delta_content, output_finished)` that:
  1. Fast-path: bail if no `<` seen yet.
  2. Calls `tool_parser.extract_tool_calls_streaming(prev, curr, delta_content)`.
  3. Returns one of: `"suppress"` (None from parser), `{"tool_calls": [...]}`, or
     `{"content": str}`.
- In the reasoning branch (lines 2065-2090), feed `delta_msg.content` through the
  helper before building the chunk; build a `tool_calls` chunk when the helper
  returns tool_calls; skip emission on `"suppress"`.
- Unify the final-chunk `finish_reason` calculation: if `tool_calls_detected`
  and `output.finished`, set `finish_reason="tool_calls"`.
- Keep reasoning tokens (`delta_msg.reasoning`) flowing through unchanged.

### Change 2 — `vllm_mlx/tool_parsers/hermes_tool_parser.py`
- Minor: `extract_tool_calls_streaming` already handles the `<tool_call>` case
  correctly; verify the Nemotron XML sub-case (`<function=...>` inside a
  `<tool_call>` wrapper) does not double-fire the `"<function=" in current_text`
  branch when a full `<tool_call>` already exists. If it does, gate the bare
  `<function=` branch on `open_count == 0`.
- No API signature changes.

### Change 3 — `tests/test_tool_parsers.py`
- Add parametrized streaming tests driving chunks through
  `extract_tool_calls_streaming` and asserting:
  - Pure content → `{"content": delta}` only.
  - Single tool_call → one `{"tool_calls": [...]}` payload, no `<tool_call>` leak.
  - Content + tool_call mixed → content delta then tool_calls delta, never mixed.
  - Incomplete tool_call (no closing tag) → only suppression (None), no leak.

### Change 4 — New integration-ish test
- Add `tests/test_streaming_tool_calls.py` that drives a canned token sequence
  (reasoning + visible + `<tool_call>...<function=...>...</function></tool_call>`)
  through a lightweight harness calling into `stream_chat_completion`'s helper,
  or through the parser directly when the server harness is unavailable.

## Milestones (priority, not time)

- **Priority High**: Change 1 (server SSE loop routing), Change 3 (parser tests).
- **Priority High**: Change 2 (parser guard) only if parametrized tests show the
  double-fire bug; otherwise skip.
- **Priority Medium**: Change 4 (harness-level test) and coverage cleanup.

## Risks

- `delta_msg.content` from the reasoning parser may include partial chars inside
  a `<tool_call>` tag spanning chunks → helper must rely on the parser's tag
  counting against the growing `tool_accumulated_text`, not on `delta_text`.
- `finish_reason` currently computed in two branches; consolidating risks behavior
  drift for non-tool streams → covered by REQ-N2 regression test.

## Dependencies
- Depends on: SPEC-FIX-QWEN36-RUNTIME (reasoning parser already working).
- Unblocks: reliable Qwen3.6 tool use with any OpenAI-compatible client (Qwen-Agent,
  openai-python, LiteLLM, LangChain).

## Expert Consultation
- **expert-backend**: primary author for SSE routing and the helper signature.
- **expert-testing**: author parametrized parser-state tests.
