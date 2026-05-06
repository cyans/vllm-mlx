---
id: SPEC-FIX-QWEN36-TOOL-CALL-STREAMING
tags:
  - "@ACCEPT:FIX-QWEN36-TOOL-CALL-STREAMING"
---

# Acceptance — FIX-QWEN36-TOOL-CALL-STREAMING

## Definition of Done

All P1 criteria below pass on the `feature/spec-migrate-qwen36` branch against a
live Qwen3.6 server launched with `start-server-qwen36.sh`.

## Priority 1 (HARD)

### P1-AC1 — Streaming emits structured tool_calls
- **Given** an OpenAI-compatible client issues a streaming chat request with
  `tools=[tavily_search]` and a prompt known to trigger tool use
- **When** the server streams SSE chunks
- **Then** at least one chunk MUST have non-empty `choices[0].delta.tool_calls`
  **AND** the final chunk MUST have `choices[0].finish_reason == "tool_calls"`.

### P1-AC2 — No XML leak in content
- **Given** the same streaming request as P1-AC1
- **When** all `delta.content` values across all chunks are concatenated
- **Then** the concatenated string MUST NOT contain the substrings `<tool_call>`,
  `</tool_call>`, `<function=`, or `</function>`.

### P1-AC3 — Non-streaming regression guard
- **Given** a non-streaming request identical to P1-AC1 (minus `stream=True`)
- **When** the response returns
- **Then** `message.tool_calls[0].function.name` MUST equal the expected tool name
  **AND** `finish_reason` MUST equal `"tool_calls"` (matches current behavior).

### P1-AC4 — Parser streaming state-machine tests
- **Given** new parametrized tests in `tests/test_tool_parsers.py` covering:
  - Pure content chunks → parser returns `{"content": delta}`
  - Complete `<tool_call>...</tool_call>` → parser returns `{"tool_calls": [...]}`
  - Mid-tool chunk (no closing tag yet) → parser returns `None`
  - Mixed content + tool_call in same stream → content and tool_calls emitted
    as separate deltas
- **When** `pytest tests/test_tool_parsers.py -v` runs
- **Then** all new tests MUST pass.

### P1-AC5 — Quality gates
- `ruff check vllm_mlx/tool_parsers/hermes_tool_parser.py vllm_mlx/server.py` clean.
- `pytest tests/ -x` green.
- Coverage ≥ 85% on `vllm_mlx/tool_parsers/hermes_tool_parser.py` (streaming path).

### P1-AC6 — Launcher script syntax
- `bash -n start-server.sh`, `bash -n start-server-qwen36.sh`,
  `bash -n start-server-122b-mint.sh` all exit 0.

## Priority 2 (SOFT)

### P2-AC1 — `tool_choice="none"` honored
- **Given** client passes `tool_choice="none"` with tools defined
- **When** the model emits `<tool_call>` XML anyway
- **Then** the XML SHOULD pass through as `delta.content` with
  `finish_reason="stop"` (REQ-O1).

### P2-AC2 — Multiple sequential tool_calls in one stream
- **Given** the model emits two `<tool_call>...</tool_call>` blocks in one response
- **When** streamed
- **Then** two separate `delta.tool_calls` frames with `index=0` and `index=1`
  are emitted and no XML leaks into content.

## Test Commands

```bash
# Unit tests
pytest tests/test_tool_parsers.py -v

# Coverage
pytest tests/test_tool_parsers.py --cov=vllm_mlx.tool_parsers --cov-report=term-missing

# Live server verification (P1-AC1..AC3) — reuse existing harness from
# SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE verification scripts.
```

## Out of Scope (deferred)
- Partial streaming of `arguments` mid-JSON (OpenAI emits incremental
  `function.arguments` deltas); we emit a single complete tool_call frame.
- Multi-parser fallback chains.
