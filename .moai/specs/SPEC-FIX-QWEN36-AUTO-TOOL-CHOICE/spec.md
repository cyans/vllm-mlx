---
id: SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE
title: Enable auto tool choice for Qwen3.6 launchers
status: planned
priority: high
created: 2026-04-17
mode: ddd
predecessors:
  - SPEC-MIGRATE-QWEN36
  - SPEC-FIX-QWEN36-RUNTIME
lifecycle: spec-anchored
tags: [qwen36, tool-calling, launcher, bash, mcp]
---

# SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE

`@SPEC:FIX-QWEN36-AUTO-TOOL-CHOICE`

## Environment

- Launcher scripts: `start-server.sh`, `start-server-qwen36.sh`
- Server entrypoint: `vllm_mlx/server.py` (argparse already accepts `--enable-auto-tool-choice`)
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit`
- Tool-call parser: `qwen` (registered, verified working end-to-end)
- Chat template: cached Qwen3.6 `chat_template.jinja` emits `<tool_call>...</tool_call>` XML

## Assumptions

- A1: `qwen` parser correctly converts XML tool_calls to OpenAI JSON (observed live: `finish_reason=tool_calls`, function `web-search__tavily_search`, 1 tool_call).
- A2: HF model card for Qwen3.6 prescribes `--enable-auto-tool-choice` as the canonical flag.
- A3: Missing the flag causes clients that omit `tool_choice` to receive "planning prose only" instead of tool invocations.
- A4: Operators retain the ability to disable the flag via an env var without editing scripts.

## Requirements (EARS)

- REQ-U1 (Ubiquitous): Launcher scripts `start-server.sh` and `start-server-qwen36.sh` SHALL pass `--enable-auto-tool-choice` to `vllm-mlx serve` by default.
- REQ-E1 (Event-driven): WHEN the operator sets `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE=1`, the launcher SHALL NOT append `--enable-auto-tool-choice`.
- REQ-S1 (State-driven): WHILE `--enable-auto-tool-choice` is active AND a chat request arrives with MCP tools registered but no `tool_choice` field, the server SHALL offer the tools to the model (auto selection).
- REQ-N1 (Unwanted): The change SHALL NOT modify `start-server-122b-mint.sh`.
- REQ-N2 (Unwanted): The change SHALL NOT alter behavior for clients that explicitly pass `tool_choice="none"`.

## Specifications

- Launcher addition follows existing EXTRA_FLAGS pattern; env gate uses plain bash conditional.
- Single-line bash shellcheck-clean (`bash -n` clean).
- Test at `tests/test_launcher_flags.py` exercises flag propagation through `subprocess` mock or grep assertion.
- Live verification via server restart + MCP tool-use prompt without explicit `tool_choice`.

## Traceability

- Implementation: `@CODE:FIX-QWEN36-AUTO-TOOL-CHOICE/launcher` in both launcher scripts
- Tests: `@TEST:FIX-QWEN36-AUTO-TOOL-CHOICE` in `tests/test_launcher_flags.py`
- Plan: `@PLAN:FIX-QWEN36-AUTO-TOOL-CHOICE`
- Acceptance: `@ACCEPT:FIX-QWEN36-AUTO-TOOL-CHOICE`

## Out of Scope

- Switching parser from `qwen` to `qwen3_coder` (vLLM-only, not needed).
- Thinking/answer heuristic for Qwen36ReasoningParser (see SPEC-FIX-QWEN36-RUNTIME REQ-O2).
- Chat template modification, Qwen-Agent framework installation.
- `start-server-122b-mint.sh` (different model, separate SPEC).
