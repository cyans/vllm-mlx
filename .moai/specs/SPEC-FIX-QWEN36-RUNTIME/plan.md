---
id: SPEC-FIX-QWEN36-RUNTIME
artifact: plan
status: Planned
development_mode: ddd
tags:
  - "@PLAN:FIX-QWEN36-RUNTIME"
  - "@SPEC:FIX-QWEN36-RUNTIME"
---

# Plan — SPEC-FIX-QWEN36-RUNTIME

`@PLAN:FIX-QWEN36-RUNTIME` — Implementation plan for Qwen3.6 streaming reasoning parser + opt-in MCP tool auto-injection.

> Methodology: **DDD** (ANALYZE → PRESERVE → IMPROVE) per `.moai/config/sections/quality.yaml`.
> Two independent phases; each phase follows a full DDD cycle.
> **No time estimates** — milestones are priority-ordered.

## 1. Implementation Approach

### 1.1 Strategy

- Two **independent** phases, each deliverable separately.
- **Phase 1** is higher priority (streaming is user-visible broken for Qwen3.6).
- **Phase 2** is a pre-existing feature gap surfaced by validation; opt-in by design (REQ-U2).
- Each phase uses the **DDD cycle**: ANALYZE existing behavior → PRESERVE via characterization tests → IMPROVE with new code.

### 1.2 Architectural Principles

- **Isolation over patching**: Bug 1 is fixed by creating a new parser class (`Qwen36ReasoningParser`), not by modifying the existing one. This honors REQ-N1 and avoids regression risk for Qwen3.5.
- **Opt-in over default**: Bug 2 is fixed behind a flag (REQ-U2, REQ-N2). Zero behavior change when the flag is off.
- **Reuse existing helpers**: `MCPClientManager.get_merged_tools` and `.merge_tools` already exist and are unchanged.
- **Configuration resolution at one site**: `resolve_reasoning_parser()` in `vllm_mlx/config/models.py` becomes the single source of truth for model-ID → parser mapping, mirroring `matches_eos_patch`.

## 2. Milestones (Priority-Ordered)

### Primary Goal — Phase 1: Streaming Reasoning Parser Fix (Priority High)

Unblocks every OpenAI-streaming client talking to Qwen3.6.

**Sub-milestones (ordered)**:

1. **Primary Goal 1.1**: ANALYZE — locate the reasoning parser registry; inventory Qwen3.5 behavior snapshots
2. **Primary Goal 1.2**: PRESERVE — characterization tests for `qwen3` parser on Qwen3.5 inputs; snapshot current `think_parser` streaming behavior
3. **Primary Goal 1.3**: IMPROVE — implement `Qwen36ReasoningParser`; register it; add `resolve_reasoning_parser()` helper
4. **Primary Goal 1.4**: Unit tests for the new parser (streaming default routing, optional heuristic if included)
5. **Primary Goal 1.5**: Integration smoke test — live Qwen3.6 streaming request emits non-null `delta.content`
6. **Primary Goal 1.6**: Launcher update — `start-server.sh`, `start-server-qwen36.sh` pick parser dynamically
7. **Primary Goal 1.7**: TRUST 5 validation — ruff, pytest, coverage ≥ 85% on new module

### Secondary Goal — Phase 2: Opt-in MCP Tool Auto-Injection (Priority Medium)

Closes a pre-existing gap; off by default, so zero blast radius when not enabled.

**Sub-milestones (ordered)**:

1. **Secondary Goal 2.1**: ANALYZE — inspect `create_chat_completion()` at `server.py:~1483`; read `MCPClientManager.get_merged_tools` and `.merge_tools` signatures
2. **Secondary Goal 2.2**: PRESERVE — characterization test: with flag OFF and no `tools`, response has no `tool_calls` for an MCP-relevant prompt (bit-for-bit parity with pre-change)
3. **Secondary Goal 2.3**: IMPROVE — add `--auto-inject-mcp-tools` flag; wire into `create_chat_completion`; branch on flag state
4. **Secondary Goal 2.4**: Startup INFO log emits enabled/disabled status
5. **Secondary Goal 2.5**: Unit tests — (a) flag off = no-op, (b) flag on + no client tools = auto-inject, (c) flag on + client tools = merge with client-precedence on collision
6. **Secondary Goal 2.6**: Integration smoke test — flag on, prompt that needs a tool triggers a `tool_call`
7. **Secondary Goal 2.7**: Launcher env-var passthrough — `VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1` adds flag
8. **Secondary Goal 2.8**: TRUST 5 validation — ruff, pytest, coverage target met

### Final Goal — Integration & Handoff (Priority Medium)

1. **Final Goal 3.1**: Full test suite green on `feature/spec-migrate-qwen36` (or successor branch) with both phases applied
2. **Final Goal 3.2**: Update `.moai/reports/qwen36-validation-2026-04/report.md` with a "resolved" note referencing this SPEC
3. **Final Goal 3.3**: `/moai:3-sync` for docs propagation (API doc, README server-flag table, CHANGELOG entry)

### Optional Goal — Enhanced Heuristic (Priority Low)

- **Optional 4.1**: If MVP streaming (everything → `content`) is merged and works, add the thinking/answer boundary heuristic behind a `--qwen36-split-thinking` flag or similar
- **Optional 4.2**: Collect more real-world Qwen3.6 outputs to validate the heuristic before enabling by default

## 3. Technical Approach

### 3.1 Phase 1 — Qwen36ReasoningParser

**ANALYZE (PRESERVE preparation)**:
- Read `vllm_mlx/reasoning/think_parser.py` fully (focus on lines 133–140 and surrounding state machine).
- Read `vllm_mlx/reasoning/qwen3_parser.py` fully (focus on lines 60–61 for non-streaming default).
- Locate the parser registration site — candidates in order:
  1. `vllm_mlx/reasoning/__init__.py`
  2. `vllm_mlx/server.py` argparse site for `--reasoning-parser`
  3. `vllm_mlx/cli.py`
- Grep for `"qwen3"` and `"reasoning-parser"` to map all dispatch points.
- Inspect `vllm_mlx/config/models.py` for the `matches_eos_patch` analogue.

**PRESERVE (characterization tests)**:
- Capture Qwen3.5 non-streaming input→output fixtures with the current `qwen3` parser.
- Capture Qwen3.5 streaming input→output fixtures with the current parser.
- Commit these as golden tests BEFORE any code change.
- These ensure REQ-N1 is not violated.

**IMPROVE**:
- **New file**: `vllm_mlx/reasoning/qwen36_parser.py`
  - Class `Qwen36ReasoningParser` with same interface as `BaseReasoningParser`.
  - `extract_reasoning` — mirrors `qwen3_parser.py:60-61` (no marker → pure content).
  - `extract_reasoning_streaming` — **inverts** the default: no marker → emit to `delta.content`. This is the one-line fix for Bug 1.
  - Optional: thinking-prefix detection (REQ-O2) — defer if risky.
- **Registration**: add `"qwen36"` to the parser dispatch map. Keep `"qwen3"` in place untouched.
- **Resolver helper**: `vllm_mlx/config/models.py::resolve_reasoning_parser(model_id, explicit)` — returns `explicit` if set, else `"qwen36"` if `"qwen3.6"` in model_id lowercased, else the existing default.
- **Wire resolver** at the server's config-resolution site.

**Launcher update**:
- In `start-server.sh` and `start-server-qwen36.sh`, remove any hardcoded `--reasoning-parser qwen3` and let the server's auto-resolver pick the right parser based on the model ID passed via `--model`. If an explicit parser is still desired in the script, source it from a `$REASONING_PARSER` env var.

### 3.2 Phase 2 — MCP Auto-Injection

**ANALYZE**:
- Read `vllm_mlx/server.py:1470-1520` (around `create_chat_completion`).
- Read `vllm_mlx/mcp/client.py` — confirm `get_merged_tools` signature (does it accept client tools? write a small merge helper if not — do NOT modify the public API).
- Read argparse site in `vllm_mlx/cli.py` (or wherever the server flags are parsed).

**PRESERVE**:
- Fixture: chat completion request WITHOUT `tools`, prompt that could benefit from an MCP tool.
- Snapshot response: should have no `tool_calls` when flag is off. Commit as regression guard (REQ-N2).

**IMPROVE**:
- **argparse**: add `--auto-inject-mcp-tools` (store_true, default False) with the help text from spec §4.5.
- **Server init**: carry the flag into a `self.auto_inject_mcp_tools` attribute on the server/handler class.
- **`create_chat_completion`** change, focused on the line ~1483 neighborhood:
  ```
  (existing code reads request.tools)
  tools = request.tools
  if self.auto_inject_mcp_tools:
      if not tools:
          tools = self._mcp_manager.get_merged_tools()
      else:
          tools = merge_with_client_precedence(tools, self._mcp_manager.get_merged_tools())
  ```
  - `merge_with_client_precedence` either calls an existing `MCPClientManager.merge_tools` variant or is a thin local helper preserving client-tool ordering and dropping duplicate MCP entries by `function.name`.
- **Startup log**: INFO line at server boot reporting `auto_inject_mcp_tools=True/False` and the resolved reasoning parser.
- **Launcher env passthrough**: in both launcher scripts, if `${VLLM_MLX_AUTO_INJECT_MCP_TOOLS:-0}` equals `1`, append `--auto-inject-mcp-tools` to the server invocation.

### 3.3 Testing Strategy

- **Unit tests** (Phase 1): `tests/reasoning/test_qwen36_parser.py`
  - Streaming: single-chunk plain text → `delta.content` populated, `delta.reasoning_content` null/empty.
  - Streaming: multi-chunk plain text → all content routed to `delta.content` across chunks.
  - Non-streaming: plain text → `content` populated, `reasoning_content` empty.
  - Non-streaming: text with thinking prefix (if REQ-O2 implemented) → prefix in `reasoning_content`, rest in `content`.
  - Regression: `qwen3` parser on Qwen3.5 fixtures unchanged (REQ-N1).
- **Unit tests** (Phase 2): `tests/mcp/test_auto_inject.py`
  - Flag off: request.tools None → handler sees None (no injection).
  - Flag off: request.tools populated → handler sees client's list unchanged.
  - Flag on + no client tools → handler sees MCP tools list.
  - Flag on + client tools, no collision → handler sees `client_tools + mcp_tools`.
  - Flag on + client tools, name collision → handler sees client tool; MCP duplicate dropped.
- **Integration smoke** (both phases):
  - Start server with `--model Qwen/Qwen3.6-...` + `--auto-inject-mcp-tools`.
  - Streaming chat completion → at least one `delta.content` chunk before `finish_reason`.
  - Tool-using prompt → response has `tool_calls` referencing an MCP tool.
- **Coverage**: pytest-cov gate ≥ 85% on new modules.

### 3.4 Quality Gates (TRUST 5)

| Gate | Mechanism |
|------|-----------|
| Tested | pytest + characterization tests + unit tests + integration smoke; ≥ 85% coverage on new code |
| Readable | ruff (already project-wide); clear docstrings on parser class and merge helper |
| Unified | Follow existing module layout conventions of `vllm_mlx/reasoning/*.py` and `vllm_mlx/config/models.py` |
| Secured | No new attack surface; auto-injection is opt-in; collisions prefer client (no unexpected tool spoofing) |
| Trackable | Conventional commits referencing `SPEC-FIX-QWEN36-RUNTIME`; traceability tags in all new files |

## 4. Architecture Direction

### 4.1 Module Boundaries

```
vllm_mlx/
├── reasoning/
│   ├── think_parser.py          # UNCHANGED (REQ-N1)
│   ├── qwen3_parser.py          # UNCHANGED (REQ-N1)
│   ├── qwen36_parser.py         # NEW — Bug 1 fix
│   └── __init__.py              # REGISTRATION (add qwen36)
├── config/
│   └── models.py                # ADD resolve_reasoning_parser()
├── mcp/
│   └── client.py                # UNCHANGED public API
├── cli.py                       # ADD --auto-inject-mcp-tools
└── server.py
    └── create_chat_completion   # Bug 2 fix: optional merge branch

start-server.sh                  # Dynamic parser resolution + env passthrough
start-server-qwen36.sh           # Dynamic parser resolution + env passthrough
```

### 4.2 Dependency Direction

- `server.py` depends on `reasoning/*` (via parser dispatch) and `mcp/client.py` (via `_mcp_manager`).
- `reasoning/qwen36_parser.py` depends on the same base types as `qwen3_parser.py`; no new third-party deps.
- `config/models.py::resolve_reasoning_parser` is pure (takes strings, returns strings) — no runtime dependencies.

### 4.3 Extension Points

- Adding a future Qwen3.7 parser follows the exact same pattern: new `qwen37_parser.py`, register, extend `resolve_reasoning_parser`.
- Adding more env-var → CLI-flag passthroughs in launcher scripts follows the `VLLM_MLX_AUTO_INJECT_MCP_TOOLS` pattern.

## 5. Risks and Mitigation

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|------------|--------|------------|
| R1 | Heuristic for thinking/answer boundary produces false positives, routing real content to `reasoning_content` | Medium | Medium (user-visible) | MVP ships without heuristic (everything → `content`); heuristic becomes an opt-in flag (REQ-O2, Optional Goal 4.1) |
| R2 | Parser registry site is spread across multiple files and an incomplete hook leaves `qwen36` unreachable | Medium | High (feature dead on arrival) | ANALYZE phase grep audit; integration smoke test is the go/no-go gate |
| R3 | MCP tool injection conflicts with tools the model was fine-tuned to refuse | Low | Medium | Opt-in flag (off by default); client-precedence on collision; DEBUG log on drops |
| R4 | `get_merged_tools()` signature differs from what Phase 2 assumes (e.g., no client-tools argument) | Medium | Low | Write thin local merge helper in server.py rather than modifying MCP API (REQ — "no MCP API changes") |
| R5 | Launcher-script change breaks existing users relying on hardcoded `qwen3` | Low | Medium | Preserve `$REASONING_PARSER` env-var override path in scripts; document migration in commit message |
| R6 | Qwen3.5 regression slips through because characterization tests are weak | Low | High | Snapshot tests over 5+ Qwen3.5 fixture inputs in PRESERVE step; run them in CI |
| R7 | Streaming fix subtly changes finish_reason timing or token count | Low | Medium | Integration smoke test asserts `finish_reason=stop` still fires and content length matches expected envelope |
| R8 | Docs drift — users discover the flag but launcher scripts don't know about the env var | Low | Low | Phase 2 final goal includes launcher update; `/moai:3-sync` picks up docs |

## 6. Validation Plan

- **Phase 1**: after IMPROVE, rerun the exact streaming request from `.moai/reports/qwen36-validation-2026-04/report.md` and verify `delta.content` is populated.
- **Phase 2**: start server with `--auto-inject-mcp-tools`, issue a chat completion that the model should resolve via an MCP tool, and verify `tool_calls` appears in the response.
- **Regression**: rerun the non-streaming Paris example (from validation report §Bug 1 context) and verify `content` / `reasoning_content` split remains correct.
- **Qwen3.5 regression**: rerun any Qwen3.5 fixture available in the repo (or use LEGACY_MODEL_ID path) and verify output unchanged.

## 7. Dependencies & Ordering

- Phase 1 → can merge independently.
- Phase 2 → can merge independently.
- Final Goal 3.1 depends on both phases being merged.
- No external dependencies (no library bumps, no new third-party tools).

## 8. Traceability

| Tag | Links to |
|-----|----------|
| `@PLAN:FIX-QWEN36-RUNTIME` | This plan.md |
| `@SPEC:FIX-QWEN36-RUNTIME` | `spec.md` (sibling) |
| `@ACCEPT:FIX-QWEN36-RUNTIME` | `acceptance.md` (sibling) |

See `spec.md` §6 for the complete artifact→code traceability table.
