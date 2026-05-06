---
id: SPEC-FIX-QWEN36-RUNTIME
title: Fix Qwen3.6 Runtime Issues — Streaming Reasoning Parser & MCP Tool Auto-Injection
status: Planned
priority: High
created: 2026-04-17
development_mode: ddd
lifecycle: spec-anchored
predecessor: SPEC-MIGRATE-QWEN36
validation_report: .moai/reports/qwen36-validation-2026-04/report.md
tags:
  - "@SPEC:FIX-QWEN36-RUNTIME"
---

# SPEC-FIX-QWEN36-RUNTIME — Qwen3.6 Runtime Fixes

`@SPEC:FIX-QWEN36-RUNTIME`

## 1. Environment

### 1.1 Runtime Context

- Repository: `vllm-mlx` (Apple Silicon MLX inference server)
- Branch of record: `feature/spec-migrate-qwen36` (11 commits applied from SPEC-MIGRATE-QWEN36)
- Server entrypoint: `vllm_mlx/server.py` (FastAPI-based OpenAI-compatible API)
- Reasoning parser registry: `vllm_mlx/reasoning/` (currently `think_parser.py`, `qwen3_parser.py`)
- MCP manager: `vllm_mlx/mcp/client.py` (`MCPClientManager.get_merged_tools`, `merge_tools` helpers already implemented)
- Model configuration: `vllm_mlx/config/models.py` (hosts `matches_eos_patch`-style ID matchers)
- Launcher scripts: `start-server.sh`, `start-server-qwen36.sh`

### 1.2 Development Methodology

- **DDD** (per `.moai/config/sections/quality.yaml`: `development_mode: "ddd"`)
- Cycle: ANALYZE → PRESERVE → IMPROVE
- Coverage target: 85% on new/modified modules
- Quality gates: TRUST 5 framework (ruff, pytest, coverage, security, traceability)

### 1.3 Language & Tooling

- Python 3.11+ (project-wide)
- Test framework: pytest + pytest-asyncio
- Lint/format: ruff
- Type checks: as already configured in the repo

## 2. Assumptions

### 2.1 Confirmed Assumptions (Evidence: live validation & expert-debug analysis)

| # | Assumption | Confidence | Evidence |
|---|------------|------------|----------|
| A1 | Qwen3.6 emits plain-text "thinking" prose without `<think>` / `</think>` tags | High | Live validation (`.moai/reports/qwen36-validation-2026-04/report.md`); response contains literal `"Here's a thinking process:\n\n1. ..."` |
| A2 | Non-streaming parser path at `vllm_mlx/reasoning/qwen3_parser.py:60-61` defaults to "content" when no marker seen, and works correctly for Qwen3.6 | High | Live non-streaming test returned clean `message.content` / `message.reasoning_content` split (Paris example, `finish_reason=stop`) |
| A3 | Streaming parser at `vllm_mlx/reasoning/think_parser.py:133-140` defaults to "reasoning" when no closing marker seen | High | Code inspection + streaming test: every delta chunk had `delta.content=null` and text in `delta.reasoning_content` |
| A4 | `MCPClientManager.get_merged_tools()` correctly converts MCP native tool schema to OpenAI function schema | High | Code inspection of `vllm_mlx/mcp/client.py`; helper already in use elsewhere in the codebase |
| A5 | `create_chat_completion()` at `vllm_mlx/server.py:~1483` does NOT call the MCP merger today | High | Code inspection; only `request.tools` is consulted |
| A6 | Qwen3.5 legacy users currently rely on `--reasoning-parser qwen3` and must not be impacted | High | LEGACY path preserved by SPEC-MIGRATE-QWEN36; auto-selection for qwen3.5 remains in place |

### 2.2 Working Assumptions (Validate during ANALYZE phase)

| # | Assumption | Validation Method |
|---|------------|-------------------|
| A7 | A heuristic boundary (e.g., "Final Answer:" / numbered-list → prose transition) can separate thinking from answer with acceptable accuracy | Inspect 5+ real Qwen3.6 outputs collected during validation; decide MVP vs enhanced |
| A8 | `merge_tools` / `get_merged_tools` signature supports client-first precedence for name collisions (client tools win) | Read `vllm_mlx/mcp/client.py` during PRESERVE phase; write tests for collision behavior |
| A9 | `start-server*.sh` can read an env var (`VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1`) and append the CLI flag conditionally | Shell-script pattern is well-established in this repo's launcher family |

### 2.3 Out-of-Scope Assumptions (Not to validate here)

- No changes to the `qwen3` parser for Qwen3.5 behavior (explicitly prohibited by REQ-N1)
- No changes to the `MCPClientManager` public API
- No multimodal activation
- No A/B benchmarking

## 3. Requirements (EARS)

### 3.1 Ubiquitous Requirements (Always Active)

**REQ-U1**: The system **shall** expose at least one reasoning parser per supported Qwen generation, and the `qwen36` parser **shall** route plain-text output to `content` by default in streaming mode.

**REQ-U2**: MCP tool auto-injection into `/v1/chat/completions` **shall** be opt-in via an explicit CLI flag or equivalent environment-variable configuration, defaulting to disabled.

**REQ-U3**: The system **shall** preserve existing behavior for all pre-existing reasoning parsers (`qwen3`, `think`, and any others) for all Qwen generations other than Qwen3.6.

### 3.2 Event-Driven Requirements

**REQ-E1**: WHEN a chat completion request targets a model whose ID matches the Qwen3.6 pattern (lowercase substring match on `"qwen3.6"` or equivalent configured pattern) AND no explicit `--reasoning-parser` override is provided, the server **shall** select the `qwen36` parser automatically.

**REQ-E2**: WHEN `--auto-inject-mcp-tools` is enabled AND an incoming `/v1/chat/completions` request has no `tools` field (or an empty list), the server **shall** inject all registered MCP tools converted to OpenAI function schema via `MCPClientManager.get_merged_tools()`.

**REQ-E3**: WHEN `--auto-inject-mcp-tools` is enabled AND an incoming request has a non-empty `tools` field, the server **shall** merge client-supplied tools with MCP tools such that client tools take precedence on name collision.

**REQ-E4**: WHEN the server starts, it **shall** emit an INFO-level log entry stating whether MCP tool auto-injection is enabled or disabled.

### 3.3 State-Driven Requirements

**REQ-S1**: WHILE the server is running with `--reasoning-parser qwen36` (explicit or auto-selected), streaming responses **shall** emit non-thinking content tokens in `delta.content` and non-null content **shall** appear before `finish_reason` is emitted.

**REQ-S2**: WHILE `--auto-inject-mcp-tools` is disabled (default), the server **shall** behave bit-for-bit identically to pre-SPEC-FIX-QWEN36-RUNTIME behavior for all chat completion requests.

### 3.4 Unwanted-Behavior Requirements

**REQ-N1**: The `qwen3` parser (`vllm_mlx/reasoning/qwen3_parser.py`) and `BaseThinkingReasoningParser` (`vllm_mlx/reasoning/think_parser.py`) **shall not** be modified in a way that changes their observable behavior for Qwen3.5 models or any model currently matched by `LEGACY_MODEL_ID`.

**REQ-N2**: With `--auto-inject-mcp-tools` off, the server **shall not** inject any MCP tools into chat completion requests (bit-for-bit behavior match with pre-SPEC behavior).

**REQ-N3**: The `qwen36` parser **shall not** depend on the presence of `<think>` / `</think>` tags; presence of such tags in output is acceptable but not required.

**REQ-N4**: The auto-injection merge logic **shall not** silently drop or rewrite client-supplied tool definitions on name collision; client tools must win and the collision **may** be logged at DEBUG level.

### 3.5 Optional Requirements

**REQ-O1**: WHERE the environment variable `VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1` is set, the launcher scripts (`start-server.sh`, `start-server-qwen36.sh`) **shall** pass `--auto-inject-mcp-tools` to the server invocation.

**REQ-O2**: WHERE Qwen3.6 output begins with a recognizable thinking prefix (e.g., `"Here's a thinking process:"`, `"Let me think"`, or a configured pattern), the `qwen36` parser **may** route that prefix to `reasoning_content` until an answer boundary heuristic fires (e.g., `"Final Answer:"`, `"답:"`, or empty-line-after-numbered-list transition). MVP behavior (everything → `content` in streaming) is acceptable if the heuristic proves unreliable.

**REQ-O3**: WHERE the server is started without an explicit `--reasoning-parser` flag, model-ID–based auto-selection **shall** resolve via a helper in `vllm_mlx/config/models.py` analogous to `matches_eos_patch`.

## 4. Specifications

### 4.1 Component: `Qwen36ReasoningParser` (new)

- **Location**: `vllm_mlx/reasoning/qwen36_parser.py`
- **Interface**: Matches `BaseReasoningParser`
  - `extract_reasoning(text: str) -> tuple[str, str]` — non-streaming; returns `(content, reasoning_content)`
  - `extract_reasoning_streaming(delta_text: str, state: ParserState) -> StreamingDelta` — streaming; returns `delta` with `content` and/or `reasoning_content` fields populated
- **Default routing (streaming)**: plain text → `content` (mirrors non-streaming behavior; fixes Bug 1)
- **Optional heuristic**: recognize thinking prefixes and answer-boundary transitions to split the stream between `reasoning_content` and `content`

### 4.2 Parser Registration

- **Location**: reasoning parser registry (to be located in Phase 1 ANALYZE — likely in `vllm_mlx/reasoning/__init__.py` or `vllm_mlx/server.py` argparse plumbing)
- **Behavior**: `--reasoning-parser qwen36` works; model-ID auto-detection resolves to `qwen36` when substring `"qwen3.6"` (lowercase) appears in the active model ID AND no explicit override is given

### 4.3 Config Helper in `vllm_mlx/config/models.py`

- **New helper**: `resolve_reasoning_parser(model_id: str, explicit: str | None) -> str`
  - Returns `explicit` if provided
  - Else returns `"qwen36"` if `"qwen3.6"` in `model_id.lower()`
  - Else returns existing default (likely `"qwen3"`)
- **Analogue**: mirrors existing `matches_eos_patch` style (see predecessor SPEC-MIGRATE-QWEN36)

### 4.4 Launcher Scripts

- **Files**: `start-server.sh`, `start-server-qwen36.sh`
- **Change**: dynamically resolve the `--reasoning-parser` flag (remove hardcoded `qwen3` if present), and optionally append `--auto-inject-mcp-tools` when `VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1`

### 4.5 CLI Flag: `--auto-inject-mcp-tools`

- **Location**: argparse site in `vllm_mlx/cli.py` and/or `vllm_mlx/server.py`
- **Type**: boolean flag, `action="store_true"`
- **Default**: `False`
- **Help text**: "Auto-inject MCP-registered tools into /v1/chat/completions requests (client tools take precedence on name collision)."

### 4.6 Chat Completion Handler Changes

- **Location**: `vllm_mlx/server.py`, `create_chat_completion()` around line 1483
- **Logic** (pseudocode expressed as narrative):
  1. If `self.auto_inject_mcp_tools` is False, proceed with `request.tools` unchanged (existing behavior).
  2. If True and `request.tools` is None or empty, set tools to `self._mcp_manager.get_merged_tools()`.
  3. If True and `request.tools` is non-empty, set tools to the result of merging client tools with MCP tools using existing helper semantics (client-precedence).
- **Invariant**: in the False branch, the code path and resulting request are byte-identical to the pre-change implementation.

### 4.7 Startup Logging

- At server startup, after argparse and config resolution, emit:
  - `INFO` log: `"MCP tool auto-injection: <enabled|disabled>"`
  - `INFO` log: `"Reasoning parser: <resolved_parser> (source: <explicit|auto-selected-by-model-id>)"`

### 4.8 Phase Decomposition

| Phase | Scope | Bug | Deliverables |
|-------|-------|-----|--------------|
| Phase 1 | Reasoning parser fix | Bug 1 | `qwen36_parser.py`, registry hook, `resolve_reasoning_parser` helper, launcher-script update for parser, unit tests, integration smoke test |
| Phase 2 | MCP auto-injection (opt-in) | Bug 2 | CLI flag, `create_chat_completion` merge branch, startup log, launcher-script env passthrough, unit tests for merge behavior, integration smoke test |

Phases are independent and may be implemented and merged separately. Phase 1 is higher priority (Bug 1 blocks streaming clients).

## 5. Out of Scope

- Modifying the `qwen3` parser or `BaseThinkingReasoningParser` (REQ-N1 prohibits)
- Changes to `MCPClientManager` public API (reuse `get_merged_tools`, `merge_tools`)
- Multimodal activation for Qwen3.6
- A/B performance benchmarking (deferred to a follow-up SPEC)
- Changes to the `/v1/mcp/tools` endpoint (unchanged; client workaround remains valid)

## 6. Traceability

| TAG | Artifact |
|---|---|
| `@SPEC:FIX-QWEN36-RUNTIME` | `.moai/specs/SPEC-FIX-QWEN36-RUNTIME/spec.md` (this file) |
| `@PLAN:FIX-QWEN36-RUNTIME` | `.moai/specs/SPEC-FIX-QWEN36-RUNTIME/plan.md` |
| `@ACCEPT:FIX-QWEN36-RUNTIME` | `.moai/specs/SPEC-FIX-QWEN36-RUNTIME/acceptance.md` |
| `@CODE:FIX-QWEN36-RUNTIME/parser` | `vllm_mlx/reasoning/qwen36_parser.py` (to be created) |
| `@CODE:FIX-QWEN36-RUNTIME/parser-registry` | reasoning parser registration site (to be located in Phase 1 ANALYZE) |
| `@CODE:FIX-QWEN36-RUNTIME/parser-resolver` | `vllm_mlx/config/models.py` (new helper `resolve_reasoning_parser`) |
| `@CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject` | `vllm_mlx/server.py` (`create_chat_completion`, line ~1483) + argparse site in `vllm_mlx/cli.py` |
| `@CODE:FIX-QWEN36-RUNTIME/launcher` | `start-server.sh`, `start-server-qwen36.sh` (flag passthrough) |
| `@TEST:FIX-QWEN36-RUNTIME/parser-unit` | unit tests for `Qwen36ReasoningParser` streaming + non-streaming |
| `@TEST:FIX-QWEN36-RUNTIME/parser-regression` | regression test ensuring `qwen3` parser behavior unchanged for Qwen3.5 |
| `@TEST:FIX-QWEN36-RUNTIME/mcp-merge` | unit tests for merge logic (no tools, merge, collision) |
| `@TEST:FIX-QWEN36-RUNTIME/integration` | integration smoke tests against live Qwen3.6 + MCP server |
| `@REPORT:FIX-QWEN36-RUNTIME/validation` | `.moai/reports/qwen36-validation-2026-04/report.md` (upstream evidence) |

## 7. Expert Consultation

- **expert-backend**: Core changes in `vllm_mlx/server.py` and the new parser module. Review server/parser integration, argparse wiring, and FastAPI handler edits.
- **expert-testing**: Unit test coverage for parser state machine (thinking → content boundary) and merge logic (name-collision behavior, empty vs populated `tools`).
- **Not needed**: frontend, design, security, performance (no attack surface change, no perf-critical code, no UI).

## 8. References

- Predecessor SPEC: `.moai/specs/SPEC-MIGRATE-QWEN36/`
- Live validation report: `.moai/reports/qwen36-validation-2026-04/report.md`
- Expert-debug root-cause analysis (conversational; assumptions codified above in §2.1)
- Affected files:
  - `vllm_mlx/reasoning/think_parser.py:133-140` (Bug 1 root cause)
  - `vllm_mlx/reasoning/qwen3_parser.py:60-61` (non-streaming reference behavior)
  - `vllm_mlx/server.py:~1483` (Bug 2 root cause)
  - `vllm_mlx/mcp/client.py` (existing helpers to reuse)
  - `vllm_mlx/config/models.py` (new helper location)
