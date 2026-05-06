---
id: SPEC-FIX-QWEN36-RUNTIME
artifact: acceptance
status: Planned
development_mode: ddd
tags:
  - "@ACCEPT:FIX-QWEN36-RUNTIME"
  - "@SPEC:FIX-QWEN36-RUNTIME"
---

# Acceptance Criteria — SPEC-FIX-QWEN36-RUNTIME

`@ACCEPT:FIX-QWEN36-RUNTIME` — Given/When/Then scenarios, quality gates, and Definition of Done for the Qwen3.6 runtime fixes.

> Every HARD criterion must pass before the SPEC may be marked Complete.
> Scenarios map directly to EARS requirements in `spec.md`.

## 1. Phase 1 — Streaming Reasoning Parser (Bug 1)

### P1-AC1 [HARD] — Parser module exists and conforms to interface

**Maps to**: REQ-U1, spec §4.1
**Given** the repository on branch `feature/spec-migrate-qwen36` (or successor)
**When** Phase 1 implementation is complete
**Then** the file `vllm_mlx/reasoning/qwen36_parser.py` **shall** exist AND
**And** it **shall** export `Qwen36ReasoningParser` class implementing the reasoning-parser interface (`extract_reasoning`, `extract_reasoning_streaming`)
**And** `ruff check vllm_mlx/reasoning/qwen36_parser.py` **shall** exit 0
**And** the file **shall** carry the `@CODE:FIX-QWEN36-RUNTIME/parser` TAG in a module-level comment or docstring

### P1-AC2 [HARD] — Parser registration and auto-selection

**Maps to**: REQ-E1, REQ-O3, spec §4.2, §4.3
**Given** a built server binary / Python entry point
**When** the server is launched with `--reasoning-parser qwen36`
**Then** the server **shall** resolve the parser to `Qwen36ReasoningParser` AND **shall not** raise an "unknown parser" error
**And** when the server is launched with `--model Qwen/Qwen3.6-...` and no explicit `--reasoning-parser` flag
**Then** `resolve_reasoning_parser()` in `vllm_mlx/config/models.py` **shall** return `"qwen36"` AND the server **shall** use `Qwen36ReasoningParser`
**And** when the server is launched with a non-Qwen3.6 model and no explicit parser flag
**Then** `resolve_reasoning_parser()` **shall** return the pre-existing default (e.g., `"qwen3"`) and pre-existing behavior **shall** be preserved

### P1-AC3 [HARD] — Streaming emits non-null delta.content

**Maps to**: REQ-S1, REQ-U1, Bug 1 root cause
**Given** a running server with the `qwen36` parser active against a Qwen3.6 model
**When** a client issues a streaming `/v1/chat/completions` request with a simple prompt (e.g., `"What is the capital of France?"`)
**Then** at least one streamed `delta` chunk **shall** have non-null, non-empty `delta.content`
**And** the final chunk **shall** have `finish_reason` equal to `"stop"` (or an equivalent terminal reason)
**And** a unit test (streaming fixture) OR an integration test **shall** encode this behavior

### P1-AC4 [HARD] — Non-streaming regression guard

**Maps to**: REQ-U3
**Given** a running server with the `qwen36` parser active
**When** a client issues a non-streaming `/v1/chat/completions` request equivalent to the "Paris example" in `.moai/reports/qwen36-validation-2026-04/report.md`
**Then** `message.content` **shall** contain the answer (non-empty)
**And** `message.reasoning_content` **shall** contain the thinking prose (non-empty if the model produced a thinking prefix)
**And** `finish_reason` **shall** equal `"stop"`

### P1-AC5 [HARD] — Qwen3.5 regression guard

**Maps to**: REQ-N1, REQ-U3
**Given** characterization fixtures captured during the PRESERVE step from the existing `qwen3` parser on Qwen3.5 inputs
**When** the test suite runs after Phase 1 is merged
**Then** every Qwen3.5 fixture **shall** produce output byte-identical (or semantically equivalent under documented tolerance) to its pre-change snapshot
**And** `git diff vllm_mlx/reasoning/qwen3_parser.py vllm_mlx/reasoning/think_parser.py` **shall** be empty (files unchanged)

### P1-AC6 [HARD] — Quality gates (Phase 1)

**Maps to**: TRUST 5
**Given** Phase 1 implementation is merged
**Then** `ruff check vllm_mlx/reasoning/ vllm_mlx/config/models.py` **shall** exit 0
**And** `pytest tests/reasoning/` **shall** exit 0 with all tests green
**And** coverage on `vllm_mlx/reasoning/qwen36_parser.py` **shall** be ≥ 85%
**And** `vllm_mlx/config/models.py::resolve_reasoning_parser` **shall** have ≥ 85% coverage on its new branches

### P1-AC7 [HARD] — Launcher scripts updated

**Maps to**: spec §4.4
**Given** `start-server.sh` and `start-server-qwen36.sh` after Phase 1
**When** either script is invoked with a Qwen3.6 model
**Then** the invocation **shall not** hardcode `--reasoning-parser qwen3`
**And** the scripts **shall** either (a) omit the flag and rely on auto-resolution, OR (b) honor a `$REASONING_PARSER` env var that defaults to auto-resolution when unset

## 2. Phase 2 — MCP Tool Auto-Injection (Bug 2)

### P2-AC1 [HARD] — CLI flag parses

**Maps to**: REQ-U2, spec §4.5
**Given** the Phase 2 implementation is merged
**When** a user runs `python -m vllm_mlx --help` (or equivalent)
**Then** `--auto-inject-mcp-tools` **shall** appear in the help output with descriptive text
**And** when the flag is omitted from the command line, the server **shall** default to auto-injection DISABLED
**And** when the flag is provided, the server **shall** set `self.auto_inject_mcp_tools` to True

### P2-AC2 [HARD] — Flag-off regression guard (zero-change behavior)

**Maps to**: REQ-N2, REQ-S2
**Given** a running server started WITHOUT `--auto-inject-mcp-tools`
**And** an MCP server registered and healthy
**When** a client issues `/v1/chat/completions` with no `tools` field, using a prompt that would benefit from an MCP tool
**Then** the response **shall not** contain any `tool_calls` attributable to MCP auto-injection
**And** the request object reaching the downstream inference path **shall** have `tools` unchanged (byte-identical to pre-change code path)

### P2-AC3 [HARD] — Flag-on, no client tools, auto-inject fires

**Maps to**: REQ-E2
**Given** a running server started WITH `--auto-inject-mcp-tools`
**And** an MCP server registered and healthy with at least one tool exposed
**When** a client issues `/v1/chat/completions` with no `tools` field and a prompt that requires that MCP tool
**Then** the request's effective `tools` list reaching inference **shall** be equal to `self._mcp_manager.get_merged_tools()`
**And** the model's response **shall** include a `tool_calls` entry targeting the injected MCP tool (integration-level assertion)

### P2-AC4 [HARD] — Flag-on, client tools present, merge preserves client precedence

**Maps to**: REQ-E3, REQ-N4
**Given** a running server started WITH `--auto-inject-mcp-tools`
**And** an MCP tool registered under function name `search_docs`
**When** a client issues `/v1/chat/completions` with `tools=[{type: "function", function: {name: "search_docs", ...(client schema)...}}]`
**Then** the effective tools list **shall** contain exactly one entry named `search_docs`
**And** that entry **shall** be the client-supplied schema (not the MCP schema)
**And** any additional MCP tools with non-colliding names **shall** appear in the merged list
**And** a DEBUG-level log **may** record the collision (but is not required)

### P2-AC5 [HARD] — Startup INFO log emits status

**Maps to**: REQ-E4
**Given** a server launched with any combination of `--auto-inject-mcp-tools` and `--reasoning-parser`
**When** the server completes startup
**Then** the stdout/stderr log **shall** contain an INFO-level line of the form `"MCP tool auto-injection: enabled"` or `"MCP tool auto-injection: disabled"`
**And** the log **shall** contain an INFO-level line naming the active reasoning parser

### P2-AC6 [HARD] — Quality gates (Phase 2)

**Maps to**: TRUST 5
**Given** Phase 2 implementation is merged
**Then** `ruff check vllm_mlx/server.py vllm_mlx/cli.py` **shall** exit 0
**And** `pytest tests/mcp/ tests/server/` (or equivalent) **shall** exit 0
**And** the new merge logic in `create_chat_completion` **shall** have ≥ 85% coverage on its new branches
**And** no pre-existing test **shall** regress

### P2-AC7 [SOFT] — Launcher env-var passthrough

**Maps to**: REQ-O1
**Given** `start-server.sh` or `start-server-qwen36.sh` updated per Phase 2
**When** the script is invoked with `VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1 ./start-server.sh`
**Then** the resulting server invocation **shall** include `--auto-inject-mcp-tools`
**And** when the env var is unset or `0`, the invocation **shall not** include the flag

## 3. Cross-Phase Acceptance

### C-AC1 [HARD] — Traceability tags present

**Maps to**: spec §6
**Given** all files touched by this SPEC
**Then** every new file (`qwen36_parser.py`, new test files) **shall** carry a `@CODE:FIX-QWEN36-RUNTIME/*` or `@TEST:FIX-QWEN36-RUNTIME/*` TAG in a module-level comment or docstring
**And** the SPEC artifacts (`spec.md`, `plan.md`, `acceptance.md`) **shall** carry matching `@SPEC`, `@PLAN`, `@ACCEPT` tags

### C-AC2 [HARD] — Conventional commit references

**Given** the Git history for changes landing under this SPEC
**Then** each commit message **shall** reference `SPEC-FIX-QWEN36-RUNTIME` (e.g., `fix(reasoning): add Qwen3.6 streaming parser (SPEC-FIX-QWEN36-RUNTIME)`)

### C-AC3 [HARD] — Full test suite green

**Given** both phases merged
**When** `pytest` (or the project's standard test command) runs in CI
**Then** the suite **shall** exit 0

### C-AC4 [SOFT] — Validation report annotated

**Maps to**: plan Final Goal 3.2
**Given** both phases merged
**Then** `.moai/reports/qwen36-validation-2026-04/report.md` **shall** carry a "Resolved by SPEC-FIX-QWEN36-RUNTIME" note for each of the two bugs

## 4. Test Scenarios

### 4.1 Unit — `tests/reasoning/test_qwen36_parser.py`

```
test_streaming_plain_text_routes_to_content()
    given: text chunk "Paris is the capital..." with no think tags
    when:  extract_reasoning_streaming is called
    then:  delta.content = "Paris is the capital..." AND delta.reasoning_content in (None, "")

test_streaming_multi_chunk_all_content()
    given: three sequential chunks of plain text
    when:  extract_reasoning_streaming is called for each
    then:  concatenated delta.content = full text AND reasoning_content stays empty across chunks

test_nonstreaming_plain_text_to_content()
    given: full response "Paris is the capital of France."
    when:  extract_reasoning is called
    then:  content = full text, reasoning_content = ""

test_nonstreaming_with_thinking_prefix()  # OPTIONAL (REQ-O2)
    given: "Here's a thinking process:\n1. Identify...\n\nFinal Answer: Paris."
    when:  extract_reasoning is called with heuristic enabled
    then:  reasoning_content contains thinking, content contains "Paris."

test_qwen3_5_regression()
    given: each Qwen3.5 snapshot fixture in tests/reasoning/fixtures/qwen3_5/
    when:  the Qwen3 (NOT qwen36) parser runs
    then:  output matches the snapshot byte-for-byte
```

### 4.2 Unit — `tests/mcp/test_auto_inject.py`

```
test_flag_off_no_tools_unchanged()
    given: server with auto_inject_mcp_tools=False, request.tools=None
    when:  create_chat_completion preprocessing runs
    then:  effective tools list is None (unchanged)

test_flag_off_client_tools_unchanged()
    given: server with auto_inject_mcp_tools=False, request.tools=[client_tool]
    when:  preprocessing runs
    then:  effective tools list == [client_tool]

test_flag_on_no_client_tools_injects_mcp()
    given: auto_inject=True, request.tools=None, mcp_manager has 2 tools
    when:  preprocessing runs
    then:  effective tools list == mcp_manager.get_merged_tools()

test_flag_on_client_no_collision_merge()
    given: auto_inject=True, request.tools=[{name: "client_a"}], mcp tools=[{name: "mcp_b"}]
    when:  preprocessing runs
    then:  effective tools list contains both, with "client_a" first

test_flag_on_client_collision_client_wins()
    given: auto_inject=True, client tool {name: "shared", client_schema}, mcp tool {name: "shared", mcp_schema}
    when:  preprocessing runs
    then:  effective tools list has exactly one entry named "shared" AND it matches client_schema
```

### 4.3 Integration — Smoke

```
integration_streaming_qwen36_emits_content()
    given: live server with Qwen3.6 model and qwen36 parser
    when:  streaming /v1/chat/completions with "What is the capital of France?"
    then:  at least one delta chunk has non-null delta.content AND finish_reason=stop on the last chunk

integration_mcp_auto_inject_triggers_tool_call()
    given: live server with --auto-inject-mcp-tools and at least one MCP tool registered
    when:  /v1/chat/completions with a prompt designed to invoke that tool, no client tools
    then:  response contains tool_calls referencing the MCP tool
```

## 5. Definition of Done

The SPEC is **Done** when all the following hold:

1. **All [HARD] criteria above pass** (P1-AC1 through P1-AC7, P2-AC1 through P2-AC6, C-AC1 through C-AC3).
2. **DDD cycle completed for each phase**: ANALYZE logged, PRESERVE characterization tests committed, IMPROVE implementation merged.
3. **TRUST 5 quality gates** all green:
   - Tested: pytest passes, coverage ≥ 85% on new code
   - Readable: ruff clean
   - Unified: module structure consistent with existing `vllm_mlx/reasoning/` and `vllm_mlx/config/` conventions
   - Secured: no new uncontrolled surface area; opt-in flag off by default
   - Trackable: all commits reference SPEC-ID; traceability tags present
4. **Integration smoke tests** (4.3) executed at least once against a live Qwen3.6 server and recorded.
5. **Qwen3.5 regression** verified (P1-AC5) — no behavior change for legacy users.
6. **Launcher scripts** updated (P1-AC7, P2-AC7).
7. **Validation report** annotated (C-AC4 [SOFT]).
8. **`/moai:3-sync`** executed to propagate documentation (README server-flag table, CHANGELOG entry).

## 6. Traceability

| AC ID | EARS Req | Code Artifact |
|-------|----------|---------------|
| P1-AC1 | REQ-U1 | `@CODE:FIX-QWEN36-RUNTIME/parser` |
| P1-AC2 | REQ-E1, REQ-O3 | `@CODE:FIX-QWEN36-RUNTIME/parser-registry`, `@CODE:FIX-QWEN36-RUNTIME/parser-resolver` |
| P1-AC3 | REQ-S1 | `@CODE:FIX-QWEN36-RUNTIME/parser`, `@TEST:FIX-QWEN36-RUNTIME/parser-unit`, `@TEST:FIX-QWEN36-RUNTIME/integration` |
| P1-AC4 | REQ-U3 | `@CODE:FIX-QWEN36-RUNTIME/parser`, `@TEST:FIX-QWEN36-RUNTIME/parser-unit` |
| P1-AC5 | REQ-N1, REQ-U3 | `@TEST:FIX-QWEN36-RUNTIME/parser-regression` |
| P1-AC6 | TRUST 5 | (all Phase 1 code) |
| P1-AC7 | spec §4.4 | `@CODE:FIX-QWEN36-RUNTIME/launcher` |
| P2-AC1 | REQ-U2 | `@CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject` |
| P2-AC2 | REQ-N2, REQ-S2 | `@CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject`, `@TEST:FIX-QWEN36-RUNTIME/mcp-merge` |
| P2-AC3 | REQ-E2 | `@CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject`, `@TEST:FIX-QWEN36-RUNTIME/integration` |
| P2-AC4 | REQ-E3, REQ-N4 | `@CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject`, `@TEST:FIX-QWEN36-RUNTIME/mcp-merge` |
| P2-AC5 | REQ-E4 | `@CODE:FIX-QWEN36-RUNTIME/mcp-auto-inject` |
| P2-AC6 | TRUST 5 | (all Phase 2 code) |
| P2-AC7 | REQ-O1 | `@CODE:FIX-QWEN36-RUNTIME/launcher` |
| C-AC1 | spec §6 | (all artifacts) |
| C-AC2 | Git conventions | commit messages |
| C-AC3 | TRUST 5 | CI pipeline |
| C-AC4 | plan 3.2 | `@REPORT:FIX-QWEN36-RUNTIME/validation` |

End of acceptance criteria.
