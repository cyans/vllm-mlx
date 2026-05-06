---
id: SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE
phase: acceptance
---

# Acceptance Criteria

`@ACCEPT:FIX-QWEN36-AUTO-TOOL-CHOICE`

## Phase 1 — HARD (must pass before merge)

### P1-AC1: `start-server.sh` passes the flag by default

- Given: `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE` is unset.
- When: the script reaches the `vllm-mlx serve` invocation.
- Then: the resolved argv contains `--enable-auto-tool-choice`.

### P1-AC2: `start-server-qwen36.sh` passes the flag by default

- Given: `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE` is unset.
- When: the script reaches the `vllm-mlx serve` invocation.
- Then: the resolved argv contains `--enable-auto-tool-choice`.

### P1-AC3: Both scripts are syntactically valid

- Given: the modified launcher scripts.
- When: `bash -n start-server.sh` and `bash -n start-server-qwen36.sh` are executed.
- Then: both return exit code 0 with no output.

### P1-AC4: Env opt-out suppresses the flag

- Given: `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE=1` is exported.
- When: either launcher script is invoked.
- Then: the resolved argv does NOT contain `--enable-auto-tool-choice`.
- And: all other flags are unchanged.

### P1-AC5 (LIVE): Auto tool choice works end-to-end

- Given: the server is restarted with `./start-server-qwen36.sh`.
- And: the operator sends POST `/v1/chat/completions` with:
  - MCP tools registered (e.g., `web-search__tavily_search`).
  - No `tool_choice` field in the request body.
  - Prompt: "Search the web for MLX release notes".
  - `max_tokens: 2048`.
- When: the response returns.
- Then: `finish_reason == "tool_calls"`.
- And: `len(choices[0].message.tool_calls) >= 1`.
- And: at least one `tool_call.function.name` matches a registered MCP tool.

## Phase 1 — SOFT (recommended, non-blocking)

### P1-AC6: Launcher flag test exists and passes

- Given: `tests/test_launcher_flags.py` (`@TEST:FIX-QWEN36-AUTO-TOOL-CHOICE`).
- When: `pytest tests/test_launcher_flags.py -v` is run.
- Then: all cases pass, covering:
  - Default env → flag appears in constructed argv.
  - `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE=1` → flag absent.
  - 122B launcher unchanged (REQ-N1 regression guard, optional).

## Out-of-Scope Verifications (REQ-N1/N2)

- `start-server-122b-mint.sh` diff: zero changes (REQ-N1).
- Explicit `tool_choice="none"` client request: returns no tool_calls (REQ-N2; relies on server semantics, no launcher behavior change).

## Definition of Done

- [ ] P1-AC1 through P1-AC5 all pass.
- [ ] P1-AC6 passes OR has documented deferral rationale.
- [ ] No changes to `start-server-122b-mint.sh` (verified by `git diff --stat`).
- [ ] Commit tagged with `@CODE:FIX-QWEN36-AUTO-TOOL-CHOICE/launcher` in message body.
