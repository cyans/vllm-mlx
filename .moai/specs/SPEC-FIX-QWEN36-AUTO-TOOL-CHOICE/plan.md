---
id: SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE
phase: plan
---

# Implementation Plan

`@PLAN:FIX-QWEN36-AUTO-TOOL-CHOICE`

## Approach

Append `--enable-auto-tool-choice` to the vllm-mlx serve invocation in both Qwen3.6 launchers, gated by a `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE` env var for opt-out. Use the existing EXTRA_FLAGS-style construction to keep the diff minimal. Cover the flag-construction logic with a lightweight unit test that asserts the env-gate branches both behave correctly.

## Milestones (priority-ordered, no time estimates)

### Primary Goal (P0)

- M1: Edit `start-server.sh` — add env-gated flag append block (`@CODE:FIX-QWEN36-AUTO-TOOL-CHOICE/launcher`).
- M2: Edit `start-server-qwen36.sh` — same block with identical semantics.
- M3: Syntactic validation — `bash -n` on both scripts.

### Secondary Goal (P1)

- M4: Add `tests/test_launcher_flags.py` (`@TEST:FIX-QWEN36-AUTO-TOOL-CHOICE`) exercising:
  - Default invocation → flag present.
  - `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE=1` → flag absent.
  Implementation option: shell out with `bash -x` and grep the trace, or source the script with a stubbed `exec` wrapper.

### Final Goal (P2, live)

- M5: Live restart — kill current server, run `./start-server-qwen36.sh`, verify `--enable-auto-tool-choice` in process args (`ps` or server log).
- M6: Live request — POST `/v1/chat/completions` with MCP tools and without `tool_choice`, prompt "Search the web for MLX release notes", `max_tokens=2048`. Expect `finish_reason=tool_calls` and ≥1 `tool_calls` entry.

## Technical Approach

Bash block (conceptual, for both launchers):

```
EXTRA_ARGS=()
if [ "${VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE:-0}" != "1" ]; then
  EXTRA_ARGS+=(--enable-auto-tool-choice)
fi
# ... append to existing exec vllm-mlx serve command
```

Exact splicing point is the existing `serve` invocation in each script. Preserve any pre-existing `--tool-call-parser qwen` argument; the two flags are complementary.

## Architecture Notes

- No Python changes required — argparse already accepts the flag.
- No config-file changes — opt-out is env-var-only by design (no silent defaults drift).
- No cross-launcher coupling — each script carries its own copy of the block to stay self-contained.

## Risks and Mitigations

- R1: Flag name typo → silent no-op. Mitigation: unit test asserts exact string.
- R2: Client explicitly sends `tool_choice="none"` and expects no tools. Mitigation: REQ-N2 covered by server-side semantics (flag enables auto, does not override explicit choice).
- R3: Operator forgets env opt-out is available. Mitigation: inline comment above the block in both launchers.
- R4: Regression in 122B launcher. Mitigation: REQ-N1 — 122B script untouched; visual diff review.

## Dependencies

- Upstream: SPEC-FIX-QWEN36-RUNTIME (runtime parity baseline must hold).
- Downstream: none.

## Expert Consultation (optional)

- expert-devops: shell-level flag + env gate review.
- expert-testing: test harness shape for bash-level assertion.
