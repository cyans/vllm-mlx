# Session Summary — Qwen3.6 Migration, Runtime Fix, and Tool-Call Hardening

**Date**: 2026-04-17
**Branch**: `feature/spec-migrate-qwen36` (24 commits ahead of `main`, not pushed, not merged)
**Working directory**: `/Users/mac4/claude_apps/vllm-mlx`
**SPECs executed**: `SPEC-MIGRATE-QWEN36`, `SPEC-FIX-QWEN36-RUNTIME`, `SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE`, `SPEC-FIX-QWEN36-TOOL-CALL-STREAMING`

---

## 1. Executive Summary

**Verdict**: This session migrated the `vllm-mlx` default model from Qwen3.5-35B-A3B to Qwen3.6-35B-A3B end-to-end, then discovered and fixed four distinct runtime issues (streaming reasoning parser, MCP tool auto-injection, auto tool-choice flag, streaming tool-call XML leak) — all verified live against running servers on port 8001. The server now handles both reasoning and tool-calling correctly for OpenAI-compatible streaming and non-streaming clients.

**Scoreboard**:
- SPECs authored: **4** — MIGRATE-QWEN36 (Phases 0-3), FIX-QWEN36-RUNTIME (Phases 1-2), FIX-QWEN36-AUTO-TOOL-CHOICE (Phase 1), FIX-QWEN36-TOOL-CALL-STREAMING (Phase 1).
- Phases executed: **9** total.
- Commits landed: **24** on `feature/spec-migrate-qwen36`, not yet pushed.
- Live verifications: **4** (Qwen3.6 streaming `delta.content`; MCP auto-inject `tool_calls`; auto tool-choice without explicit `tool_choice`; streaming `delta.tool_calls` without XML leak).
- Operational mode switches: perf tuning (prefix-cache ON + KV 0.4), then auto-inject OFF for plugin co-existence.

**Consolidated quality gate**: `SESSION GATE: CLEARED` (46/46 HARD acceptance items PASS at first checkpoint; additional 11 HARD items from AUTO-TOOL-CHOICE and TOOL-CALL-STREAMING SPECs verified live, see Section 5).

---

## 2. Timeline

1. User's initial ask: "Qwen3.6-35B-A3B이 Qwen3.5-35B-A3B를 대체할 수 있는지 확인 + 계획" → verify feasibility and plan the migration.
2. Phase 0 feasibility checks (`expert-devops`, read-only): MLX 4-bit quant exists, mlx-lm 0.30.7 supports Gated DeltaNet hybrid, `model_type=qwen3_5_moe`, EOS token `<|im_end|>` unchanged → GATE PASS.
3. `SPEC-MIGRATE-QWEN36` authored → Phase 1 centralization → Phase 2 parallel 3.6 support → Phase 3 default cutover.
4. Deeper validation (`expert-testing` A+C tracks) discovered 2 runtime bugs: streaming reasoning parser all-to-reasoning, MCP tools never auto-injected.
5. `SPEC-FIX-QWEN36-RUNTIME` authored → Phase 1 Qwen36ReasoningParser + auto-resolver → Phase 2 opt-in MCP auto-inject flag. Live verified.
6. Performance diagnosis: TTFT 3.38s due to `--disable-prefix-cache` + 2K MCP schema re-prefill every request. Launchers tuned: prefix cache ON + `--cache-memory-percent 0.4`.
7. User reported streaming tool-call loop with Obsidian client → Phase 3 investigation revealed HF-required `--enable-auto-tool-choice` flag missing.
8. `SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE` authored → Phase 1 launcher flag + qwen3_coder parser resolution. Live verified: `tool_calls` emitted without explicit `tool_choice`.
9. User reported the loop again → streaming-specific diagnosis revealed `<tool_call>` XML leaking into `delta.content` because the reasoning parser branch in `stream_chat_completion()` bypassed the tool parser.
10. `SPEC-FIX-QWEN36-TOOL-CALL-STREAMING` authored → Phase 1 `chain_reasoning_and_tool_parsers()` helper unifies both branches. Live verified.
11. Consulting: reviewed user's `SPEC-PLUGIN-001` (Obsidian Vault Agent Plugin, planning status) and `SPEC-VAULT-SUMMARY v1.2.0` (summary/writing-material tools) — feedback delivered for their own review cycle.
12. Consulting: reviewed MCP OAuth Setup Guide doc (Claude Code v2.1.30+) — gap analysis delivered.
13. Final operational switch: restarted server with `VLLM_MLX_AUTO_INJECT_MCP_TOOLS` unset so the user's Obsidian Copilot plugin can use its own web-search tool without collision with MCP's `web-search__tavily_search`.

---

## 3. SPEC-MIGRATE-QWEN36 — Summary Table

| Phase | Title | Agent | Files | Tests added | Acceptance | Commit SHAs |
|---|---|---|---|---|---|---|
| 0 | Feasibility gate | expert-devops | 0 (report only) | 0 | P0-AC1..6 | planning commit `b607f29` |
| 1 | Centralization (behavior-preserving) | manager-ddd | 10 | 33 | P1-AC1..9 | `277a244`, `f317989`, `d3f0d42`, `d90477b`, `e0afd00` |
| 2 | Parallel 3.6 support | manager-ddd | 5 | 11 | P2-AC1..8 | `af2e570`, `315e522`, `36b8844` |
| 3 | Default cutover | manager-ddd | 3 | 5 | P3-AC1..5 | `4c43253`, `c330072` |
| 4 | A/B benchmark | deferred | — | — | P4-AC1..5 | not executed |
| 5 | Optional cleanup | deferred | — | — | P5-AC1..3 | not executed |

---

## 4. SPEC-FIX-QWEN36-RUNTIME — Summary Table

| Phase | Title | Agent | Files | Tests added | Acceptance | Commit SHAs |
|---|---|---|---|---|---|---|
| 0.5 | Dual diagnosis (Bug 1 + Bug 2) | 2× expert-debug (parallel, read-only) | 0 | 0 | diagnosis → REQ-* | planning commit `8781657` |
| 1 | Qwen36 reasoning parser + auto-resolve | manager-ddd | 9 | 20 | P1-AC1..7 | `8781657`, `a82cab0`, `d37bf1b`, `5da0d84` |
| 2 | Opt-in MCP auto-inject flag | manager-ddd | 7 | 21 | P2-AC1..7 | `e517939`, `da99dcb`, `be3949e` |

---

## 4.1 SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE — Summary Table

Addresses tool-call "planning prose loop" observed with the Obsidian Copilot client. Root cause: Qwen3.6's `chat_template.jinja` already instructs XML `<tool_call>` output and the server's existing `qwen` parser converts it to OpenAI `tool_calls` — but the HF-prescribed `--enable-auto-tool-choice` flag was missing from the launcher, so clients that omit `tool_choice` never got reliable tool invocation.

| Phase | Title | Agent | Files | Tests added | Acceptance | Commit SHAs |
|---|---|---|---|---|---|---|
| 0 | Planning | manager-spec | 3 (spec/plan/acceptance) | 0 | — | `63c4d2d` |
| 1 | Launcher flag + parser resolution + env opt-out | manager-ddd | 3 (2 modified + 1 new test) | 4 | P1-AC1..P1-AC6 | `2bb91fa` |

Key result: `tool_call_parser` now auto-resolved to `qwen3_coder` (preferred-available); request without `tool_choice` returns `finish_reason=tool_calls` with a valid `web-search__tavily_search` invocation.

---

## 4.2 SPEC-FIX-QWEN36-TOOL-CALL-STREAMING — Summary Table

Addresses a streaming-specific leak discovered after AUTO-TOOL-CHOICE was deployed: the XML `<tool_call>...</tool_call>` block was being emitted on the wire as `delta.content` text rather than as `delta.tool_calls`. Root cause: `stream_chat_completion()` had two mutually-exclusive branches for reasoning parser vs tool parser, so when `--reasoning-parser qwen36` was active the tool parser was never invoked in the streaming loop.

| Phase | Title | Agent | Files | Tests added | Acceptance | Commit SHAs |
|---|---|---|---|---|---|---|
| 0 | Planning + ANALYZE (diagnostic findings in spec.md §2) | manager-spec | 3 (spec/plan/acceptance) | 0 | — | `ce74705` |
| 1 | Parser chain helper + SSE loop unification | manager-ddd | 2 (1 modified + 1 new test) | 5 | P1-AC1..P1-AC6 | `d8f1d25` |

Key result: streaming request for a tool-triggering prompt now emits `delta.tool_calls[0].function.name="web-search__tavily_search"`, concatenated `delta.content` contains no `<tool_call>` substring, final `finish_reason="tool_calls"`.

---

## 5. Acceptance Matrix (consolidated roll-up)

| SPEC | HARD | PASS | DEFERRED | FAIL |
|---|---|---|---|---|
| SPEC-MIGRATE-QWEN36 | 29 | 29 | 0 | 0 |
| SPEC-FIX-QWEN36-RUNTIME | 17 | 17 | 0 | 0 |
| SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE | 5 | 5 | 0 | 0 |
| SPEC-FIX-QWEN36-TOOL-CALL-STREAMING | 6 | 6 | 0 | 0 |
| **Combined** | **57** | **57** | **0** | **0** |

Full per-item matrix for the first two SPECs is in `quality-gate.md`; the two newer SPECs' acceptance items are verified inline in their respective `acceptance.md` files under `.moai/specs/SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE/` and `.moai/specs/SPEC-FIX-QWEN36-TOOL-CALL-STREAMING/`.

---

## 6. Live Verification Evidence

### 6.1 Phase 1 live — Qwen3.6 default + `qwen36` reasoning parser
```
./start-server.sh
# Server log:
#   INFO:vllm_mlx.cli:Reasoning parser enabled: qwen36
#   INFO:vllm_mlx.server:Starting server at http://0.0.0.0:8001

curl -sN -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit",
       "messages":[{"role":"user","content":"Say hello"}],
       "max_tokens":20,"stream":true}'
# Verified: at least one SSE chunk with non-null delta.content.
# Pre-fix: delta.content always null (all tokens in reasoning_content).
# Post-fix: delta.content populated; finish_reason="stop".
```

### 6.2 Phase 2 live — MCP tool auto-inject (opt-in)
```
VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1 ./start-server.sh
# Server log:
#   INFO:vllm_mlx.api.mcp_inject:MCP auto-injection: enabled
#   INFO:vllm_mlx.server:MCP initialized with 5 tools

curl -sS -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit",
       "messages":[{"role":"user","content":"Search the web for MLX release notes"}],
       "max_tokens":256}'
# Verified output:
#   finish_reason: "tool_calls"
#   tool_calls[0].function.name: "web-search__tavily_search"
```

### 6.3 AUTO-TOOL-CHOICE live — no explicit `tool_choice` needed
```
./start-server.sh
# Server log:
#   INFO:vllm_mlx.cli:Tool parser selection: user=qwen3_coder, resolved=qwen3_coder (preferred-available)
#   INFO:vllm_mlx.cli:Tool calling: ENABLED (parser: qwen3_coder)

curl -sS -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit",
       "messages":[{"role":"user","content":"Search the web for MLX release notes"}],
       "max_tokens":2048}'
# Request omits tool_choice. With --enable-auto-tool-choice + MCP auto-inject:
#   finish_reason: "tool_calls"
#   tool_calls[0].function.name: "web-search__tavily_search"
#   tool_calls[0].function.arguments: {"query":"MLX framework release notes","max_results":5}
```

### 6.4 TOOL-CALL-STREAMING live — no XML leak in streaming
```
./start-server.sh   # (auto-tool-choice + qwen3_coder + auto-inject on)
curl -sN -X POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen3.6-35B-A3B-4bit",
       "messages":[{"role":"user","content":"Search the web for Qwen 3.6 release notes"}],
       "max_tokens":512,"stream":true}'
# Verified SSE behavior post-fix:
#   At least one chunk has non-empty delta.tool_calls
#   Concatenated delta.content has NO "<tool_call>" substring
#   Final chunk: finish_reason="tool_calls"
# Pre-fix (before d8f1d25): concat content contained full <tool_call>...</tool_call> XML,
# delta.tool_calls was always empty, OpenAI-compatible streaming clients saw raw XML.
```

### 6.5 Emergency rollback to Qwen3.5
```
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit   # (deleted this session)
VLLM_MLX_MODEL_ID=mlx-community/Qwen3.5-35B-A3B-4bit ./start-server.sh
# resolve_reasoning_parser() auto-selects the legacy qwen3 parser for the 3.5 ID.
```

Validation report (full detail): `.moai/reports/qwen36-validation-2026-04/report.md`.

---

## 7. Environment State

### Runtime
- Python 3.11.14
- mlx 0.31.0, mlx-lm 0.30.7, mlx-vlm 0.3.12
- pytest 9.0.3, pytest-cov 7.1.0, ruff 0.15.11
- gh 2.90.0 (authenticated as `cyans`)

### HuggingFace cache
- `mlx-community/Qwen3.6-35B-A3B-4bit` — 19 GB (active default)
- `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` — 590 MB
- Deleted this session to free disk space:
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (~19 GB)
  - `mlx-community/Qwen3.5-122B-A10B-MINT-3bit-MLX` (~52 GB)

### Disk
- Free: ~82 GB (was ~11 GB pre-cleanup)

### Final launcher flags (both `start-server.sh` and `start-server-qwen36.sh`)
- `--reasoning-parser qwen36` (auto-resolved by model id)
- `--enable-auto-tool-choice` + `--tool-call-parser qwen3_coder` (auto-resolved, preferred-available)
- `--cache-memory-percent 0.4` (raised from 0.08)
- Prefix cache: **enabled** (previously `--disable-prefix-cache` was set)
- Env opt-outs:
  - `VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE=1` to suppress the auto tool-choice flag
  - `VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1` to opt IN to server-side MCP tool injection (default OFF)
  - `VLLM_MLX_MODEL_ID=<id>` to override the default model
  - `VLLM_MLX_REASONING_PARSER=<name>` to override parser auto-resolution
  - `VLLM_MLX_TOOL_CALL_PARSER=<name>` same for the tool-call parser

### Current server (at session end)
- PID 2688, RSS ~18.85 GB
- `./start-server.sh` invocation (no env vars) → **MCP auto-injection DISABLED**
- Chosen to avoid collision with Obsidian Copilot plugin's own `web_search` tool; plugin now owns web-search in the client context.

---

## 8. Remaining Work / Deferred Items

- **PR creation**: 24 commits on `feature/spec-migrate-qwen36` ready for PR; `gh auth` valid (`cyans`). Needs fork configuration against `waybarrios/vllm-mlx` or a user-owned fork push.
- **SPEC-MIGRATE-QWEN36 Phase 4** (A/B benchmark 3.5 vs 3.6): requires re-downloading 19 GB Qwen3.5 cache.
- **SPEC-FIX-QWEN36-RUNTIME REQ-O2**: optional thinking/answer separation heuristic in `Qwen36ReasoningParser` (e.g., "Final Answer:" / double-newline detection).
- **Qwen3.5-122B-MINT re-download**: if `PLAN.md` TurboQuant work needs it.
- **Obsidian plugin (SPEC-PLUGIN-001 / SPEC-VAULT-SUMMARY)**: planning docs only; no code yet. Reviewed this session, ready for v1.3.0 revision with P0+P1 improvements (see review notes in conversation history).
- **Plugin-side tool handling**: Obsidian plugin review delegated to a separate review agent to debug why the plugin doesn't present Qwen3.6's `tool_calls` response correctly — suspected Option 2 (tool_call emitted but plugin fails to send `tool_result` back, causing model to loop). Waiting for plugin-side diagnostic logs.
- **MCP OAuth doc review**: 9 gap items identified (CLI identity, sandbox/keychain quirks, OAuth flow type unspecified, env-var resolution unclear, refresh mechanism, multi-account, scope semantics, revoke path, `.moai/config/mcp-servers.yaml` reference).
- **Validation report annotation (SOFT)**: add "Resolved by SPEC-FIX-QWEN36-RUNTIME" footer to bugs A1/A4 section of `qwen36-validation-2026-04/report.md`.

---

## 9. Commit Log

```
feature/spec-migrate-qwen36 (24 commits ahead of main)
│
├── SPEC-MIGRATE-QWEN36 (11 commits)
│   ├─ Phase 0 (planning)
│   │   └── b607f29  docs(spec): add SPEC-MIGRATE-QWEN36 planning documents
│   ├─ Phase 1 (centralize configuration, behavior-preserving)
│   │   ├── 277a244  feat(config): add vllm_mlx.config.models single-source-of-truth
│   │   ├── f317989  refactor(models/llm): route EOS-patch decision through config
│   │   ├── d3f0d42  refactor(cli,server): resolve tool-call parser via config
│   │   ├── d90477b  refactor(launcher): read model id and parser from config
│   │   └── e0afd00  refactor(examples,scripts): import model ids from config
│   ├─ Phase 2 (parallel 3.6 support)
│   │   ├── af2e570  feat(config): add QWEN36_PROFILE and tighten EOS patterns
│   │   ├── 315e522  feat(launcher): add start-server-qwen36.sh opt-in launcher
│   │   └── 36b8844  feat(examples): add Qwen3.6 MLLM example
│   └─ Phase 3 (default cutover)
│       ├── 4c43253  feat(models): cutover default to Qwen3.6-35B-A3B
│       └── c330072  docs: add CHANGELOG.md documenting Qwen3.6 cutover + rollback
│
├── SPEC-FIX-QWEN36-RUNTIME (7 commits)
│   ├─ Phase 1 (Qwen36 reasoning parser + auto-resolve)
│   │   ├── 8781657  docs(spec): add SPEC-FIX-QWEN36-RUNTIME + validation evidence
│   │   ├── a82cab0  feat(reasoning): add Qwen36ReasoningParser for tag-less thinking
│   │   ├── d37bf1b  feat(config): add resolve_reasoning_parser with model-id auto-select
│   │   └── 5da0d84  refactor(launcher): resolve reasoning parser via config helper
│   └─ Phase 2 (opt-in MCP auto-inject flag)
│       ├── e517939  feat(mcp): add merge_tool_lists + resolve_effective_tools helpers
│       ├── da99dcb  feat(server,cli): wire --auto-inject-mcp-tools flag
│       └── be3949e  refactor(launcher): propagate VLLM_MLX_AUTO_INJECT_MCP_TOOLS env
│
├── Performance tuning + session reports (2 commits)
│   ├── e04551c  perf(launcher): enable prefix cache and raise KV cache memory to 0.4
│   └── 9aaeea3  docs(reports): add Qwen3.6 session summary and quality gate report
│
├── SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE (2 commits)
│   ├── 63c4d2d  docs(spec): add SPEC-FIX-QWEN36-AUTO-TOOL-CHOICE planning documents
│   └── 2bb91fa  feat(launcher): enable auto-tool-choice by default with parser resolution
│
└── SPEC-FIX-QWEN36-TOOL-CALL-STREAMING (2 commits)
    ├── ce74705  docs(spec): add SPEC-FIX-QWEN36-TOOL-CALL-STREAMING planning documents
    └── d8f1d25  feat(server): chain reasoning and tool-call parsers in streaming path
```

---

## 10. Consulting Activities (no commits — external to this repo)

### 10.1 Obsidian plugin SPEC reviews
- **SPEC-PLUGIN-001** (Obsidian Vault Agent Plugin, Planned): 5-tool design (Vault Search, Web Search, Write-to-File, Replace-in-File, YouTube Transcription). Review identified 3 P0 issues (default endpoint is Ollama's 11434, should be 8001; tool-calling strategy §5.3 needs Qwen3.6-aware branch; web-search toggle must coordinate with server MCP to avoid duplication).
- **SPEC-VAULT-SUMMARY v1.2.0** (Reviewed status): vault summarize + writing-material extraction via `vault_read_contents`, `vault_summarize`, `outline_organize`. Review identified 3 P0 issues (undefined "Obsidian CLI" identity, missing output-language requirement, missing cross-reference to SPEC-PLUGIN-001 dependency) + 4 P1 recommendations (combined_summary semantics, chunking algorithm, all-fail scenario, Qwen3.6 tool-calling compat note).
- Delivered back to user for their own v1.3.0 revision cycle.

### 10.2 MCP OAuth Setup Guide review
- Input: "MCP OAuth Setup Guide" v1.0.0 (2026-02-20) documenting `.mcp.json` OAuth field (`clientId`, `callbackPort`) with keychain-stored secrets.
- Identified 9 gaps: CLI identity missing, sandbox/keychain interaction (user hit this with gh earlier), OAuth flow type unspecified (Auth Code vs PKCE), env-var resolution subject unclear, token refresh mechanism opaque, no multi-account guidance, scope semantics misleading, no revoke/logout path, `.moai/config/mcp-servers.yaml` referenced but not present in user's project.

### 10.3 Architecture consultation — server-side MCP vs plugin-internal tools
- Decision framework delivered: both simultaneously ON is the direct cause of the "I will use X. Query: ..." loop (tool name collision + `tool_result` round-trip failure).
- For single-user Obsidian Copilot context → plugin-side tool ownership is the right choice; server MCP auto-inject should be OFF.
- For future SPEC-PLUGIN-001 self-built plugin → server MCP delegation is preferable (single source of tool config, reusable across clients).
- User chose to keep the server auto-inject OFF and delegate the plugin-side diagnostic to a separate review agent.

---

## 11. Next-Step Recommendations

**Option A — Create PR**: push branch to a user-owned fork, then `gh pr create --base main` against `waybarrios/vllm-mlx`. All 24 commits ready for review.

**Option B — Complete A/B benchmark** (SPEC-MIGRATE-QWEN36 Phase 4): re-download Qwen3.5 (~19 GB), write `scripts/benchmark_qwen35_vs_qwen36.py`, emit A/B report.

**Option C — Implement REQ-O2 heuristic** (Qwen36ReasoningParser): add thinking/answer split detection ("Final Answer:" / double-newline boundary).

**Option D — Start Obsidian plugin implementation**: scaffold TypeScript project per SPEC-PLUGIN-001 + SPEC-VAULT-SUMMARY (after v1.3.0 revision). Leverage server-side MCP delegation or plugin-internal tools per user preference.

**Option E — Wait for plugin-side diagnostic logs** from the separate review agent to decide next server-side fixes (if any).

---

<moai>DONE</moai>
<moai>COMPLETE</moai>
