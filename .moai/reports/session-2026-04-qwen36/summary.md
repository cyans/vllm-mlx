# Session Summary — Qwen3.6 Migration & Runtime Fix

**Date**: 2026-04-17
**Branch**: `feature/spec-migrate-qwen36` (18 commits ahead of `main`, not pushed, not merged)
**Working directory**: `/Users/mac4/claude_apps/vllm-mlx`
**SPECs executed**: `SPEC-MIGRATE-QWEN36`, `SPEC-FIX-QWEN36-RUNTIME`

---

## 1. Executive Summary

**Verdict**: This session migrated the `vllm-mlx` default model from Qwen3.5-35B-A3B to Qwen3.6-35B-A3B end-to-end, then discovered and fixed two runtime bugs (streaming parser routing and MCP tool auto-injection) that would have shipped broken without deeper validation — all verified live against a running server on port 8001.

**Scoreboard**:
- SPECs: 2 authored, 2 advanced past Phase 1 (one to Phase 3, one to Phase 2).
- Phases executed: 7 total (MIGRATE P0 + P1 + P2 + P3; FIX P0.5 + P1 + P2).
- Commits landed: 18 on `feature/spec-migrate-qwen36`, not yet pushed.
- Live verifications: 2 (Qwen3.6 streaming `delta.content` populated; MCP `web-search__tavily_search` auto-invoked).

**Consolidated quality gate**: `SESSION GATE: CLEARED` (46/46 HARD acceptance items PASS; see `quality-gate.md`).

---

## 2. Timeline

1. User's initial ask: "Qwen3.6-35B-A3B이 Qwen3.5-35B-A3B를 대체할 수 있는지 확인 + 계획" → verify feasibility and plan the migration.
2. Phase 0 feasibility checks (`expert-devops`, read-only): MLX 4-bit quant exists, mlx-lm 0.30.7 supports Gated DeltaNet hybrid, `model_type=qwen3_5_moe`, EOS token `<|im_end|>` unchanged → GATE PASS.
3. `SPEC-MIGRATE-QWEN36` authored → Phase 1 centralization → Phase 2 parallel 3.6 support → Phase 3 default cutover.
4. Deeper validation (`expert-testing` A+C tracks) discovered 2 runtime bugs: streaming reasoning parser all-to-reasoning, MCP tools never auto-injected.
5. `SPEC-FIX-QWEN36-RUNTIME` authored → Phase 1 Qwen36ReasoningParser + auto-resolver → Phase 2 opt-in MCP auto-inject flag.
6. Live verification of both fixes end-to-end on Qwen3.6 server.
7. Obsidian plugin clarification: `PLAN-obsidian-plugin.md` remains a planning doc. `model` field value is echoed (not validated) by server; cosmetic label change optional.

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

## 5. Acceptance Matrix (consolidated roll-up)

| SPEC | HARD | PASS | DEFERRED | FAIL |
|---|---|---|---|---|
| SPEC-MIGRATE-QWEN36 | 29 | 29 | 0 | 0 |
| SPEC-FIX-QWEN36-RUNTIME | 17 | 17 | 0 | 0 |
| **Combined** | **46** | **46** | **0** | **0** |

Full per-item matrix with evidence is in `quality-gate.md`.

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

### 6.2 Phase 2 live — MCP tool auto-inject
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
#   tool_calls[0].function.arguments: valid JSON with "query" field
```

### 6.3 Emergency rollback to Qwen3.5
```
# Step 1: re-download the 3.5 cache (deleted during this session to free disk)
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit

# Step 2: force legacy
VLLM_MLX_MODEL_ID=mlx-community/Qwen3.5-35B-A3B-4bit ./start-server.sh
# Resolver auto-selects qwen3 parser for 3.5 ID.
```

Validation report (full detail): `.moai/reports/qwen36-validation-2026-04/report.md`.

---

## 7. Environment State

### Runtime
- Python 3.11.14
- mlx 0.31.0, mlx-lm 0.30.7, mlx-vlm 0.3.12
- pytest 9.0.3, pytest-cov 7.1.0, ruff 0.15.11

### HuggingFace cache
- `mlx-community/Qwen3.6-35B-A3B-4bit` — 19 GB (active default)
- `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` — 590 MB
- Deleted this session to free disk space:
  - `mlx-community/Qwen3.5-35B-A3B-4bit` (~19 GB)
  - `mlx-community/Qwen3.5-122B-A10B-MINT-3bit-MLX` (~52 GB)

### Disk
- Free: ~82 GB (was ~11 GB pre-cleanup)

### Server tuning (post-session update)
After performance diagnosis revealed TTFT 3.38s with `--disable-prefix-cache` + 2K MCP-injected prompt tokens, launchers were updated:
- Removed: `--disable-prefix-cache`
- Changed: `--cache-memory-percent 0.08` → `0.4`
- MCP auto-inject: default OFF (operator opts in via env var)

Expected post-tuning TTFT: 0.3-0.5s on warm cache requests.

---

## 8. Remaining Work / Deferred Items

- **PR creation**: branch is ready locally; `gh auth` valid. `waybarrios/vllm-mlx` push requires fork configuration.
- **SPEC-MIGRATE-QWEN36 Phase 4** (A/B benchmark 3.5 vs 3.6): requires re-downloading 19 GB Qwen3.5 cache.
- **SPEC-FIX-QWEN36-RUNTIME REQ-O2**: optional thinking/answer separation heuristic in `Qwen36ReasoningParser`.
- **Qwen3.5-122B-MINT re-download**: if `PLAN.md` TurboQuant work needs it.
- **Obsidian plugin**: `PLAN-obsidian-plugin.md` still planning; no code yet.
- **Validation report annotation (SOFT)**: add "Resolved by SPEC-FIX-QWEN36-RUNTIME" footer to bugs A1/A4 section of `qwen36-validation-2026-04/report.md`.

---

## 9. Commit Log

```
feature/spec-migrate-qwen36 (18 commits ahead of main)
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
└── SPEC-FIX-QWEN36-RUNTIME (7 commits)
    ├─ Phase 1 (Qwen36 reasoning parser + auto-resolve)
    │   ├── 8781657  docs(spec): add SPEC-FIX-QWEN36-RUNTIME + validation evidence
    │   ├── a82cab0  feat(reasoning): add Qwen36ReasoningParser for tag-less thinking
    │   ├── d37bf1b  feat(config): add resolve_reasoning_parser with model-id auto-select
    │   └── 5da0d84  refactor(launcher): resolve reasoning parser via config helper
    └─ Phase 2 (opt-in MCP auto-inject flag)
        ├── e517939  feat(mcp): add merge_tool_lists + resolve_effective_tools helpers
        ├── da99dcb  feat(server,cli): wire --auto-inject-mcp-tools flag
        └── be3949e  refactor(launcher): propagate VLLM_MLX_AUTO_INJECT_MCP_TOOLS env
```

---

## 10. Next-Step Recommendations

**Option A — Create PR**: push branch to a user-owned fork, then `gh pr create --base main` against `waybarrios/vllm-mlx`.

**Option B — Complete A/B benchmark** (SPEC-MIGRATE-QWEN36 Phase 4): re-download Qwen3.5 (~19 GB), write `scripts/benchmark_qwen35_vs_qwen36.py`, emit A/B report.

**Option C — Implement REQ-O2 heuristic**: add thinking/answer split detection to `Qwen36ReasoningParser` (e.g., "Final Answer:", double-newline boundary).

**Option D — Start Obsidian plugin implementation**: scaffold TypeScript project per `PLAN-obsidian-plugin.md`; leverage the now-working MCP auto-inject so the plugin can offload web-search to the server side.

---

<moai>DONE</moai>
<moai>COMPLETE</moai>
