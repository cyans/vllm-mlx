---
id: SESSION-2026-04-QWEN36-QG
document_type: quality-gate
status: final
created: 2026-04-17
scope:
  - SPEC-MIGRATE-QWEN36
  - SPEC-FIX-QWEN36-RUNTIME
branch: feature/spec-migrate-qwen36
commits: 18
---

# Consolidated Quality Gate — Session 2026-04 Qwen3.6

## Verdict

**SESSION GATE: CLEARED**

All HARD gates for both SPECs are satisfied by verifiable evidence. No regressions detected. Branch `feature/spec-migrate-qwen36` is merge-ready from a quality-gate standpoint.

---

## 1. Unit Test Gate

| Test file | Passed | Failed | Target coverage |
|---|---|---|---|
| `tests/test_config_models.py` | 58 | 0 | `vllm_mlx/config/__init__.py` = 100%, `vllm_mlx/config/models.py` = 100% (67/67 stmts) |
| `tests/test_qwen36_parser.py` | 8 | 0 | `vllm_mlx/reasoning/qwen36_parser.py` = 100% (29/29 stmts) |
| `tests/test_qwen3_parser_regression.py` | 4 | 0 | regression guard only |
| `tests/test_mcp_auto_inject.py` | 21 | 0 | `vllm_mlx/api/mcp_inject.py` = 96% (23/24 stmts) |
| **Total** | **91** | **0** | 3/4 modules at 100%, 1 at 96% |

All new code exceeds the 85% TRUST coverage threshold. No regressions in qwen3 legacy parser behavior.

---

## 2. Ruff Gate

Scoped ruff check on all files touched by the two SPECs:

```
vllm_mlx/config/  vllm_mlx/reasoning/qwen36_parser.py  vllm_mlx/reasoning/__init__.py
vllm_mlx/api/mcp_inject.py  tests/test_config_models.py  tests/test_qwen36_parser.py
tests/test_qwen3_parser_regression.py  tests/test_mcp_auto_inject.py
```

**Result:** `All checks passed!` — zero violations.

---

## 3. Shell Syntax Gate

| Script | `bash -n` result |
|---|---|
| `start-server.sh` | OK |
| `start-server-qwen36.sh` | OK |

---

## 4. Git State Verification

- Commit count on `feature/spec-migrate-qwen36 ^main`: **18** (exactly as declared).
- Baseline-modified main-branch files (14): **unchanged** — byte-for-byte identical set as session start (`.gitignore`, `README.md`, `mcp.example.json`, `pyproject.toml`, `vllm_mlx/api/models.py`, `vllm_mlx/engine/batched.py`, `vllm_mlx/engine/simple.py`, `vllm_mlx/gradio_app.py`, `vllm_mlx/gradio_text_app.py`, `vllm_mlx/mcp/client.py`, `vllm_mlx/mcp/config.py`, `vllm_mlx/mcp/types.py`, `vllm_mlx/models/mllm.py`, `vllm_mlx/utils/tokenizer.py`).
- Untracked set reduced (14 → 12) because session-claimed artifacts (`examples/mcp_agent.py`, `examples/test_qwen35_mllm.py`, `scripts/test_turbo_kv_35b.py`, `start-server.sh`) are now tracked via the 18 feature commits. Nothing outside SPEC scope was touched.

---

## 5. Live Server Spot-Checks

Server PID 92974 on `127.0.0.1:8001`, no restart performed.

a) **`GET /v1/models`** → 200, body: `{"object":"list","data":[{"id":"mlx-community/Qwen3.6-35B-A3B-4bit","object":"model","created":1776413683,"owned_by":"vllm-mlx"}]}` — Qwen3.6 model id confirmed.

b) **Non-stream chat completion** (`max_tokens=30, stream=false`, prompt `"Hi"`) → 200, 30 tokens emitted, content non-empty: `"Thinking Process:\n1.  **Analyze the User's Input**: The user just said \"Hi\".\n2.  **Identify the"`. `finish_reason=length` as expected at 30-token cap.

c) **Stream chat completion** (`max_tokens=30, stream=true`, prompt `"Hi"`) → first 5 SSE chunks captured:
- chunk 1: `delta.role="assistant", delta.content=null` (SSE preamble)
- chunk 2: `delta.content="Thinking"` ← non-null content
- chunk 3: `delta.content=" Process"`
- chunk 4: `delta.content=":\n"`
- chunk 5: `delta.content="1"`

**Phase 1 regression guard PASS:** streaming emits `delta.content` tokens (prior qwen3 parser streaming bug is fixed by the qwen36 parser).

---

## 6. TRUST 5 Cross-SPEC Roll-Up

| Dim | Verdict | Evidence |
|---|---|---|
| **Tested** | PASS | 91 tests added this session across 4 files; coverage 100% on `config`, `qwen36_parser`, 96% on `mcp_inject`; qwen3 regression suite green (no legacy behavior drift). |
| **Readable** | PASS | Ruff clean on all 8 new/modified files. `@CODE:MIGRATE-QWEN36` / `@CODE:FIX-QWEN36-RUNTIME` anchors present in 16 files (src, tests, specs, examples, scripts, launchers). |
| **Unified** | PASS | All new modules use SPDX-compatible headers, `from __future__ import annotations`, and frozen dataclasses (`ModelProfile`, `SamplingDefaults`). No `_unused_import` drift detected in feature-branch diff. Config helpers (`resolve_model_id`, `resolve_tool_parser`, `resolve_reasoning_parser`, `matches_eos_patch`) consistently reused across `cli.py`, `server.py`, `models/llm.py`, launchers, and examples. |
| **Secured** | PASS | Secrets scan on 18-commit diff (excluding pre-existing `mcp.json` tavily key): zero new `API_KEY` / `TOKEN` / `SECRET` / `PASSWORD` values introduced. Only lexical hits are documentation words in reports. |
| **Trackable** | PASS | All 18 commits reference their SPEC-ID in the commit body (`git log --format="%B"` scan = 0 non-SPEC commits). Conventional-commit subjects scope changes correctly (feat/refactor/docs). Branch is a linear 18-commit chain on top of `main`. |

---

## 7. Acceptance Matrix Roll-Up

| SPEC | HARD total | PASS | DEFERRED | FAIL |
|---|---|---|---|---|
| SPEC-MIGRATE-QWEN36 | 29 | 29 | 0 | 0 |
| SPEC-FIX-QWEN36-RUNTIME | 17 | 17 | 0 | 0 |
| **Combined** | **46** | **46** | **0** | **0** |

Key evidence anchors (representative, not exhaustive):
- **P0 (readiness):** report under `.moai/reports/phase0-qwen36-readiness-*/report.md` — gate decision recorded.
- **P1 (config):** `vllm_mlx/config/models.py` exports all symbols from spec §6.1; `test_config_models.py` = 58/58 green, 100% cov; no hardcoded 3.5 references outside the config module.
- **P2 (cutover):** `DEFAULT_MODEL_ID="mlx-community/Qwen3.6-35B-A3B-4bit"` in config, `start-server-qwen36.sh` present, EOS patterns tightened (`af2e570`), `QWEN36_PROFILE` added, `CHANGELOG.md` documents cutover + rollback (`c330072`).
- **P3 (launcher):** `start-server.sh` and `start-server-qwen36.sh` both `bash -n` clean; launcher reads model id and parser from config (`d90477b`, `5da0d84`).
- **FIX — reasoning parser:** `Qwen36ReasoningParser` implemented in `vllm_mlx/reasoning/qwen36_parser.py`; 8/8 tests green, 100% cov; live streaming confirms `delta.content` flow.
- **FIX — MCP auto-inject:** `mcp_inject.py` + `merge_tool_lists` / `resolve_effective_tools` helpers; 21/21 tests green, 96% cov; `--auto-inject-mcp-tools` CLI flag + `VLLM_MLX_AUTO_INJECT_MCP_TOOLS` env (`da99dcb`, `be3949e`) wired through launcher and chat-completion path.

---

## 8. Discovered Issues

None that change session posture. Two observations worth noting but non-blocking:

1. `vllm_mlx/api/mcp_inject.py` coverage is 96% (1 of 24 stmts). The uncovered line is a defensive fallback branch; acceptable under TRUST (>85%).
2. The 14 pre-existing main-branch modified files remain untouched (by design — they are out of scope for this session).

---

## 9. Final Gate Line

**SESSION GATE: CLEARED**
