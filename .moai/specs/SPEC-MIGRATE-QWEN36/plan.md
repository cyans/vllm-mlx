---
id: SPEC-MIGRATE-QWEN36
document_type: plan
status: draft
created: 2026-04-17
development_mode: ddd
---

# Implementation Plan — SPEC-MIGRATE-QWEN36

## @PLAN:MIGRATE-QWEN36

Migrate the `vllm-mlx` default inference model from Qwen3.5-35B-A3B to Qwen3.6-35B-A3B while simultaneously introducing a centralized model-configuration module that eliminates ≥ 10 hardcoded model ID references scattered across launchers, engine code, examples, benchmarks, and docs.

This plan follows the project's **DDD (ANALYZE-PRESERVE-IMPROVE)** development mode (per `.moai/config/sections/quality.yaml`). Characterization tests capture current Qwen3.5 behavior before any refactor; behavior is preserved through Phase 1; Qwen3.6 support is added in parallel in Phase 2; cutover occurs only in Phase 3 after the A/B evaluation of Phase 4.

---

## 1. Phase Overview & Milestones (priority-based, no time estimates)

| Phase | Priority | Goal | Gate condition to advance |
|---|---|---|---|
| **Phase 0** | HARD GATE | Verify Qwen3.6 is loadable by `mlx-lm` on target hardware; confirm MLX quant availability; confirm `qwen3_coder` parser availability | All Phase 0 acceptance items pass OR user explicitly approves a blocker-workaround scope expansion |
| **Phase 1** | Primary | Centralize configuration (no behavior change) | `ruff` + existing test suite green; characterization tests captured & passing |
| **Phase 2** | Primary | Add Qwen3.6 in parallel (Qwen3.5 remains default) | `start-server-qwen36.sh` serves 3.6 end-to-end; `start-server.sh` still serves 3.5 identically |
| **Phase 3** | Primary | Cutover default to Qwen3.6; retain rollback path | `./start-server.sh` serves 3.6 by default; `VLLM_MLX_MODEL_ID` rollback verified |
| **Phase 4** | Final | Quality validation + A/B benchmark | TRUST 5 gates pass; A/B report published |
| **Phase 5** | Optional | Post-migration cleanup (legacy markers, deprecation notes) | User approval |

> Phase 0 is a **HARD GATE**. No work on Phases 1-4 may be merged to `main` until Phase 0 passes.

---

## 2. Phase 0 — Architecture & Availability Gate

### 2.1 Objectives
Confirm the three unknowns from `spec.md` Assumption A1-A4 before committing engineering effort.

### 2.2 Tasks
1. **MLX quant discovery** — query HuggingFace Hub API for repos matching `mlx-community/Qwen3.6-35B-A3B*`; record URL and available quantizations.
2. **Architecture load test** — on the target Apple Silicon machine, run:
   ```python
   from mlx_lm import load
   model, tokenizer = load("<discovered-mlx-quant-id>")
   ```
   Capture any `ValueError`, `NotImplementedError`, or missing-module error.
3. **First-token smoke test** — if load succeeds, generate a single short completion ("Hello, ") and confirm coherent output (not pure garbage / not repeating BOS).
4. **Tokenizer EOS inspection** — download `tokenizer_config.json` from HF for `Qwen/Qwen3.6-35B-A3B`; record `eos_token`, `added_tokens_decoder` entries matching `<|im_end|>` and `<|endoftext|>`.
5. **vLLM parser registry inspection** — in the currently-installed `vllm-mlx` environment, enumerate the registered tool-call parsers; confirm presence or absence of `qwen3_coder`.
6. **Gate decision** — document findings in `.moai/reports/phase0-qwen36-readiness-{YYYY-MM}/report.md` (NOT in `.moai/specs/` — per SPEC vs Report classification rules). Decision matrix:
   - All 5 checks pass → advance to Phase 1.
   - MLX quant missing but architecture supported via upstream weights → user decision: add self-conversion to scope, OR wait.
   - Architecture unsupported by `mlx-lm` → PAUSE SPEC; recommend upstreaming or waiting for `mlx-lm` release.
   - `qwen3_coder` parser missing → document fallback-only policy; no SPEC pause.

### 2.3 Agent Assignment
- **manager-spec** (this agent): draft Phase 0 plan & questions
- **expert-backend**: execute the architecture/load test & parser-registry inspection on target hardware
- **Explore agent**: tokenizer_config.json retrieval & diff vs Qwen3.5

### 2.4 Exit artifacts
- `.moai/reports/phase0-qwen36-readiness-{YYYY-MM}/report.md` with findings table
- Updated assumption confidence levels in `spec.md` (A1-A4 promoted from Low/Medium → High or SPEC paused)

---

## 3. Phase 1 — Centralize Model Configuration (behavior-preserving refactor)

### 3.1 Objectives
Create `vllm_mlx/config/models.py` and replace every hardcoded model-ID / parser-flag literal in the repo (except the 122B variant) with imports from this module — **without changing any runtime behavior**. This is the ANALYZE → PRESERVE steps of the DDD cycle.

### 3.2 ANALYZE
1. Re-audit the file list from `spec.md` §1.4 via `Grep` for drift since the initial audit.
2. Map each hardcoded reference to its replacement symbol (DEFAULT_MODEL_ID, REASONING_PARSER, TOOL_PARSER_*, ModelProfile).
3. Identify any reference that is semantically different (e.g., a test that explicitly wants the 3.5 model for regression purposes → replace with `LEGACY_MODEL_ID` rather than `DEFAULT_MODEL_ID`).

### 3.3 PRESERVE — Characterization tests
Write characterization tests BEFORE touching any file:
1. **Server boot test**: start `start-server.sh` → POST a fixed prompt to `/v1/chat/completions` → capture (status code, response JSON shape, first 20 tokens). Store as golden output under `tests/golden/qwen35_boot.json`.
2. **EOS patch test**: unit-test `llm.py` with a mock model name `"qwen3-test"` → assert `eos_token == "<|im_end|>"` after patch. Preserves current behavior.
3. **Parser-flag test**: boot server with current defaults → assert via `/v1/models` (or debug endpoint) that reasoning parser is `qwen3` and tool-call parser is `qwen`.
4. **Example script smoke test**: `python examples/test_qwen35_mllm.py --dry-run` (if dry-run exists) or minimal invocation to confirm model-ID string equals `mlx-community/Qwen3.5-35B-A3B-4bit`.

Target coverage: ≥ 85% on the new `vllm_mlx/config/models.py` module; characterization tests for existing touched files must pass identically before & after Phase 1 changes.

### 3.4 IMPROVE — Refactor sweep
Order of file edits (atomic commits, one logical unit per commit):

1. **Create** `vllm_mlx/config/__init__.py` (empty or re-export) + `vllm_mlx/config/models.py` with the API from `spec.md` §6.1. Initial values:
   - `DEFAULT_MODEL_ID = "mlx-community/Qwen3.5-35B-A3B-4bit"` (unchanged)
   - `LEGACY_MODEL_ID = "mlx-community/Qwen3.5-35B-A3B-4bit"` (same as default for Phase 1; divergence in Phase 3)
   - `EOS_PATCH_MODEL_PATTERNS = ("qwen3",)` (unchanged for Phase 1)
2. **Refactor** `vllm_mlx/models/llm.py:85-87` to call `matches_eos_patch(self.model_name)`.
3. **Refactor** `vllm_mlx/server.py` to use `resolve_tool_parser(...)`; preserve current output (qwen) via the fallback path.
4. **Refactor** shell launchers to read `DEFAULT_MODEL_ID` via the helper Python call described in `spec.md` §6.2.
5. **Refactor** `examples/test_qwen35_mllm.py` → import `LEGACY_MODEL_ID` (intentionally the legacy anchor — this file tests the 3.5 path by name).
6. **Refactor** `examples/mcp_agent.py:24` → import `DEFAULT_MODEL_ID`.
7. **Refactor** `scripts/test_turbo_kv_35b.py` → remove `MODEL_DEFAULT` constant; add `--model` CLI arg with `DEFAULT_MODEL_ID` as the fallback default.
8. **Run characterization tests**. Any red test is a regression — revert and fix before continuing.

### 3.5 Exit criteria (see acceptance.md §Phase 1)
- Characterization tests green
- `ruff check .` clean
- `mypy vllm_mlx/config` clean (if mypy is configured)
- `grep -rnE "mlx-community/Qwen3\.5" vllm_mlx/ examples/ scripts/ *.sh` returns only expected sites (config module, tests targeting legacy, optional README historical notes)

### 3.6 Agent assignment
- **expert-refactoring**: drive the sweep across ≥ 10 files
- **expert-backend**: review `vllm_mlx/config/models.py` API contract
- **expert-testing**: author characterization tests
- **manager-quality**: run TRUST 5 validation at end of phase

---

## 4. Phase 2 — Add Qwen3.6 in Parallel

### 4.1 Objectives
Introduce Qwen3.6 support alongside Qwen3.5 with **no change to default behavior**. Users must opt in via `start-server-qwen36.sh` or `VLLM_MLX_MODEL_ID`.

### 4.2 Tasks
1. Extend `EOS_PATCH_MODEL_PATTERNS` → `("qwen3.5", "qwen3.6")` (tighten from `"qwen3"` now that we can enumerate both; reduces false-positive risk on future Qwen variants).
2. Add `QWEN36_PROFILE` to `vllm_mlx/config/models.py` with:
   - `model_id = "<verified Qwen3.6 MLX quant>"` from Phase 0
   - `max_context_tokens = 262144`
   - `supports_tool_calling = True`
   - `supports_multimodal = True` (capability flag only; runtime still runs text-only)
3. Create `start-server-qwen36.sh`:
   - Sources shared config
   - Passes `--model "$(python -c 'from vllm_mlx.config.models import QWEN36_PROFILE; print(QWEN36_PROFILE.model_id)')"`
   - Passes `--reasoning-parser qwen3`
   - Resolves tool-call parser at server startup (no hardcode in shell)
   - Passes `--language-model-only`
4. Create `examples/test_qwen36_mllm.py` mirroring the 3.5 test structure; all model-ID references go through `QWEN36_PROFILE.model_id`.
5. Update `vllm_mlx/server.py` logging: on startup, print active model ID, reasoning parser, tool-call parser, and whether the tool parser came from preferred or fallback branch.
6. Run **smoke tests**:
   - `./start-server-qwen36.sh &` → `curl /v1/chat/completions` with a short prompt → assert HTTP 200 and non-empty choice
   - Kill 3.6 server; `./start-server.sh &` → same request → assert identical shape as pre-Phase-2 baseline (regression guard)

### 4.3 Risk mitigations in this phase
- **EOS correctness for 3.6**: Add an explicit unit test asserting that with `model_name="Qwen/Qwen3.6-35B-A3B"`, `matches_eos_patch(...)` returns `True` AND the patched EOS matches the tokenizer's actual `<|im_end|>` ID.
- **Tool parser divergence**: Capture a short tool-calling conversation against 3.6; confirm the chosen parser (preferred or fallback) parses function-call JSON correctly.

### 4.4 Exit criteria (see acceptance.md §Phase 2)
- 3.6 server boots and generates coherent output
- 3.5 server unchanged (regression guard passes)
- EOS emission verified for 3.6 (no runaway generation on 200-token prompt)

### 4.5 Agent assignment
- **expert-backend**: server.py parser resolution + launcher design
- **expert-testing**: smoke-test harness
- **expert-devops**: shell launcher Python-shell boundary

---

## 5. Phase 3 — Cutover to Qwen3.6 as Default

### 5.1 Objectives
Flip `DEFAULT_MODEL_ID` to Qwen3.6. Preserve the Qwen3.5 rollback path via `VLLM_MLX_MODEL_ID` and `LEGACY_MODEL_ID`.

### 5.2 Tasks
1. Update `vllm_mlx/config/models.py`:
   - `DEFAULT_MODEL_ID = QWEN36_PROFILE.model_id`
   - `LEGACY_MODEL_ID = QWEN35_PROFILE.model_id` (now genuinely divergent)
2. Update `start-server.sh` to drop any residual 3.5 assumptions; it now merely invokes the shared launcher logic and picks up whichever `DEFAULT_MODEL_ID` is in the config.
3. Update docs:
   - `README.md:28` → replace model label with Qwen3.6
   - `PLAN.md` lines 11, 284, 291, 342, 363 → update Phase 3 headings referencing the new model
   - `PLAN-obsidian-plugin.md:172` → default model label to "Qwen3.6:35B"
4. Run regression: `VLLM_MLX_MODEL_ID=$(python -c 'from vllm_mlx.config.models import LEGACY_MODEL_ID; print(LEGACY_MODEL_ID)') ./start-server.sh` must still load & serve 3.5.
5. Announce deprecation window in `CHANGELOG` (if present) — "3.5 remains available via env override for the next minor release."

### 5.3 Exit criteria (see acceptance.md §Phase 3)
- Fresh clone + `./start-server.sh` (no args, no env) loads 3.6
- Env-var rollback to 3.5 passes the same smoke test
- All docs reflect 3.6 as default

---

## 6. Phase 4 — Quality Validation & A/B Benchmark

### 6.1 Objectives
Deliver the TRUST 5 quality gate and the data-driven comparison promised by Story 4 in `spec.md`.

### 6.2 Tasks
1. **TRUST 5 validation** via manager-quality:
   - Tested: coverage ≥ 85 % on `vllm_mlx/config/models.py`; characterization + smoke tests green
   - Readable: `ruff` clean
   - Unified: code style consistent with existing repo norms
   - Secured: no secrets introduced; model IDs are public HF URLs
   - Trackable: conventional commits per logical unit; each commit references `SPEC-MIGRATE-QWEN36`
2. **A/B benchmark script** under `scripts/benchmark_qwen35_vs_qwen36.py`:
   - CLI: `--prompts <file>` (JSONL), `--modes instruct,thinking`, `--output <dir>`
   - Iterates ≥ 5 shared prompts; for each mode, runs once against `LEGACY_MODEL_ID` and once against `DEFAULT_MODEL_ID`
   - Captures: time-to-first-token, total latency, output text, token count
   - Emits a Markdown report with side-by-side output and a latency summary table
3. **LSP regression check**: diff LSP diagnostics snapshot vs end-of-Phase-1 baseline (per `.moai/config/sections/quality.yaml`:run.allow_regression=false)
4. **Publish report** to `.moai/reports/qwen35-vs-qwen36-ab-{YYYY-MM}/report.md`.

### 6.3 Agent assignment
- **manager-quality**: TRUST 5 gate
- **expert-performance**: benchmark design (latency measurement methodology)
- **expert-testing**: report formatting

---

## 7. Risk Register & Mitigations

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R1 | `mlx-lm` does not yet support Hybrid Gated DeltaNet architecture | Medium | Critical | Phase 0 HARD GATE; if blocked, SPEC pauses; option to upstream to `mlx-lm` (much larger scope → separate SPEC) | expert-backend |
| R2 | MLX quant of Qwen3.6 not yet published on HF | Medium | High | Phase 0 detection; user decides whether to add self-conversion (`mlx_lm.convert`) scope | manager-spec |
| R3 | `qwen3_coder` tool-call parser not in pinned vllm-mlx version | Medium | Medium | Runtime fallback to `qwen` parser; document required `vllm` version bump in release notes | expert-backend |
| R4 | Tokenizer EOS semantics diverge for 3.6 (unlikely but possible) | Low | High | Read `tokenizer_config.json` in Phase 0; add explicit EOS-ID unit test in Phase 2 | expert-testing |
| R5 | Multimodal tensors load into RAM despite `--language-model-only` | Low | Medium | Phase 2 smoke test monitors resident memory; document expected footprint | expert-devops |
| R6 | Hidden downstream consumer hardcodes `"Qwen3.5"` in client-side prompts | Medium | Low | Repo-wide grep already performed; external consumers are out of scope; release note to call out the change | manager-docs |
| R7 | A/B benchmark prompts not representative → misleading comparison | Low | Medium | Use the same prompt file committed to the repo; publish raw outputs alongside the report | expert-performance |
| R8 | Refactor sweep introduces subtle regression not caught by tests | Medium | High | Characterization tests cover server boot, tokenizer patch, parser flags, and example invocation paths; run full suite after every commit | expert-refactoring |

---

## 8. Rollback Strategy

### 8.1 Forward rollback (preferred)
- Set `VLLM_MLX_MODEL_ID=mlx-community/Qwen3.5-35B-A3B-4bit` in the shell environment OR the process manager (systemd / launchd / Docker env).
- Restart the server. No code change needed. No git revert needed.
- Verified by acceptance item P3-AC4.

### 8.2 Backward rollback (git)
- Revert the Phase 3 commit (identifiable via its message `feat(models): cutover default to Qwen3.6 — SPEC-MIGRATE-QWEN36`).
- `DEFAULT_MODEL_ID` returns to 3.5; rebuild / redeploy.
- Phase 1 and Phase 2 changes remain (they are behavior-preserving for 3.5 users).

### 8.3 Hard rollback (full revert)
- Revert all commits tagged with `SPEC-MIGRATE-QWEN36`.
- Restores pre-migration state exactly.
- Only needed if a non-obvious regression is found in Phase 1's refactor sweep.

---

## 9. Dependencies

### 9.1 Upstream
- `mlx-lm` release supporting Hybrid Gated DeltaNet (Phase 0 verifies)
- `mlx-community/Qwen3.6-35B-A3B-<quant>` MLX weights on HuggingFace (Phase 0 verifies)
- `vllm-mlx` dependency version exposing `qwen3_coder` tool-call parser (Phase 0 detects; fallback if absent)

### 9.2 Internal
- No other active SPEC touches the files in `spec.md` §1.4 at the time of writing (should be re-verified in Phase 1 ANALYZE).
- 122B-MINT migration is tracked separately and does not block this SPEC.

---

## 10. Technical Approach — Summary

1. **Introduce one narrow abstraction** (`vllm_mlx/config/models.py`) rather than many scattered ones. Single import surface for callers.
2. **Refactor first, migrate second**: Phase 1 is behavior-preserving so the diff between "Qwen3.5 with centralized config" and "Qwen3.6 with centralized config" is minimal and reviewable.
3. **Opt-in before cutover**: Phase 2 proves 3.6 works end-to-end while Phase 1 users stay on 3.5.
4. **Reversible cutover**: Phase 3's default flip is one line in `models.py`; `VLLM_MLX_MODEL_ID` gives operators instant rollback with zero code change.
5. **Data-driven sign-off**: Phase 4's A/B benchmark report is the evidence basis for declaring the migration successful.

---

## 11. Agent Assignment Summary

| Phase | Primary | Secondary | Review |
|---|---|---|---|
| 0 | expert-backend | Explore agent | manager-spec |
| 1 | expert-refactoring | expert-testing | manager-quality, expert-backend |
| 2 | expert-backend | expert-devops, expert-testing | manager-quality |
| 3 | expert-backend | manager-docs | manager-quality |
| 4 | manager-quality | expert-performance, expert-testing | manager-spec |

Git operations (branch creation, commits, PR management) for all phases are delegated to **manager-git** per the single-responsibility rule.

---

## 12. Open Questions (resolved or deferred)

| # | Question | Resolution |
|---|---|---|
| Q1 | Which MLX quant of Qwen3.6 to target (4-bit, 6-bit, 8-bit)? | Deferred to Phase 0 discovery; prefer 4-bit to match 3.5 parity unless quality delta is material |
| Q2 | Shell-Python boundary: helper script vs generated `.env`? | Deferred to Phase 1 implementation; default to inline `python -c` invocation for simplicity |
| Q3 | Should `start-server.sh` be renamed post-cutover? | No. Preserve filename for user muscle memory; the default model it serves is what changes |
| Q4 | Retire `start-server-qwen36.sh` after Phase 3? | Deferred to Phase 5 (optional cleanup); keep for a deprecation window |
