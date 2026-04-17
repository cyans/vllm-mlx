---
id: SPEC-MIGRATE-QWEN36
document_type: acceptance
status: draft
created: 2026-04-17
development_mode: ddd
---

# Acceptance Criteria — SPEC-MIGRATE-QWEN36

## @ACCEPT:MIGRATE-QWEN36

This document defines the verifiable exit criteria for each phase of the Qwen3.6 migration. Each item is written in Given-When-Then form where applicable and is checkable without reference to implementation details.

Phases are gated: no phase may be declared complete unless **all** HARD items in that phase pass. SOFT items are strongly recommended but may be deferred with explicit justification captured in the Phase-exit commit or PR description.

Legend:
- [HARD] — blocking; must pass to advance to the next phase
- [SOFT] — non-blocking; deferral requires justification

---

## Phase 0 — Architecture & Availability Gate

> Gate: every [HARD] item below MUST pass before any Phase 1 work is merged to `main`.

### P0-AC1 [HARD] — MLX quant availability
- **Given** the HuggingFace Hub is reachable
- **When** we query for repos matching `mlx-community/Qwen3.6-35B-A3B*`
- **Then** at least one quantized MLX repo is found AND its URL + quantization format are recorded in `.moai/reports/phase0-qwen36-readiness-*/report.md`
- **OR** no MLX quant is found AND the user has explicitly approved either (a) expanding scope to include self-conversion via `mlx_lm.convert`, OR (b) pausing this SPEC pending community upload. The approval MUST be captured in the Phase 0 report.

### P0-AC2 [HARD] — Architecture loadability
- **Given** the chosen MLX quant ID from P0-AC1
- **When** a developer runs `python -c "from mlx_lm import load; load('<quant-id>')"` on the target Apple Silicon dev machine
- **Then** the call completes without `ValueError: Unsupported architecture`, `NotImplementedError`, or any missing-module import error
- **And** the returned `model` and `tokenizer` objects are non-None

### P0-AC3 [HARD] — First-token coherence
- **Given** the loaded model from P0-AC2
- **When** the developer generates a completion for the prompt `"Hello, "` with max_tokens=20, temperature=0.7
- **Then** the output is non-empty, is not a repeating BOS/PAD/EOS sequence, and is recognizable as coherent English (ASCII-printable + whitespace)

### P0-AC4 [HARD] — Tokenizer EOS inspection
- **Given** the HuggingFace repo `Qwen/Qwen3.6-35B-A3B`
- **When** the developer retrieves `tokenizer_config.json`
- **Then** the `eos_token` field and the `added_tokens_decoder` entries for `<|im_end|>` and `<|endoftext|>` are recorded in the Phase 0 report
- **And** the report states explicitly whether the current `"qwen3" in model_name` EOS patch in `vllm_mlx/models/llm.py:85-87` is still correct for 3.6 (expected: yes — `<|im_end|>`)

### P0-AC5 [HARD] — vLLM `qwen3_coder` parser availability
- **Given** the `vllm-mlx` dependency version currently pinned in `pyproject.toml`
- **When** the developer inspects the registered tool-call parsers
- **Then** the report states one of:
  - (a) `qwen3_coder` IS registered — no fallback needed
  - (b) `qwen3_coder` is NOT registered — runtime fallback to `qwen` parser is the documented policy, with a recommended `vllm` version bump noted for future work

### P0-AC6 [HARD] — Gate decision recorded
- The Phase 0 report file `.moai/reports/phase0-qwen36-readiness-*/report.md` exists, is committed, and contains an explicit `## Gate Decision` section stating PASS / BLOCKED / SCOPE-EXPANDED with the user's sign-off.

---

## Phase 1 — Centralize Configuration (Behavior-Preserving)

### P1-AC1 [HARD] — Config module exists
- The file `vllm_mlx/config/models.py` exists and exports **all** symbols listed in `spec.md` §6.1:
  - `DEFAULT_MODEL_ID`, `LEGACY_MODEL_ID`, `REASONING_PARSER`, `TOOL_PARSER_PREFERRED`, `TOOL_PARSER_FALLBACK`, `EOS_PATCH_MODEL_PATTERNS`
  - `ModelProfile` dataclass, `SamplingDefaults` dataclass
  - `QWEN35_PROFILE`, `PROFILES` dict
  - `resolve_model_id`, `resolve_tool_parser`, `matches_eos_patch` functions

### P1-AC2 [HARD] — Characterization tests captured and green
- **Given** the Qwen3.5 baseline before any refactor
- **When** the developer starts `start-server.sh`, POSTs a fixed prompt to `/v1/chat/completions`, and runs `python examples/test_qwen35_mllm.py`
- **Then** the captured outputs (first 20 tokens of a chat response; script exit code and stdout) are stored as golden fixtures under `tests/golden/` or equivalent
- **And** the same tests, re-run after Phase 1 refactor, produce byte-identical (or tokenization-deterministic) results

### P1-AC3 [HARD] — All hardcoded 3.5 references removed from non-config modules
- **Given** the Phase 1 refactor is complete
- **When** the developer runs:
  ```
  grep -rnE "mlx-community/Qwen3\.5-35B-A3B" vllm_mlx/ examples/ scripts/ start-server.sh 2>/dev/null | grep -v "config/models.py"
  ```
- **Then** the only remaining matches are in files that intentionally target the legacy model by name (e.g., `examples/test_qwen35_mllm.py` after refactor imports `LEGACY_MODEL_ID`)
- **And** `start-server-122b-mint.sh` is explicitly excluded (out of scope; will match but is allowed)

### P1-AC4 [HARD] — `llm.py` EOS patch uses config module
- `vllm_mlx/models/llm.py` imports `matches_eos_patch` from `vllm_mlx.config.models` and no longer contains the literal string `"qwen3" in`.
- Running the existing test suite plus new EOS unit tests passes.

### P1-AC5 [HARD] — `server.py` uses `resolve_tool_parser`
- `vllm_mlx/server.py` imports and calls `resolve_tool_parser(...)` at startup
- Server still boots against Qwen3.5 and logs the chosen parser at INFO level

### P1-AC6 [HARD] — Launcher scripts no longer hardcode model ID
- `start-server.sh` does not contain the literal string `mlx-community/Qwen3.5-35B-A3B-4bit`
- The model ID is obtained via the shell-Python helper described in `spec.md` §6.2
- Running `./start-server.sh` still starts a server serving Qwen3.5 (since `DEFAULT_MODEL_ID` is unchanged in Phase 1)

### P1-AC7 [HARD] — `scripts/test_turbo_kv_35b.py` parameterized
- The `MODEL_DEFAULT` constant is removed
- The script accepts `--model <id>` CLI arg; when omitted, falls back to `DEFAULT_MODEL_ID` imported from the config module

### P1-AC8 [HARD] — Quality gates
- `ruff check .` exits 0 (or with only pre-existing warnings, no new ones introduced)
- `mypy vllm_mlx/config/` exits 0 (only if mypy is configured for the project)
- Existing test suite (`pytest`) exits 0
- New config-module tests achieve ≥ 85 % line coverage on `vllm_mlx/config/models.py`

### P1-AC9 [SOFT] — Commit granularity
- Changes are split into logical commits, each referencing `SPEC-MIGRATE-QWEN36`
- No single commit touches more than ~5 files unless it is purely a mechanical rename

---

## Phase 2 — Parallel Support for Qwen3.6

### P2-AC1 [HARD] — `QWEN36_PROFILE` present
- `vllm_mlx/config/models.py` exports a `QWEN36_PROFILE: ModelProfile` whose `model_id` matches the MLX quant confirmed in Phase 0.
- `PROFILES` dict contains the new profile keyed by its `model_id`.

### P2-AC2 [HARD] — `EOS_PATCH_MODEL_PATTERNS` tightened
- The tuple value is `("qwen3.5", "qwen3.6")` (or case-normalized equivalent)
- Unit test: `matches_eos_patch("Qwen/Qwen3.6-35B-A3B")` returns `True`
- Unit test: `matches_eos_patch("Qwen/Qwen3.5-35B-A3B")` returns `True`
- Unit test: `matches_eos_patch("some-other-model")` returns `False`

### P2-AC3 [HARD] — `start-server-qwen36.sh` exists and works
- **Given** the Phase 2 build
- **When** the operator runs `./start-server-qwen36.sh` in a clean shell
- **Then** the server starts without error, logs the Qwen3.6 model ID as active, and responds to `curl -X POST http://localhost:<port>/v1/chat/completions` with a non-empty assistant message
- **And** the response does not run away (i.e., a 200-token cap is respected — EOS is recognized)

### P2-AC4 [HARD] — `examples/test_qwen36_mllm.py` exists and runs
- The file exists, imports `QWEN36_PROFILE.model_id`, and mirrors the structure of the 3.5 example
- Running the script end-to-end produces a successful response (no exceptions)
- The test may keep `--language-model-only` semantics (text-only) per scope assumption A9

### P2-AC5 [HARD] — Regression: Qwen3.5 path unchanged
- **Given** Phase 2 code merged to `main`
- **When** the operator runs `./start-server.sh` with no env var overrides
- **Then** the server still serves Qwen3.5 (since `DEFAULT_MODEL_ID` has not yet been flipped)
- **And** the golden-output characterization test from P1-AC2 still passes byte-identically

### P2-AC6 [HARD] — Tool-parser resolution logged
- On Qwen3.6 server startup, the log contains a line of the form `tool-call parser: <name> (source: preferred | fallback)` where `<name>` is either `qwen3_coder` or `qwen`

### P2-AC7 [HARD] — EOS emission verified for 3.6
- **Given** a running Qwen3.6 server
- **When** a chat completion is requested with `max_tokens=200` and a prompt that would normally produce a short answer (e.g., "What is 2+2?")
- **Then** the response terminates before `max_tokens` is hit because the model emits `<|im_end|>`
- **And** there is no runaway generation of empty / repeating tokens past the intended answer

### P2-AC8 [SOFT] — Tool-calling smoke test
- A short tool-calling conversation against Qwen3.6 parses correctly with the chosen parser (preferred or fallback). The captured conversation is attached to the Phase 2 completion note.

---

## Phase 3 — Default Cutover to Qwen3.6

### P3-AC1 [HARD] — `DEFAULT_MODEL_ID` flipped
- `vllm_mlx/config/models.py` has `DEFAULT_MODEL_ID == QWEN36_PROFILE.model_id`
- `LEGACY_MODEL_ID == QWEN35_PROFILE.model_id` and the two are now genuinely different

### P3-AC2 [HARD] — Default launcher serves 3.6
- **Given** a fresh git clone at the Phase 3 tip with no `VLLM_MLX_MODEL_ID` set
- **When** the operator runs `./start-server.sh`
- **Then** the server loads `Qwen/Qwen3.6-35B-A3B` (or its MLX quant) and responds to a chat completion successfully

### P3-AC3 [HARD] — Rollback path verified
- **Given** the Phase 3 build
- **When** the operator runs:
  ```
  VLLM_MLX_MODEL_ID=mlx-community/Qwen3.5-35B-A3B-4bit ./start-server.sh
  ```
- **Then** the server loads Qwen3.5, the `qwen` tool parser is used (via fallback or preferred), and a chat completion succeeds
- **And** the golden-output characterization test from P1-AC2 passes byte-identically

### P3-AC4 [HARD] — Documentation updated
- `README.md:28` references Qwen3.6 as the default
- `PLAN.md` lines referenced in `spec.md` §1.4 are updated
- `PLAN-obsidian-plugin.md:172` default model label says `Qwen3.6:35B`
- `grep -rE "Qwen3\.5" README.md PLAN.md PLAN-obsidian-plugin.md` returns only (a) historical migration notes clearly marked as such, or (b) references to the rollback env var

### P3-AC5 [SOFT] — Deprecation notice
- A `CHANGELOG` entry (if a CHANGELOG exists in the repo) announces the default cutover and the rollback mechanism
- A release note documents any required `vllm-mlx` dependency bump (if Phase 0 found `qwen3_coder` absent)

---

## Phase 4 — Quality Validation & A/B Benchmark

### P4-AC1 [HARD] — TRUST 5 gates pass
- **Tested**: total test suite ≥ 85 % coverage; `vllm_mlx/config/models.py` module ≥ 85 % coverage; all characterization & smoke tests green
- **Readable**: `ruff check .` clean; naming conventions followed
- **Unified**: code style consistent with existing repo (no new linter rules violated)
- **Secured**: no secrets in repo; no plaintext credentials; no PII in example prompts
- **Trackable**: every commit introduced by this SPEC references `SPEC-MIGRATE-QWEN36`; a conventional-commits log can be produced

### P4-AC2 [HARD] — LSP regression check
- An LSP diagnostics snapshot taken at the end of Phase 4 shows zero new errors, zero new type errors, and zero new lint errors compared to the end-of-Phase-1 baseline
- Warning count delta ≤ 10 (per `.moai/config/sections/quality.yaml` sync thresholds)

### P4-AC3 [HARD] — A/B benchmark script exists
- `scripts/benchmark_qwen35_vs_qwen36.py` exists and accepts:
  - `--prompts <file>` — JSONL prompt file
  - `--modes instruct,thinking` — comma-separated mode list
  - `--output <dir>` — output directory for the report
- Script runs end-to-end against a committed prompt fixture with at least 5 prompts

### P4-AC4 [HARD] — A/B report published
- `.moai/reports/qwen35-vs-qwen36-ab-*/report.md` exists and contains:
  - A latency summary table (TTFT + total latency, median + p95, for each model × mode)
  - Side-by-side output text for each of the ≥ 5 prompts × 2 modes
  - A short qualitative summary paragraph noting any obvious quality / behavior deltas
  - The commit SHA of the tested build

### P4-AC5 [SOFT] — Review sign-off
- The A/B report has been reviewed by the user or a designated reviewer; their comment or approval is linked from the report

---

## Phase 5 — Optional Post-Migration Cleanup

### P5-AC1 [SOFT] — Retire `start-server-qwen36.sh`
- If the deprecation window has elapsed AND no users report reliance on the dedicated launcher, `start-server-qwen36.sh` may be removed with a commit message referencing this SPEC and the acceptance item.

### P5-AC2 [SOFT] — Tighten legacy references
- Remove `LEGACY_MODEL_ID` if no code / test / doc still references it.
- Requires grep-verification first.

### P5-AC3 [SOFT] — Companion SPECs drafted
- A follow-up SPEC placeholder exists for:
  - 122B-MINT migration (out-of-scope here)
  - Multimodal activation (out-of-scope here)

---

## Definition of Done (overall)

The migration is considered DONE when:
1. All Phase 0 [HARD] items have PASSED
2. All Phase 1 [HARD] items have PASSED
3. All Phase 2 [HARD] items have PASSED
4. All Phase 3 [HARD] items have PASSED
5. All Phase 4 [HARD] items have PASSED
6. The A/B report (P4-AC4) has been shared with the user
7. The user has accepted the cutover OR explicitly invoked the rollback path (P3-AC3)

Phase 5 items are explicitly **not** required for DONE; they are opportunistic follow-ups.

---

## Verification Commands Reference

Quick commands the operator or CI can use to verify items above:

- **Find stragglers of the 3.5 model ID**:
  ```
  grep -rnE "mlx-community/Qwen3\.5-35B-A3B" vllm_mlx/ examples/ scripts/ *.sh 2>/dev/null | grep -v "config/models.py" | grep -v "test_qwen35"
  ```
- **Smoke-test 3.6 server**:
  ```
  ./start-server-qwen36.sh &
  sleep 30  # wait for model load
  curl -sS -X POST http://localhost:<PORT>/v1/chat/completions \
       -H 'Content-Type: application/json' \
       -d '{"model":"<id>","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":200}'
  ```
- **Rollback verification**:
  ```
  VLLM_MLX_MODEL_ID=mlx-community/Qwen3.5-35B-A3B-4bit ./start-server.sh
  ```
- **EOS patch unit test target**:
  ```
  pytest tests/ -k "matches_eos_patch" -v
  ```
- **Coverage check on config module**:
  ```
  pytest --cov=vllm_mlx.config --cov-report=term-missing tests/
  ```
