---
id: SPEC-MIGRATE-QWEN36
title: Migrate vllm-mlx default model from Qwen3.5-35B-A3B to Qwen3.6-35B-A3B with centralized model configuration
status: draft
priority: high
created: 2026-04-17
author: manager-spec
development_mode: ddd
lifecycle: spec-anchored
related_specs: []
labels: [migration, model-upgrade, refactor, configuration]
---

# SPEC-MIGRATE-QWEN36: Qwen3.6-35B-A3B Migration + Model Configuration Centralization

## 1. Environment

### 1.1 Runtime Environment
- **Host**: Apple Silicon (M-series) with MLX runtime
- **Framework**: `vllm-mlx` (this repository) — MLX-based inference server wrapping `mlx-lm`
- **Python**: 3.11+ (per `pyproject.toml`)
- **Inference engine**: `mlx-lm` (version pinned in `pyproject.toml`); vLLM-compatible HTTP layer in `vllm_mlx/server.py`
- **Protocol**: OpenAI-compatible chat/completions + tool-calling + reasoning-parser extensions

### 1.2 Current Baseline (model in use)
- **Default model ID**: `mlx-community/Qwen3.5-35B-A3B-4bit` (hardcoded)
- **Reasoning parser flag**: `--reasoning-parser qwen3`
- **Tool-call parser flag**: `--tool-call-parser qwen`
- **Tokenizer special case**: `vllm_mlx/models/llm.py:85-87` force-sets `eos_token = "<|im_end|>"` when model name contains substring `qwen3`

### 1.3 Target Baseline (post-migration)
- **Default model ID**: `Qwen/Qwen3.6-35B-A3B` (upstream) OR `mlx-community/Qwen3.6-35B-A3B-<quant>` (MLX quant, availability TBC in Phase 0)
- **Architecture**: Hybrid Gated DeltaNet + Gated Attention + MoE (NOT a plain transformer)
  - 10 blocks of (3× GatedDeltaNet→MoE + 1× GatedAttention→MoE), 40 hidden layers
  - 256 experts total, 8 routed + 1 shared activated per token
  - 35B total params / 3B activated
  - Native context 262,144 tokens; up to ~1M via YaRN
- **Modalities**: multimodal (text + image + video); text-only enforced via `--language-model-only`
- **Reasoning parser flag**: `--reasoning-parser qwen3` (unchanged in Phase 2; revisit in Phase 3 if upstream emits a new parser name for 3.6)
- **Tool-call parser flag**: `--tool-call-parser qwen3_coder` when supported by installed `vllm-mlx`; fallback to `qwen`
- **License**: Apache-2.0

### 1.4 Relevant Source Files (audit baseline)
Shell launch scripts:
- `start-server.sh:19` — hardcoded 3.5 model ID and `qwen3` reasoning parser
- `start-server-122b-mint.sh:19` — 122B variant (OUT OF SCOPE; noted for future centralization)

Core engine:
- `vllm_mlx/models/llm.py:85-87` — EOS patch branch keyed on `"qwen3" in self.model_name.lower()`
- `vllm_mlx/server.py` — 6 references to `qwen3` / `qwen` parser flags

Tests & examples:
- `scripts/test_turbo_kv_35b.py:22` — `MODEL_DEFAULT` constant
- `examples/test_qwen35_mllm.py` — 4 occurrences (lines 3, 9, 36, 243)
- `examples/mcp_agent.py:24`

Documentation:
- `PLAN.md` (lines 11, 284, 291, 342, 363 — Phase 3 headings)
- `PLAN-obsidian-plugin.md:172` (default model label "Qwen3.5:35B")
- `README.md:28`

Total: **≥ 10 distinct references** across 9 files (excluding the 122B variant and tooling docs under `.moai/`).

---

## 2. Assumptions

### 2.1 Technical Assumptions
| # | Assumption | Confidence | Evidence | Risk if wrong | Validation method |
|---|---|---|---|---|---|
| A1 | `mlx-lm` can load the Qwen3.6 Hybrid Gated-DeltaNet + MoE architecture without custom modeling code | **Low** | HF model card lists `model_type="qwen"`, but Gated DeltaNet is a recent architecture that may require `mlx-lm` ≥ a future release | Blocks entire migration; must fork `mlx-lm` or wait for upstream | Phase 0 hard gate: `from mlx_lm import load; load("<mlx-quant-id>")` on dev machine |
| A2 | An MLX quantization of Qwen3.6-35B-A3B exists on HuggingFace (e.g., `mlx-community/Qwen3.6-35B-A3B-4bit`) | **Low** | HF docs for upstream model do NOT reference MLX ports | Phase 0 must either wait for community port OR expand scope to include self-conversion via `mlx_lm.convert` | Phase 0 check: `huggingface_hub.list_repo_files` search under `mlx-community/` |
| A3 | The vLLM `qwen3_coder` tool-call parser is implemented in the currently-pinned `vllm-mlx` dependency version | **Medium** | Parser name is referenced in Qwen3.6 model card; pinning inside this repo is older | Fallback to `qwen` parser works but may mis-parse function-call JSON | Grep installed `vllm` package for `qwen3_coder` parser registration |
| A4 | Tokenizer EOS for 3.6 is still `<|im_end|>` (same as 3.5 ChatML), so the existing patch in `llm.py` remains correct after expanding the substring match | **Medium** | Qwen3.6 inherits ChatML template; likely unchanged but MUST be verified | Runaway generation (no stop on assistant turn end) | Phase 0: read `tokenizer_config.json` from HF repo; confirm `eos_token`, `special_tokens_map` |
| A5 | Text-only mode via `--language-model-only` disables the vision encoder cleanly with no residual state | **Medium** | Flag exists in `vllm_mlx/server.py` for 3.5 MLLM variants | Start-up crash or silent loading of vision weights into RAM | Phase 2 smoke test: start server with flag, confirm VRAM/RAM footprint |

### 2.2 Operational Assumptions
- A6: Users who currently depend on Qwen3.5 will accept an opt-out via `VLLM_MLX_MODEL_ID` env var or explicit CLI flag (rollback path preserved).
- A7: No downstream consumer hardcodes `"Qwen3.5"` in client-side tool-schema prompts (checked only within this repo; external consumers out of scope).

### 2.3 Scope Assumptions
- A8: 122B-MINT variant is a **separate SPEC** and its shell script remains untouched.
- A9: Multimodal activation (vision / video) is **out of scope** for this SPEC; text-only via `--language-model-only` is the Phase 2 & 3 default.

---

## 3. Requirements (EARS Format)

### 3.1 Ubiquitous Requirements (always active)

**REQ-U1** — The system SHALL expose a single Python module `vllm_mlx/config/models.py` as the sole source of truth for:
- `DEFAULT_MODEL_ID: str` — the active default model ID used when no env var / CLI override is provided
- `LEGACY_MODEL_ID: str` — the previous generation model ID retained for rollback
- `REASONING_PARSER: str` — the server flag value for the reasoning parser
- `TOOL_PARSER: str` — the server flag value for the tool-call parser (3.6 preference, 3.5 fallback resolved at runtime)
- `EOS_PATCH_MODEL_PATTERNS: tuple[str, ...]` — substring patterns triggering the `<|im_end|>` EOS override
- `ModelProfile` — a `@dataclass(frozen=True)` bundling sampling defaults per thinking / instruct mode (temperature, top_p, top_k, max_tokens, repetition_penalty)

**REQ-U2** — All source files, shell scripts, and example scripts in this repository (excluding the 122B-MINT variant) SHALL import model identifiers and parser flags from `vllm_mlx/config/models.py` OR read them from environment variables. No hardcoded `"Qwen3.5"` or `"Qwen3.6"` string literals SHALL remain in non-config modules after Phase 1 completion.

**REQ-U3** — The project SHALL retain Apache-2.0 license compatibility; no model weights or tokenizer files SHALL be bundled in-repo.

### 3.2 Event-Driven Requirements

**REQ-E1** — WHEN a user launches `start-server-qwen36.sh`, THEN the server SHALL load the configured Qwen3.6-35B-A3B MLX quant with:
- `--reasoning-parser qwen3`
- `--tool-call-parser qwen3_coder` if the installed `vllm-mlx` registers that parser; ELSE fall back to `qwen`
- `--language-model-only` (text-only, per A9)

**REQ-E2** — WHEN the server starts and the loaded model name matches any pattern in `EOS_PATCH_MODEL_PATTERNS` (which SHALL include both `qwen3.5` and `qwen3.6` substrings after Phase 2), THEN the EOS token SHALL be overridden to `<|im_end|>` at tokenizer-load time.

**REQ-E3** — WHEN the user launches the legacy `start-server.sh` after Phase 3 cutover WITHOUT any override, THEN the server SHALL load `DEFAULT_MODEL_ID` (= Qwen3.6).

### 3.3 State-Driven Requirements

**REQ-S1** — WHILE `DEFAULT_MODEL_ID` is set to the Qwen3.6 identifier, a default invocation of `start-server.sh` (no env var, no CLI arg) SHALL start a server serving Qwen3.6.

**REQ-S2** — WHILE the environment variable `VLLM_MLX_MODEL_ID` is set to a non-empty string, any shell launcher in this repo (after Phase 2) SHALL load the model ID specified by the env var, overriding `DEFAULT_MODEL_ID`.

**REQ-S3** — WHILE the Phase 0 architecture gate has NOT been passed, no code changes targeting Qwen3.6 SHALL be merged into `main`.

### 3.4 Unwanted-Behavior Requirements

**REQ-N1** — IF `mlx-lm` raises `ValueError: Unsupported architecture` (or equivalent) on a Qwen3.6 load attempt, THEN the server SHALL fail fast with a structured error message referencing:
- The offending model ID
- The Phase 0 verification script name (`scripts/verify_phase1_env.py` or new equivalent)
- A pointer to this SPEC (`SPEC-MIGRATE-QWEN36`)

**REQ-N2** — The server SHALL NOT silently fall back to Qwen3.5 when a Qwen3.6 load fails. Any fallback SHALL require explicit user action (env var or CLI flag).

**REQ-N3** — The system SHALL NOT emit requests exceeding the model's declared context window; WHEN a request exceeds `model_profile.max_context_tokens`, the server SHALL reject with HTTP 400 and a clear error (already existing behavior; must be preserved).

**REQ-N4** — No test, example, or documentation SHALL retain a hardcoded `mlx-community/Qwen3.5-35B-A3B-4bit` literal after Phase 1 completion, EXCEPT:
- `vllm_mlx/config/models.py` (as `LEGACY_MODEL_ID`)
- Migration notes in `PLAN.md` historical sections (clearly marked as historical)

### 3.5 Optional Requirements

**REQ-O1** — WHERE the `VLLM_MLX_MODEL_ID` environment variable is set, the server SHALL override `DEFAULT_MODEL_ID` with its value and propagate the override through all launcher scripts.

**REQ-O2** — WHERE the operator sets `VLLM_MLX_THINKING_MODE=1`, the sampling defaults from `ModelProfile.thinking` SHALL be applied; otherwise `ModelProfile.instruct` SHALL apply.

**REQ-O3** — WHERE a future SPEC enables multimodal mode, the `--language-model-only` flag SHALL be removed from the Qwen3.6 launcher (out of scope for this SPEC).

---

## 4. User Stories

### Story 1 — Operator: default upgrade
> As an operator running `./start-server.sh` after pulling `main` post-Phase-3, I want the server to serve Qwen3.6-35B-A3B with sensible defaults, so that I benefit from the new model's quality and context window without touching config.

**Acceptance**: `./start-server.sh` (no args, no env) starts and responds to an OpenAI-compatible `/v1/chat/completions` request with Qwen3.6 output.

### Story 2 — Power user: rollback to Qwen3.5
> As a power user who has tuned prompts for Qwen3.5, I want a single-environment-variable rollback path, so that I can keep serving Qwen3.5 while evaluating 3.6 in parallel.

**Acceptance**: `VLLM_MLX_MODEL_ID=mlx-community/Qwen3.5-35B-A3B-4bit ./start-server.sh` loads the 3.5 model successfully, with `qwen` tool-parser preserved.

### Story 3 — Developer: centralized model config
> As a contributor adding a new example script, I want to import `DEFAULT_MODEL_ID` from `vllm_mlx.config.models`, so that my example automatically tracks whichever model the project currently defaults to.

**Acceptance**: A new example file can be created using `from vllm_mlx.config.models import DEFAULT_MODEL_ID` with zero further string literals; `grep -r "mlx-community/Qwen" examples/` after the migration returns only import statements referencing the config module (or the legacy-test file pointing at `LEGACY_MODEL_ID`).

### Story 4 — Operator: A/B evaluation
> As an operator comparing Qwen3.5 vs Qwen3.6 quality, I want a reproducible benchmark script that runs the same prompt set against both models, so that I can make a data-driven cutover decision.

**Acceptance**: A benchmark script (new or extended) accepts `--model` as CLI arg and emits side-by-side latency + output logs for both models on ≥ 5 shared prompts in thinking + instruct modes.

---

## 5. Scope

### 5.1 In Scope
1. Create `vllm_mlx/config/models.py` with the API defined in REQ-U1.
2. Replace all hardcoded 3.5 references with imports from the new config module (≥ 10 files).
3. Extend `vllm_mlx/models/llm.py` EOS-patch branch to match both `qwen3.5` and `qwen3.6` substrings, using `EOS_PATCH_MODEL_PATTERNS` from the config module. Verify via `tokenizer_config.json` of the 3.6 model before edit.
4. Add `start-server-qwen36.sh` launcher that leaves `start-server.sh` pointing at 3.5 until Phase 3 cutover.
5. Add `examples/test_qwen36_mllm.py` mirroring the 3.5 multimodal smoke test (structure only; vision encoder remains off, per A9).
6. Parameterize turbo-KV benchmark scripts (`scripts/test_turbo_kv_35b.py`, `scripts/test_turbo_kv_small.py`, `scripts/test_turbo_kv_122b.py` only where safe) to accept `--model <id>` CLI arg; remove the `MODEL_DEFAULT` constant from `scripts/test_turbo_kv_35b.py`.
7. Update `vllm_mlx/server.py` to resolve the tool-call parser dynamically: prefer `qwen3_coder` if available in the installed `vllm-mlx` parser registry, else fall back to `qwen`.
8. Phase 3 flip: set `DEFAULT_MODEL_ID = "<Qwen3.6 MLX quant ID>"`; keep `LEGACY_MODEL_ID = "mlx-community/Qwen3.5-35B-A3B-4bit"`.
9. Update `README.md`, `PLAN.md`, `PLAN-obsidian-plugin.md` so Qwen3.6 is the default and 3.5 is labelled legacy.
10. Deliver a Phase 4 A/B benchmark report per Story 4.

### 5.2 Out of Scope
- Migration of `start-server-122b-mint.sh` or any Qwen3.5-122B-MINT code path (future SPEC).
- Activation of the vision / video encoder by default (text-only via `--language-model-only`; future SPEC).
- Custom implementation of Gated DeltaNet in this repo. If `mlx-lm` does not support the architecture, Phase 0 fails the gate and this SPEC pauses pending upstream support.
- Changes to the OpenAI-compatible HTTP schema beyond parser-flag defaults.
- New multimodal examples beyond the structural placeholder in `examples/test_qwen36_mllm.py`.

---

## 6. Specifications

### 6.1 API of `vllm_mlx/config/models.py`

Module-level constants (strings):
- `DEFAULT_MODEL_ID` — the active default (Phase 2: 3.5 quant; Phase 3: 3.6 quant)
- `LEGACY_MODEL_ID` — the previous generation ID (Phase 3: 3.5 quant; before Phase 3: empty or equal to default)
- `REASONING_PARSER` — default `"qwen3"` for both 3.5 and 3.6
- `TOOL_PARSER_PREFERRED` — `"qwen3_coder"`
- `TOOL_PARSER_FALLBACK` — `"qwen"`
- `EOS_PATCH_MODEL_PATTERNS` — tuple of lowercase substrings; Phase 1: `("qwen3",)`; Phase 2: `("qwen3.5", "qwen3.6")` (tightened to avoid false positives once 3.6 is verified)

Dataclass:
- `ModelProfile(frozen=True)` fields: `model_id: str`, `instruct: SamplingDefaults`, `thinking: SamplingDefaults`, `max_context_tokens: int`, `supports_tool_calling: bool`, `supports_multimodal: bool`
- `SamplingDefaults(frozen=True)` fields: `temperature: float`, `top_p: float`, `top_k: int`, `repetition_penalty: float`, `max_tokens: int`

Module-level profiles (read-only):
- `QWEN35_PROFILE: ModelProfile`
- `QWEN36_PROFILE: ModelProfile`
- `PROFILES: dict[str, ModelProfile]` keyed by model ID for runtime lookup

Functions:
- `resolve_model_id(env: Mapping[str, str] | None = None) -> str` — returns `env["VLLM_MLX_MODEL_ID"]` if set, else `DEFAULT_MODEL_ID`
- `resolve_tool_parser(registered_parsers: Iterable[str]) -> str` — returns preferred if in iterable, else fallback
- `matches_eos_patch(model_id: str) -> bool` — returns `True` if any pattern in `EOS_PATCH_MODEL_PATTERNS` is a substring of `model_id.lower()`

### 6.2 Launcher Script Contract
Both `start-server.sh` and `start-server-qwen36.sh` SHALL:
1. Read `VLLM_MLX_MODEL_ID` from env and pass it through to `python -m vllm_mlx.server ...`.
2. Not hardcode any 3.5 / 3.6 literal; instead shell out to a tiny helper (e.g., `python -c "from vllm_mlx.config.models import DEFAULT_MODEL_ID; print(DEFAULT_MODEL_ID)"`) OR source a generated `.env` produced at build time. Implementation choice is deferred to plan.md.
3. Pass `--language-model-only` for Phase 2 & 3.
4. Not invoke any subprocess that would download model weights silently; weight download remains `mlx-lm`'s responsibility via first-run cache.

### 6.3 Server Tool-Parser Resolution
`vllm_mlx/server.py` SHALL:
1. Import `resolve_tool_parser` from `vllm_mlx.config.models`.
2. On startup, discover the set of registered vLLM tool-call parsers (existing vllm-mlx API; if absent, fall back to a hardcoded capability table bounded by `TOOL_PARSER_PREFERRED` / `TOOL_PARSER_FALLBACK`).
3. Log the chosen parser at INFO level with rationale (preferred-available vs fallback).

### 6.4 Tokenizer EOS Patch
`vllm_mlx/models/llm.py` SHALL:
1. Replace the hardcoded `"qwen3" in self.model_name.lower()` check with `matches_eos_patch(self.model_name)`.
2. Use the `<|im_end|>` string from the tokenizer's `added_tokens_decoder` if available; fall back to literal `"<|im_end|>"` only if lookup fails.
3. Emit a WARN log when the patch is applied with the matched pattern and tokenizer-reported EOS before override.

### 6.5 Traceability (TAG BLOCK)
| TAG | Artifact |
|---|---|
| `@SPEC:MIGRATE-QWEN36` | This spec.md |
| `@PLAN:MIGRATE-QWEN36` | plan.md |
| `@ACCEPT:MIGRATE-QWEN36` | acceptance.md |
| `@CODE:MIGRATE-QWEN36/config` | `vllm_mlx/config/models.py` (to be created) |
| `@CODE:MIGRATE-QWEN36/eos-patch` | `vllm_mlx/models/llm.py:85-87` (to be extended) |
| `@CODE:MIGRATE-QWEN36/server` | `vllm_mlx/server.py` (tool-parser resolution) |
| `@CODE:MIGRATE-QWEN36/launcher` | `start-server.sh`, `start-server-qwen36.sh` |
| `@TEST:MIGRATE-QWEN36` | Characterization tests (Phase 1), smoke tests (Phase 2), A/B benchmark (Phase 4) |

---

## 7. Expert Consultation Recommendation

This SPEC involves:
- **Backend / engine changes** (model loading, tokenizer patching, tool-parser resolution) — recommend **expert-backend** consultation for API-contract review of the `vllm_mlx/config/models.py` module boundary.
- **Refactoring across ≥ 10 files** — recommend **expert-refactoring** consultation for the centralization sweep plan in Phase 1.
- **Infrastructure / deployment scripts** (launcher shell scripts, env-var contracts) — recommend **expert-devops** consultation for the shell-Python boundary design.

No UI, accessibility, or design-system concerns identified. `design-uiux` consultation not required.

---

## 8. Constitution Alignment

Verified against `.moai/project/tech.md` (if present) and repo norms:
- Python version: compatible with existing `pyproject.toml` (3.11+).
- No new forbidden libraries introduced.
- Follows existing module naming (`vllm_mlx/<subpackage>/<module>.py`).
- License Apache-2.0 preserved (inherited from upstream model).
- Logging conventions follow existing `vllm_mlx/server.py` patterns.
