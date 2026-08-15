# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Default model flipped from Qwen3.6-35B-A3B to Qwen3.8-27B (dense VLM).**
  The server now loads `mlx-community/Qwen3.8-27B-4bit` (~15 GB) by default.
  Qwen3.8-27B is a dense model with a native vision encoder — token
  generation is slower per token than the A3B MoE quants (27B active vs 3B
  active) but single-request quality on agentic coding tasks is substantially
  higher (SWE-bench Pro 53.5 → 61.7, Terminal Bench 63.4 → 73.0 per the
  Qwen3.8 model card). Reasoning format (`<think>...</think>`) and
  tool-call format are unchanged, so the existing `qwen36` reasoning parser
  and `qwen3_coder` tool parser carry over as-is. See SPEC-MIGRATE-QWEN38.

### Added
- `QWEN38_PROFILE` registered in `vllm_mlx.config.models.PROFILES` with
  official 3.8 sampling defaults, including the new instruct-mode
  `presence_penalty=1.5` recommendation (`SamplingDefaults.presence_penalty`,
  default `0.0` — additive metadata, older profiles unaffected).
- `EOS_PATCH_MODEL_PATTERNS` extended to `("qwen3.5", "qwen3.6", "qwen3.8")`.
- `resolve_reasoning_parser` maps `qwen3.8` model ids to the shared `qwen36`
  parser (identical `<think>` format).

### Rollback
- Operators preferring Qwen3.6 can set the `VLLM_MLX_MODEL_ID` environment
  variable to override the default:
  ```
  VLLM_MLX_MODEL_ID=mlx-community/Qwen3.6-35B-A3B-4bit ./start-server.sh
  ```
  `LEGACY_MODEL_ID` now points at the Qwen3.6 quant (one-step rollback);
  `QWEN35_PROFILE` / `QWEN36_PROFILE` hold their own literal ids.
- **Default model flipped from Qwen3.5-35B-A3B to Qwen3.6-35B-A3B.**
  `./start-server.sh` now loads `mlx-community/Qwen3.6-35B-A3B-4bit` by default.
  First-time invocation will download ~20.4 GB of MLX-quantized weights.
  See SPEC-MIGRATE-QWEN36 for the full migration rationale.

### Added
- `vllm_mlx.config.models` — single-source-of-truth module for model IDs,
  parser flags, EOS patch patterns, and sampling profiles (`QWEN35_PROFILE`,
  `QWEN36_PROFILE`, `PROFILES`).
- `start-server-qwen36.sh` — opt-in launcher for Qwen3.6 (retained post-cutover
  as a convenience; `start-server.sh` now uses the same default).
  *(Consolidated into `start-server.sh` on 2026-04-27; the dedicated launcher
  was retired because its `--language-model-only` flag was never wired into
  `vllm_mlx/server.py`.)*
- `examples/test_qwen36_mllm.py` — text-only smoke test example for Qwen3.6.

### Rollback
- Operators preferring Qwen3.5 can set the `VLLM_MLX_MODEL_ID` environment
  variable to override the default:
  ```
  VLLM_MLX_MODEL_ID=mlx-community/Qwen3.5-35B-A3B-4bit ./start-server.sh
  ```
  The legacy model ID is also exposed as `LEGACY_MODEL_ID` from
  `vllm_mlx.config.models` for use in scripts and examples that intentionally
  target the previous generation.

### Deferred
- `vllm` optional dependency may lack the `qwen3_coder` tool-call parser at the
  currently-pinned version. The server falls back to the `qwen` parser
  transparently. A future `vllm` bump will enable the preferred parser.
- Multimodal (vision + audio) capability of Qwen3.6 is not activated by the
  default launchers; text-only via `--language-model-only` is the Phase 3
  configuration. A follow-up SPEC will address multimodal enablement.
