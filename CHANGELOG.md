# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
