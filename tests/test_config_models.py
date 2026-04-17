# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for ``vllm_mlx.config.models``.

These tests freeze the Phase 1 contract of the centralized model
configuration module introduced by SPEC-MIGRATE-QWEN36. They are
unit-level: no server boot, no model download, no ``mlx_lm`` load. The
goal is to lock down the public API so that Phase 2/3 changes that flip
``DEFAULT_MODEL_ID`` or tighten ``EOS_PATCH_MODEL_PATTERNS`` must be
explicit and reviewed.

@TEST:MIGRATE-QWEN36
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vllm_mlx.config import models as config_models
from vllm_mlx.config.models import (
    DEFAULT_MODEL_ID,
    EOS_PATCH_MODEL_PATTERNS,
    LEGACY_MODEL_ID,
    PROFILES,
    QWEN35_PROFILE,
    REASONING_PARSER,
    TOOL_PARSER_FALLBACK,
    TOOL_PARSER_PREFERRED,
    ModelProfile,
    SamplingDefaults,
    matches_eos_patch,
    resolve_model_id,
    resolve_tool_parser,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
class TestModelIdentifiers:
    """Freeze the Phase 1 model-identifier contract."""

    def test_default_model_id_is_qwen35(self) -> None:
        # REQ-U1: DEFAULT_MODEL_ID must remain the 3.5 quant in Phase 1.
        assert DEFAULT_MODEL_ID == "mlx-community/Qwen3.5-35B-A3B-4bit"

    def test_legacy_model_id_equals_default_in_phase1(self) -> None:
        # Phase 1 invariant: legacy and default coincide. Phase 3 will
        # diverge them — that change must break this test deliberately.
        assert LEGACY_MODEL_ID == DEFAULT_MODEL_ID
        assert LEGACY_MODEL_ID == "mlx-community/Qwen3.5-35B-A3B-4bit"

    def test_reasoning_parser_is_qwen3(self) -> None:
        assert REASONING_PARSER == "qwen3"

    def test_tool_parser_preferred_is_qwen3_coder(self) -> None:
        assert TOOL_PARSER_PREFERRED == "qwen3_coder"

    def test_tool_parser_fallback_is_qwen(self) -> None:
        assert TOOL_PARSER_FALLBACK == "qwen"

    def test_eos_patch_patterns_phase1_permissive(self) -> None:
        # Phase 1 keeps the single permissive pattern so that the refactor
        # is a pure no-op relative to the pre-existing inline check.
        assert EOS_PATCH_MODEL_PATTERNS == ("qwen3",)

    def test_eos_patch_patterns_is_tuple(self) -> None:
        # Immutability guard: must be a tuple, not a list.
        assert isinstance(EOS_PATCH_MODEL_PATTERNS, tuple)


# ---------------------------------------------------------------------------
# matches_eos_patch
# ---------------------------------------------------------------------------
class TestMatchesEosPatch:
    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen3.5-35B-A3B",
            "mlx-community/Qwen3.5-35B-A3B-4bit",
            "qwen3-anything",
            "QWEN3.6-PREVIEW",  # case-insensitive
            "mlx-community/qwen3.6-coder-30b",
        ],
    )
    def test_matches_eos_patch_positive(self, model_id: str) -> None:
        assert matches_eos_patch(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "llama-3",
            "gpt-4",
            "meta-llama/Llama-3-70B",
            "mistralai/Mistral-7B",
            "",
        ],
    )
    def test_matches_eos_patch_negative(self, model_id: str) -> None:
        assert matches_eos_patch(model_id) is False


# ---------------------------------------------------------------------------
# resolve_model_id
# ---------------------------------------------------------------------------
class TestResolveModelId:
    def test_respects_env_override(self) -> None:
        env = {"VLLM_MLX_MODEL_ID": "foo"}
        assert resolve_model_id(env) == "foo"

    def test_falls_back_to_default_on_missing_key(self) -> None:
        assert resolve_model_id({}) == DEFAULT_MODEL_ID

    def test_falls_back_to_default_on_empty_string(self) -> None:
        # An empty env var value must NOT override — it is indistinguishable
        # from "unset" in shell convention.
        assert resolve_model_id({"VLLM_MLX_MODEL_ID": ""}) == DEFAULT_MODEL_ID

    def test_none_env_returns_default(self) -> None:
        # Passing None disables env lookup entirely.
        assert resolve_model_id(None) == DEFAULT_MODEL_ID

    def test_default_argument_is_none(self) -> None:
        # No-arg call is the 99% case in launcher scripts.
        assert resolve_model_id() == DEFAULT_MODEL_ID


# ---------------------------------------------------------------------------
# resolve_tool_parser
# ---------------------------------------------------------------------------
class TestResolveToolParser:
    def test_preferred_available(self) -> None:
        assert resolve_tool_parser({"qwen3_coder", "qwen"}) == "qwen3_coder"

    def test_fallback_when_preferred_absent(self) -> None:
        assert resolve_tool_parser({"qwen"}) == "qwen"

    def test_empty_registry_returns_fallback(self) -> None:
        # Per spec §6.3, the fallback is the final default even when the
        # registry is empty. Caller is responsible for validating
        # availability before instantiating the parser.
        assert resolve_tool_parser(set()) == TOOL_PARSER_FALLBACK

    def test_accepts_generator(self) -> None:
        # Common call pattern: pass a generator from
        # ToolParserManager.list_registered().
        def gen():
            yield "qwen3_coder"
            yield "qwen"

        assert resolve_tool_parser(gen()) == "qwen3_coder"

    def test_accepts_list(self) -> None:
        assert resolve_tool_parser(["mistral", "qwen"]) == "qwen"


# ---------------------------------------------------------------------------
# Dataclass immutability
# ---------------------------------------------------------------------------
class TestImmutability:
    def test_sampling_defaults_is_frozen(self) -> None:
        sd = SamplingDefaults(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            max_tokens=4096,
        )
        with pytest.raises(FrozenInstanceError):
            sd.temperature = 0.9  # type: ignore[misc]

    def test_model_profile_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            QWEN35_PROFILE.max_context_tokens = 99  # type: ignore[misc]

    def test_profiles_contains_qwen35(self) -> None:
        assert DEFAULT_MODEL_ID in PROFILES
        assert PROFILES[DEFAULT_MODEL_ID] is QWEN35_PROFILE

    def test_qwen35_profile_fields_present(self) -> None:
        # Spot-check shape: all six top-level fields and both nested
        # SamplingDefaults instances are wired up.
        assert QWEN35_PROFILE.model_id == DEFAULT_MODEL_ID
        assert isinstance(QWEN35_PROFILE.instruct, SamplingDefaults)
        assert isinstance(QWEN35_PROFILE.thinking, SamplingDefaults)
        assert QWEN35_PROFILE.max_context_tokens > 0
        assert QWEN35_PROFILE.supports_tool_calling is True
        assert QWEN35_PROFILE.supports_multimodal is False

    def test_sampling_defaults_sane_ranges(self) -> None:
        # Light invariant check so a future edit can't silently set
        # nonsensical values.
        for sd in (QWEN35_PROFILE.instruct, QWEN35_PROFILE.thinking):
            assert 0.0 < sd.temperature <= 2.0
            assert 0.0 < sd.top_p <= 1.0
            assert sd.top_k > 0
            assert sd.repetition_penalty >= 1.0
            assert sd.max_tokens > 0


# ---------------------------------------------------------------------------
# Package-level re-exports
# ---------------------------------------------------------------------------
class TestPackageExports:
    def test_config_package_reexports(self) -> None:
        from vllm_mlx import config

        assert config.DEFAULT_MODEL_ID == DEFAULT_MODEL_ID
        assert config.resolve_model_id is config_models.resolve_model_id
        assert config.matches_eos_patch is config_models.matches_eos_patch
