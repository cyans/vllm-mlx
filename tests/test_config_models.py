# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for ``vllm_mlx.config.models``.

These tests freeze the Phase 1, Phase 2, and Phase 3 contract of the
centralized model configuration module introduced by SPEC-MIGRATE-QWEN36.
They are unit-level: no server boot, no model download, no ``mlx_lm``
load. The goal is to lock down the public API so that future changes
affecting default model selection are explicit and reviewed.

Phase 2 changes locked in by this suite:

* ``EOS_PATCH_MODEL_PATTERNS`` is tightened from the Phase 1 permissive
  ``("qwen3",)`` to ``("qwen3.5", "qwen3.6")``. Consequence: bare
  ``"qwen3"`` strings (without a version suffix) no longer match.
* A new :data:`~vllm_mlx.config.models.QWEN36_PROFILE` is registered in
  :data:`~vllm_mlx.config.models.PROFILES` alongside ``QWEN35_PROFILE``.

Phase 3 changes locked in by this suite:

* :data:`~vllm_mlx.config.models.DEFAULT_MODEL_ID` is flipped from the
  Qwen3.5 quant to the Qwen3.6 quant.
* :data:`~vllm_mlx.config.models.LEGACY_MODEL_ID` continues to point at
  the Qwen3.5 quant (they now diverge — this is the rollback handle).
* ``resolve_model_id`` without an override returns the Qwen3.6 quant.
* Setting ``VLLM_MLX_MODEL_ID=LEGACY_MODEL_ID`` restores Qwen3.5.

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
    QWEN36_PROFILE,
    REASONING_PARSER,
    TOOL_PARSER_FALLBACK,
    TOOL_PARSER_PREFERRED,
    SamplingDefaults,
    matches_eos_patch,
    resolve_model_id,
    resolve_tool_parser,
)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
class TestModelIdentifiers:
    """Freeze the Phase 1 / Phase 2 model-identifier contract."""

    def test_default_model_id_is_qwen36_after_phase3(self) -> None:
        # REQ-U1 (Phase 3 cutover): DEFAULT_MODEL_ID is now the 3.6 quant.
        # Duplicate of TestPhase3Cutover.test_default_model_id_is_qwen36
        # kept here so TestModelIdentifiers remains a complete snapshot of
        # the module-level identifier contract.
        assert DEFAULT_MODEL_ID == "mlx-community/Qwen3.6-35B-A3B-4bit"

    def test_legacy_model_id_stays_on_qwen35(self) -> None:
        # Phase 3 invariant: LEGACY_MODEL_ID remains pinned to the 3.5
        # quant and now diverges from DEFAULT_MODEL_ID. This is the
        # rollback handle documented in CHANGELOG.md.
        assert LEGACY_MODEL_ID == "mlx-community/Qwen3.5-35B-A3B-4bit"
        assert LEGACY_MODEL_ID != DEFAULT_MODEL_ID

    def test_reasoning_parser_is_qwen3(self) -> None:
        assert REASONING_PARSER == "qwen3"

    def test_tool_parser_preferred_is_qwen3_coder(self) -> None:
        assert TOOL_PARSER_PREFERRED == "qwen3_coder"

    def test_tool_parser_fallback_is_qwen(self) -> None:
        assert TOOL_PARSER_FALLBACK == "qwen"

    def test_eos_patch_patterns_phase2_tightened(self) -> None:
        # Phase 2 tightens the Phase 1 permissive ``("qwen3",)`` to an
        # explicit pair of versioned substrings. Bare ``"qwen3"`` tokens
        # (without ``.5`` or ``.6``) no longer trigger the EOS patch.
        assert EOS_PATCH_MODEL_PATTERNS == ("qwen3.5", "qwen3.6")

    def test_eos_patch_patterns_is_tuple(self) -> None:
        # Immutability guard: must be a tuple, not a list.
        assert isinstance(EOS_PATCH_MODEL_PATTERNS, tuple)


# ---------------------------------------------------------------------------
# matches_eos_patch — Phase 2 tightened patterns
# ---------------------------------------------------------------------------
class TestMatchesEosPatch:
    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen3.5-35B-A3B",
            "mlx-community/Qwen3.5-35B-A3B-4bit",
            "QWEN3.5-PREVIEW",  # case-insensitive
        ],
    )
    def test_matches_eos_patch_qwen35_positive(self, model_id: str) -> None:
        assert matches_eos_patch(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "Qwen/Qwen3.6-35B-A3B",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
            "QWEN3.6-PREVIEW",  # case-insensitive
        ],
    )
    def test_matches_eos_patch_qwen36_positive(self, model_id: str) -> None:
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

    @pytest.mark.parametrize(
        "model_id",
        [
            "qwen3",  # bare family name, no version
            "qwen3-anything",  # version-less variant
            "Qwen/Qwen3-7B",  # Phase 1 would have matched; Phase 2 does not
        ],
    )
    def test_matches_eos_patch_qwen3_only_no_version_is_now_negative(
        self, model_id: str
    ) -> None:
        """Phase 2 deliberate behavior change.

        Under Phase 1's permissive ``("qwen3",)`` pattern these strings
        returned ``True``. Under Phase 2's tightened
        ``("qwen3.5", "qwen3.6")`` they must return ``False`` because the
        EOS patch is only correct for the 3.5 / 3.6 tokenizer profile.
        """
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
# Dataclass immutability and profile shape (Qwen3.5)
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
        # Phase 3: QWEN35_PROFILE is keyed by LEGACY_MODEL_ID (the 3.5
        # quant) now that DEFAULT_MODEL_ID has flipped to 3.6.
        assert LEGACY_MODEL_ID in PROFILES
        assert PROFILES[LEGACY_MODEL_ID] is QWEN35_PROFILE

    def test_qwen35_profile_fields_present(self) -> None:
        # Spot-check shape: all six top-level fields and both nested
        # SamplingDefaults instances are wired up.
        # Phase 3: the 3.5 profile's model_id is LEGACY_MODEL_ID, not
        # DEFAULT_MODEL_ID (DEFAULT now resolves to 3.6).
        assert QWEN35_PROFILE.model_id == LEGACY_MODEL_ID
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
# QWEN36_PROFILE — Phase 2 addition
# ---------------------------------------------------------------------------
class TestQwen36Profile:
    """Phase 2 locks in the Qwen3.6 profile alongside the 3.5 profile."""

    def test_qwen36_profile_present(self) -> None:
        # Core shape: model_id, context window, capability flags.
        assert QWEN36_PROFILE.model_id == "mlx-community/Qwen3.6-35B-A3B-4bit"
        assert QWEN36_PROFILE.max_context_tokens == 262_144
        assert QWEN36_PROFILE.supports_tool_calling is True
        # Qwen3.6 ships with vision/audio/video special tokens in its
        # tokenizer config; we advertise multimodal support at the profile
        # level even though Phase 2 launchers stay text-only
        # (``--language-model-only``).
        assert QWEN36_PROFILE.supports_multimodal is True

    def test_qwen36_profile_is_frozen(self) -> None:
        with pytest.raises(FrozenInstanceError):
            QWEN36_PROFILE.max_context_tokens = 99  # type: ignore[misc]

    def test_profiles_contains_both(self) -> None:
        # Both entries must coexist so callers can look up either model id.
        assert QWEN35_PROFILE.model_id in PROFILES
        assert QWEN36_PROFILE.model_id in PROFILES
        assert PROFILES[QWEN36_PROFILE.model_id] is QWEN36_PROFILE

    def test_qwen36_thinking_defaults(self) -> None:
        # Per HF model card "Thinking Mode — General Tasks":
        # temperature=1.0, top_p=0.95, top_k=20, repetition_penalty=1.0.
        thinking = QWEN36_PROFILE.thinking
        assert thinking.temperature == 1.0
        assert thinking.top_p == 0.95
        assert thinking.top_k == 20
        assert thinking.repetition_penalty == 1.0
        assert thinking.max_tokens == 32_768

    def test_qwen36_instruct_defaults(self) -> None:
        # Per HF model card "Instruct Mode — General":
        # temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0.
        instruct = QWEN36_PROFILE.instruct
        assert instruct.temperature == 0.7
        assert instruct.top_p == 0.8
        assert instruct.top_k == 20
        assert instruct.repetition_penalty == 1.0
        assert instruct.max_tokens == 32_768

    def test_qwen36_sampling_sane_ranges(self) -> None:
        for sd in (QWEN36_PROFILE.instruct, QWEN36_PROFILE.thinking):
            assert 0.0 < sd.temperature <= 2.0
            assert 0.0 < sd.top_p <= 1.0
            assert sd.top_k > 0
            assert sd.repetition_penalty >= 1.0
            assert sd.max_tokens > 0


# ---------------------------------------------------------------------------
# Phase 3 cutover — DEFAULT_MODEL_ID flipped to Qwen3.6
# ---------------------------------------------------------------------------
class TestPhase3Cutover:
    """Phase 3 locks in the default flip from Qwen3.5 to Qwen3.6.

    @TEST:MIGRATE-QWEN36/phase3
    """

    def test_default_model_id_is_qwen36(self) -> None:
        # REQ-U1 (Phase 3): DEFAULT_MODEL_ID flips from 3.5 to 3.6.
        assert QWEN36_PROFILE.model_id == DEFAULT_MODEL_ID
        assert DEFAULT_MODEL_ID == "mlx-community/Qwen3.6-35B-A3B-4bit"

    def test_legacy_model_id_is_qwen35(self) -> None:
        # Phase 3: LEGACY_MODEL_ID continues to point at the 3.5 quant for
        # rollback via VLLM_MLX_MODEL_ID.
        assert LEGACY_MODEL_ID == "mlx-community/Qwen3.5-35B-A3B-4bit"

    def test_legacy_differs_from_default_phase3(self) -> None:
        # Phase 3 invariant: default and legacy diverge (3.6 vs 3.5).
        assert LEGACY_MODEL_ID != DEFAULT_MODEL_ID

    def test_resolve_model_id_default_is_qwen36(self) -> None:
        # Explicit assertion: no env override => 3.6 quant is returned.
        assert resolve_model_id({}) == "mlx-community/Qwen3.6-35B-A3B-4bit"

    def test_resolve_model_id_env_rollback_to_qwen35(self) -> None:
        # Documented rollback path: operators set VLLM_MLX_MODEL_ID to
        # LEGACY_MODEL_ID to opt back into the previous generation.
        env = {"VLLM_MLX_MODEL_ID": LEGACY_MODEL_ID}
        assert resolve_model_id(env) == "mlx-community/Qwen3.5-35B-A3B-4bit"


# ---------------------------------------------------------------------------
# Package-level re-exports
# ---------------------------------------------------------------------------
class TestPackageExports:
    def test_config_package_reexports(self) -> None:
        from vllm_mlx import config

        assert config.DEFAULT_MODEL_ID == DEFAULT_MODEL_ID
        assert config.resolve_model_id is config_models.resolve_model_id
        assert config.matches_eos_patch is config_models.matches_eos_patch

    def test_config_package_reexports_qwen36(self) -> None:
        # Phase 2: QWEN36_PROFILE must be reachable via the package-level
        # shortcut just like QWEN35_PROFILE.
        from vllm_mlx import config

        assert config.QWEN36_PROFILE is QWEN36_PROFILE
