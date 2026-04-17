# SPDX-License-Identifier: Apache-2.0
"""Configuration package for vllm-mlx.

Exports the single source of truth for model identifiers, parser flags,
EOS-patch patterns, and sampling profiles. See :mod:`vllm_mlx.config.models`
for the full API.
"""

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
    ModelProfile,
    SamplingDefaults,
    matches_eos_patch,
    resolve_model_id,
    resolve_reasoning_parser,
    resolve_tool_parser,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "LEGACY_MODEL_ID",
    "REASONING_PARSER",
    "TOOL_PARSER_PREFERRED",
    "TOOL_PARSER_FALLBACK",
    "EOS_PATCH_MODEL_PATTERNS",
    "ModelProfile",
    "SamplingDefaults",
    "QWEN35_PROFILE",
    "QWEN36_PROFILE",
    "PROFILES",
    "resolve_model_id",
    "resolve_reasoning_parser",
    "resolve_tool_parser",
    "matches_eos_patch",
]
