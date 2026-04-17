# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for vllm-mlx model configuration.

This module centralizes every model identifier, parser flag, EOS-patch
pattern, and sampling default the rest of the codebase depends on. See
``SPEC-MIGRATE-QWEN36`` (``.moai/specs/SPEC-MIGRATE-QWEN36/spec.md``) §6.1
for the full API contract.

The module supports three consumer patterns:

1. Direct constant import:
   ``from vllm_mlx.config.models import DEFAULT_MODEL_ID``
2. Environment-aware resolution:
   ``model = resolve_model_id(os.environ)``
3. Profile lookup for sampling defaults:
   ``profile = PROFILES[model_id]``

Phase 1 keeps the Qwen3.5 quant as ``DEFAULT_MODEL_ID``. Phase 3 will flip
the default to a Qwen3.6 identifier while ``LEGACY_MODEL_ID`` continues to
point at the Qwen3.5 quant for rollback purposes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
# @CODE:MIGRATE-QWEN36/config
# @CODE:MIGRATE-QWEN36/phase3 — Phase 3 cutover: DEFAULT flipped to Qwen3.6.
# Note: this literal must stay in sync with ``QWEN36_PROFILE.model_id`` below.
# Because ``QWEN36_PROFILE`` is defined after this constant, we cannot
# reference it directly here without reordering the module. The module-level
# ``assert`` after the profile block enforces that they never drift.
DEFAULT_MODEL_ID: str = "mlx-community/Qwen3.6-35B-A3B-4bit"
"""Active default model ID.

Used when neither the ``VLLM_MLX_MODEL_ID`` environment variable nor an
explicit CLI override is provided. Phase 3 flips this from the Qwen3.5
quant to the Qwen3.6 quant. Operators can opt back into Qwen3.5 by setting
``VLLM_MLX_MODEL_ID`` to :data:`LEGACY_MODEL_ID` or the literal id string.
"""

LEGACY_MODEL_ID: str = "mlx-community/Qwen3.5-35B-A3B-4bit"
"""Previous-generation model ID retained for rollback.

Phase 1 and Phase 2 had this equal to :data:`DEFAULT_MODEL_ID`. Phase 3
diverges them: :data:`DEFAULT_MODEL_ID` becomes the Qwen3.6 quant while
this constant continues to point at the Qwen3.5 quant. Set
``VLLM_MLX_MODEL_ID`` to this value to opt back into the previous
generation without editing launcher scripts.
"""

# ---------------------------------------------------------------------------
# Parser flag values
# ---------------------------------------------------------------------------
REASONING_PARSER: str = "qwen3"
"""Default ``--reasoning-parser`` flag value.

The ``qwen3`` reasoning parser handles both Qwen3.5 and Qwen3.6 output
formats (``<think>`` / ``</think>`` channels), so this constant is stable
across Phase 1 through Phase 3.
"""

TOOL_PARSER_PREFERRED: str = "qwen3_coder"
"""Preferred ``--tool-call-parser`` flag value.

This parser is registered in ``vllm_mlx/tool_parsers/hermes_tool_parser.py``
(alongside the ``hermes`` / ``nous`` aliases) and is preferred for Qwen3.x
models because it handles the Qwen-Coder-style function-call JSON emitted
by Qwen3.6.
"""

TOOL_PARSER_FALLBACK: str = "qwen"
"""Fallback ``--tool-call-parser`` flag value.

Used when the installed ``vllm-mlx`` parser registry does not expose the
preferred parser. The ``qwen`` parser handles the classic
``<tool_call>`` / ``[Calling tool:]`` formats used by Qwen3.5.
"""

# ---------------------------------------------------------------------------
# EOS-patch substring patterns
# ---------------------------------------------------------------------------
EOS_PATCH_MODEL_PATTERNS: tuple[str, ...] = ("qwen3.5", "qwen3.6")
"""Substrings that, when present in a model id, trigger the ``<|im_end|>``
EOS override documented in ``vllm_mlx/models/llm.py``.

Phase 2 tightens the Phase 1 permissive ``("qwen3",)`` pattern to the
explicit versioned pair ``("qwen3.5", "qwen3.6")``. Consequence: bare
``"qwen3"`` tokens (for example ``Qwen/Qwen3-7B``) no longer trigger the
EOS patch — the ``<|im_end|>`` override is specific to the 3.5 / 3.6
tokenizer profile and should not silently be applied to future unrelated
members of the Qwen3 family.

Matching is case-insensitive and substring-based; see
:func:`matches_eos_patch` for the comparison semantics.
"""


# ---------------------------------------------------------------------------
# Sampling profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingDefaults:
    """Immutable bundle of sampling hyperparameters for a single mode."""

    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_tokens: int


@dataclass(frozen=True)
class ModelProfile:
    """Immutable bundle of sampling defaults and capability flags per model."""

    model_id: str
    instruct: SamplingDefaults
    thinking: SamplingDefaults
    max_context_tokens: int
    supports_tool_calling: bool
    supports_multimodal: bool


# Qwen3.5 community-recommended defaults.
# - Instruct mode: temperature=0.7, top_p=0.8, top_k=20 (Qwen docs).
# - Thinking mode: temperature=1.0, top_p=0.95, top_k=40 (higher diversity
#   so the chain-of-thought does not collapse to deterministic loops).
# Sources: https://github.com/QwenLM/Qwen2.5 README sampling section.
QWEN35_PROFILE: ModelProfile = ModelProfile(
    # @CODE:MIGRATE-QWEN36/phase3 — Phase 3 flipped DEFAULT to 3.6, so the
    # 3.5 profile must be bound to ``LEGACY_MODEL_ID`` (the 3.5 quant)
    # rather than ``DEFAULT_MODEL_ID`` to preserve its identity.
    model_id=LEGACY_MODEL_ID,
    instruct=SamplingDefaults(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        max_tokens=8192,
    ),
    thinking=SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.0,
        max_tokens=16384,
    ),
    max_context_tokens=32_768,
    supports_tool_calling=True,
    supports_multimodal=False,
)

# Qwen3.6 Hugging-Face model-card defaults.
# - Instruct mode ("General"):  temperature=0.7, top_p=0.8,  top_k=20,
#                               repetition_penalty=1.0.
# - Thinking mode ("General Tasks"): temperature=1.0, top_p=0.95, top_k=20,
#                                    repetition_penalty=1.0.
# Max context per the 3.6 config is 262_144 tokens. The MLX-community
# 4-bit quant ships with vision / audio / video special tokens in the
# tokenizer config, so we advertise ``supports_multimodal=True`` at the
# profile level; Phase 2 launchers still opt into text-only mode via
# ``--language-model-only``.
QWEN36_PROFILE: ModelProfile = ModelProfile(
    model_id="mlx-community/Qwen3.6-35B-A3B-4bit",
    instruct=SamplingDefaults(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        max_tokens=32_768,
    ),
    thinking=SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.0,
        max_tokens=32_768,
    ),
    max_context_tokens=262_144,
    supports_tool_calling=True,
    supports_multimodal=True,
)

PROFILES: dict[str, ModelProfile] = {
    QWEN35_PROFILE.model_id: QWEN35_PROFILE,
    QWEN36_PROFILE.model_id: QWEN36_PROFILE,
}
"""Read-only map from model id to :class:`ModelProfile`.

Phase 2 registers both the Qwen3.5 and Qwen3.6 profiles so callers can
look up sampling defaults and capability flags for either quant without
conditional logic. Phase 3 flips :data:`DEFAULT_MODEL_ID` to the 3.6
entry; :data:`LEGACY_MODEL_ID` continues to point at the 3.5 entry
for rollback.
"""

# @CODE:MIGRATE-QWEN36/phase3 — Module-level coupling guard.
# ``DEFAULT_MODEL_ID`` is defined as a literal (before ``QWEN36_PROFILE``
# exists), so this assertion pins the two together and fails import-time
# if a future edit breaks the invariant.
assert QWEN36_PROFILE.model_id == DEFAULT_MODEL_ID, (
    "DEFAULT_MODEL_ID must equal QWEN36_PROFILE.model_id "
    f"(got {DEFAULT_MODEL_ID!r} vs {QWEN36_PROFILE.model_id!r})"
)


# ---------------------------------------------------------------------------
# Resolver functions
# ---------------------------------------------------------------------------
def resolve_model_id(env: Mapping[str, str] | None = None) -> str:
    """Return the model id to load for this process.

    If ``env`` contains a non-empty ``VLLM_MLX_MODEL_ID`` entry it takes
    precedence; otherwise :data:`DEFAULT_MODEL_ID` is returned. Passing
    ``None`` (the default) disables env lookup entirely and always returns
    :data:`DEFAULT_MODEL_ID`, which makes the function trivially unit-testable
    without touching :data:`os.environ`.
    """
    if env is None:
        return DEFAULT_MODEL_ID
    override = env.get("VLLM_MLX_MODEL_ID", "")
    if override:
        return override
    return DEFAULT_MODEL_ID


def resolve_tool_parser(registered_parsers: Iterable[str]) -> str:
    """Return the best available tool-call parser name.

    Prefers :data:`TOOL_PARSER_PREFERRED` when present in the registry,
    otherwise falls back to :data:`TOOL_PARSER_FALLBACK`. The fallback is
    returned unconditionally as the final default even if it is absent from
    the registry, so callers always receive a non-empty string; the caller
    is responsible for validating parser availability before use.
    """
    # Materialize in case the caller passes a generator so we can iterate
    # multiple times safely.
    available = set(registered_parsers)
    if TOOL_PARSER_PREFERRED in available:
        return TOOL_PARSER_PREFERRED
    return TOOL_PARSER_FALLBACK


# @CODE:FIX-QWEN36-RUNTIME/parser-resolver — per-model reasoning-parser
# resolver. Phase 1 of SPEC-MIGRATE-QWEN36 hard-coded ``REASONING_PARSER``
# ("qwen3") in the launcher; Phase 1 of SPEC-FIX-QWEN36-RUNTIME introduces
# this resolver so the 3.6 quant picks up the new ``qwen36`` parser
# automatically while 3.5 continues to use the unchanged ``qwen3`` parser.
_REASONING_PARSER_QWEN36 = "qwen36"


def resolve_reasoning_parser(
    model_id: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return the ``--reasoning-parser`` flag value for ``model_id``.

    Precedence (highest first):

    1. ``VLLM_MLX_REASONING_PARSER`` env override — wins unconditionally
       when present and non-empty. Matches the "unset = empty string"
       shell convention used by :func:`resolve_model_id`.
    2. Model-id pattern match (case-insensitive substring):
       * ``"qwen3.6"`` → ``"qwen36"``
       * ``"qwen3.5"`` or bare ``"qwen3"`` → ``"qwen3"``
    3. Module-level :data:`REASONING_PARSER` default.

    Passing ``env=None`` (the default) disables env lookup entirely so the
    function is trivially unit-testable without touching :data:`os.environ`.
    An empty ``model_id`` skips the pattern-match step and falls through to
    the default.
    """
    # Step 1: env override.
    if env is not None:
        override = env.get("VLLM_MLX_REASONING_PARSER", "")
        if override:
            return override

    # Step 2: model-id pattern match.
    if model_id:
        needle = model_id.lower()
        if "qwen3.6" in needle:
            return _REASONING_PARSER_QWEN36
        # "qwen3.5" and bare "qwen3" both resolve to the existing qwen3
        # parser. Ordering matters: check "qwen3.6" first so the more
        # specific pattern wins.
        if "qwen3" in needle:
            return REASONING_PARSER

    # Step 3: default.
    return REASONING_PARSER


def matches_eos_patch(model_id: str) -> bool:
    """Return ``True`` if ``model_id`` triggers the ``<|im_end|>`` EOS patch.

    The comparison is case-insensitive and substring-based: any pattern in
    :data:`EOS_PATCH_MODEL_PATTERNS` that appears anywhere inside
    ``model_id.lower()`` causes the function to return ``True``. An empty
    ``model_id`` always returns ``False``.
    """
    if not model_id:
        return False
    needle = model_id.lower()
    return any(pattern in needle for pattern in EOS_PATCH_MODEL_PATTERNS)
