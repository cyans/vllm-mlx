#!/usr/bin/env python3
"""TurboQuant KV Cache 35B(MoE) 벤치마크 — PLAN.md Phase 3.

Baseline과 TurboQuant는 patch_attention() 전역 패치로 서로 간섭하므로
  1) 모델 로드 → 모든 Baseline 실행 → 언로드
  2) 모델 재로드 → patch_attention() → 모든 TurboQuant 실행
순서로 측정한다.
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate, load as _mlx_load
from mlx_lm.models.cache import KVCache, make_prompt_cache

# Qwen3.6-35B-A3B-4bit 같은 VL 가중치가 섞인 체크포인트도 텍스트 경로만으로
# 로드할 수 있도록, mlx_lm.load 호출 동안만 nn.Module.load_weights 를 lenient
# 모드로 살짝 바꾼다. patch_attention() 은 mlx_lm.SDPA 를 패치하므로
# 텍스트 경로에서 그대로 유효하다.
_VL_FILTER_TOKENS = (
    "vision_tower",
    "patch_embed",
    "merger.linear_fc",
    "merger.norm",
    "pos_embed.weight",
)


def _load(model_id: str) -> Any:
    """mlx_lm.load — VL 가중치는 무시하고 strict=False 로 로딩."""
    original = nn.Module.load_weights

    def _lenient(self, weights, strict=True):  # type: ignore[no-untyped-def]
        if isinstance(weights, list):
            weights = [
                (k, v)
                for k, v in weights
                if not any(tok in k for tok in _VL_FILTER_TOKENS)
            ]
        return original(self, weights, strict=False)

    nn.Module.load_weights = _lenient  # type: ignore[assignment]
    try:
        return _mlx_load(model_id)
    finally:
        nn.Module.load_weights = original  # type: ignore[assignment]


# Backwards-compatible alias for the rest of the script.
load = _load

# @CODE:MIGRATE-QWEN36 — the default model id is owned by
# vllm_mlx.config.models, not hardcoded here. Use --model on the CLI to
# override (e.g., to target the Qwen3.6 quant once available).
from vllm_mlx.config.models import DEFAULT_MODEL_ID

PROMPTS: dict[str, str] = {
    "short": "What is machine learning?",
    "medium": (
        "Write a detailed explanation of how transformer neural networks work, "
        "including the self-attention mechanism, positional encoding, and the "
        "encoder-decoder architecture. Explain each component step by step."
    ),
    "long": (
        "You are a senior software architect. Design a complete microservices "
        "architecture for an e-commerce platform that handles user authentication, "
        "product catalog, shopping cart, order processing, payment integration, "
        "inventory management, and notification services. For each service, describe "
        "the API endpoints, data models, inter-service communication patterns, "
        "database choices, caching strategies, and error handling approaches. "
        "Also discuss deployment strategies, monitoring, and scaling considerations."
    ),
}


def _text_model_args(model: Any):
    if hasattr(model, "language_model") and hasattr(model.language_model, "args"):
        return model.language_model.args
    if hasattr(model, "args"):
        return model.args
    return None


def _head_dim_from_args(model: Any) -> int:
    args = _text_model_args(model)
    if args is not None:
        hd = getattr(args, "head_dim", None)
        if hd is not None:
            return int(hd)
        hs = getattr(args, "hidden_size", None)
        nh = getattr(args, "num_attention_heads", None)
        if hs is not None and nh:
            return int(hs) // int(nh)
    return 128


def _mem_gb() -> tuple[float, float]:
    try:
        return mx.get_active_memory() / 1e9, mx.get_peak_memory() / 1e9
    except Exception:
        return float("nan"), float("nan")


def _release_model(model: Any, tokenizer: Any) -> None:
    del model, tokenizer
    gc.collect()
    try:
        mx.clear_cache()
        mx.reset_peak_memory()
    except Exception:
        pass


def _run_baseline(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    label: str,
) -> dict[str, Any]:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass
    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    active, peak = _mem_gb()
    text = response or ""
    print(f"\n[Baseline] {label}")
    print(f"  시간: {elapsed:.2f}s | peak: {peak:.2f} GB | active: {active:.2f} GB | 글자수: {len(text)}")
    return {
        "time": elapsed,
        "active_gb": active,
        "peak_gb": peak,
        "response_len": len(text),
    }


def _build_turbo_cache(model: Any, TurboQuantKVCache: type) -> tuple[list[Any], int]:
    """make_prompt_cache 후 self-attn 레이어만 TurboQuantKVCache로 교체."""
    prompt_cache = make_prompt_cache(model)
    head_dim = _head_dim_from_args(model)
    n = 0
    for i, layer in enumerate(model.layers):
        if isinstance(prompt_cache[i], KVCache) and hasattr(layer, "self_attn"):
            prompt_cache[i] = TurboQuantKVCache(
                head_dim=head_dim,
                bits=4,
                seed=42 + i,
            )
            n += 1
    return prompt_cache, n


def _run_turbo(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    label: str,
    TurboQuantKVCache: type,
) -> dict[str, Any]:
    prompt_cache, n_layers = _build_turbo_cache(model, TurboQuantKVCache)
    print(f"  TurboQuantKVCache 레이어: {n_layers} (head_dim={_head_dim_from_args(model)})")
    try:
        mx.reset_peak_memory()
    except Exception:
        pass
    t0 = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    active, peak = _mem_gb()
    text = response or ""
    print(f"\n[TurboQuant] {label}")
    print(f"  시간: {elapsed:.2f}s | peak: {peak:.2f} GB | active: {active:.2f} GB | 글자수: {len(text)}")
    return {
        "time": elapsed,
        "active_gb": active,
        "peak_gb": peak,
        "response_len": len(text),
    }


def _long_context_prompt(repeats: int) -> str:
    chunk = "The quick brown fox jumps over the lazy dog. "
    return chunk * repeats


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: TurboQuant 35B-A3B 벤치마크")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"mlx HF 모델 id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="프롬프트당 생성 토큰 (기본 256, PLAN은 500)")
    parser.add_argument(
        "--prompts",
        default="all",
        help="쉼표 구분: short,medium,long 또는 all",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Baseline만 실행",
    )
    parser.add_argument(
        "--turbo-only",
        action="store_true",
        help="TurboQuant만 실행 (patch 후)",
    )
    parser.add_argument(
        "--long-context",
        action="store_true",
        help="긴 프롬프트(반복 문장) 한 번 더 TurboQuant로 생성",
    )
    parser.add_argument(
        "--long-context-repeats",
        type=int,
        default=500,
        help="--long-context 시 문장 반복 횟수",
    )
    parser.add_argument(
        "--long-context-tokens",
        type=int,
        default=100,
        help="--long-context 시 max_tokens",
    )
    args = parser.parse_args()

    if args.baseline_only and args.turbo_only:
        print("--baseline-only 와 --turbo-only 는 동시에 쓸 수 없습니다.", file=sys.stderr)
        sys.exit(2)
    if args.long_context and args.baseline_only:
        print(
            "경고: --long-context 는 TurboQuant 패스에만 적용됩니다. --baseline-only 이므로 건너뜁니다.",
            file=sys.stderr,
        )

    raw = args.prompts.strip().lower()
    if raw == "all":
        names = list(PROMPTS.keys())
    else:
        names = [x.strip() for x in raw.split(",") if x.strip()]
        for n in names:
            if n not in PROMPTS:
                print(f"알 수 없는 프롬프트 키: {n}", file=sys.stderr)
                sys.exit(2)

    selected = {k: PROMPTS[k] for k in names}

    print("TurboQuant 35B 벤치마크")
    print(f"  model: {args.model}")
    print(f"  MLX 디바이스: {mx.default_device()}")
    print(f"  프롬프트: {', '.join(names)}")
    print(f"  max_tokens: {args.max_tokens}")
    print()

    results: dict[str, dict[str, Any]] = {k: {} for k in names}

    if not args.turbo_only:
        print("=" * 60)
        print("Pass 1 — Baseline (patch 없음, 모델 1회 로드)")
        print("=" * 60)
        print("모델 로딩 중...")
        model, tokenizer = load(args.model)
        a0, p0 = _mem_gb()
        print(f"로드 직후 peak 메모리(참고): {p0:.2f} GB\n")

        for label, prompt in selected.items():
            print(f"\n{'─' * 60}\n>>> {label}\n{'─' * 60}")
            results[label]["baseline"] = _run_baseline(
                model, tokenizer, prompt, args.max_tokens, label
            )

        _release_model(model, tokenizer)

    if not args.baseline_only:
        print("\n" + "=" * 60)
        print("Pass 2 — TurboQuant (모델 재로드 + patch_attention)")
        print("=" * 60)
        try:
            from optiq.core.turbo_kv_cache import TurboQuantKVCache, patch_attention
        except ImportError as e:
            print("optiq(mlx-optiq) 미설치:", e, file=sys.stderr)
            print("설치: pip install -e \".[turboquant]\"", file=sys.stderr)
            sys.exit(1)

        print("모델 로딩 중...")
        model, tokenizer = load(args.model)
        patch_attention()
        print("patch_attention() 완료\n")

        for label, prompt in selected.items():
            print(f"\n{'─' * 60}\n>>> {label}\n{'─' * 60}")
            results[label]["turbo"] = _run_turbo(
                model,
                tokenizer,
                prompt,
                args.max_tokens,
                label,
                TurboQuantKVCache,
            )

        if args.long_context:
            lc_prompt = _long_context_prompt(args.long_context_repeats)
            print(f"\n{'─' * 60}\n>>> long_context ({args.long_context_repeats} repeats)\n{'─' * 60}")
            results["long_context"] = {
                "turbo": _run_turbo(
                    model,
                    tokenizer,
                    lc_prompt,
                    args.long_context_tokens,
                    "long_context",
                    TurboQuantKVCache,
                )
            }

        _release_model(model, tokenizer)

    # 요약 표
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    hdr = f"{'프롬프트':<14} {'Baseline(s)':<12} {'Turbo(s)':<12} {'Δ%':<10} {'Δpeak GB':<12}"
    print(hdr)
    print("-" * len(hdr))

    summary_order = list(names)
    if "long_context" in results:
        summary_order.append("long_context")

    for label in summary_order:
        row = results[label]
        b = row.get("baseline")
        t = row.get("turbo")
        if b and t:
            diff_pct = (t["time"] - b["time"]) / b["time"] * 100 if b["time"] else 0.0
            d_peak = b["peak_gb"] - t["peak_gb"]
            if not math.isfinite(b["peak_gb"]) or not math.isfinite(t["peak_gb"]):
                d_peak_str = "n/a"
            else:
                d_peak_str = f"{d_peak:+.2f}"
            print(
                f"{label:<14} {b['time']:<12.2f} {t['time']:<12.2f} {diff_pct:>+6.1f}%    {d_peak_str:<12}"
            )
        elif b:
            print(f"{label:<14} {b['time']:<12.2f} {'—':<12} {'—':<10} {'—':<12}")
        elif t:
            print(f"{label:<14} {'—':<12} {t['time']:<12.2f} {'—':<10} {'—':<12}")

    print()


if __name__ == "__main__":
    main()
