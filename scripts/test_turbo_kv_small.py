#!/usr/bin/env python3
"""TurboQuant KV Cache 소형 모델 검증 (PLAN.md Phase 2).

Qwen3.5-0.8B-OptiQ-4bit 기준: Baseline(KVCache) vs TurboQuantKVCache + patch_attention.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.models.cache import KVCache, make_prompt_cache

MODEL_DEFAULT = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
TEST_PROMPT_DEFAULT = "Explain quantum computing in simple terms:"


def _text_model_args(model):
    """로드된 nn.Module에서 TextModelArgs 유사 객체 찾기."""
    if hasattr(model, "language_model") and hasattr(model.language_model, "args"):
        return model.language_model.args
    if hasattr(model, "args"):
        return model.args
    return None


def _head_dim_from_args(model) -> int:
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


def _metal_mem_gb() -> tuple[float, float]:
    """(active_gb, peak_gb) — GPU 메모리 (mlx 권장 API)."""
    try:
        active = mx.get_active_memory() / 1e9
        peak = mx.get_peak_memory() / 1e9
        return active, peak
    except Exception:
        return float("nan"), float("nan")


def _release_model(model, tokenizer) -> None:
    del model, tokenizer
    gc.collect()
    try:
        mx.clear_cache()
        mx.reset_peak_memory()
    except Exception:
        pass


def test_baseline(model_id: str, test_prompt: str, max_tokens: int) -> float:
    """TurboQuant 없이 기본 KV 캐시로 생성."""
    print("=" * 60)
    print("[1] Baseline (TurboQuant 미적용, mlx-lm 기본 캐시)")
    print("=" * 60)

    model, tokenizer = load(model_id)
    start = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=test_prompt,
        max_tokens=max_tokens,
        verbose=True,
    )
    elapsed = time.perf_counter() - start

    preview = (response or "")[:200]
    print(f"\n응답 일부: {preview!r}...")
    print(f"소요 시간: {elapsed:.2f}초")
    a, p = _metal_mem_gb()
    print(f"메모리(active): {a:.2f} GB | peak: {p:.2f} GB")

    _release_model(model, tokenizer)
    return elapsed


def test_turbo_quant(model_id: str, test_prompt: str, max_tokens: int) -> float:
    """patch_attention + TurboQuantKVCache (self-attn 레이어만 교체)."""
    print("\n" + "=" * 60)
    print("[2] TurboQuant (4-bit KV, rotated-space attention 패치)")
    print("=" * 60)

    try:
        from optiq.core.turbo_kv_cache import TurboQuantKVCache, patch_attention
    except ImportError as e:
        print("optiq(mlx-optiq) 미설치:", e, file=sys.stderr)
        print("설치: pip install -e \".[turboquant]\"", file=sys.stderr)
        raise

    model, tokenizer = load(model_id)
    patch_attention()
    print("patch_attention() 적용 완료")

    prompt_cache = make_prompt_cache(model)
    head_dim = _head_dim_from_args(model)

    turbo_layers = 0
    for i, layer in enumerate(model.layers):
        # Qwen3.5: GatedDeltaNet 레이어는 ArraysCache — 교체하지 않음
        if isinstance(prompt_cache[i], KVCache) and hasattr(layer, "self_attn"):
            prompt_cache[i] = TurboQuantKVCache(
                head_dim=head_dim,
                bits=4,
                seed=42 + i,
            )
            turbo_layers += 1

    print(f"TurboQuantKVCache 적용 레이어 수: {turbo_layers} (head_dim={head_dim})")

    start = time.perf_counter()
    response = generate(
        model,
        tokenizer,
        prompt=test_prompt,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        verbose=True,
    )
    elapsed = time.perf_counter() - start

    preview = (response or "")[:200]
    print(f"\n응답 일부: {preview!r}...")
    print(f"소요 시간: {elapsed:.2f}초")
    a, p = _metal_mem_gb()
    print(f"메모리(active): {a:.2f} GB | peak: {p:.2f} GB")

    _release_model(model, tokenizer)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: TurboQuant 소형 모델 검증")
    parser.add_argument("--model", default=MODEL_DEFAULT, help="HF mlx 모델 id")
    parser.add_argument(
        "--prompt",
        default=TEST_PROMPT_DEFAULT,
        help="생성 프롬프트",
    )
    parser.add_argument("--max-tokens", type=int, default=200, help="생성 토큰 상한")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="TurboQuant 경로만 실행 (디버그용)",
    )
    args = parser.parse_args()

    print("TurboQuant KV Cache 검증")
    print(f"  model: {args.model}")
    print(f"  MLX 디바이스: {mx.default_device()}")
    a0, _ = _metal_mem_gb()
    if a0 == a0:  # not NaN
        print(f"  Metal active memory: {a0:.2f} GB")
    print()

    if not args.skip_baseline:
        baseline_time = test_baseline(args.model, args.prompt, args.max_tokens)
    else:
        baseline_time = float("nan")

    turbo_time = test_turbo_quant(args.model, args.prompt, args.max_tokens)

    print("\n" + "=" * 60)
    print("[결과 비교]")
    print("=" * 60)
    if baseline_time == baseline_time:  # not NaN
        print(f"Baseline 시간:   {baseline_time:.2f}초")
        print(f"TurboQuant 시간: {turbo_time:.2f}초")
        if baseline_time > 0:
            pct = (turbo_time - baseline_time) / baseline_time * 100
            print(f"차이: {pct:+.1f}% (PLAN 권장: 오버헤드 ~10% 이내)")
    else:
        print(f"TurboQuant 시간: {turbo_time:.2f}초")


if __name__ == "__main__":
    main()
