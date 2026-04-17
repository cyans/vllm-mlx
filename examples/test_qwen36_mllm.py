#!/usr/bin/env python3
"""Example script targeting Qwen3.6 explicitly.

This example mirrors ``examples/test_qwen35_mllm.py`` but imports
:data:`vllm_mlx.config.models.QWEN36_PROFILE` so the example exercises
the 3.6 code path regardless of the project-wide :data:`DEFAULT_MODEL_ID`
default (Phase 2 keeps the default pointing at Qwen 3.5).

Requires the 3.6 MLX quant to be cached locally or downloadable at
runtime. The module itself performs no model load at import time — all
network / runtime work happens inside ``main()``, invoked only under the
``__main__`` guard. That keeps the module safely importable by unit
tests and documentation generators that must not trigger a 20 GB
download.

Usage:
    1. Start server (Phase 2 launcher, text-only):
       ./start-server-qwen36.sh

       or explicitly:
       vllm-mlx serve $(python -c 'from vllm_mlx.config.models import QWEN36_PROFILE; print(QWEN36_PROFILE.model_id)') \\
           --language-model-only --port 8001

    2. Run this example:
       python examples/test_qwen36_mllm.py

@CODE:MIGRATE-QWEN36/examples
"""

from __future__ import annotations

import json
import sys

import httpx

from vllm_mlx.config.models import QWEN36_PROFILE

BASE_URL = "http://localhost:8001"
MODEL_ID = QWEN36_PROFILE.model_id


def check_server() -> None:
    """Check server status and surface the /v1/models payload."""
    print("=" * 60)
    print("Step 1: Server Status Check")
    print("=" * 60)

    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
        models = resp.json()
        print("  Server: OK")
        print(f"  Models: {json.dumps(models, indent=2)}")
    except httpx.ConnectError:
        print("  ERROR: Server not running.")
        print(
            f"  Start with: vllm-mlx serve {MODEL_ID} "
            "--language-model-only --port 8001"
        )
        sys.exit(1)

    # Health probe — Phase 2 runs text-only, so we just report the
    # advertised model_type without enforcing a specific value.
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=10)
        health = resp.json()
        model_type = health.get("model_type", "unknown")
        print(f"  Model type: {model_type}")
    except Exception as exc:  # pragma: no cover — network diag path
        print(f"  Health check failed: {exc}")

    print()


def test_text_only() -> None:
    """Sanity-check text-only generation against the 3.6 endpoint."""
    print("=" * 60)
    print("Step 2: Text-Only Generation")
    print("=" * 60)

    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2? Answer in one word.",
                }
            ],
            "max_tokens": 50,
        },
        timeout=60,
    )

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print("  Q: What is 2+2?")
        print(f"  A: {content}")
        print("  Status: PASS")
    else:
        print(f"  ERROR: {resp.status_code} - {resp.text}")
        print("  Status: FAIL")

    print()


def test_thinking_mode() -> None:
    """Exercise the reasoning parser on a prompt that should elicit
    a visible thinking trace (Qwen 3.6 emits ``<think>...</think>`` via
    the same ``qwen3`` reasoning parser the 3.5 quant uses)."""
    print("=" * 60)
    print("Step 3: Thinking Mode (reasoning parser)")
    print("=" * 60)

    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Briefly explain why the sky appears blue. "
                        "Think step by step."
                    ),
                }
            ],
            "max_tokens": 256,
            # Thinking-mode defaults from QWEN36_PROFILE.thinking.
            "temperature": QWEN36_PROFILE.thinking.temperature,
            "top_p": QWEN36_PROFILE.thinking.top_p,
        },
        timeout=120,
    )

    if resp.status_code == 200:
        result = resp.json()
        message = result["choices"][0]["message"]
        reasoning = message.get("reasoning_content") or message.get(
            "reasoning", ""
        )
        content = message.get("content", "")
        print(f"  Reasoning (first 200 chars): {str(reasoning)[:200]}")
        print(f"  Answer   (first 200 chars): {content[:200]}")
        print("  Status: PASS")
    else:
        print(f"  ERROR: {resp.status_code} - {resp.text}")
        print("  Status: FAIL")

    print()


def test_tool_calling() -> None:
    """Tool calling smoke test (requires --enable-auto-tool-choice)."""
    print("=" * 60)
    print("Step 4: Tool Calling (optional)")
    print("=" * 60)

    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Seoul?",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "City name",
                                    },
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ],
                "max_tokens": 200,
            },
            timeout=60,
        )

        if resp.status_code == 200:
            result = resp.json()
            message = result["choices"][0]["message"]
            if message.get("tool_calls"):
                tc = message["tool_calls"][0]
                print(f"  Tool called: {tc['function']['name']}")
                print(f"  Arguments: {tc['function']['arguments']}")
                print("  Status: PASS")
            else:
                content = message.get("content", "")
                print(f"  No tool call (text response): {content[:200]}")
                print("  Status: SKIP (--enable-auto-tool-choice not set?)")
        else:
            print(f"  ERROR: {resp.status_code}")
            print("  Status: FAIL")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print("  Status: FAIL")

    print()


def main() -> None:
    """Entry point. Intentionally kept free of module-level side effects."""
    print()
    print(f"{MODEL_ID} Qwen 3.6 Test")
    print("=" * 60)
    print()

    check_server()
    test_text_only()
    test_thinking_mode()
    test_tool_calling()

    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
