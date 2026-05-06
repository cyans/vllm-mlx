#!/usr/bin/env python3
"""
Test: Qwen3.5-35B-A3B MLLM mode on vllm-mlx

Verifies that Qwen3.5 works in multimodal mode with --mllm flag.
This example INTENTIONALLY targets the legacy Qwen3.5 quant; see
SPEC-MIGRATE-QWEN36 §5.1 (LEGACY_MODEL_ID is retained for this case).

Usage:
    1. Start server:
       vllm-mlx serve $(python -c 'from vllm_mlx.config.models import LEGACY_MODEL_ID; print(LEGACY_MODEL_ID)') --mllm --port 8000

    2. Run this test:
       python examples/test_qwen35_mllm.py
"""

import json
import sys

import httpx

from vllm_mlx.config.models import LEGACY_MODEL_ID

BASE_URL = "http://localhost:8000"


def check_server():
    """Check server status and verify MLLM mode."""
    print("=" * 60)
    print("Step 1: Server Status Check")
    print("=" * 60)

    try:
        resp = httpx.get(f"{BASE_URL}/v1/models", timeout=10)
        models = resp.json()
        print(f"  Server: OK")
        print(f"  Models: {json.dumps(models, indent=2)}")
    except httpx.ConnectError:
        print("  ERROR: Server not running.")
        print(f"  Start with: vllm-mlx serve {LEGACY_MODEL_ID} --mllm --port 8000")
        sys.exit(1)

    # Check if MLLM mode is active
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=10)
        health = resp.json()
        model_type = health.get("model_type", "unknown")
        print(f"  Model type: {model_type}")
        if model_type != "mllm":
            print("  WARNING: Server is NOT in MLLM mode!")
            print("  Restart with --mllm flag")
            sys.exit(1)
        print("  MLLM mode: ACTIVE")
    except Exception as e:
        print(f"  Health check failed: {e}")

    print()


def test_text_only():
    """Test text-only generation (MLLM should handle this too)."""
    print("=" * 60)
    print("Step 2: Text-Only Generation")
    print("=" * 60)

    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
            "max_tokens": 50,
        },
        timeout=60,
    )

    if resp.status_code == 200:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"  Q: What is 2+2?")
        print(f"  A: {content}")
        print(f"  Status: PASS")
    else:
        print(f"  ERROR: {resp.status_code} - {resp.text}")
        print(f"  Status: FAIL")

    print()


def test_image_url():
    """Test image analysis via URL."""
    print("=" * 60)
    print("Step 3: Image Analysis (URL)")
    print("=" * 60)

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    print(f"  Image: Cat photo from Wikipedia")

    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What animal is in this image? Answer briefly."},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": 100,
            },
            timeout=120,
        )

        if resp.status_code == 200:
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            print(f"  Q: What animal is in this image?")
            print(f"  A: {content}")
            print(f"  Status: PASS")
        else:
            print(f"  ERROR: {resp.status_code}")
            print(f"  Response: {resp.text[:500]}")
            print(f"  Status: FAIL")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Status: FAIL")

    print()


def test_image_base64():
    """Test image analysis via base64."""
    print("=" * 60)
    print("Step 4: Image Analysis (Base64)")
    print("=" * 60)

    try:
        from PIL import Image
        import base64
        import io

        # Create a simple test image: blue square
        img = Image.new("RGB", (100, 100), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print(f"  Image: 100x100 blue square (generated)")

        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What color is this image?"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                "max_tokens": 50,
            },
            timeout=120,
        )

        if resp.status_code == 200:
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            print(f"  Q: What color is this image?")
            print(f"  A: {content}")
            print(f"  Status: PASS")
        else:
            print(f"  ERROR: {resp.status_code}")
            print(f"  Response: {resp.text[:500]}")
            print(f"  Status: FAIL")
    except ImportError:
        print("  SKIP: PIL not installed")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Status: FAIL")

    print()


def test_tool_calling():
    """Test tool calling (if --enable-auto-tool-choice was also set)."""
    print("=" * 60)
    print("Step 5: Tool Calling (optional)")
    print("=" * 60)

    try:
        resp = httpx.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "What's the weather in Seoul?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get current weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "City name"},
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
                print(f"  Status: PASS")
            else:
                content = message.get("content", "")
                print(f"  No tool call (text response): {content[:200]}")
                print(f"  Status: SKIP (--enable-auto-tool-choice not set?)")
        else:
            print(f"  ERROR: {resp.status_code}")
            print(f"  Status: FAIL")
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Status: FAIL")

    print()


def main():
    print()
    print(f"{LEGACY_MODEL_ID} MLLM Mode Test")
    print("=" * 60)
    print()

    check_server()
    test_text_only()
    test_image_url()
    test_image_base64()
    test_tool_calling()

    print("=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
