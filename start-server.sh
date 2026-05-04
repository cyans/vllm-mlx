#!/usr/bin/env bash
# Canonical vLLM-MLX server launcher (Qwen3.6 + MCP + reasoning/tool-call parsers).
#
# ============================================================================
# PRE-LAUNCH MEMORY CHECKLIST (read this before starting on a 64 GB Mac)
# ----------------------------------------------------------------------------
# A 35B-A3B 4-bit model with continuous batching at --cache-memory-percent 0.4
# routinely runs at 28-34 GB unified memory. If the host is also running
# Chrome with 80 tabs, Slack, Docker Desktop, etc., free RAM can collapse and
# the M-series watchdog will trip a kernel panic. To avoid that:
#
#   1. Close heavy GUI apps before launching (browsers, IDEs you are not
#      actively using, Docker Desktop, video calls).
#   2. Aim for ≥ 8 GB free RAM BEFORE you start the server. Check with
#      `vm_stat` or `memory_pressure -Q`.
#   3. The server now enforces a soft headroom guardrail via
#      --memory-headroom-gb (default 6 GiB). It logs structured warnings
#      when free RAM drops below the target; it does NOT kill itself.
#      Monitor live state with: `curl localhost:8001/memory/budget`.
#   4. If you see repeated headroom warnings, lower --cache-memory-percent
#      or close more apps; do not raise --memory-headroom-gb past your
#      actual free-RAM ceiling.
# ============================================================================
#
# Defaults: model=mlx-community/Qwen3.6-35B-A3B-4bit (resolved via
# vllm_mlx.config.models.resolve_model_id), port 8001, host 0.0.0.0,
# continuous batching, prefix cache, KV cache 40%, 180s request timeout.
#
# Reasoning parser is auto-resolved from the model id (qwen36 for 3.6,
# qwen3 for 3.5). Auto-tool-choice is ON by default with the tool-call
# parser resolved against the live ToolParserManager registry.
#
# Common overrides (env vars):
#   VLLM_MLX_MODEL_ID=...                  pin a different model id (e.g. Qwen3.5)
#   VLLM_MLX_REASONING_PARSER=...          force a specific reasoning parser
#   VLLM_MLX_TOOL_CALL_PARSER=...          force a specific tool-call parser
#   VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE=1    revert to pre-Phase-1 (no auto tool choice)
#   VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1       inject MCP tools when client sends none
#
# History (2026-04-27): consolidated start-server-qwen36.sh into this file.
# The retired sibling carried a `--language-model-only` flag that was never
# wired into vllm_mlx/server.py and produced an argparse error at runtime
# (see docs/blog/qwen36-on-mac-mini-m4pro.md §6). This unified launcher is
# the only supported entry point going forward.

set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "가상환경이 없습니다. 먼저 실행하세요: python3.11 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

source .venv/bin/activate
# --host 0.0.0.0: 같은 네트워크의 다른 기기에서 접속 가능 (기본값이지만 명시)
# 성능 튜닝 (Qwen 3.6 + MCP auto-inject 관찰 결과 반영):
# - prefix cache 활성화: system prompt / MCP tool schema 재사용으로 TTFT 크게 감소
# - KV cache memory를 40%로 상향: continuous batching에서 MLX unified memory 활용도 증대
# - request timeout은 180초 유지
# @CODE:MIGRATE-QWEN36/launcher — spec §6.2: model id and reasoning parser
# are owned by vllm_mlx.config.models, not hardcoded here.
# @CODE:FIX-QWEN36-RUNTIME/parser-resolver — REASONING_PARSER now flows
# through resolve_reasoning_parser(model_id, env) so a 3.6 model id picks
# up the "qwen36" parser automatically while 3.5 keeps "qwen3".
MODEL_ID="${VLLM_MLX_MODEL_ID:-$(.venv/bin/python -c 'from vllm_mlx.config.models import resolve_model_id; print(resolve_model_id())')}"
REASONING_PARSER="$(_RESOLVER_MODEL_ID="$MODEL_ID" .venv/bin/python -c 'import os; from vllm_mlx.config.models import resolve_reasoning_parser; print(resolve_reasoning_parser(os.environ.get("_RESOLVER_MODEL_ID", ""), env=os.environ))')"

# @CODE:FIX-QWEN36-RUNTIME/launcher — opt-in MCP tool auto-injection.
# Operators can set VLLM_MLX_AUTO_INJECT_MCP_TOOLS=1 to surface
# MCP-registered tools to the model when the client does not supply any.
# Default stays OFF so bit-for-bit compat with Qwen 3.5 is preserved
# (SPEC-FIX-QWEN36-RUNTIME REQ-N2).
EXTRA_FLAGS=()
if [[ "${VLLM_MLX_AUTO_INJECT_MCP_TOOLS:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--auto-inject-mcp-tools)
fi

# @CODE:FIX-QWEN36-AUTO-TOOL-CHOICE/launcher — enable auto-tool-choice by
# default so clients do not need to pass tool_choice="auto" explicitly.
# Operators who want the pre-Phase-1 behavior can set
# VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE=1. --enable-auto-tool-choice requires
# --tool-call-parser, so we resolve the parser via
# vllm_mlx.config.models.resolve_tool_parser against the live
# ToolParserManager registry (prefers "qwen3_coder", falls back to "qwen"),
# and let operators force a specific value via VLLM_MLX_TOOL_CALL_PARSER.
if [[ "${VLLM_MLX_DISABLE_AUTO_TOOL_CHOICE:-0}" != "1" ]]; then
  TOOL_CALL_PARSER="${VLLM_MLX_TOOL_CALL_PARSER:-$(.venv/bin/python -c 'from vllm_mlx.tool_parsers import ToolParserManager; from vllm_mlx.config.models import resolve_tool_parser; print(resolve_tool_parser(list(ToolParserManager.tool_parsers.keys())))')}"
  EXTRA_FLAGS+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

exec vllm-mlx serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8001 \
  --continuous-batching \
  --reasoning-parser "$REASONING_PARSER" \
  --mcp-config mcp.json \
  --max-tokens 8192 \
  --cache-memory-percent 0.4 \
  --timeout 180 \
  --memory-headroom-gb 6 \
  "${EXTRA_FLAGS[@]}"
