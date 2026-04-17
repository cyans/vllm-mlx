#!/usr/bin/env bash
# Qwen 3.6 35B-A3B-4bit launcher for vLLM-MLX (opt-in; default stays 3.5)
#
# This is the Phase 2 sibling of ./start-server.sh. The default launcher
# still points at the Qwen 3.5 quant via DEFAULT_MODEL_ID; this script
# explicitly targets QWEN36_PROFILE.model_id so an operator can boot a
# Qwen 3.6 server without flipping the project-wide default.
#
# Phase 2 keeps the server text-only (--language-model-only) even though
# QWEN36_PROFILE advertises supports_multimodal=True at the profile level.
# The 3.6 MLX quant ships vision/audio/video special tokens in its
# tokenizer config, but the MLLM text+image code path is out of scope for
# Phase 2; see SPEC-MIGRATE-QWEN36 §5.1 item 3 and the Phase 2 scope
# assumption A9 in plan.md §4.
#
# Env overrides:
#   VLLM_MLX_MODEL_ID  — force a specific model id (e.g. a different 3.6
#                        community quant). If unset, we read
#                        QWEN36_PROFILE.model_id from the single source
#                        of truth in vllm_mlx.config.models.
#
# @CODE:MIGRATE-QWEN36/launcher

set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "가상환경이 없습니다. 먼저 실행하세요: python3.11 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

source .venv/bin/activate
# 성능 튜닝 (start-server.sh와 동일 패턴):
# - prefix cache 활성화: system prompt / MCP tool schema 재사용으로 TTFT 크게 감소
# - KV cache memory를 40%로 상향: continuous batching에서 MLX unified memory 활용도 증대
# - request timeout은 180초 유지
# @CODE:FIX-QWEN36-RUNTIME/parser-resolver — route REASONING_PARSER
# through resolve_reasoning_parser(model_id, env) so the 3.6 quant picks
# up the new "qwen36" parser automatically. Operators can still force a
# specific parser via VLLM_MLX_REASONING_PARSER.
MODEL_ID="${VLLM_MLX_MODEL_ID:-$(.venv/bin/python -c 'from vllm_mlx.config.models import QWEN36_PROFILE; print(QWEN36_PROFILE.model_id)')}"
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

exec vllm-mlx serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8001 \
  --language-model-only \
  --continuous-batching \
  --reasoning-parser "$REASONING_PARSER" \
  --mcp-config mcp.json \
  --max-tokens 8192 \
  --cache-memory-percent 0.4 \
  --timeout 180 \
  "${EXTRA_FLAGS[@]}"
