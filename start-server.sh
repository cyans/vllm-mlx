#!/usr/bin/env bash
# 가상환경 활성화 후 vLLM-MLX 서버 시작 (Qwen 3.5 35B + MCP + reasoning)

set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "가상환경이 없습니다. 먼저 실행하세요: python3.11 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

source .venv/bin/activate
# Qwen 3.5 35B는 텍스트 전용이므로 --mllm 제거 (MLLM 경로는 tools 미전달 이슈 있음)
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

exec vllm-mlx serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8001 \
  --continuous-batching \
  --reasoning-parser "$REASONING_PARSER" \
  --mcp-config mcp.json \
  --max-tokens 8192 \
  --cache-memory-percent 0.4 \
  --timeout 180 \
  "${EXTRA_FLAGS[@]}"
