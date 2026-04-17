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
# 안정성 우선 설정:
# - 긴 reasoning 응답으로 인한 체감 멈춤을 줄이기 위해 max tokens 축소
# - 디스크 prefix cache 자동 복구를 끄고 캐시 메모리 점유를 낮춤
# - request timeout을 줄여 비정상적으로 오래 걸리는 요청을 빨리 정리
# @CODE:MIGRATE-QWEN36/launcher — spec §6.2: model id and reasoning parser
# are owned by vllm_mlx.config.models, not hardcoded here.
# @CODE:FIX-QWEN36-RUNTIME/parser-resolver — REASONING_PARSER now flows
# through resolve_reasoning_parser(model_id, env) so a 3.6 model id picks
# up the "qwen36" parser automatically while 3.5 keeps "qwen3".
MODEL_ID="${VLLM_MLX_MODEL_ID:-$(.venv/bin/python -c 'from vllm_mlx.config.models import resolve_model_id; print(resolve_model_id())')}"
REASONING_PARSER="$(_RESOLVER_MODEL_ID="$MODEL_ID" .venv/bin/python -c 'import os; from vllm_mlx.config.models import resolve_reasoning_parser; print(resolve_reasoning_parser(os.environ.get("_RESOLVER_MODEL_ID", ""), env=os.environ))')"
exec vllm-mlx serve "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8001 \
  --continuous-batching \
  --reasoning-parser "$REASONING_PARSER" \
  --mcp-config mcp.json \
  --max-tokens 8192 \
  --cache-memory-percent 0.08 \
  --disable-prefix-cache \
  --timeout 180
