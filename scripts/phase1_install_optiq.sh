#!/usr/bin/env bash
# PLAN.md Phase 1: vllm-mlx .venv에 mlx-optiq[all] 설치 및 버전 기록
# 사용: ./scripts/phase1_install_optiq.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  echo "오류: .venv 가 없습니다. 먼저: python3.11 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "========== Phase 1: 설치 전 버전 (기록용) =========="
python -c "import mlx; from importlib.metadata import version as v; print('mlx:', getattr(mlx, '__version__', None) or v('mlx'))"
python -c "import mlx_lm; print(f'mlx-lm: {mlx_lm.__version__}')" || true
pip list | grep -E "mlx|optiq|vllm-mlx" || true

echo ""
echo "========== pyproject [turboquant] extra (mlx-optiq[all]) 설치 =========="
pip install -e ".[turboquant]"

echo ""
echo "========== 설치 확인 =========="
python scripts/verify_phase1_env.py

echo ""
echo "========== 설치 후 mlx / mlx-lm (변경 여부 확인) =========="
python -c "import mlx; from importlib.metadata import version as v; print('mlx:', getattr(mlx, '__version__', None) or v('mlx'))"
python -c "import mlx_lm; print(f'mlx-lm: {mlx_lm.__version__}')"

echo ""
echo "Phase 1 pip 단계 완료. 서버 확인은 별도 터미널에서:"
echo "  ./start-server.sh   # 또는 기존 방식"
echo "  curl -s http://localhost:8001/v1/models | head"
