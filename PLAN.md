# TurboQuant KV Cache 통합 구현 계획 (Qwen3.6-35B-A3B)

> Apple Silicon M4 Pro 64GB에서 vllm-mlx + mlx-optiq TurboQuant로
> Qwen3.6-35B-A3B의 컨텍스트 윈도우 확장 및 KV 메모리 절감.

## 목차

- [동기 (왜 35B에 TurboQuant?)](#동기-왜-35b에-turboquant)
- [아키텍처 개요](#아키텍처-개요)
- [메모리 예산 분석](#메모리-예산-분석)
- [Phase 1: 환경 준비 (완료)](#phase-1-환경-준비-완료)
- [Phase 2: 0.8B 스모크 (완료)](#phase-2-08b-스모크-완료)
- [Phase 3: 35B-A3B 스모크](#phase-3-35b-a3b-스모크)
- [Phase 4: vllm-mlx 서버 통합](#phase-4-vllm-mlx-서버-통합)
- [롤백 계획](#롤백-계획)
- [참고 자료](#참고-자료)

---

## 동기 (왜 35B에 TurboQuant?)

이 프로젝트의 기본 모델은 `mlx-community/Qwen3.6-35B-A3B-4bit`로, 64GB 통합 메모리 환경에서
가중치(~22 GB)와 4-bit FP16 KV 캐시(32K 컨텍스트 기준 ~3 GB)만으로도 충분한 여유 공간(~30+ GB)이
남는다. 즉, **TurboQuant는 모델을 "끼워 맞추기 위한" 수단이 아니라, 더 긴 컨텍스트(64K, 128K)를
안정적으로 굴리기 위한 수단**이다. 이는 더 큰 모델을 메모리에 욱여넣기 위해 KV 양자화를 동원하는
전형적인 동기와는 다른 출발점이다.

KV 양자화 자체에 대한 연구 산출물(회전 공간 어텐션, Needle 검색 100%, perplexity 손실 < 2%)은
이전 검토 단계에서 축적되었고, 그 자산을 35B에 그대로 이식한다. 또한 최근 추가된 메모리 가드레일
(`MEMORY_HEADROOM_GB` 환경변수, `--memory-headroom-gb` CLI 플래그, `/memory/budget` 엔드포인트,
커밋 `7460119`)이 있어, 향후 워크로드가 헤드룸을 침범하는 드문 케이스도 안전하게 잡힌다.

요약:
1. 35B에서 TurboQuant의 가치는 **롱컨텍스트 지원**이지 "모델 들어맞추기"가 아니다.
2. 이전에 축적한 KV 양자화 연구 산출물을 35B에 재활용한다.
3. 메모리 가드레일이 안전망 역할을 한다 — TurboQuant는 그 위의 최적화 레이어.

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                  Apple Silicon M4 Pro 64GB                       │
│                     Unified Memory                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌───────────────────────────────────┐  │
│  │   vllm-mlx       │    │  mlx-optiq (TurboQuant)           │  │
│  │   v0.2.6         │    │                                   │  │
│  │                   │    │  ┌─────────────────────────────┐  │  │
│  │  OpenAI API       │───>│  │  patch_attention()           │  │  │
│  │  호환 서버        │    │  │  (회전 공간 어텐션 설치)     │  │  │
│  │  :8001            │    │  └─────────────────────────────┘  │  │
│  │                   │    │                                   │  │
│  └──────────────────┘    │  ┌─────────────────────────────┐  │  │
│                           │  │  TurboQuantKVCache           │  │  │
│  ┌──────────────────┐    │  │  - 4-bit 양자화 KV cache     │  │  │
│  │  MLX Engine       │    │  │  - 4~5x 압축률              │  │  │
│  │  mlx 0.31.0       │<──│  │  - Perplexity 손실 < 2%     │  │  │
│  │  mlx-lm 0.30.7   │    │  │  - Needle 검색 100%          │  │  │
│  │                   │    │  └─────────────────────────────┘  │  │
│  └──────────────────┘    └───────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    모델 가중치                             │   │
│  │  Qwen3.6-35B-A3B (MoE, ~3B activated)                    │   │
│  │  - GatedDeltaNet 레이어 (선형 어텐션, KV 캐시 불필요)     │   │
│  │  - Self-Attention 레이어 (TurboQuant KV 적용 대상)        │   │
│  │  4-bit 양자화 (mlx-community 사전 빌드)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────┐  ┌────────────────────────────────┐    │
│  │  모델 가중치 ~22GB   │  │  KV Cache (TurboQuant 4-bit)  │    │
│  │                      │  │  Self-Attn 레이어만            │    │
│  │                      │  │  ~0.8GB (32K 컨텍스트)        │    │
│  └─────────────────────┘  └────────────────────────────────┘    │
│                                                                  │
│  남은 메모리: OS + Framework + 헤드룸 ~32GB                      │
└─────────────────────────────────────────────────────────────────┘
```

### TurboQuant 작동 원리

```
일반 KV Cache (FP16):
  Key/Value 텐서 → FP16 저장 → 메모리 O(n * d * 2bytes)

TurboQuant KV Cache (4-bit):
  Key/Value 텐서 → 회전 변환(patch_attention) → 4-bit 양자화 저장
                                                 → 메모리 O(n * d * 0.5bytes)

  압축률: ~4x (FP16 대비)
  품질: Perplexity 손실 < 2%
  속도: -2% 오버헤드 (거의 없음)
  Needle 검색: 100% (affine 80% 대비 우수)
```

---

## 메모리 예산 분석

### 64GB 통합 메모리 배분 (Qwen3.6-35B-A3B-4bit)

| 구성 요소 | 용량 |
|-----------|-----:|
| 모델 가중치 | ~22.0 GB |
| KV Cache (FP16, 32K ctx) | ~3.0 GB |
| KV Cache (TurboQuant 4-bit, 32K ctx) | ~0.8 GB |
| MLX framework + Python runtime | ~3.0 GB |
| 메모리 가드레일 헤드룸 (기본 6 GB) | 6.0 GB |
| **합계 (32K ctx, baseline)** | **~34.0 GB** |
| **합계 (32K ctx, TurboQuant)** | **~31.8 GB** |
| **여유 메모리 (32K ctx, TurboQuant)** | **~32.2 GB** |

> **참고:** 헤드룸이 매우 크기 때문에, 35B에서 TurboQuant의 가치는 **롱컨텍스트 지원**(64K, 128K)에
> 있다. 더 큰 모델을 64GB에 욱여넣기 위한 KV 양자화 동기와는 **다른 출발점**이다.

### KV Cache 메모리 상세 (Qwen3.6-35B-A3B)

```
Qwen3.x 계열은 GatedDeltaNet(선형 어텐션, KV 불필요)과 Self-Attention 레이어가 혼합되어 있어
TurboQuant의 효과는 Self-Attention 레이어에 한정된다. GatedDeltaNet 레이어는 ArraysCache로
유지되며 양자화 대상이 아니다.

대략적인 추산 (32K 컨텍스트):
  FP16 KV Cache       : ~3.0 GB (Self-Attention 레이어 합계)
  TurboQuant 4-bit KV : ~0.8 GB (~3.7x 절감)

64K 컨텍스트로 확장하면 FP16은 ~6 GB, TurboQuant는 ~1.5 GB로 스케일하며
헤드룸 안에서 안전하게 동작한다.
```

### 권장 구성

| 우선순위 | 구성 | 컨텍스트 | 안정성 |
|---------|------|--------|--------|
| 1순위 | 35B-A3B-4bit + TurboQuant | 32K~64K | 매우 높음 (여유 ~30+ GB) |
| 2순위 | 35B-A3B-4bit + TurboQuant | 128K | 높음 (가드레일 검증 필요) |
| 3순위 | 35B-A3B-4bit (TurboQuant off) | 32K | 매우 높음 (베이스라인) |

---

## Phase 1: 환경 준비 (완료)

**Commit:** `1cd3f34 chore(turboquant): Phase 1 — install scripts for mlx-optiq`

**Result:**
- `mlx-optiq 0.1.0` 설치 완료, `from optiq.core.turbo_kv_cache import TurboQuantKVCache, patch_attention` import OK.
- 기존 `mlx 0.31.0`, `mlx-lm 0.30.7` 버전 변동 없음.
- 80/80 스모크 테스트 통과 (기존 vllm-mlx 회귀 없음).

**Scripts:**
- `scripts/phase1_install_optiq.sh` — `.venv` 활성화 후 `pip install -e ".[turboquant]"` 실행 + 설치 검증.
- `scripts/verify_phase1_env.py` — mlx / mlx-lm / optiq 버전, import 확인.

**재실행 (필요 시):**

```bash
cd /Users/mac4/claude_apps/vllm-mlx
./scripts/phase1_install_optiq.sh
python scripts/verify_phase1_env.py
```

---

## Phase 2: 0.8B 스모크 (완료)

**Commit:** `7491893 chore(turboquant): Phase 2 — 0.8B smoke for TurboQuantKVCache`

**Result (M4 Pro 64GB):**
- 6개 self-attention 레이어가 `TurboQuantKVCache`로 패치됨 (head_dim=256).
- GatedDeltaNet 레이어의 `ArraysCache`는 그대로 유지 (Qwen3.x MoE 구조 호환 확인).
- Baseline: 240 tok/s, peak 0.705 GB.
- TurboQuant: 214 tok/s, peak 0.684 GB → **+6.5% wall-clock 오버헤드** (PLAN 목표 ≤10% 충족).

**Script:** `scripts/test_turbo_kv_small.py`

- Baseline: `mlx_lm.models.cache.make_prompt_cache` 기본 동작.
- TurboQuant: `patch_attention()` 후 `KVCache` 슬롯만 `TurboQuantKVCache`로 교체.
- 메모리: `mx.get_active_memory()` / `mx.get_peak_memory()` (mlx 권장 API).

**재실행:**

```bash
cd /Users/mac4/claude_apps/vllm-mlx
source .venv/bin/activate
python scripts/test_turbo_kv_small.py
# 빠른 확인: python scripts/test_turbo_kv_small.py --max-tokens 48
```

---

## Phase 3: 35B-A3B 스모크

### 목표

기본 모델인 `mlx-community/Qwen3.6-35B-A3B-4bit`에 TurboQuant를 적용하여 성능·메모리·롱컨텍스트
안정성을 측정한다. Phase 2의 0.8B에서 검증된 패치 경로(self-attention 레이어만 교체, GatedDeltaNet
유지)를 그대로 35B에 적용한다.

### 사전 조건

- Phase 1·2 완료 (mlx-optiq 설치, 0.8B 스모크 통과).
- `mlx-community/Qwen3.6-35B-A3B-4bit` 모델 캐시 또는 다운로드 가능한 상태.

### 테스트 스크립트 (저자 필요)

**파일:** `scripts/test_turbo_kv_35b.py` *(아직 존재하지 않음 — Phase 3 착수 시 작성)*

`scripts/test_turbo_kv_small.py`의 35B 카운터파트로 작성한다. 핵심 설계:

- **Baseline 패스**: `patch_attention()` 없이 모델을 1회 로드 → short / medium / long 프롬프트 순서로 생성.
- **TurboQuant 패스**: 모델 언로드 후 재로드 → `patch_attention()` 호출 → `make_prompt_cache` 결과의
  `KVCache` 슬롯만 `TurboQuantKVCache`로 교체. `ArraysCache`(GatedDeltaNet 슬롯)는 손대지 않는다.
- **두 패스를 분리**하는 것이 중요하다. `patch_attention()`은 mlx-lm SDPA 전역 패치이므로 단일 로드에서
  교차 실행하면 baseline이 오염된다.
- 메모리: `mx.get_active_memory()` / `mx.get_peak_memory()`.
- 출력: 패치된 self-attention 레이어 수, head_dim, baseline vs TurboQuant 토큰/초·피크 메모리·생성
  텍스트 샘플.

### 메모리 친화적 스모크 (권장 첫 실행)

35B는 0.8B보다 훨씬 큰 모델이므로 첫 실행은 다음과 같이 보수적으로 시작한다:

```bash
cd /Users/mac4/claude_apps/vllm-mlx
source .venv/bin/activate

# 1) 가장 짧은 스모크 — 한 프롬프트, 64 토큰만 생성
python scripts/test_turbo_kv_35b.py --prompts short --max-tokens 64

# 2) 정상 작동 확인 후 표준 스모크
python scripts/test_turbo_kv_35b.py 2>&1 | tee test_turbo_kv_35b_results.txt

# 3) 롱컨텍스트(반복 문장) 확인 — TurboQuant 패스 끝에서만
python scripts/test_turbo_kv_35b.py --turbo-only --long-context --prompts short --max-tokens 128
```

### 기대 결과

- 35B 모델 로딩 성공 (~22 GB 가중치).
- TurboQuant 패치된 self-attention 레이어 수 ≈ 모델 config의 self-attn 레이어 수 (보통 24개 부근,
  config로 변동).
- 속도 오버헤드 ≤ 5% (0.8B의 +6.5%보다 모델 크기 효과로 더 좋아질 가능성).
- 피크 메모리: TurboQuant on이 baseline보다 KV cache 분만큼 작거나 동일 컨텍스트에서 더 안정.
- 생성 텍스트 품질 유지 (육안 검토).

### 성공 기준

- [ ] `scripts/test_turbo_kv_35b.py` 작성 및 첫 실행 성공.
- [ ] 35B 모델에서 `patch_attention()` + 슬롯 교체 정상 작동.
- [ ] 두 패스 분리 후 baseline 오염 없음.
- [ ] 속도 오버헤드 5% 이내.
- [ ] 피크 메모리 측정값이 PLAN 예산표(±2 GB)와 일치.
- [ ] 32K 입력에서 OOM 없이 생성 완료 (가드레일 `--memory-headroom-gb 6` 기본값 준수).

### 문제 해결

**모델 레이어 구조 확인이 필요할 때:**

```bash
python -c "
from mlx_lm import load
model, _ = load('mlx-community/Qwen3.6-35B-A3B-4bit')
for i, layer in enumerate(model.layers[:5]):
    print(f'Layer {i}: {type(layer).__name__}')
    for name, child in layer.named_children():
        print(f'  {name}: {type(child).__name__}')
"
```

**메모리 부족이 의심될 때:** `--max-tokens`를 줄이고, `MEMORY_HEADROOM_GB`를 8 정도로 올려 가드레일이
먼저 작동하는지 확인.

---

## Phase 4: vllm-mlx 서버 통합

> 원 PLAN의 §5 기술 설계를 그대로 이어받되, 모델 가정을 35B로 변경하고 대형 모델 64GB 캐비어트를
> 제거한다.

### 목표

TurboQuant KV Cache를 vllm-mlx 서버 경로에 통합하여 OpenAI 호환 API(`/v1/chat/completions`,
스트리밍, MCP 자동 주입)로 35B-A3B를 더 긴 컨텍스트에서 안정적으로 서빙한다.

### 4.1 현재 추론 경로 (요약)

- `vllm-mlx serve` → 모델 로드 후 **`Scheduler` + mlx-lm `BatchGenerator`** 가 텍스트 LLM의 연속 배칭을 담당한다.
- 프롬프트별 KV는 **`mlx_lm.models.cache.make_prompt_cache(model)`** 로 생성되고, 배치 시 **`KVCache.merge()`** 등으로 합쳐진다 (`scheduler.py`, `mllm_batch_generator.py` 등).
- Prefix 캐시는 **`MemoryAwarePrefixCache`** 등으로 **재사용·압축(기존 `--kv-cache-quantization`)** 이 걸리며, 이는 **TurboQuant(optiq)와 다른 축**이다.

### 4.2 TurboQuant가 요구하는 것

- 프로세스당 1회: **`optiq.core.turbo_kv_cache.patch_attention()`** (mlx-lm SDPA 전역 패치).
- 요청(또는 레이어)별: **`TurboQuantKVCache`** 로 **기존 `KVCache` 슬롯만** 교체 (Qwen3.x 계열은 GatedDeltaNet 레이어는 `ArraysCache` 유지 — Phase 2·3 스크립트와 동일).
- **효과 범위:** 활성 KV·prefix에 저장되는 KV 성격의 텐서. **가중치(~22 GB)는 변하지 않음.**

### 4.3 통합 시 핵심 리스크

| 리스크 | 설명 |
|--------|------|
| **BatchGenerator 호환** | 연속 배칭은 `KVCache` 기준 `merge`/`extract` 경로에 의존. `TurboQuantKVCache`가 동일 계약을 만족하지 않으면 배치 깨짐 또는 런타임 오류. |
| **MLLM** | 기존 코드에 **표준 `KVCache`만 가정**하는 분기가 있음 (예: merge 실패 시). 멀티모달+TurboQuant는 후순위. |
| **패치 순서** | `patch_attention()`은 **모델 로드·첫 forward 전**에 적용하는 것이 안전. |
| **prefix 캐시와 중복** | 서버 KV 양자화 옵션과 TurboQuant가 **동시에** 켜질 때 품질·메모리 상호작용을 검증해야 함. |

### 4.4 권장 구현 단계

#### 4A — 부트스트랩 (저위험)

- CLI: `--enable-turboquant` (이름 가칭).
- 서버 `load_model` 직후 **`patch_attention()`만** 호출 (optiq 미설치 시 no-op 로그).
- **아직 캐시 클래스는 교체하지 않음** → 회귀 테스트용. 이후 4B에서 캐시 교체.

#### 4B — 캐시 팩토리 (핵심)

- `make_prompt_cache` 호출 직후 한 곳에서만 후처리:
  `for i, layer in enumerate(model.layers):`
  `isinstance(cache[i], KVCache) and hasattr(layer, "self_attn")` → `TurboQuantKVCache(...)`.
- 구현 위치 후보: **`Scheduler`가 `BatchGenerator`를 만들 때** 넘기는 캐시 생성 경로, 또는 **배치 생성 전용 래퍼** (단일 진입점 유지).
- **텍스트 LLM + continuous batching** 조합부터 통과시키고, 실패 시 플래그로 끄기.

#### 4C — 관측·가드

- 로그: TurboQuant 적용 레이어 수, `head_dim`, 옵트인 여부.
- OOM 시: TurboQuant 비활성화 폴백 또는 `--max-tokens` 클램프(서버 옵션)와 연동 검토.
- 신규 메모리 가드레일(`MEMORY_HEADROOM_GB` / `--memory-headroom-gb` / `/memory/budget`,
  커밋 `7460119`)와의 상호작용 확인.

#### 4D (선택) — MLLM / mlx-lm 단독 서버

- vllm-mlx 밖으로는 기존 PLAN의 **`mlx_lm` 서버 래퍼**가 구현 비용은 낮으나, **도구·배칭·MCP**를 쓰는 사용자에는 부적합할 수 있음.

### 4.5 성공 기준 (통합 완료 시)

- 35B-A3B는 64GB에 충분한 헤드룸을 갖는 만큼, **성공의 의미는 TurboQuant on/off A/B에서 피크 메모리
  또는 롱컨텍스트(예: 32K → 64K) 안정성이 개선**되는 것이다. `/v1/chat/completions`·스트리밍·(선택)
  MCP에서 회귀가 없어야 한다.
- 가드레일과의 통합: `/memory/budget`에 TurboQuant 활성 상태와 추정 절감량이 표시되거나, 최소한
  서버 시작 로그에 한 번 기록된다.

---

## 롤백 계획

### TurboQuant 제거 시

```bash
# 1. 서버 중지 (Ctrl+C 또는 kill)

# 2. 기존 start-server.sh로 복원 (35B 기본 경로)
cd /Users/mac4/claude_apps/vllm-mlx
./start-server.sh

# 3. (선택) mlx-optiq 제거
pip uninstall mlx-optiq optiq
```

### 단계별 롤백 포인트

| 상황 | 롤백 방법 |
|------|----------|
| Phase 1 실패 (설치 문제) | `pip uninstall mlx-optiq`, 기존 환경 유지 |
| Phase 2 실패 (소형 모델 오류) | mlx-optiq 업데이트 대기 또는 이슈 리포트 |
| Phase 3 실패 (35B 호환성) | `start-server.sh`로 복원, TurboQuant 비활성화로 운영 |
| Phase 4 실패 (서버 통합) | `--enable-turboquant` 미사용, baseline KV 경로로 폴백 |

### 비상 복구

```bash
# 전체 환경 초기화 (최후 수단)
cd /Users/mac4/claude_apps/vllm-mlx
pip install -e .                  # 원래 의존성 복원
./start-server.sh                 # 기존 서버 시작
```

---

## 참고 자료

- [mlx-optiq PyPI](https://pypi.org/project/mlx-optiq/)
- [mlx-optiq GitHub](https://github.com/argmaxinc/mlx-optiq)
- [TurboQuant 논문/블로그](https://github.com/argmaxinc/mlx-optiq#turboquant)
- Qwen3.6-35B-A3B (mlx-community)
- [vllm-mlx GitHub](https://github.com/vllm-project/vllm-mlx)
- [mlx-lm 문서](https://github.com/ml-explore/mlx-examples/tree/main/llms)

---

> 작성일: 2026-05-05 (재작성)
> 대상 환경: Apple Silicon M4 Pro 64GB, macOS 26.3.1
> 상태: Phase 1·2 완료, Phase 3 다음 단계
