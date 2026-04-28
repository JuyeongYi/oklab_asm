# ok_color — 다중 구현 비교 프로젝트

Björn Ottosson 의 [Oklab 색공간](https://bottosson.github.io/posts/oklab/) 단일-헤더 라이브러리 [`ok_color.h`](https://bottosson.github.io/misc/ok_color.h) 를 기반으로, **scalar → SIMD 4-wide → SIMD 16-wide → hybrid → CUDA** 까지 단계적으로 재구현하며 각 단계의 **성능·정확도·트레이드오프**를 측정한 비교 프로젝트.

## 디렉토리 구조

```
ok_color/
├── README.md                         # 이 파일 (전체 정리)
├── build.sh                          # 통합 빌드 스크립트
│
├── original/                         # Björn Ottosson 의 원본 single-header
├── scalar_vec3/                      # 헤더/구현 분리 + vec3 + 멤버 메서드
├── scalar_vec4/                      # vec3 → float4 (alignas 16) 실험 (반증 자료)
├── simd_sse/                         # SSE4.1 + FMA, 4-pixel SoA batch
├── simd_avx512_vl/                   # 같은 소스, AVX-VL EVEX 빌드 (비교 측정)
├── simd_avx512/                      # AVX-512, 16-pixel SoA batch
├── hybrid/                           # 함수별 최적 라우팅 (CPU 추천)
├── cuda/                             # CUDA, thread 당 1 픽셀
│
├── tests/                            # 정확성 검증 (orig vs 각 구현)
└── benches/                          # 성능 측정
```

각 디렉토리의 `INDEX.md` 에 해당 구현의 상세 설명·성능·사용 시점이 정리되어 있다.

## 구현체 한눈에 보기

| 구현 | 처리 방식 | 위치 | 빌드 요구 |
|---|---|---|---|
| **original** | inline single-header | `original/` | 어디서든 |
| **scalar_vec3** | 한 픽셀씩, vec3 | `scalar_vec3/` | C++17 |
| **scalar_vec4** | 한 픽셀씩, float4 (16-byte 정렬) | `scalar_vec4/` | C++17 |
| **simd_sse** | 4-pixel SoA batch | `simd_sse/` | SSE4.1 + FMA |
| **simd_avx512_vl** | 4-pixel batch (EVEX 인코딩) | `simd_avx512_vl/` | AVX-512VL |
| **simd_avx512** | 16-pixel SoA batch | `simd_avx512/` | AVX-512F+DQ+BW+VL |
| **hybrid** ⭐ | 함수별 최적 라우팅 | `hybrid/` | 위 SIMD 모두 |
| **cuda** | thread 당 1 픽셀 (SIMT) | `cuda/` | CUDA + NVIDIA GPU |

## 측정 결과

### 1) per-call throughput — `benches/master_bench.cpp`

CPU 입력풀 4096, 단일 호출 ns/pixel (Zen 5 / AMD Ryzen 9 9950X).

| 함수 | orig | vec3 | vec4 | SSE×4 | AVX-VL×4 | AVX512×16 | **Hybrid** |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear_srgb→oklab | 26.0 | 26.0 | 27.2 | 2.07 | 2.07 | 2.38 | **1.99** |
| oklab→linear_srgb | 2.62 | 2.27 | 6.72 | 0.67 | 0.67 | 1.59 | **0.61** |
| okhsl→srgb | 110 | 101 | 106 | 28.6 | 28.9 | 9.26 | **9.17** |
| srgb→okhsl | 137 | 122 | 127 | 29.5 | 29.6 | 9.53 | **9.54** |
| okhsv→srgb | 125 | 115 | 126 | 25.8 | 25.9 | 8.14 | **8.16** |
| srgb→okhsv | 155 | 136 | 147 | 28.1 | 28.3 | 9.22 | **9.27** |

Hybrid 가 모든 함수에서 동률 1위 — scalar 대비 9-15x 가속.

### 2) 2048² (4M) 픽셀 한 방 시나리오 — `benches/master_all.cu` (CPU + GPU)

2K 이미지 한 장 변환을 한 번에 한다고 가정. 총 처리 시간 (ms).
CPU 는 300ms timeout — 그 안에 못 끝내면 처리한 픽셀 수와 함께 조기 종료.

| 함수 | orig | vec3 | vec4 | SSE×4 | AVXVL×4 | AVX×16 | Hybrid | **GPU(K)** | GPU(RT) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_srgb→oklab | 110 | 109 | 112 | 9.76 | 9.70 | 20.6 | **9.54** | 0.22 | 10.86 |
| oklab→linear_srgb | 10.5 | 9.94 | 17.4 | 5.73 | 5.43 | 17.8 | **5.49** | 0.22 | 11.47 |
| okhsl→srgb | TO 2.71M | TO 2.83M | TO 2.67M | 126 | 124 | **46.9** | 47.7 | 0.22 | 12.8 |
| srgb→okhsl | TO 2.18M | TO 2.42M | TO 2.26M | 132 | 133 | **49.9** | 50.3 | 0.21 | 11.6 |
| okhsv→srgb | TO 2.50M | TO 2.59M | TO 2.36M | 110 | 109 | **43.1** | 43.2 | 0.20 | 11.7 |
| srgb→okhsv | TO 2.03M | TO 2.21M | TO 2.00M | 120 | 122 | **49.3** | 48.8 | 0.22 | 10.3 |

표기:
- **GPU(K)** = Kernel only, 데이터가 이미 GPU 에 있다고 가정.
- **GPU(RT)** = Round-Trip, `cudaMemcpy` H2D + 커널 + D2H 전체 시간.
- **TO X.XM** = CPU 가 timeout(300ms) 안에 4M 픽셀을 못 끝내고 처리한 픽셀 수 (M 단위) 보고.

### 핵심 관찰

1. **scalar 는 HSL/HSV 에서 timeout**: 4M × 156ns/px = 654ms 걸려 300ms 한도 초과. ~50% 처리 후 중단.
2. **GPU(RT) 가 HSL/HSV 에서 CPU Hybrid 의 4배 빠름** (11ms vs 47ms). 100K 에선 2.5배였는데 이미지가 클수록 격차 확대 — PCIe 비용은 상수, 변환 비용은 픽셀에 비례.
3. **GPU(RT) 의 11ms 가 PCIe 자체**: 4M × 16 byte × 2 방향 = 128MB / 12 GB/s = 10.7ms. kernel 0.2ms 는 무의미하고 *전송이 거의 전부*.
4. **Lab↔RGB 는 4M 에서도 CPU 우세**: Hybrid 5.5-9.5ms < GPU(RT) 11ms. 변환 자체가 가벼워 PCIe 못 상쇄. 8K (33M) 픽셀 정도부터 break-even.

## 정확성

`tests/` 에서 각 구현 검증:
- **scalar (vec3/vec4)**: 1e-7 이내 (1 ULP 수준)
- **SIMD (SSE/AVX512)**: 1e-5 ~ 1e-3 (packed cbrt/log/exp/sincos 다항식 근사)
- **CUDA**: 1e-6 이내 (HW transcendental 사용, SIMD 보다 정확)
- 모든 1100+ 테스트 케이스 통과

## 단계별 핵심 통찰

### 1) **분리·정리만으로 ~5-15% 가속**
원본 `inline` 함수들을 헤더/구현으로 나누고 행렬을 named constexpr 로 추출하면 컴파일러가 더 잘 최적화한다.

### 2) **vec4 (16-byte 정렬) 가 더 느릴 수 있다** ❗
직관과 반대. `.w = 0` padding lane 의 *낭비된* scalar FMA 와 `alignas(16)` 의 stack alignment 비용이 커, AoS 단일-색에서 vec3 (12-byte) 가 더 빠름.

### 3) **packed cbrt 가 진짜 게임 체인저**
원본 `cbrtf` 가 30-60 cycle. bit-hack + Newton 2회 packed 구현으로 lane 당 ~10 cycle. Lab 변환에서 **11.9x 가속** 의 핵심.

### 4) **AVX-512 가 만능이 아니다**
- HSL/HSV (transcendental 무거움): SSE 의 3배 빠름.
- Lab↔RGB (가벼운 행렬): gather/scatter overhead 가 본체보다 커서 **SSE 보다 느림** (oklab→linear_srgb 가 2.5배 느려짐).
- 결론: 함수별 라우팅이 정답.

### 5) **단일-색 OoO 도 매우 빠름**
`scalar_vec3::to_linear_srgb` 1.89 ns/op = ~5.7 cycle. 모던 OoO 엔진의 ILP 가 인접 반복을 파이프라인하면서 SIMD 와 경쟁 가능. SIMD 의 진짜 가치는 batch (이미지 변환) 에서 나옴.

### 6) **GPU 의 진짜 비용은 PCIe 전송**
RTX 4080 SUPER 에서 100K 픽셀 변환:
- Kernel only: 4-8 μs (모든 함수가 비슷 — **memory bandwidth bound**)
- Round-trip: 380-480 μs (PCIe 4.0 의 1.6MB × 2 전송이 거의 전부)

→ 가벼운 함수 (Lab↔RGB) 는 **CPU Hybrid 가 GPU 보다 빠름**. transcendental 무거운 HSL/HSV 는 GPU round-trip 가 2.5배 빠름. GPU 가 진짜 빛나는 영역은 *데이터가 이미 GPU 에 있는* 파이프라인 중간.

## 빌드

```bash
./build.sh           # CPU 전체 (정확성 + 성능)
./build.sh master    # CPU master_bench 만
./build.sh cuda      # CUDA 빌드 (cuda_bench + master_all)
./build.sh all       # CPU + CUDA 모두
./build.sh clean     # build/ 제거
```

요구 사항:
- g++ 또는 clang++ (C++17)
- 최소: SSE4.1 + FMA3
- 권장: AVX-512F + DQ + BW + VL (Zen 4+, Ice Lake+, Sapphire Rapids+)
- CUDA: nvcc + NVIDIA GPU (compute capability 7.0 이상). 기본 `sm_89` (Ada).

CUDA 아키텍처 변경:
```bash
NVCC_ARCH=sm_80 ./build.sh cuda  # Ampere
NVCC_ARCH=sm_90 ./build.sh cuda  # Hopper
```

## 사용 시 추천

| 시나리오 | 추천 |
|---|---|
| **GUI / color picker** (단일 픽셀) | `scalar_vec3` (1.89 ns/op, 컴파일러 친화) |
| **이미지 일괄 변환, AVX-512 가능 CPU** | `hybrid` (모든 함수 best-of-breed) |
| **호환성 우선** (구형 CPU 포함) | `simd_sse` (3-12x 가속, 어디서든 동작) |
| **GPU 파이프라인 중간** (디코더 → 색 변환 → ML) | `cuda` (kernel only 100x+) |
| **GPU 단발성 변환** (CPU 데이터 → GPU → CPU) | 가벼운 함수는 `hybrid`, 무거운 함수는 `cuda` (PCIe 비용 고려) |
| **학습·이해** | 폴더 순서대로 — 단계별 통찰 |

## 라이선스

원본은 MIT (Björn Ottosson, 2021). 본 프로젝트의 재구현·실험·측정 코드도 동일 정신 유지.

## 참고 자료

- [Oklab — A perceptual color space](https://bottosson.github.io/posts/oklab/)
- [How software gets color wrong](https://bottosson.github.io/posts/colorpicker/)
- [Gamut clipping](https://bottosson.github.io/posts/gamutclipping/)
- 원본 헤더: <https://bottosson.github.io/misc/ok_color.h>
