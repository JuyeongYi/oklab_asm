# ok_color — 다중 구현 비교 프로젝트

Björn Ottosson 의 [Oklab 색공간](https://bottosson.github.io/posts/oklab/) 단일-헤더 라이브러리 [`ok_color.h`](https://bottosson.github.io/misc/ok_color.h) 를 기반으로, **scalar → SIMD 4-wide → SIMD 16-wide → hybrid** 까지 단계적으로 재구현하며 각 단계의 **성능·정확도·트레이드오프**를 측정한 비교 프로젝트.

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
├── hybrid/                           # 함수별 최적 라우팅 (production 추천)
│
├── tests/                            # 정확성 검증 (orig vs 각 구현)
└── benches/                          # 성능 측정 (master_bench 가 메인)
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
| **hybrid** ⭐ | 함수별 최적 라우팅 | `hybrid/` | 위 모두 |

## 측정 결과 (Zen 5 / AMD Ryzen 9 9950X)

`benches/master_bench.cpp` 출력. 단위 = **ns/pixel** (작을수록 빠름).

| 함수 | orig | vec3 | vec4 | SSE×4 | AVX-VL×4 | AVX512×16 | **Hybrid** |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear_srgb→oklab | 26.0 | 26.0 | 27.2 | 2.07 | 2.07 | 2.38 | **1.99** |
| oklab→linear_srgb | 2.62 | 2.27 | 6.72 | 0.67 | 0.67 | 1.59 | **0.61** |
| okhsl→srgb | 110 | 101 | 106 | 28.6 | 28.9 | 9.26 | **9.17** |
| srgb→okhsl | 137 | 122 | 127 | 29.5 | 29.6 | 9.53 | **9.54** |
| okhsv→srgb | 125 | 115 | 126 | 25.8 | 25.9 | 8.14 | **8.16** |
| srgb→okhsv | 155 | 136 | 147 | 28.1 | 28.3 | 9.22 | **9.27** |

**Hybrid 가 모든 함수에서 동률 1위 또는 노이즈 안 (±2%).** scalar 대비 9-15x 가속.

## 정확성

`tests/` 에서 각 구현 검증:
- **scalar (vec3/vec4)**: 1e-7 이내 (1 ULP 수준)
- **SIMD (SSE/AVX512)**: 1e-5 ~ 1e-3 (packed cbrt/log/exp/sincos 다항식 근사 누적)
- 모든 1100+ 테스트 케이스 통과

## 단계별 핵심 통찰

### 1. **분리·정리만으로 ~5-15% 가속**
원본 `inline` 함수들을 헤더/구현으로 나누고 행렬을 named constexpr 로 추출하면 컴파일러가 더 잘 최적화한다.

### 2. **vec4 (16-byte 정렬) 가 더 느릴 수 있다** ❗
직관과 반대. `.w = 0` padding lane 의 *낭비된* scalar FMA 와 `alignas(16)` 의 stack alignment 비용이 커, AoS 단일-색에서 vec3 (12-byte) 가 더 빠름.

### 3. **packed cbrt 가 진짜 게임 체인저**
원본 `cbrtf` 가 30-60 cycle. bit-hack + Newton 2회 packed 구현으로 lane 당 ~10 cycle. Lab 변환에서 **11.9x 가속** 의 핵심.

### 4. **AVX-512 가 만능이 아니다**
- HSL/HSV (transcendental 무거움): SSE 의 3배 빠름.
- Lab↔RGB (가벼운 행렬): gather/scatter overhead 가 본체보다 커서 **SSE 보다 느림** (oklab→linear_srgb 가 2.5배 느려짐).
- 결론: 함수별 라우팅이 정답.

### 5. **단일-색 OoO 도 매우 빠름**
`scalar_vec3::to_linear_srgb` 1.89 ns/op = ~5.7 cycle. 모던 OoO 엔진의 ILP 가 인접 반복을 파이프라인하면서 SIMD 와 경쟁 가능. SIMD 의 진짜 가치는 batch (이미지 변환) 에서 나옴.

## 빌드

```bash
./build.sh           # 전체 빌드 + 정확성 + 성능 (기본)
./build.sh master    # master_bench 만
./build.sh tests     # 정확성만
./build.sh benches   # 성능만
./build.sh clean     # build/ 제거
```

요구 사항:
- g++ 또는 clang++ (C++17)
- 최소: SSE4.1 + FMA3 (2013년 이후 거의 모든 x86)
- 최대: AVX-512F + DQ + BW + VL (Zen 4+, Ice Lake+, Sapphire Rapids+)

플래그:
```
-O3 -flto -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1
```

## 사용 시 추천

| 시나리오 | 추천 |
|---|---|
| Zen 4+/SPR+ production | **hybrid** (모든 함수에서 best-of-breed) |
| 단일-픽셀 GUI / color picker | **scalar_vec3** (1.89 ns/op, 컴파일러 친화) |
| 호환성 우선 (구형 CPU 포함) | **simd_sse** (3-12x 가속, 어디서든 동작) |
| AVX-512 가능한 이미지 일괄 처리 | **hybrid** (단일 SIMD 폭 고정보다 빠름) |
| 학습·이해 | 폴더 순서대로 — 단계별 통찰 |

## 라이선스

원본은 MIT (Björn Ottosson, 2021). 본 프로젝트의 재구현·실험·측정 코드도 동일 정신 유지.

## 참고 자료

- [Oklab — A perceptual color space](https://bottosson.github.io/posts/oklab/)
- [How software gets color wrong](https://bottosson.github.io/posts/colorpicker/) — OkHSL/OkHSV 설계
- [Gamut clipping](https://bottosson.github.io/posts/gamutclipping/) — 5가지 clipping 전략
- 원본 헤더: <https://bottosson.github.io/misc/ok_color.h>
