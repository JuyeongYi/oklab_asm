# simd_avx512 — AVX-512F + DQ + BW + VL, 16-pixel SoA batch

512-bit `__m512` 로 16 픽셀을 동시 변환. 처리 폭이 simd_sse 의 4배.

## 핵심 차이 (simd_sse 대비)

| 항목 | simd_sse (4-wide) | simd_avx512 (16-wide) |
|---|---|---|
| 처리 폭 | 4 픽셀/배치 | 16 픽셀/배치 |
| 데이터 타입 | `__m128` | `__m512` |
| Mask 타입 | `__m128` (lane 별 0xFF / 0x00) | `__mmask16` (16-bit 정수) |
| Blend 인자 순서 | `_mm_blendv_ps(a, b, mask)` | `_mm512_mask_blend_ps(k, a, b)` ← 인자 순서 다름! |
| AoS↔SoA | `_MM_TRANSPOSE4_PS` | `_mm512_i32gather_ps` / `i32scatter_ps` |
| FMA throughput | 2 fma/cycle × 4 lane = 8 floats/cycle | 2 fma/cycle × 16 lane = **32 floats/cycle** |

## 결과는 두 영역으로 갈림

**transcendental 무거운 함수 (HSL/HSV)**: 16-wide 가 압도적. SSE 대비 **3배 빠름**.

| 함수 | SSE×4 | AVX×16 | 가속 |
|---|---:|---:|---:|
| okhsl→srgb | 28.6 | **9.26** | 3.09x |
| srgb→okhsv | 28.5 | **9.23** | 3.09x |

**가벼운 함수 (Lab↔RGB)**: 16-wide 가 *오히려 느림*. gather/scatter overhead 가 본체보다 큼.

| 함수 | SSE×4 | AVX×16 | 변화 |
|---|---:|---:|---:|
| linear_srgb→oklab | 2.07 | 2.38 | -15% |
| oklab→linear_srgb | 0.67 | 1.59 | **-137%** ❗ |

## 왜 oklab→linear_srgb 가 2.5배 느린가

함수 본체가 너무 가벼움 (6 FMA + 3 mul = ~9 ops). 16-wide gather/scatter 의 fixed cost (~80-120 cycle for scatter × 4 channels) 가 본체보다 훨씬 큼. SSE 의 `_MM_TRANSPOSE4_PS + 4 store` 는 16-byte aligned cache line 1 줄 정확히 채우는 패턴이라 메모리 시스템이 가장 잘 처리함.

## 정확성

simd_sse 와 거의 동일 (1e-5 ~ 1e-3 이내). packed cbrt/log/exp/pow/sincos/atan2 의 다항식 계수가 lane 폭과 무관하게 동일.

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_avx512.h` | PackedRGB512/Lab512/HSL512/HSV512 정의 + API (16-pixel batch) |
| `ok_color_avx512.cpp` | AVX512 SIMD 수학 헬퍼 + 색공간 변환 |

## 빌드 옵션

```
-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1
```

## 사용 시점

- **HSL/HSV 변환 중심**의 워크로드 (이미지 색조 조정, color picker 일괄 변환).
- 입력이 16 픽셀 이상의 배수로 나뉘는 경우 (그렇지 않으면 tail 처리 필요).
- AVX-512 가능 CPU 한정 (Zen 4+, Ice Lake+, Sapphire Rapids+).
- **Lab↔RGB 만 쓴다면 simd_sse 가 더 빠름** — 이때는 hybrid 라이브러리를 쓰면 자동 라우팅.
