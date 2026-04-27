# simd_sse — SSE4.1 + FMA3, 4-pixel SoA batch

128-bit `__m128` 으로 4 픽셀을 동시 변환. 본 라이브러리의 *진짜* 첫 SIMD 구현.

## 핵심 설계

1. **SoA (Struct of Arrays)**: `PackedRGB { __m128 r, g, b; }` 한 변수가 4 픽셀의 같은 성분을 담음. 4 lane 이 *완전 독립* 이라 진정한 4-way ILP.
2. **packed transcendental**: `cbrt`, `log`, `exp`, `pow`, `sin`, `cos`, `atan2` 모두 packed 구현 (다항식 + bit-hack).
3. **분기 → mask + blend**: scalar 의 if-else 가 `_mm_blendv_ps` 로. lane divergence 없이 항상 모든 분기 계산.
4. **AoS↔SoA transpose**: `_MM_TRANSPOSE4_PS` 매크로 활용. 4 RGB4 (16 floats) ↔ packed (R, G, B) 변환.
5. **공용체 + 멤버 메서드**: ok_color.h 와 동일한 API 규칙 (`rgb.to_oklab()` 형태).

## packed cbrt 의 비밀

원본 scalar `cbrtf` 는 ~30-60 cycle. 우리 `cbrt_packed` 는 **bit-hack 초기치 + Newton 2회**:
- 초기치: `i_y = (i_x + 0x7F000000) / 3`. IEEE float 비트의 지수부를 1/3 로 나누는 트릭.
- Newton: `y ← (2y + x/y²) / 3`. 2회 반복으로 1e-6 정확도.
- 4 lane 동시 처리 → 픽셀당 cbrt 비용 ~10 cycle (scalar 대비 6배 가속).

## 정확성

`tests/simd_verify.cpp` 에서 1100 케이스 (11 함수 × 100 입력) 모두 원본과 1e-3 이내 일치.
- Lab 변환: 1e-5 이내 (cbrt 근사)
- HSL/HSV: 1e-3 이내 (sincos/pow/atan2 누적)

## 성능 (Zen 5)

| 함수 | scalar | SSE×4 | 가속 |
|---|---:|---:|---:|
| linear_srgb→oklab | 26.2 ns | **2.07 ns** | 12.6x |
| oklab→linear_srgb | 2.5 ns | **0.67 ns** | 3.7x |
| okhsl→srgb | 110 ns | **28.6 ns** | 3.8x |
| srgb→okhsv | 156 ns | **28.5 ns** | 5.5x |

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_simd.h` | PackedRGB/Lab/HSL/HSV 정의 + API |
| `ok_color_simd.cpp` | SIMD 수학 헬퍼 + 색공간 변환 + gamut clipping |

## 빌드 옵션

```
-msse4.1 -mfma  # 최소
-mavx512f -mavx512dq -mavx512bw -mavx512vl  # 권장 (EVEX 인코딩으로 ~3% 추가 가속)
```

## 사용 시점

- **이미지/스트림 색 변환**: 한 번에 여러 픽셀.
- **호환성 우선**: SSE4.1 + FMA 만 요구. 2013년 이후 거의 모든 x86 에서 동작.
- **단일-색 SIMD 실험**의 출발점.
