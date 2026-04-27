# hybrid — 함수별 최적 SIMD 폭으로 라우팅

`ok_color_simd` (SSE 4-wide) 와 `ok_color_avx512` (AVX512 16-wide) 의 thin wrapper. 각 함수에서 측정 결과 더 빠른 변형을 호출.

## 라우팅 테이블 (Zen 5 측정 기반)

| 함수 | 라우팅 | 이유 |
|---|---|---|
| `linear_srgb_to_oklab` | SSE×4 | 본체 가벼움, gather overhead 부담 |
| `oklab_to_linear_srgb` | SSE×4 | 가장 큰 차이 (AVX×16 의 2.5배) |
| `okhsl_to_srgb` | AVX512×16 | transcendental 무거워 16-wide 이득 큼 |
| `srgb_to_okhsl` | AVX512×16 | 동일 |
| `okhsv_to_srgb` | AVX512×16 | 동일 |
| `srgb_to_okhsv` | AVX512×16 | 동일 |

## 왜 이 라우팅인가

함수 본체의 **compute density** (lane 당 op 수) 가 transpose / gather / scatter 비용과의 비율을 결정:
- 비율 < 1: 좁은 SIMD (SSE×4) 가 유리
- 비율 > 5: 넓은 SIMD (AVX512×16) 가 유리

Lab↔RGB 는 비율 ≈ 0.3, HSL/HSV 는 비율 ≈ 5-8.

## API 형태

16-pixel batch 만 제공 (SSE-routing 의 경우 내부적으로 4-batch × 4 호출).

```cpp
namespace ok_color_best {
    using RGB4 = ok_color_simd::RGB4;  // 양 라이브러리 layout 동일
    using Lab4 = ok_color_simd::Lab4;

    void linear_srgb_to_oklab_batch16(const RGB4* in, Lab4* out);
    void oklab_to_linear_srgb_batch16(const Lab4* in, RGB4* out);
    void okhsl_to_srgb_batch16(const HSL4* in, RGB4* out);
    void srgb_to_okhsl_batch16(const RGB4* in, HSL4* out);
    void okhsv_to_srgb_batch16(const HSV4* in, RGB4* out);
    void srgb_to_okhsv_batch16(const RGB4* in, HSV4* out);
}
```

## 검증

`tests/best_asm_verify.cpp` 에서 6 함수 모두:
- 정확성: scalar 와 1e-3 이내 일치 (모두 통과)
- 성능: 함수별로 SSE / AVX512 의 *최저값* 그대로 달성 → best-of-breed 입증

## 의존성

이 라이브러리는 simd_sse + simd_avx512 두 라이브러리 모두 필요. 단독으로 빌드 불가. CPU 가 둘 다 지원해야 함 (Zen 4+ / Sapphire Rapids+).

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_best_asm.h` | 6개 함수의 thin wrapper API |
| `ok_color_best_asm.cpp` | 라우팅 구현 (각 ~3 줄) |

## 빌드 옵션

```
-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1 -O3 -flto
```

## 사용 시점

- **Zen 5 / Sapphire Rapids+ 환경에서 최고 성능**이 필요할 때.
- 호환성 제약 없는 production 라이브러리.
- CPU feature detection 추가 (cpuid + 함수 포인터 dispatch) 시 다른 CPU 도 자동 fallback 가능.

## 다음 진화 단계

런타임 CPU feature detection:
```cpp
if (cpu_has_avx512) {
    fp_okhsl = avx512::okhsl_to_srgb_batch16;
} else if (cpu_has_sse41) {
    fp_okhsl = simd::okhsl_to_srgb_batch4;
} else {
    fp_okhsl = scalar::okhsl_to_srgb;  // tail-handling wrapper
}
```
