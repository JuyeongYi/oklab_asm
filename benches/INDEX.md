# benches — 성능 측정

각 구현의 ns/pixel 측정. **`master_bench.cpp`** 가 메인 — 모든 구현을 한 표에 정리.

## 파일

| 파일 | 비교 대상 |
|---|---|
| `master_bench.cpp` ⭐ | **orig vs vec3 vs vec4 vs SSE×4 vs AVX-VL×4 vs AVX×16 vs Hybrid** (전체) |
| `bench.cpp` | orig vs vec3 |
| `bench_three.cpp` | orig vs vec3 vs vec4 |
| `bench_cpp_sse_avx512.cpp` | vec4 vs SSE×4 vs AVX×16 |
| `bench_hybrid.cpp` | SSE vs AVX vs Hybrid |
| `bench_four.cpp` | CPP vs SSE×4 vs AVX-VL×4 vs AVX×16 |
| `bench_simd.cpp` | 단일-색 SIMD 변형들 (DPPS, mulps+haddps, column-major) |
| `bench_batch.cpp` | scalar 4-at-a-time vs SSE batch vs AVX2 batch |
| `bench_batch_packedcbrt.cpp` | packed cbrt 의 효과 (cbrt 병목 분쇄 실험) |
| `bench_handsimd.cpp` | hand SIMD `to_linear_srgb` (cbrt 없는 함수) |

## 실행

```bash
./build.sh master       # master_bench 만
./build.sh benches      # 전체 + master_bench 출력
```

## 측정 방식

- **min ns/pixel**: 라운드별 최저값. 시스템 잡음 (interrupt, 캐시 miss) robust.
- 라운드 수 9, iter 1M~4M.
- 입력 풀 4096 픽셀 (캐시 안에 들어옴).
- `volatile` accumulator 로 dead-code elimination 차단.

## 빌드 플래그

```
-O3 -flto -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1 -DNDEBUG
```

`-flto` 없으면 hybrid wrapper 가 인라인 안 되어 함수 호출 오버헤드가 살아남음.

## 결과 요약 (Zen 5, AMD Ryzen 9 9950X)

자세한 표는 README.md 참고. 주요 관찰:

1. **Hybrid 가 모든 함수에서 동률 1위** (또는 최저값 ±2% 안).
2. 단순 함수 (Lab↔RGB) 는 SSE×4 가 최적 (gather/scatter 비용 회피).
3. 복잡 함수 (HSL/HSV) 는 AVX512×16 가 압도 (3x 가속).
4. vec4 (16-byte 정렬) 가 vec3 (12-byte) 보다 *느림* — 직관 반증.
5. scalar `to_linear_srgb` 의 1.89 ns/op 는 OoO ILP 만으로 SIMD 와 경쟁 가능 수준.
