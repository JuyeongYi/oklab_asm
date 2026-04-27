# tests — 정확성 검증

각 구현의 출력이 원본 `orig` namespace 와 일치하는지 검증.

## 파일

| 파일 | 검증 대상 | 케이스 수 | 허용오차 |
|---|---|---:|---|
| `smoke_test.cpp` | scalar_vec3 | ~50 | 1e-4 (round-trip 누적) |
| `compare_test.cpp` | scalar_vec3 (전체 11 함수) | 1100 (11 × 100) | 1e-6 |
| `simd_verify.cpp` | simd_sse + AoS batch | 1100 | 1e-3 ~ 1e-5 |
| `best_asm_verify.cpp` | hybrid (best_asm) | 64 (4 batches × 16) | 1e-3 ~ 1e-5 |

## 실행

```bash
./build.sh tests
```

또는 개별:
```bash
./build/compare_test
./build/simd_verify
./build/best_asm_verify
```

## 통과 기준

- **Lab 변환**: 1e-5 이내 (cbrt 근사 한계)
- **HSL/HSV**: 1e-3 이내 (sin/cos/atan2/pow 누적)
- **gamut clip**: 1e-4 이내

## 입력 패턴 (sampler)

- **코너**: (0,0,0), (1,1,1), R/G/B primaries, CMY secondaries — 8개
- **경계**: hue 별 풀채도, l/v 0/1, 다양한 mid 값 — 약 10개
- **무작위**: `mt19937` 고정 시드, [0.05, 0.95] 균등분포 (회색 회피)

회색 (R=G=B) 입력은 hue 가 수학적으로 정의되지 않아 (chroma=0 → 0/0) 검증에서 제외.
