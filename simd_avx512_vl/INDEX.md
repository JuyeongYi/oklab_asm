# simd_avx512_vl — AVX-512VL 로 빌드된 4-wide SIMD

`simd_sse` 와 **소스 코드는 동일** (namespace 만 `ok_color_simd` → `ok_color_avx512_4`). 다만 빌드 시 `-mavx512vl` 플래그로 컴파일러가 EVEX 인코딩을 자유롭게 쓰도록 허용.

## 핵심 차이

소스 코드는 100% 같지만 컴파일 결과가 다름:

| 항목 | simd_sse (VEX) | simd_avx512_vl (EVEX) |
|---|---|---|
| 인스트럭션 인코딩 | VEX (3-byte prefix) | EVEX (4-byte prefix) |
| 사용 가능 register | 16 XMM | 32 XMM (k 마스크 7개 추가) |
| mask blend | `vblendvps` | `vblendvps` 또는 mask k-reg |
| 명령어 길이 | 짧음 | 약간 김 (1 byte prefix) |

## 측정 결과

EVEX 인코딩으로 ~**3% 추가 가속**:

| 함수 | SSE×4 (VEX) | AVX-VL×4 (EVEX) | 차이 |
|---|---:|---:|---:|
| linear_srgb→oklab | 2.37 ns | **2.29 ns** | -3.4% |
| oklab→linear_srgb | 0.75 ns | **0.73 ns** | -2.7% |
| okhsl→srgb | 35.1 ns | 35.2 ns | +0.3% |

차이가 작은 이유: 우리 코드의 register pressure 가 16 XMM 으로 충분 (32 까지 필요 없음). EVEX 의 *큰* 이점은 mask 레지스터, embedded broadcast, mask predication 등인데 우리 코드는 이를 적극 활용 안 함.

## 의의

이 구현체는 **두 가지 사실을 측정**하기 위해 존재:
1. 같은 소스라도 EVEX 빌드는 ~3% 빠르다 (= "공짜 가속").
2. 진짜 큰 차이는 **명시적으로 16-wide AVX512 (`__m512`) 를 쓸 때만** 나옴 → simd_avx512 참고.

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_avx512_4.h` | simd_sse/ok_color_simd.h 의 namespace 만 변경 |
| `ok_color_avx512_4.cpp` | 동일 (sed 치환) |

## 빌드 옵션

```
-mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1
```

## 사용 시점

- AVX-512 환경에서 build flag 만으로 무료 ~3% 가속을 원할 때.
- 실제로는 simd_sse 를 같은 flag 로 빌드하면 동일 결과 — 별도 namespace 가 필요한 이유는 *측정 비교* 용.
