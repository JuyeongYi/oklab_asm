# scalar_vec4 — float4 (16-byte 정렬) 실험

`vec3` 를 16-byte 정렬된 `float4` 로 확장하면 컴파일러가 자동으로 SIMD 를 쓸까? 확인용 실험 구현.

## 핵심 변경

`scalar_vec3` 에서 `vec3 (12 bytes)` → `float4 (16 bytes, alignas(16), .w = 0 padding)` 로 교체.
- 모든 행렬 상수도 4-wide (`{a, b, c, 0.f}`)
- dot 도 4-wide (`x*o.x + y*o.y + z*o.z + w*o.w`)
- API 형태는 scalar_vec3 와 동일

## 가설 (실패)

> 16-byte 정렬 + 4-wide layout 이면 컴파일러가 자동으로 packed SIMD 를 쓰고 빨라질 것이다.

## 실측 결과

**오히려 느려짐**:

| 함수 | scalar_vec3 | scalar_vec4 | 변화 |
|---|---:|---:|---:|
| linear_srgb→oklab | 26.0 | 27.2 | -5% |
| oklab→linear_srgb | 2.27 | 6.72 | **-196%** |
| okhsl→srgb | 100.6 | 106.2 | -6% |

## 왜 느려졌나

1. **컴파일러는 `.w == 0` 을 모름**: ctor 에서 0 으로 초기화해도 함수 경계를 못 넘어 (LTO 도 IPA constant prop 까진 안 함), 매 dot 마다 `.w * coef + prev` 같은 *낭비된* scalar FMA 가 추가됨 (lane 당 +1 fma).
2. **stack alignment 비용**: `alignas(16)` 타입이 함수 인자/지역변수로 쓰이면 prologue 에 `and rsp, ~31` 같은 정렬 명령이 추가됨. 짧은 함수일수록 비중 큼.
3. **컴파일러는 여전히 scalar FMA**: AoS 단일-색에서 자동 벡터화 안 일어남. 그래서 추가 비용만 지불.

## 교훈

- `alignas(16)` + padding 만으로는 SIMD 가속 안 됨.
- 진짜 SIMD 이득은 명시적 intrinsic + SoA 데이터 레이아웃 + batch 처리에서만.
- 이 구현은 production 용이 아니라 **반증 자료**.

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_v4.h` | float4 기반 선언 |
| `ok_color_v4.cpp` | 구현 |

## 사용 시점

**거의 없음.** 이 실험은 "더 큰 정렬 = 더 빠름" 직관이 틀리다는 걸 증명한 데이터로서 가치가 있을 뿐, production 에선 scalar_vec3 또는 SIMD batch 를 쓰는 게 맞다.
