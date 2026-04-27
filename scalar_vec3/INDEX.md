# scalar_vec3 — 헤더/구현 분리 + vec3 + 멤버 메서드

원본을 `.h` (선언) 와 `.cpp` (구현) 로 분리한 첫 리팩터링.

## 핵심 변경

1. **헤더/구현 분리**: 원본 single-header 의 `inline` 함수들을 `.h` 의 prototype + `.cpp` 의 정의로 분리.
2. **`vec3` 자료형 도입**: 3개 float 짜리 공통 벡터 타입. dot, +, -, * 등 산술 연산 보유.
3. **익명 union**: Lab/RGB/HSV/HSL 이 각자 `struct { float L,a,b; }` 와 `vec3 vector` 를 익명 union 으로 묶어 메모리 공유.
4. **`<x>_to_<y>` → 멤버 메서드**: `linear_srgb_to_oklab(rgb)` → `rgb.to_oklab()`.
5. **행렬 → named constexpr vec3[3]**: `M_RGB_TO_LMS`, `M_LMS_TO_LAB` 등 4개 행렬을 익명 namespace 의 `inline constexpr vec3 M_*[3]` 로 추출. 행렬-벡터 곱이 `row.dot(input)` 으로 표현됨.
6. **`this->` 일관 적용**: 멤버 변수 접근 시 모두 `this->` 사용 (가독성).

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color.h` | 선언 + 간략 설명 (266줄) |
| `ok_color.cpp` | 구현 + 단계별 상세 설명 (~900줄) |

## 성능 특성

- 단일-색 호출 API. 한 픽셀씩 처리.
- OoO 엔진의 ILP 만으로도 매우 빠름 (예: `to_linear_srgb` ≈ 2.2 ns/op).
- 원본 대비 전반적으로 동등하거나 5-15% 빠름 (분리·헬퍼 정리로 컴파일러가 더 잘 최적화).

## 사용 예

```cpp
#include "ok_color.h"
using namespace ok_color;

RGB rgb{0.5f, 0.7f, 0.2f};
Lab lab = rgb.to_oklab();
RGB rgb_back = lab.to_linear_srgb();
HSL hsl = rgb.to_okhsl();
RGB clipped = gamut_clip_preserve_chroma({1.5f, -0.2f, 0.8f});
```

## 정확성

`tests/compare_test.cpp` 에서 1100 케이스 (11 함수 × 100 입력) 모두 원본과 비트-동일 또는 1 ULP 안. 수치적 정합성 우수.

## 사용 시점

- **단일-색 변환** 위주 (color picker, GUI 슬라이더, 한 픽셀 변환).
- 가독성과 유지보수성이 중요한 곳.
- SIMD 셋업 비용이 부담스러운 짧은 함수.
