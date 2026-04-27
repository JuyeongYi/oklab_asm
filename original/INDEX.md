# original — 원본 single-header

Björn Ottosson 의 [ok_color.h](https://bottosson.github.io/misc/ok_color.h) 원본.
다른 모든 구현의 비교 기준 (truth) 으로 사용된다.

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color.original.h` | 원본 (ISO-8859 인코딩) |
| `ok_color.original.utf8.h` | UTF-8 변환 사본 |
| `orig_ok_color.h` | `namespace ok_color → namespace orig` 로 sed 치환한 사본. 다른 구현과 같은 TU 에서 동시 비교를 위함. |

## API 형태

- 단일-헤더 (header-only)
- `namespace ok_color`
- 6개 구조체 (Lab/RGB/HSV/HSL/LC/ST/Cs)
- 자유 함수: `linear_srgb_to_oklab`, `okhsl_to_srgb`, `gamut_clip_*` 등 약 25개
- 모든 구현이 `inline` (헤더에 정의)

## 알고리즘 요지

1. **Oklab 변환**: linear sRGB → LMS (3×3 행렬) → cbrt → Lab (3×3 행렬). 정확한 역변환 가능.
2. **gamut clipping**: sRGB 색역 밖 색을 5가지 전략으로 안전하게 끌어당김.
3. **OkHSL/OkHSV**: 지각적으로 균일한 hue, saturation, lightness 슬라이더. cusp 기반 보간.

## 라이선스

MIT (2021 © Björn Ottosson). 자유 사용·수정·배포·재라이선스 가능 (저작권 표기 유지 조건).
