#pragma once
// -----------------------------------------------------------------------------
// ok_color_best_asm — 함수별 최적 SIMD 변형으로 라우팅하는 thin wrapper.
//
//   라우팅 결정 (Zen 5 측정 기반):
//     linear_srgb→oklab  →  SSE  (gather/scatter overhead > 본체)
//     oklab→linear_srgb  →  SSE  (본체가 매우 가벼워 SSE 의 transpose+store 가 압승)
//     okhsl ↔ srgb       →  AVX512  (transcendental 무거워 16-wide 이득 큼)
//     okhsv ↔ srgb       →  AVX512  (동일)
//
//   이 라이브러리는 SSE4.1 + FMA3 + AVX512F+DQ+BW+VL 를 모두 요구한다.  CPU 가 둘 중
//   하나만 지원한다면 ok_color_simd 또는 ok_color_avx512 를 직접 쓰면 된다.
//
//   API: 16-pixel batch.  입력/출력은 16-byte 정렬된 RGB4 / Lab4 / HSL4 / HSV4 배열.
// -----------------------------------------------------------------------------

#include "ok_color_simd.h"
#include "ok_color_avx512.h"

namespace ok_color_best
{

// AoS 타입은 ok_color_simd 의 정의를 그대로 사용.  ok_color_avx512 와 메모리 layout
// 동일 (alignas(16) float×4) 이므로 reinterpret 안전.
using RGB4 = ok_color_simd::RGB4;
using Lab4 = ok_color_simd::Lab4;
using HSL4 = ok_color_simd::HSL4;
using HSV4 = ok_color_simd::HSV4;

// 16-pixel batch APIs — 함수마다 최적 변형으로 라우팅.
void linear_srgb_to_oklab_batch16(const RGB4* in, Lab4* out);  // → SSE×4
void oklab_to_linear_srgb_batch16(const Lab4* in, RGB4* out);  // → SSE×4
void okhsl_to_srgb_batch16       (const HSL4* in, RGB4* out);  // → AVX512
void srgb_to_okhsl_batch16       (const RGB4* in, HSL4* out);  // → AVX512
void okhsv_to_srgb_batch16       (const HSV4* in, RGB4* out);  // → AVX512
void srgb_to_okhsv_batch16       (const RGB4* in, HSV4* out);  // → AVX512

} // namespace ok_color_best
