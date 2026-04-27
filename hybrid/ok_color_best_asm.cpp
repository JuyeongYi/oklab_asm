// ok_color_best_asm.cpp — Hybrid 라우팅 wrapper 의 구현.
//
//   각 wrapper 는 인라인 가능하지만 LTO 환경 외에서도 안전하도록 .cpp 로 분리.
//   본체는 ok_color_simd / ok_color_avx512 호출 한 두 줄.

#include "ok_color_best_asm.h"

namespace ok_color_best
{

void linear_srgb_to_oklab_batch16(const RGB4* in, Lab4* out)
{
	// SSE 가 더 빠른 영역 (본체가 가벼워 transpose+store 가 scatter 보다 우수).
	ok_color_simd::linear_srgb_to_oklab_batch4(in,    out);
	ok_color_simd::linear_srgb_to_oklab_batch4(in+4,  out+4);
	ok_color_simd::linear_srgb_to_oklab_batch4(in+8,  out+8);
	ok_color_simd::linear_srgb_to_oklab_batch4(in+12, out+12);
}

void oklab_to_linear_srgb_batch16(const Lab4* in, RGB4* out)
{
	// SSE 가 압도적인 영역 (본체 = 6 FMA + 3 mul 매우 가벼움).
	ok_color_simd::oklab_to_linear_srgb_batch4(in,    out);
	ok_color_simd::oklab_to_linear_srgb_batch4(in+4,  out+4);
	ok_color_simd::oklab_to_linear_srgb_batch4(in+8,  out+8);
	ok_color_simd::oklab_to_linear_srgb_batch4(in+12, out+12);
}

void okhsl_to_srgb_batch16(const HSL4* in, RGB4* out)
{
	// transcendental 무거워 AVX512 16-wide 가 압도적.
	ok_color_avx512::okhsl_to_srgb_batch16(
		reinterpret_cast<const ok_color_avx512::HSL4*>(in),
		reinterpret_cast<ok_color_avx512::RGB4*>(out));
}

void srgb_to_okhsl_batch16(const RGB4* in, HSL4* out)
{
	ok_color_avx512::srgb_to_okhsl_batch16(
		reinterpret_cast<const ok_color_avx512::RGB4*>(in),
		reinterpret_cast<ok_color_avx512::HSL4*>(out));
}

void okhsv_to_srgb_batch16(const HSV4* in, RGB4* out)
{
	ok_color_avx512::okhsv_to_srgb_batch16(
		reinterpret_cast<const ok_color_avx512::HSV4*>(in),
		reinterpret_cast<ok_color_avx512::RGB4*>(out));
}

void srgb_to_okhsv_batch16(const RGB4* in, HSV4* out)
{
	ok_color_avx512::srgb_to_okhsv_batch16(
		reinterpret_cast<const ok_color_avx512::RGB4*>(in),
		reinterpret_cast<ok_color_avx512::HSV4*>(out));
}

} // namespace ok_color_best
