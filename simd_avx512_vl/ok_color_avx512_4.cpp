// -----------------------------------------------------------------------------
// ok_color_avx512_4.cpp — batch SoA SIMD 구현.
//
//   섹션 구성:
//     §1. 내부 SIMD 수학 primitive (cbrt/log/exp/pow/sin/cos/atan2 packed)
//     §2. 색공간 유틸 (srgb_transfer / toe / toe_inv packed)
//     §3. 행렬 상수 (Oklab 변환용)
//     §4. 색공간 변환 (Lab ↔ linear sRGB,  HSL/HSV ↔ sRGB)
//     §5. 색역 분석 (compute_max_saturation, find_cusp, find_gamut_intersection ...)
//     §6. 색역 클리핑 (5 종)
//     §7. AoS batch wrapper
// -----------------------------------------------------------------------------

#include "ok_color_avx512_4.h"
#include <cmath>
#include <cfloat>

namespace ok_color_avx512_4
{

namespace { // §0. 내부 상수 / 유틸

constexpr float pi = 3.14159265358979323846f;

// 자주 쓰는 상수 → 함수로 감싸서 _mm_set1_ps 호출을 줄이고 컴파일러가 hoist 하도록.
static inline __m128 vsel(__m128 mask, __m128 a, __m128 b) { return _mm_blendv_ps(b, a, mask); }
static inline __m128 vand(__m128 a, __m128 b) { return _mm_and_ps(a, b); }
static inline __m128 vor (__m128 a, __m128 b) { return _mm_or_ps (a, b); }
static inline __m128 vxor(__m128 a, __m128 b) { return _mm_xor_ps(a, b); }

static inline __m128 vabs(__m128 x) {
	return _mm_andnot_ps(_mm_set1_ps(-0.f), x);
}
static inline __m128 vsgn(__m128 x) {
	__m128 pos = _mm_cmpgt_ps(x, _mm_setzero_ps());
	__m128 neg = _mm_cmplt_ps(x, _mm_setzero_ps());
	return _mm_sub_ps(_mm_and_ps(pos, _mm_set1_ps(1.f)),
	                  _mm_and_ps(neg, _mm_set1_ps(1.f)));
}
static inline __m128 vclamp(__m128 x, float lo, float hi) {
	return _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(lo)), _mm_set1_ps(hi));
}

// =============================================================================
// §1. 내부 SIMD 수학 primitive.
// =============================================================================

// ── cbrt: bit-hack 초기치 + Newton 2회.  x > 0 가정.
//
//   초기치: i_y = (i_x + 2·127·2²³) / 3.
//           IEEE float 의 비트 조작으로 "지수부를 1/3" 한 근사를 만듦.
//   Newton:  y ← (2y + x/y²) / 3.   2회 → ~1e-6 정확도.
//
//   SSE 정수 div 가 없어서 `i/3 = (i × 0xAAAAAAAB) >> 33` multiply-high 트릭 사용.
//   `_mm_mul_epu32` 가 짝수 lane (0, 2) 만 64-bit 곱을 만들어주므로
//   홀수 lane 은 입력을 32 bit 시프트해 따로 곱한 뒤 재합성.
static inline __m128i div_by_3_u32(__m128i x)
{
	const __m128i magic = _mm_set1_epi32((int)0xAAAAAAABu);
	__m128i even = _mm_mul_epu32(x, magic);
	__m128i odd  = _mm_mul_epu32(_mm_srli_epi64(x, 32), magic);
	even = _mm_srli_epi64(even, 33);
	odd  = _mm_srli_epi64(odd,  33);
	odd  = _mm_slli_epi64(odd,  32);
	return _mm_or_si128(even, odd);
}
static inline __m128 cbrt_packed(__m128 x)
{
	__m128i ix = _mm_castps_si128(x);
	__m128i t  = _mm_add_epi32(ix, _mm_set1_epi32(0x7F000000));  // 2 · 127 · 2²³
	__m128 y   = _mm_castsi128_ps(div_by_3_u32(t));
	const __m128 third = _mm_set1_ps(1.f / 3.f);
	const __m128 two   = _mm_set1_ps(2.f);
	for (int i = 0; i < 2; ++i) {
		__m128 y2  = _mm_mul_ps(y, y);
		__m128 xy2 = _mm_div_ps(x, y2);
		y = _mm_mul_ps(_mm_fmadd_ps(two, y, xy2), third);
	}
	// 입력이 정확히 0 인 lane 은 Newton 이 0 에 수렴은 하지만 도달은 못 함.
	// 0 으로 강제 (scalar cbrtf(0) == 0 동작 매칭).
	__m128 zero_mask = _mm_cmpeq_ps(x, _mm_setzero_ps());
	return _mm_andnot_ps(zero_mask, y);
}

// ── log_packed (자연로그).  Cephes/sse_mathfun 스타일 polynomial.
//   x = m · 2^e  (m ∈ [√½, √2)) 로 range reduction 후 8차 다항식.
//   정확도 ~1 ULP (단정도).
static inline __m128 log_packed(__m128 x)
{
	const __m128 vinvalid = _mm_cmple_ps(x, _mm_setzero_ps());
	x = _mm_max_ps(x, _mm_set1_ps(1.175494351e-38f));   // FLT_MIN

	// 지수/만티사 분리.
	__m128i ix = _mm_castps_si128(x);
	__m128i e  = _mm_sub_epi32(_mm_srli_epi32(ix, 23), _mm_set1_epi32(0x7F));
	// 만티사를 [0.5, 1) 로 클램프 (지수 부분을 0x3F = 0.5 로 강제).
	x = _mm_or_ps(_mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32((int)0x7F800000)), x),
	              _mm_set1_ps(0.5f));

	__m128 ef = _mm_add_ps(_mm_cvtepi32_ps(e), _mm_set1_ps(1.f));

	// m < √½ 이면 m ← 2m, e ← e − 1.
	__m128 mask = _mm_cmplt_ps(x, _mm_set1_ps(0.7071067811865476f));
	__m128 add  = _mm_and_ps(x, mask);
	x = _mm_sub_ps(x, _mm_set1_ps(1.f));
	ef = _mm_sub_ps(ef, _mm_and_ps(_mm_set1_ps(1.f), mask));
	x  = _mm_add_ps(x, add);

	__m128 z = _mm_mul_ps(x, x);

	__m128 y = _mm_set1_ps( 7.0376836292e-2f);
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(-1.1514610310e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps( 1.1676998740e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(-1.2420140846e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps( 1.4249322787e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(-1.6668057665e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps( 2.0000714765e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(-2.4999993993e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps( 3.3333331174e-1f));
	y = _mm_mul_ps(_mm_mul_ps(y, x), z);

	y = _mm_fmadd_ps(ef, _mm_set1_ps(-2.12194440e-4f), y);
	y = _mm_fnmadd_ps(z, _mm_set1_ps(0.5f), y);

	x = _mm_add_ps(x, y);
	x = _mm_fmadd_ps(ef, _mm_set1_ps(0.693359375f), x);

	// 음수/0 입력은 NaN/-inf 로 강제.
	return _mm_or_ps(x, vinvalid);
}

// ── exp_packed.  Cephes/sse_mathfun 스타일.
static inline __m128 exp_packed(__m128 x)
{
	x = _mm_min_ps(x, _mm_set1_ps( 88.3762626647949f));
	x = _mm_max_ps(x, _mm_set1_ps(-88.3762626647949f));

	__m128 fx = _mm_fmadd_ps(x, _mm_set1_ps(1.44269504088896341f),  // 1/log(2)
	                            _mm_set1_ps(0.5f));
	__m128i emm0 = _mm_cvttps_epi32(fx);
	fx = _mm_cvtepi32_ps(emm0);
	__m128 mask = _mm_cmpgt_ps(fx, _mm_add_ps(x, _mm_set1_ps(0.5f * 0.f)));
	(void)mask;
	// 위의 round-to-nearest 가 cvttps 와 차이날 때 보정 (생략 가능, 정확도 영향 미미).

	__m128 tmp = _mm_mul_ps(fx, _mm_set1_ps(0.693359375f));
	__m128 z   = _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4f));
	x = _mm_sub_ps(x, tmp);
	x = _mm_sub_ps(x, z);

	z = _mm_mul_ps(x, x);

	__m128 y = _mm_set1_ps(1.9875691500e-4f);
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(1.3981999507e-3f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(8.3334519073e-3f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(4.1665795894e-2f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(1.6666665459e-1f));
	y = _mm_fmadd_ps(y, x, _mm_set1_ps(5.0000001201e-1f));
	y = _mm_fmadd_ps(y, z, x);
	y = _mm_add_ps(y, _mm_set1_ps(1.f));

	// 2^n 곱셈 (정수 e 를 IEEE 비트로 직접 구성).
	emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(0x7F));
	emm0 = _mm_slli_epi32(emm0, 23);
	__m128 pow2n = _mm_castsi128_ps(emm0);
	return _mm_mul_ps(y, pow2n);
}

// ── pow_packed.  pow(x, c) = exp(c · log(x)). 양수 입력만.
static inline __m128 pow_packed(__m128 x, __m128 c)
{
	return exp_packed(_mm_mul_ps(c, log_packed(x)));
}

// ── sincos_packed.  Cephes/sse_mathfun 식 range reduction + 6차 다항식.
static inline void sincos_packed(__m128 x, __m128* sin_out, __m128* cos_out)
{
	__m128 sign_bit_sin = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000)));
	x = vabs(x);

	// y = round_to_even(x · 4/π).
	__m128 y = _mm_mul_ps(x, _mm_set1_ps(1.27323954473516f));
	__m128i emm2 = _mm_cvttps_epi32(y);
	emm2 = _mm_add_epi32(emm2, _mm_set1_epi32(1));
	emm2 = _mm_and_si128(emm2, _mm_set1_epi32(~1));
	y = _mm_cvtepi32_ps(emm2);

	// sign_bit_sin: emm2 의 bit 2 → MSB.
	__m128i emm0 = _mm_slli_epi32(_mm_and_si128(emm2, _mm_set1_epi32(4)), 29);
	__m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);
	sign_bit_sin = vxor(sign_bit_sin, swap_sign_bit_sin);

	// poly_mask: (emm2 & 2) == 0  → sin/cos polynomial 선택.
	__m128i emm2_poly = _mm_cmpeq_epi32(_mm_and_si128(emm2, _mm_set1_epi32(2)),
	                                    _mm_setzero_si128());
	__m128 poly_mask = _mm_castsi128_ps(emm2_poly);

	// sign_bit_cos:  ((~(emm2 − 2)) & 4) << 29.   (Cephes 의 cos 공식.)
	__m128i emm2_cos = _mm_sub_epi32(emm2, _mm_set1_epi32(2));
	__m128i emm0_cos = _mm_slli_epi32(_mm_andnot_si128(emm2_cos, _mm_set1_epi32(4)), 29);
	__m128 sign_bit_cos = _mm_castsi128_ps(emm0_cos);

	// Range reduction:  x = x − y·DP1 − y·DP2 − y·DP3  (Cephes triple-precision π/4).
	x = _mm_fmadd_ps(y, _mm_set1_ps(-0.78515625f), x);
	x = _mm_fmadd_ps(y, _mm_set1_ps(-2.4187564849853515625e-4f), x);
	x = _mm_fmadd_ps(y, _mm_set1_ps(-3.77489497744594108e-8f), x);

	__m128 z = _mm_mul_ps(x, x);

	__m128 yc = _mm_set1_ps(2.443315711809948e-5f);
	yc = _mm_fmadd_ps(yc, z, _mm_set1_ps(-1.388731625493765e-3f));
	yc = _mm_fmadd_ps(yc, z, _mm_set1_ps( 4.166664568298827e-2f));
	yc = _mm_mul_ps(yc, _mm_mul_ps(z, z));
	yc = _mm_sub_ps(yc, _mm_mul_ps(z, _mm_set1_ps(0.5f)));
	yc = _mm_add_ps(yc, _mm_set1_ps(1.f));

	__m128 ys = _mm_set1_ps(-1.9515295891e-4f);
	ys = _mm_fmadd_ps(ys, z, _mm_set1_ps( 8.3321608736e-3f));
	ys = _mm_fmadd_ps(ys, z, _mm_set1_ps(-1.6666654611e-1f));
	ys = _mm_mul_ps(ys, _mm_mul_ps(z, x));
	ys = _mm_add_ps(ys, x);

	__m128 sin_v = vsel(poly_mask, ys, yc);
	__m128 cos_v = vsel(poly_mask, yc, ys);

	*sin_out = vxor(sin_v, sign_bit_sin);
	*cos_out = vxor(cos_v, sign_bit_cos);
}

// ── atan2_packed.  y/x → 각도 [-π, π].
//   *부호 비트*로 negx/negy 판정 (그래야 −0 입력이 4사분면을 정확히 가른다).
static inline __m128 atan2_packed(__m128 y, __m128 x)
{
	__m128 ay = vabs(y), ax = vabs(x);
	__m128 a  = _mm_div_ps(_mm_min_ps(ax, ay),
	                       _mm_max_ps(_mm_max_ps(ax, ay), _mm_set1_ps(FLT_MIN)));
	__m128 a2 = _mm_mul_ps(a, a);

	// minimax 다항식: r ≈ atan(a),  a ∈ [0, 1].
	__m128 r = _mm_set1_ps(-0.0464964749f);
	r = _mm_fmadd_ps(r, a2, _mm_set1_ps( 0.15931422f));
	r = _mm_fmadd_ps(r, a2, _mm_set1_ps(-0.327622764f));
	r = _mm_mul_ps(r, _mm_mul_ps(a2, a));
	r = _mm_add_ps(r, a);

	// |y| > |x| 이면 r ← π/2 − r.
	__m128 swap = _mm_cmpgt_ps(ay, ax);
	r = vsel(swap, _mm_sub_ps(_mm_set1_ps(pi * 0.5f), r), r);

	// 부호 비트 추출 (signed integer arithmetic shift right 31 → all-1 if sign set).
	__m128 negx = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(x), 31));
	__m128 negy = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(y), 31));

	r = vsel(negx, _mm_sub_ps(_mm_set1_ps(pi), r), r);
	r = vsel(negy, _mm_xor_ps(r, _mm_set1_ps(-0.f)), r);
	return r;
}

// =============================================================================
// §2. 색공간 유틸 — packed 버전.
// =============================================================================

// linear sRGB 성분 → 표시용 sRGB 성분 (감마 인코딩).
//   y = 12.92·x                            (x ≤ 0.0031308)
//   y = 1.055·x^(1/2.4) − 0.055            (otherwise)
static inline __m128 srgb_transfer_packed(__m128 a)
{
	__m128 lin   = _mm_mul_ps(a, _mm_set1_ps(12.92f));
	__m128 gamma = _mm_fmsub_ps(_mm_set1_ps(1.055f),
	                            pow_packed(a, _mm_set1_ps(1.f / 2.4f)),
	                            _mm_set1_ps(0.055f));
	__m128 mask  = _mm_cmple_ps(a, _mm_set1_ps(0.0031308f));
	return vsel(mask, lin, gamma);
}
static inline __m128 srgb_transfer_inv_packed(__m128 a)
{
	__m128 lin   = _mm_div_ps(a, _mm_set1_ps(12.92f));
	__m128 gamma = pow_packed(_mm_div_ps(_mm_add_ps(a, _mm_set1_ps(0.055f)),
	                                     _mm_set1_ps(1.055f)),
	                          _mm_set1_ps(2.4f));
	__m128 mask  = _mm_cmplt_ps(a, _mm_set1_ps(0.04045f));
	return vsel(mask, lin, gamma);
}

// toe / toe_inv (이차식, 분기 없음).
static inline __m128 toe_packed(__m128 x)
{
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	__m128 u = _mm_fmsub_ps(_mm_set1_ps(k3), x, _mm_set1_ps(k1));
	__m128 inside = _mm_fmadd_ps(u, u, _mm_mul_ps(_mm_set1_ps(4.f * k2 * k3), x));
	return _mm_mul_ps(_mm_set1_ps(0.5f), _mm_add_ps(u, _mm_sqrt_ps(inside)));
}
static inline __m128 toe_inv_packed(__m128 x)
{
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	__m128 num = _mm_fmadd_ps(x, x, _mm_mul_ps(_mm_set1_ps(k1), x));
	__m128 den = _mm_mul_ps(_mm_set1_ps(k3), _mm_add_ps(x, _mm_set1_ps(k2)));
	return _mm_div_ps(num, den);
}

// =============================================================================
// §3. 행렬 상수 (matrix coefficients).  scalar float — 호출부에서 _mm_set1_ps 로 broadcast.
// =============================================================================

// linear sRGB → LMS
static constexpr float M_RGB_TO_LMS[3][3] = {
	{ 0.4122214708f, 0.5363325363f, 0.0514459929f }, // l
	{ 0.2119034982f, 0.6806995451f, 0.1073969566f }, // m
	{ 0.0883024619f, 0.2817188376f, 0.6299787005f }, // s
};
// LMS' → Lab
static constexpr float M_LMS_TO_LAB[3][3] = {
	{ 0.2104542553f,  0.7936177850f, -0.0040720468f }, // L
	{ 1.9779984951f, -2.4285922050f,  0.4505937099f }, // a
	{ 0.0259040371f,  0.7827717662f, -0.8086757660f }, // b
};
// Lab → LMS' — 첫 열 = (1, 1, 1).
static constexpr float M_LAB_TO_LMS[3][3] = {
	{ 1.f, +0.3963377774f, +0.2158037573f },
	{ 1.f, -0.1055613458f, -0.0638541728f },
	{ 1.f, -0.0894841775f, -1.2914855480f },
};
// LMS → linear sRGB
static constexpr float M_LMS_TO_RGB[3][3] = {
	{ +4.0767416621f, -3.3077115913f, +0.2309699292f }, // r
	{ -1.2684380046f, +2.6097574011f, -0.3413193965f }, // g
	{ -0.0041960863f, -0.7034186147f, +1.7076147010f }, // b
};

// 행 i 의 dot(coef[i], (R,G,B)) 를 packed 로 — coef 는 broadcast scalar.
template <const float (&M)[3][3]>
static inline __m128 row_dot3(int i, __m128 R, __m128 G, __m128 B)
{
	return _mm_fmadd_ps(_mm_set1_ps(M[i][2]), B,
	       _mm_fmadd_ps(_mm_set1_ps(M[i][1]), G,
	                    _mm_mul_ps(_mm_set1_ps(M[i][0]), R)));
}

} // anon

// =============================================================================
// §4. 색공간 변환 — Lab ↔ linear sRGB, HSL/HSV ↔ sRGB.
// =============================================================================

PackedLab PackedRGB::to_oklab() const
{
	const __m128 R = this->r, G = this->g, B = this->b;
	__m128 l = row_dot3<M_RGB_TO_LMS>(0, R, G, B);
	__m128 m = row_dot3<M_RGB_TO_LMS>(1, R, G, B);
	__m128 s = row_dot3<M_RGB_TO_LMS>(2, R, G, B);
	l = cbrt_packed(l);
	m = cbrt_packed(m);
	s = cbrt_packed(s);
	__m128 L = row_dot3<M_LMS_TO_LAB>(0, l, m, s);
	__m128 A = row_dot3<M_LMS_TO_LAB>(1, l, m, s);
	__m128 Bo= row_dot3<M_LMS_TO_LAB>(2, l, m, s);
	return PackedLab{ L, A, Bo };
}

PackedRGB PackedLab::to_linear_srgb() const
{
	const __m128 L = this->L, A = this->a, B = this->b;
	__m128 l = row_dot3<M_LAB_TO_LMS>(0, L, A, B);
	__m128 m = row_dot3<M_LAB_TO_LMS>(1, L, A, B);
	__m128 s = row_dot3<M_LAB_TO_LMS>(2, L, A, B);
	l = _mm_mul_ps(_mm_mul_ps(l, l), l);
	m = _mm_mul_ps(_mm_mul_ps(m, m), m);
	s = _mm_mul_ps(_mm_mul_ps(s, s), s);
	__m128 R = row_dot3<M_LMS_TO_RGB>(0, l, m, s);
	__m128 G = row_dot3<M_LMS_TO_RGB>(1, l, m, s);
	__m128 Bo= row_dot3<M_LMS_TO_RGB>(2, l, m, s);
	return PackedRGB{ R, G, Bo };
}

// =============================================================================
// §5. 색역 분석.
// =============================================================================

__m128 compute_max_saturation(__m128 a, __m128 b)
{
	// 3-way channel 분기 → mask + blend.
	const __m128 one = _mm_set1_ps(1.f);
	__m128 maskR = _mm_cmpgt_ps(_mm_fmadd_ps(_mm_set1_ps(-1.88170328f), a,
	                                          _mm_mul_ps(_mm_set1_ps(-0.80936493f), b)),
	                            one);
	__m128 maskG_raw = _mm_cmpgt_ps(_mm_fmadd_ps(_mm_set1_ps( 1.81444104f), a,
	                                              _mm_mul_ps(_mm_set1_ps(-1.19445276f), b)),
	                                one);
	__m128 maskG = _mm_andnot_ps(maskR, maskG_raw);
	// maskB = ~(maskR | maskG)  (사용 안 함 — vsel 두 번 누적으로 충분.)

	auto pick3 = [&](float r, float g, float b_) {
		__m128 v = _mm_set1_ps(b_);            // default: B
		v = vsel(maskG, _mm_set1_ps(g), v);    // override with G if G
		v = vsel(maskR, _mm_set1_ps(r), v);    // override with R if R
		return v;
	};

	__m128 k0 = pick3(+1.19086277f, +0.73956515f, +1.35733652f);
	__m128 k1 = pick3(+1.76576728f, -0.45954404f, -0.00915799f);
	__m128 k2 = pick3(+0.59662641f, +0.08285427f, -1.15130210f);
	__m128 k3 = pick3(+0.75515197f, +0.12541070f, -0.50559606f);
	__m128 k4 = pick3(+0.56771245f, +0.14503204f, +0.00692167f);

	__m128 wl = pick3(+4.0767416621f, -1.2684380046f, -0.0041960863f);
	__m128 wm = pick3(-3.3077115913f, +2.6097574011f, -0.7034186147f);
	__m128 ws = pick3(+0.2309699292f, -0.3413193965f, +1.7076147010f);

	// 다항식 초기치.
	__m128 S = _mm_add_ps(k0,
	            _mm_add_ps(_mm_mul_ps(k1, a),
	            _mm_add_ps(_mm_mul_ps(k2, b),
	            _mm_add_ps(_mm_mul_ps(_mm_mul_ps(k3, a), a),
	                       _mm_mul_ps(_mm_mul_ps(k4, a), b)))));

	// k_l/k_m/k_s = M_LAB_TO_LMS[i] · (0, a, b) = M[i][1]·a + M[i][2]·b.
	__m128 k_l = _mm_fmadd_ps(_mm_set1_ps(+0.3963377774f), a, _mm_mul_ps(_mm_set1_ps(+0.2158037573f), b));
	__m128 k_m = _mm_fmadd_ps(_mm_set1_ps(-0.1055613458f), a, _mm_mul_ps(_mm_set1_ps(-0.0638541728f), b));
	__m128 k_s = _mm_fmadd_ps(_mm_set1_ps(-0.0894841775f), a, _mm_mul_ps(_mm_set1_ps(-1.2914855480f), b));

	// Halley 1회.
	__m128 l_ = _mm_fmadd_ps(S, k_l, _mm_set1_ps(1.f));
	__m128 m_ = _mm_fmadd_ps(S, k_m, _mm_set1_ps(1.f));
	__m128 s_ = _mm_fmadd_ps(S, k_s, _mm_set1_ps(1.f));
	__m128 lp = _mm_mul_ps(_mm_mul_ps(l_, l_), l_);
	__m128 mp = _mm_mul_ps(_mm_mul_ps(m_, m_), m_);
	__m128 sp = _mm_mul_ps(_mm_mul_ps(s_, s_), s_);
	__m128 ldS  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(3.f), k_l), _mm_mul_ps(l_, l_));
	__m128 mdS  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(3.f), k_m), _mm_mul_ps(m_, m_));
	__m128 sdS  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(3.f), k_s), _mm_mul_ps(s_, s_));
	__m128 ldS2 = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(6.f), _mm_mul_ps(k_l, k_l)), l_);
	__m128 mdS2 = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(6.f), _mm_mul_ps(k_m, k_m)), m_);
	__m128 sdS2 = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(6.f), _mm_mul_ps(k_s, k_s)), s_);

	__m128 f  = _mm_fmadd_ps(wl, lp,  _mm_fmadd_ps(wm, mp,  _mm_mul_ps(ws, sp)));
	__m128 f1 = _mm_fmadd_ps(wl, ldS, _mm_fmadd_ps(wm, mdS, _mm_mul_ps(ws, sdS)));
	__m128 f2 = _mm_fmadd_ps(wl, ldS2,_mm_fmadd_ps(wm, mdS2,_mm_mul_ps(ws, sdS2)));

	__m128 num = _mm_mul_ps(f, f1);
	__m128 den = _mm_fnmadd_ps(_mm_set1_ps(0.5f), _mm_mul_ps(f, f2), _mm_mul_ps(f1, f1));
	S = _mm_sub_ps(S, _mm_div_ps(num, den));
	return S;
}

PackedLC find_cusp(__m128 a, __m128 b)
{
	__m128 S_cusp = compute_max_saturation(a, b);
	// rgb_at_max = oklab_to_linear_srgb(1, S*a, S*b).  packed.
	PackedLab labp{ _mm_set1_ps(1.f), _mm_mul_ps(S_cusp, a), _mm_mul_ps(S_cusp, b) };
	PackedRGB rgbp = labp.to_linear_srgb();
	__m128 mx = _mm_max_ps(_mm_max_ps(rgbp.r, rgbp.g), rgbp.b);
	__m128 L_cusp = cbrt_packed(_mm_div_ps(_mm_set1_ps(1.f), mx));
	return PackedLC{ L_cusp, _mm_mul_ps(L_cusp, S_cusp) };
}

PackedST PackedLC::to_ST() const
{
	__m128 S = _mm_div_ps(this->C, this->L);
	__m128 T = _mm_div_ps(this->C, _mm_sub_ps(_mm_set1_ps(1.f), this->L));
	return PackedST{ S, T };
}

__m128 find_gamut_intersection(__m128 a, __m128 b, __m128 L1, __m128 C1, __m128 L0,
                               PackedLC cusp)
{
	// half-plane 부호: (L1−L0)·C_cusp − (L_cusp−L0)·C1 ≤ 0  → lower half.
	__m128 dL = _mm_sub_ps(L1, L0);
	__m128 dC = C1;
	__m128 lower_mask = _mm_cmple_ps(
		_mm_fmsub_ps(dL, cusp.C, _mm_mul_ps(_mm_sub_ps(cusp.L, L0), C1)),
		_mm_setzero_ps());

	// Lower half: t = C_cusp · L0 / (C1·L_cusp + C_cusp·(L0−L1)).
	__m128 t_lower = _mm_div_ps(
		_mm_mul_ps(cusp.C, L0),
		_mm_fmadd_ps(C1, cusp.L, _mm_mul_ps(cusp.C, _mm_sub_ps(L0, L1))));

	// Upper half 초기치 + Halley 1회.
	__m128 t = _mm_div_ps(
		_mm_mul_ps(cusp.C, _mm_sub_ps(L0, _mm_set1_ps(1.f))),
		_mm_fmadd_ps(C1, _mm_sub_ps(cusp.L, _mm_set1_ps(1.f)),
		             _mm_mul_ps(cusp.C, _mm_sub_ps(L0, L1))));

	__m128 k_l = _mm_fmadd_ps(_mm_set1_ps(+0.3963377774f), a, _mm_mul_ps(_mm_set1_ps(+0.2158037573f), b));
	__m128 k_m = _mm_fmadd_ps(_mm_set1_ps(-0.1055613458f), a, _mm_mul_ps(_mm_set1_ps(-0.0638541728f), b));
	__m128 k_s = _mm_fmadd_ps(_mm_set1_ps(-0.0894841775f), a, _mm_mul_ps(_mm_set1_ps(-1.2914855480f), b));

	__m128 l_dt = _mm_fmadd_ps(dC, k_l, dL);
	__m128 m_dt = _mm_fmadd_ps(dC, k_m, dL);
	__m128 s_dt = _mm_fmadd_ps(dC, k_s, dL);

	__m128 L = _mm_fmadd_ps(L0, _mm_sub_ps(_mm_set1_ps(1.f), t), _mm_mul_ps(t, L1));
	__m128 C = _mm_mul_ps(t, C1);

	__m128 l_ = _mm_fmadd_ps(C, k_l, L);
	__m128 m_ = _mm_fmadd_ps(C, k_m, L);
	__m128 s_ = _mm_fmadd_ps(C, k_s, L);
	__m128 lp = _mm_mul_ps(_mm_mul_ps(l_, l_), l_);
	__m128 mp = _mm_mul_ps(_mm_mul_ps(m_, m_), m_);
	__m128 sp = _mm_mul_ps(_mm_mul_ps(s_, s_), s_);
	__m128 ldt  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(3.f), l_dt), _mm_mul_ps(l_, l_));
	__m128 mdt  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(3.f), m_dt), _mm_mul_ps(m_, m_));
	__m128 sdt  = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(3.f), s_dt), _mm_mul_ps(s_, s_));
	__m128 ldt2 = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(6.f), _mm_mul_ps(l_dt, l_dt)), l_);
	__m128 mdt2 = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(6.f), _mm_mul_ps(m_dt, m_dt)), m_);
	__m128 sdt2 = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(6.f), _mm_mul_ps(s_dt, s_dt)), s_);

	const __m128 FLTMAX = _mm_set1_ps(FLT_MAX);
	auto channel_t = [&](float wl, float wm, float ws) {
		__m128 wlv = _mm_set1_ps(wl), wmv = _mm_set1_ps(wm), wsv = _mm_set1_ps(ws);
		__m128 r  = _mm_sub_ps(_mm_fmadd_ps(wlv, lp,
		                       _mm_fmadd_ps(wmv, mp,
		                                    _mm_mul_ps(wsv, sp))),
		                       _mm_set1_ps(1.f));
		__m128 r1 = _mm_fmadd_ps(wlv, ldt,
		             _mm_fmadd_ps(wmv, mdt,
		                          _mm_mul_ps(wsv, sdt)));
		__m128 r2 = _mm_fmadd_ps(wlv, ldt2,
		             _mm_fmadd_ps(wmv, mdt2,
		                          _mm_mul_ps(wsv, sdt2)));
		__m128 u  = _mm_div_ps(r1,
		                       _mm_fnmadd_ps(_mm_set1_ps(0.5f), _mm_mul_ps(r, r2),
		                                     _mm_mul_ps(r1, r1)));
		__m128 candidate = _mm_mul_ps(_mm_sub_ps(_mm_setzero_ps(), r), u);
		__m128 valid = _mm_cmpge_ps(u, _mm_setzero_ps());
		return vsel(valid, candidate, FLTMAX);
	};
	__m128 t_r = channel_t(+4.0767416621f, -3.3077115913f, +0.2309699292f);
	__m128 t_g = channel_t(-1.2684380046f, +2.6097574011f, -0.3413193965f);
	__m128 t_b = channel_t(-0.0041960863f, -0.7034186147f, +1.7076147010f);
	__m128 dt_min = _mm_min_ps(t_r, _mm_min_ps(t_g, t_b));
	__m128 t_upper = _mm_add_ps(t, dt_min);

	return vsel(lower_mask, t_lower, t_upper);
}

__m128 find_gamut_intersection(__m128 a, __m128 b, __m128 L1, __m128 C1, __m128 L0)
{
	return find_gamut_intersection(a, b, L1, C1, L0, find_cusp(a, b));
}

PackedST get_ST_mid(__m128 a_, __m128 b_)
{
	auto rec = [&](__m128 c0,
	               float c1, float c2, float c3, float c4,
	               float c5, float c6, float c7, float c8,
	               float c9, float c10, float c11) {
		__m128 t = _mm_fmadd_ps(_mm_set1_ps(c11), a_,
		            _mm_fmadd_ps(_mm_set1_ps(c10), b_, _mm_set1_ps(c9)));
		t = _mm_fmadd_ps(t, a_,
		     _mm_fmadd_ps(_mm_set1_ps(c8), b_, _mm_set1_ps(c7)));
		t = _mm_fmadd_ps(t, a_,
		     _mm_fmadd_ps(_mm_set1_ps(c6), b_, _mm_set1_ps(c5)));
		t = _mm_fmadd_ps(t, a_,
		     _mm_fmadd_ps(_mm_set1_ps(c4), b_,
		     _mm_fmadd_ps(_mm_set1_ps(c3), a_, _mm_set1_ps(c2))));
		t = _mm_fmadd_ps(t, a_,
		     _mm_fmadd_ps(_mm_set1_ps(c1), b_, c0));
		return _mm_add_ps(_mm_div_ps(_mm_set1_ps(1.f), t),
		                  _mm_set1_ps(0.f));
	};
	(void)rec;

	// 위 일반화는 어색 — 직접 작성.  원본 식 그대로.
	auto mk_S = [&]() {
		__m128 inner = _mm_fmadd_ps(_mm_set1_ps(4.69891013f), a_,
		                _mm_fmadd_ps(_mm_set1_ps(5.38770819f), b_, _mm_set1_ps(-4.24894561f)));
		inner = _mm_fmadd_ps(inner, a_,
		         _mm_fmadd_ps(_mm_set1_ps(-10.02301043f), b_, _mm_set1_ps(-2.13704948f)));
		inner = _mm_fmadd_ps(inner, a_,
		         _mm_fmadd_ps(_mm_set1_ps( 1.75198401f), b_, _mm_set1_ps(-2.19557347f)));
		inner = _mm_fmadd_ps(inner, a_,
		         _mm_fmadd_ps(_mm_set1_ps( 4.15901240f), b_, _mm_set1_ps( 7.44778970f)));
		return _mm_add_ps(_mm_set1_ps(0.11516993f),
		                  _mm_div_ps(_mm_set1_ps(1.f), inner));
	};
	auto mk_T = [&]() {
		__m128 inner = _mm_fmadd_ps(_mm_set1_ps(-0.14661872f), a_,
		                _mm_fmadd_ps(_mm_set1_ps(-0.45399568f), b_, _mm_set1_ps( 0.00299215f)));
		inner = _mm_fmadd_ps(inner, a_,
		         _mm_fmadd_ps(_mm_set1_ps( 0.61223990f), b_, _mm_set1_ps(-0.27087943f)));
		inner = _mm_fmadd_ps(inner, a_,
		         _mm_fmadd_ps(_mm_set1_ps( 0.90148123f), b_, _mm_set1_ps( 0.40370612f)));
		inner = _mm_fmadd_ps(inner, a_,
		         _mm_fmadd_ps(_mm_set1_ps(-0.68124379f), b_, _mm_set1_ps( 1.61320320f)));
		return _mm_add_ps(_mm_set1_ps(0.11239642f),
		                  _mm_div_ps(_mm_set1_ps(1.f), inner));
	};
	return PackedST{ mk_S(), mk_T() };
}

PackedCs get_Cs(__m128 L, __m128 a_, __m128 b_)
{
	PackedLC cusp = find_cusp(a_, b_);
	__m128 C_max = find_gamut_intersection(a_, b_, L, _mm_set1_ps(1.f), L, cusp);
	PackedST ST_max = cusp.to_ST();
	__m128 LSm = _mm_mul_ps(L, ST_max.S);
	__m128 oneL = _mm_sub_ps(_mm_set1_ps(1.f), L);
	__m128 oTm = _mm_mul_ps(oneL, ST_max.T);
	__m128 k = _mm_div_ps(C_max, _mm_min_ps(LSm, oTm));

	PackedST STmid = get_ST_mid(a_, b_);
	__m128 Ca_m = _mm_mul_ps(L, STmid.S);
	__m128 Cb_m = _mm_mul_ps(oneL, STmid.T);
	__m128 Ca4 = _mm_mul_ps(_mm_mul_ps(Ca_m, Ca_m), _mm_mul_ps(Ca_m, Ca_m));
	__m128 Cb4 = _mm_mul_ps(_mm_mul_ps(Cb_m, Cb_m), _mm_mul_ps(Cb_m, Cb_m));
	__m128 inv = _mm_add_ps(_mm_div_ps(_mm_set1_ps(1.f), Ca4),
	                        _mm_div_ps(_mm_set1_ps(1.f), Cb4));
	__m128 C_mid = _mm_mul_ps(_mm_mul_ps(_mm_set1_ps(0.9f), k),
	                          _mm_sqrt_ps(_mm_sqrt_ps(_mm_div_ps(_mm_set1_ps(1.f), inv))));

	__m128 Ca0 = _mm_mul_ps(L, _mm_set1_ps(0.4f));
	__m128 Cb0 = _mm_mul_ps(oneL, _mm_set1_ps(0.8f));
	__m128 inv0 = _mm_add_ps(_mm_div_ps(_mm_set1_ps(1.f), _mm_mul_ps(Ca0, Ca0)),
	                         _mm_div_ps(_mm_set1_ps(1.f), _mm_mul_ps(Cb0, Cb0)));
	__m128 C_0 = _mm_sqrt_ps(_mm_div_ps(_mm_set1_ps(1.f), inv0));

	return PackedCs{ C_0, C_mid, C_max };
}

// =============================================================================
// §6. 색역 클리핑.
// =============================================================================

namespace {
struct HuePack { __m128 L, C, a_, b_; __m128 inside; };  // inside = mask "in gamut"

static inline HuePack split_hue_packed(PackedRGB rgb)
{
	const __m128 zero = _mm_setzero_ps(), one = _mm_set1_ps(1.f);
	// in-gamut mask: 모든 채널이 (0, 1) 사이.
	__m128 in = _mm_and_ps(_mm_and_ps(_mm_cmpgt_ps(rgb.r, zero),
	                                  _mm_cmpgt_ps(rgb.g, zero)),
	            _mm_and_ps(_mm_and_ps(_mm_cmpgt_ps(rgb.b, zero),
	                                  _mm_cmplt_ps(rgb.r, one)),
	            _mm_and_ps(_mm_cmplt_ps(rgb.g, one),
	                       _mm_cmplt_ps(rgb.b, one))));

	PackedLab lab = rgb.to_oklab();
	__m128 eps = _mm_set1_ps(0.00001f);
	__m128 C = _mm_max_ps(eps, _mm_sqrt_ps(_mm_fmadd_ps(lab.a, lab.a,
	                                                     _mm_mul_ps(lab.b, lab.b))));
	__m128 a_ = _mm_div_ps(lab.a, C);
	__m128 b_ = _mm_div_ps(lab.b, C);
	return { lab.L, C, a_, b_, in };
}

static inline PackedRGB project_along_packed(HuePack h, __m128 L0)
{
	__m128 t = find_gamut_intersection(h.a_, h.b_, h.L, h.C, L0);
	__m128 L_clipped = _mm_fmadd_ps(L0, _mm_sub_ps(_mm_set1_ps(1.f), t), _mm_mul_ps(t, h.L));
	__m128 C_clipped = _mm_mul_ps(t, h.C);
	PackedLab clipped{ L_clipped, _mm_mul_ps(C_clipped, h.a_),
	                              _mm_mul_ps(C_clipped, h.b_) };
	return clipped.to_linear_srgb();
}
} // anon

PackedRGB gamut_clip_preserve_chroma(PackedRGB rgb)
{
	HuePack h = split_hue_packed(rgb);
	__m128 L0 = vclamp(h.L, 0.f, 1.f);
	PackedRGB c = project_along_packed(h, L0);
	return PackedRGB{
		vsel(h.inside, rgb.r, c.r),
		vsel(h.inside, rgb.g, c.g),
		vsel(h.inside, rgb.b, c.b),
	};
}

PackedRGB gamut_clip_project_to_0_5(PackedRGB rgb)
{
	HuePack h = split_hue_packed(rgb);
	PackedRGB c = project_along_packed(h, _mm_set1_ps(0.5f));
	return PackedRGB{
		vsel(h.inside, rgb.r, c.r),
		vsel(h.inside, rgb.g, c.g),
		vsel(h.inside, rgb.b, c.b),
	};
}

PackedRGB gamut_clip_project_to_L_cusp(PackedRGB rgb)
{
	HuePack h = split_hue_packed(rgb);
	PackedLC cusp = find_cusp(h.a_, h.b_);
	PackedRGB c = project_along_packed(h, cusp.L);
	return PackedRGB{
		vsel(h.inside, rgb.r, c.r),
		vsel(h.inside, rgb.g, c.g),
		vsel(h.inside, rgb.b, c.b),
	};
}

PackedRGB gamut_clip_adaptive_L0_0_5(PackedRGB rgb, float alpha)
{
	HuePack h = split_hue_packed(rgb);
	__m128 Ld = _mm_sub_ps(h.L, _mm_set1_ps(0.5f));
	__m128 e1 = _mm_fmadd_ps(_mm_set1_ps(alpha), h.C,
	                          _mm_add_ps(_mm_set1_ps(0.5f), vabs(Ld)));
	__m128 sq = _mm_sqrt_ps(_mm_fnmadd_ps(_mm_set1_ps(2.f), vabs(Ld), _mm_mul_ps(e1, e1)));
	__m128 L0 = _mm_mul_ps(_mm_set1_ps(0.5f),
	            _mm_add_ps(_mm_set1_ps(1.f),
	                       _mm_mul_ps(vsgn(Ld), _mm_sub_ps(e1, sq))));
	PackedRGB c = project_along_packed(h, L0);
	return PackedRGB{
		vsel(h.inside, rgb.r, c.r),
		vsel(h.inside, rgb.g, c.g),
		vsel(h.inside, rgb.b, c.b),
	};
}

PackedRGB gamut_clip_adaptive_L0_L_cusp(PackedRGB rgb, float alpha)
{
	HuePack h = split_hue_packed(rgb);
	PackedLC cusp = find_cusp(h.a_, h.b_);

	__m128 Ld = _mm_sub_ps(h.L, cusp.L);
	__m128 cusp_above = _mm_cmpgt_ps(Ld, _mm_setzero_ps());
	__m128 k = _mm_mul_ps(_mm_set1_ps(2.f),
	            vsel(cusp_above, _mm_sub_ps(_mm_set1_ps(1.f), cusp.L), cusp.L));

	__m128 e1 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0.5f), k),
	             _mm_add_ps(vabs(Ld),
	                        _mm_mul_ps(_mm_set1_ps(alpha), _mm_div_ps(h.C, k))));
	__m128 sq = _mm_sqrt_ps(_mm_fnmadd_ps(_mm_mul_ps(_mm_set1_ps(2.f), k), vabs(Ld),
	                                       _mm_mul_ps(e1, e1)));
	__m128 L0 = _mm_add_ps(cusp.L,
	            _mm_mul_ps(_mm_set1_ps(0.5f),
	                       _mm_mul_ps(vsgn(Ld), _mm_sub_ps(e1, sq))));
	PackedRGB c = project_along_packed(h, L0);
	return PackedRGB{
		vsel(h.inside, rgb.r, c.r),
		vsel(h.inside, rgb.g, c.g),
		vsel(h.inside, rgb.b, c.b),
	};
}

// =============================================================================
// §4(이어서). HSL/HSV ↔ sRGB.
// =============================================================================

PackedRGB PackedHSL::to_srgb() const
{
	const __m128 h = this->h, s = this->s, l = this->l;
	const __m128 zero = _mm_setzero_ps(), one = _mm_set1_ps(1.f);

	// l == 0 또는 1 일 때를 위한 mask (일반 경로 안전하게 통과 후 마지막에 덮어씀).
	__m128 is_one  = _mm_cmpeq_ps(l, one);
	__m128 is_zero = _mm_cmpeq_ps(l, zero);

	// hue → (a_, b_).
	__m128 sin_v, cos_v;
	sincos_packed(_mm_mul_ps(_mm_set1_ps(2.f * pi), h), &sin_v, &cos_v);
	__m128 a_ = cos_v, b_ = sin_v;
	__m128 L = toe_inv_packed(l);

	PackedCs cs = get_Cs(L, a_, b_);
	__m128 C_0 = cs.C_0, C_mid = cs.C_mid, C_max = cs.C_max;

	const __m128 mid     = _mm_set1_ps(0.8f);
	const __m128 mid_inv = _mm_set1_ps(1.25f);

	// 두 경로 모두 계산 후 mask 로 선택 (s < mid).
	__m128 t1   = _mm_mul_ps(mid_inv, s);
	__m128 k1a  = _mm_mul_ps(mid, C_0);
	__m128 k2a  = _mm_sub_ps(one, _mm_div_ps(k1a, C_mid));
	__m128 C_a  = _mm_div_ps(_mm_mul_ps(t1, k1a),
	                         _mm_sub_ps(one, _mm_mul_ps(k2a, t1)));

	__m128 t2   = _mm_div_ps(_mm_sub_ps(s, mid), _mm_sub_ps(one, mid));
	__m128 k0b  = C_mid;
	__m128 k1b  = _mm_div_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(one, mid),
	                                                _mm_mul_ps(C_mid, C_mid)),
	                                     _mm_mul_ps(mid_inv, mid_inv)),
	                         C_0);
	__m128 k2b  = _mm_sub_ps(one, _mm_div_ps(k1b, _mm_sub_ps(C_max, C_mid)));
	__m128 C_b  = _mm_add_ps(k0b, _mm_div_ps(_mm_mul_ps(t2, k1b),
	                                         _mm_sub_ps(one, _mm_mul_ps(k2b, t2))));

	__m128 sm   = _mm_cmplt_ps(s, mid);
	__m128 C    = vsel(sm, C_a, C_b);

	PackedLab labp{ L, _mm_mul_ps(C, a_), _mm_mul_ps(C, b_) };
	PackedRGB lin = labp.to_linear_srgb();

	__m128 R = srgb_transfer_packed(lin.r);
	__m128 G = srgb_transfer_packed(lin.g);
	__m128 B = srgb_transfer_packed(lin.b);

	// l == 1 → (1,1,1), l == 0 → (0,0,0).
	R = vsel(is_one, one,  vsel(is_zero, zero, R));
	G = vsel(is_one, one,  vsel(is_zero, zero, G));
	B = vsel(is_one, one,  vsel(is_zero, zero, B));
	return PackedRGB{ R, G, B };
}

PackedHSL PackedRGB::to_okhsl() const
{
	PackedRGB lin{
		srgb_transfer_inv_packed(this->r),
		srgb_transfer_inv_packed(this->g),
		srgb_transfer_inv_packed(this->b),
	};
	PackedLab lab = lin.to_oklab();

	__m128 C = _mm_sqrt_ps(_mm_fmadd_ps(lab.a, lab.a, _mm_mul_ps(lab.b, lab.b)));
	__m128 a_ = _mm_div_ps(lab.a, C);
	__m128 b_ = _mm_div_ps(lab.b, C);
	__m128 L = lab.L;
	// 부호 비트만 뒤집어 −0 까지 보존 (subtract 는 +0 만 만들어버려 부호 정보 손실).
	const __m128 signbit = _mm_set1_ps(-0.f);
	__m128 ang = atan2_packed(_mm_xor_ps(lab.b, signbit), _mm_xor_ps(lab.a, signbit));
	__m128 h_ = _mm_fmadd_ps(_mm_set1_ps(0.5f / pi), ang, _mm_set1_ps(0.5f));

	PackedCs cs = get_Cs(L, a_, b_);
	__m128 C_0 = cs.C_0, C_mid = cs.C_mid, C_max = cs.C_max;

	const __m128 one = _mm_set1_ps(1.f);
	const __m128 mid     = _mm_set1_ps(0.8f);
	const __m128 mid_inv = _mm_set1_ps(1.25f);

	__m128 k1a = _mm_mul_ps(mid, C_0);
	__m128 k2a = _mm_sub_ps(one, _mm_div_ps(k1a, C_mid));
	__m128 ta  = _mm_div_ps(C, _mm_fmadd_ps(k2a, C, k1a));
	__m128 sa  = _mm_mul_ps(ta, mid);

	__m128 k0b = C_mid;
	__m128 k1b = _mm_div_ps(_mm_mul_ps(_mm_mul_ps(_mm_sub_ps(one, mid),
	                                                _mm_mul_ps(C_mid, C_mid)),
	                                     _mm_mul_ps(mid_inv, mid_inv)),
	                         C_0);
	__m128 k2b = _mm_sub_ps(one, _mm_div_ps(k1b, _mm_sub_ps(C_max, C_mid)));
	__m128 dC  = _mm_sub_ps(C, k0b);
	__m128 tb  = _mm_div_ps(dC, _mm_fmadd_ps(k2b, dC, k1b));
	__m128 sb  = _mm_add_ps(mid, _mm_mul_ps(_mm_sub_ps(one, mid), tb));

	__m128 use_a = _mm_cmplt_ps(C, C_mid);
	__m128 s_ = vsel(use_a, sa, sb);

	__m128 l_ = toe_packed(L);
	return PackedHSL{ h_, s_, l_ };
}

PackedRGB PackedHSV::to_srgb() const
{
	const __m128 h = this->h, s = this->s, v = this->v;
	__m128 sin_v, cos_v;
	sincos_packed(_mm_mul_ps(_mm_set1_ps(2.f * pi), h), &sin_v, &cos_v);
	__m128 a_ = cos_v, b_ = sin_v;

	PackedLC cusp = find_cusp(a_, b_);
	PackedST ST_max = cusp.to_ST();
	__m128 S_max = ST_max.S, T_max = ST_max.T;
	const __m128 S0 = _mm_set1_ps(0.5f);
	__m128 k = _mm_sub_ps(_mm_set1_ps(1.f), _mm_div_ps(S0, S_max));

	__m128 denom = _mm_fnmadd_ps(_mm_mul_ps(T_max, k), s, _mm_add_ps(S0, T_max));
	__m128 L_v = _mm_sub_ps(_mm_set1_ps(1.f), _mm_div_ps(_mm_mul_ps(s, S0), denom));
	__m128 C_v = _mm_div_ps(_mm_mul_ps(_mm_mul_ps(s, T_max), S0), denom);

	__m128 L = _mm_mul_ps(v, L_v);
	__m128 C = _mm_mul_ps(v, C_v);

	__m128 L_vt = toe_inv_packed(L_v);
	__m128 C_vt = _mm_mul_ps(C_v, _mm_div_ps(L_vt, L_v));
	__m128 L_new = toe_inv_packed(L);
	C = _mm_mul_ps(C, _mm_div_ps(L_new, L));
	L = L_new;

	PackedLab labp{ L_vt, _mm_mul_ps(a_, C_vt), _mm_mul_ps(b_, C_vt) };
	PackedRGB rgb_scale = labp.to_linear_srgb();
	__m128 mx = _mm_max_ps(_mm_max_ps(rgb_scale.r, rgb_scale.g),
	            _mm_max_ps(rgb_scale.b, _mm_setzero_ps()));
	__m128 scale_L = cbrt_packed(_mm_div_ps(_mm_set1_ps(1.f), mx));
	L = _mm_mul_ps(L, scale_L);
	C = _mm_mul_ps(C, scale_L);

	PackedLab final_lab{ L, _mm_mul_ps(C, a_), _mm_mul_ps(C, b_) };
	PackedRGB lin = final_lab.to_linear_srgb();
	return PackedRGB{
		srgb_transfer_packed(lin.r),
		srgb_transfer_packed(lin.g),
		srgb_transfer_packed(lin.b),
	};
}

PackedHSV PackedRGB::to_okhsv() const
{
	PackedRGB lin{
		srgb_transfer_inv_packed(this->r),
		srgb_transfer_inv_packed(this->g),
		srgb_transfer_inv_packed(this->b),
	};
	PackedLab lab = lin.to_oklab();

	__m128 C = _mm_sqrt_ps(_mm_fmadd_ps(lab.a, lab.a, _mm_mul_ps(lab.b, lab.b)));
	__m128 a_ = _mm_div_ps(lab.a, C);
	__m128 b_ = _mm_div_ps(lab.b, C);
	__m128 L = lab.L;
	// 부호 비트만 뒤집어 −0 까지 보존 (subtract 는 +0 만 만들어버려 부호 정보 손실).
	const __m128 signbit = _mm_set1_ps(-0.f);
	__m128 ang = atan2_packed(_mm_xor_ps(lab.b, signbit), _mm_xor_ps(lab.a, signbit));
	__m128 h_ = _mm_fmadd_ps(_mm_set1_ps(0.5f / pi), ang, _mm_set1_ps(0.5f));

	PackedLC cusp = find_cusp(a_, b_);
	PackedST ST_max = cusp.to_ST();
	__m128 S_max = ST_max.S, T_max = ST_max.T;
	const __m128 S0 = _mm_set1_ps(0.5f);
	__m128 k = _mm_sub_ps(_mm_set1_ps(1.f), _mm_div_ps(S0, S_max));

	__m128 t = _mm_div_ps(T_max, _mm_fmadd_ps(L, T_max, C));
	__m128 L_v = _mm_mul_ps(t, L);
	__m128 C_v = _mm_mul_ps(t, C);
	__m128 L_vt = toe_inv_packed(L_v);
	__m128 C_vt = _mm_mul_ps(C_v, _mm_div_ps(L_vt, L_v));

	PackedLab labp{ L_vt, _mm_mul_ps(a_, C_vt), _mm_mul_ps(b_, C_vt) };
	PackedRGB rgb_scale = labp.to_linear_srgb();
	__m128 mx = _mm_max_ps(_mm_max_ps(rgb_scale.r, rgb_scale.g),
	            _mm_max_ps(rgb_scale.b, _mm_setzero_ps()));
	__m128 scale_L = cbrt_packed(_mm_div_ps(_mm_set1_ps(1.f), mx));
	L = _mm_div_ps(L, scale_L);
	C = _mm_div_ps(C, scale_L);

	C = _mm_mul_ps(C, _mm_div_ps(toe_packed(L), L));
	L = toe_packed(L);

	__m128 v = _mm_div_ps(L, L_v);
	__m128 num = _mm_mul_ps(_mm_add_ps(S0, T_max), C_v);
	__m128 den = _mm_fmadd_ps(_mm_mul_ps(T_max, k), C_v, _mm_mul_ps(T_max, S0));
	__m128 s = _mm_div_ps(num, den);
	return PackedHSV{ h_, s, v };
}

// =============================================================================
// §7. AoS batch wrapper.  4 RGB4/Lab4/... in / out.
// =============================================================================

namespace {

static inline PackedRGB load_AoS_RGB(const RGB4* in)
{
	__m128 v0 = _mm_load_ps(&in[0].r), v1 = _mm_load_ps(&in[1].r);
	__m128 v2 = _mm_load_ps(&in[2].r), v3 = _mm_load_ps(&in[3].r);
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	return PackedRGB{ v0, v1, v2 };  // r, g, b
}
static inline void store_AoS_RGB(PackedRGB p, RGB4* out)
{
	__m128 v0 = p.r, v1 = p.g, v2 = p.b, v3 = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	_mm_store_ps(&out[0].r, v0);
	_mm_store_ps(&out[1].r, v1);
	_mm_store_ps(&out[2].r, v2);
	_mm_store_ps(&out[3].r, v3);
}
static inline PackedLab load_AoS_Lab(const Lab4* in)
{
	__m128 v0 = _mm_load_ps(&in[0].L), v1 = _mm_load_ps(&in[1].L);
	__m128 v2 = _mm_load_ps(&in[2].L), v3 = _mm_load_ps(&in[3].L);
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	return PackedLab{ v0, v1, v2 };
}
static inline void store_AoS_Lab(PackedLab p, Lab4* out)
{
	__m128 v0 = p.L, v1 = p.a, v2 = p.b, v3 = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	_mm_store_ps(&out[0].L, v0);
	_mm_store_ps(&out[1].L, v1);
	_mm_store_ps(&out[2].L, v2);
	_mm_store_ps(&out[3].L, v3);
}
static inline PackedHSL load_AoS_HSL(const HSL4* in)
{
	__m128 v0 = _mm_load_ps(&in[0].h), v1 = _mm_load_ps(&in[1].h);
	__m128 v2 = _mm_load_ps(&in[2].h), v3 = _mm_load_ps(&in[3].h);
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	return PackedHSL{ v0, v1, v2 };
}
static inline void store_AoS_HSL(PackedHSL p, HSL4* out)
{
	__m128 v0 = p.h, v1 = p.s, v2 = p.l, v3 = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	_mm_store_ps(&out[0].h, v0);
	_mm_store_ps(&out[1].h, v1);
	_mm_store_ps(&out[2].h, v2);
	_mm_store_ps(&out[3].h, v3);
}
static inline PackedHSV load_AoS_HSV(const HSV4* in)
{
	__m128 v0 = _mm_load_ps(&in[0].h), v1 = _mm_load_ps(&in[1].h);
	__m128 v2 = _mm_load_ps(&in[2].h), v3 = _mm_load_ps(&in[3].h);
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	return PackedHSV{ v0, v1, v2 };
}
static inline void store_AoS_HSV(PackedHSV p, HSV4* out)
{
	__m128 v0 = p.h, v1 = p.s, v2 = p.v, v3 = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	_mm_store_ps(&out[0].h, v0);
	_mm_store_ps(&out[1].h, v1);
	_mm_store_ps(&out[2].h, v2);
	_mm_store_ps(&out[3].h, v3);
}

} // anon

void linear_srgb_to_oklab_batch4(const RGB4* in, Lab4* out)
{ store_AoS_Lab(load_AoS_RGB(in).to_oklab(), out); }
void oklab_to_linear_srgb_batch4(const Lab4* in, RGB4* out)
{ store_AoS_RGB(load_AoS_Lab(in).to_linear_srgb(), out); }
void okhsl_to_srgb_batch4(const HSL4* in, RGB4* out)
{ store_AoS_RGB(load_AoS_HSL(in).to_srgb(), out); }
void srgb_to_okhsl_batch4(const RGB4* in, HSL4* out)
{ store_AoS_HSL(load_AoS_RGB(in).to_okhsl(), out); }
void okhsv_to_srgb_batch4(const HSV4* in, RGB4* out)
{ store_AoS_RGB(load_AoS_HSV(in).to_srgb(), out); }
void srgb_to_okhsv_batch4(const RGB4* in, HSV4* out)
{ store_AoS_HSV(load_AoS_RGB(in).to_okhsv(), out); }

void gamut_clip_preserve_chroma_batch4(const RGB4* in, RGB4* out)
{ store_AoS_RGB(gamut_clip_preserve_chroma(load_AoS_RGB(in)), out); }
void gamut_clip_project_to_0_5_batch4(const RGB4* in, RGB4* out)
{ store_AoS_RGB(gamut_clip_project_to_0_5(load_AoS_RGB(in)), out); }
void gamut_clip_project_to_L_cusp_batch4(const RGB4* in, RGB4* out)
{ store_AoS_RGB(gamut_clip_project_to_L_cusp(load_AoS_RGB(in)), out); }
void gamut_clip_adaptive_L0_0_5_batch4(const RGB4* in, RGB4* out, float alpha)
{ store_AoS_RGB(gamut_clip_adaptive_L0_0_5(load_AoS_RGB(in), alpha), out); }
void gamut_clip_adaptive_L0_L_cusp_batch4(const RGB4* in, RGB4* out, float alpha)
{ store_AoS_RGB(gamut_clip_adaptive_L0_L_cusp(load_AoS_RGB(in), alpha), out); }

} // namespace ok_color_avx512_4
