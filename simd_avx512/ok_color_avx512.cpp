// ok_color_avx512.cpp — 16-wide SIMD (AVX512F + DQ + BW) 구현.
//   ok_color_simd.cpp 의 SSE 버전을 16-lane 으로 옮긴 것.
//   gamut clipping 은 동일 패턴이라 생략 — 변환만 포팅.

#include "ok_color_avx512.h"
#include <cmath>
#include <cfloat>

namespace ok_color_avx512
{

namespace {

constexpr float pi = 3.14159265358979323846f;

// vsel: mask=true 면 a, false 면 b.
//   AVX512 _mm512_mask_blend_ps(k, a, b): k 비트 set 이면 b 를 선택 — 인자 순서 주의.
static inline __m512 vsel(__mmask16 mask, __m512 a, __m512 b) {
	return _mm512_mask_blend_ps(mask, b, a);
}
static inline __m512 vabs(__m512 x) {
	return _mm512_andnot_ps(_mm512_set1_ps(-0.f), x);
}
static inline __m512 vsgn(__m512 x) {
	__m512 zero = _mm512_setzero_ps();
	__mmask16 pos = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OS);
	__mmask16 neg = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OS);
	__m512 one = _mm512_set1_ps(1.f);
	__m512 r = _mm512_setzero_ps();
	r = _mm512_mask_blend_ps(pos, r, one);
	r = _mm512_mask_blend_ps(neg, r, _mm512_set1_ps(-1.f));
	return r;
}
static inline __m512 vclamp(__m512 x, float lo, float hi) {
	return _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(lo)), _mm512_set1_ps(hi));
}

// =============================================================================
// SIMD 수학 primitive — 16-wide.
// =============================================================================

// integer divide by 3 for 16 unsigned 32-bit lanes via multiply-high trick.
static inline __m512i div_by_3_u32(__m512i x)
{
	const __m512i magic = _mm512_set1_epi64(0xAAAAAAABLL);
	// mul_epu32: 8 × 64-bit results from even 32-bit lanes (0, 2, 4, ..., 14).
	__m512i even = _mm512_mul_epu32(x, magic);
	__m512i odd  = _mm512_mul_epu32(_mm512_srli_epi64(x, 32), magic);
	even = _mm512_srli_epi64(even, 33);
	odd  = _mm512_srli_epi64(odd,  33);
	odd  = _mm512_slli_epi64(odd,  32);
	return _mm512_or_si512(even, odd);
}

// cbrt_packed.  bit-hack 초기치 + Newton 2회.
static inline __m512 cbrt_packed(__m512 x)
{
	__m512i ix = _mm512_castps_si512(x);
	__m512i t  = _mm512_add_epi32(ix, _mm512_set1_epi32(0x7F000000));
	__m512 y   = _mm512_castsi512_ps(div_by_3_u32(t));
	const __m512 third = _mm512_set1_ps(1.f / 3.f);
	const __m512 two   = _mm512_set1_ps(2.f);
	for (int i = 0; i < 2; ++i) {
		__m512 y2  = _mm512_mul_ps(y, y);
		__m512 xy2 = _mm512_div_ps(x, y2);
		y = _mm512_mul_ps(_mm512_fmadd_ps(two, y, xy2), third);
	}
	__mmask16 zero_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_OQ);
	return _mm512_mask_blend_ps(zero_mask, y, _mm512_setzero_ps());
}

// log_packed (자연로그). Cephes 다항식.
static inline __m512 log_packed(__m512 x)
{
	__mmask16 vinvalid = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OS);
	x = _mm512_max_ps(x, _mm512_set1_ps(1.175494351e-38f));

	__m512i ix = _mm512_castps_si512(x);
	__m512i e  = _mm512_sub_epi32(_mm512_srli_epi32(ix, 23), _mm512_set1_epi32(0x7F));
	x = _mm512_or_ps(_mm512_andnot_ps(_mm512_castsi512_ps(_mm512_set1_epi32((int)0x7F800000)), x),
	                 _mm512_set1_ps(0.5f));

	__m512 ef = _mm512_add_ps(_mm512_cvtepi32_ps(e), _mm512_set1_ps(1.f));

	__mmask16 mask = _mm512_cmp_ps_mask(x, _mm512_set1_ps(0.7071067811865476f), _CMP_LT_OS);
	__m512 add = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), x);
	x = _mm512_sub_ps(x, _mm512_set1_ps(1.f));
	ef = _mm512_sub_ps(ef, _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_set1_ps(1.f)));
	x = _mm512_add_ps(x, add);

	__m512 z = _mm512_mul_ps(x, x);

	__m512 y = _mm512_set1_ps( 7.0376836292e-2f);
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.1514610310e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 1.1676998740e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.2420140846e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 1.4249322787e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.6668057665e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 2.0000714765e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-2.4999993993e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps( 3.3333331174e-1f));
	y = _mm512_mul_ps(_mm512_mul_ps(y, x), z);

	y = _mm512_fmadd_ps(ef, _mm512_set1_ps(-2.12194440e-4f), y);
	y = _mm512_fnmadd_ps(z, _mm512_set1_ps(0.5f), y);

	x = _mm512_add_ps(x, y);
	x = _mm512_fmadd_ps(ef, _mm512_set1_ps(0.693359375f), x);

	// 음수/0 입력 → -inf.  invalid mask lane 만 −inf 로 덮어씀.
	const __m512 neg_inf = _mm512_set1_ps(-INFINITY);
	return _mm512_mask_blend_ps(vinvalid, x, neg_inf);
}

static inline __m512 exp_packed(__m512 x)
{
	x = _mm512_min_ps(x, _mm512_set1_ps( 88.3762626647949f));
	x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));

	__m512 fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f),
	                               _mm512_set1_ps(0.5f));
	__m512i emm0 = _mm512_cvttps_epi32(fx);
	fx = _mm512_cvtepi32_ps(emm0);

	__m512 tmp = _mm512_mul_ps(fx, _mm512_set1_ps(0.693359375f));
	__m512 z   = _mm512_mul_ps(fx, _mm512_set1_ps(-2.12194440e-4f));
	x = _mm512_sub_ps(x, tmp);
	x = _mm512_sub_ps(x, z);

	z = _mm512_mul_ps(x, x);

	__m512 y = _mm512_set1_ps(1.9875691500e-4f);
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.3981999507e-3f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(8.3334519073e-3f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(4.1665795894e-2f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.6666665459e-1f));
	y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(5.0000001201e-1f));
	y = _mm512_fmadd_ps(y, z, x);
	y = _mm512_add_ps(y, _mm512_set1_ps(1.f));

	emm0 = _mm512_add_epi32(emm0, _mm512_set1_epi32(0x7F));
	emm0 = _mm512_slli_epi32(emm0, 23);
	__m512 pow2n = _mm512_castsi512_ps(emm0);
	return _mm512_mul_ps(y, pow2n);
}
static inline __m512 pow_packed(__m512 x, __m512 c)
{
	return exp_packed(_mm512_mul_ps(c, log_packed(x)));
}

static inline void sincos_packed(__m512 x, __m512* sin_out, __m512* cos_out)
{
	__m512 sign_bit_sin = _mm512_and_ps(x, _mm512_castsi512_ps(_mm512_set1_epi32((int)0x80000000)));
	x = vabs(x);

	__m512 y = _mm512_mul_ps(x, _mm512_set1_ps(1.27323954473516f));
	__m512i emm2 = _mm512_cvttps_epi32(y);
	emm2 = _mm512_add_epi32(emm2, _mm512_set1_epi32(1));
	emm2 = _mm512_and_si512(emm2, _mm512_set1_epi32(~1));
	y = _mm512_cvtepi32_ps(emm2);

	__m512i emm0 = _mm512_slli_epi32(_mm512_and_si512(emm2, _mm512_set1_epi32(4)), 29);
	__m512 swap_sign_bit_sin = _mm512_castsi512_ps(emm0);
	sign_bit_sin = _mm512_xor_ps(sign_bit_sin, swap_sign_bit_sin);

	__mmask16 poly_mask = _mm512_cmp_epi32_mask(_mm512_and_si512(emm2, _mm512_set1_epi32(2)),
	                                             _mm512_setzero_si512(), _MM_CMPINT_EQ);

	__m512i emm2_cos = _mm512_sub_epi32(emm2, _mm512_set1_epi32(2));
	__m512i emm0_cos = _mm512_slli_epi32(_mm512_andnot_si512(emm2_cos, _mm512_set1_epi32(4)), 29);
	__m512 sign_bit_cos = _mm512_castsi512_ps(emm0_cos);

	x = _mm512_fmadd_ps(y, _mm512_set1_ps(-0.78515625f), x);
	x = _mm512_fmadd_ps(y, _mm512_set1_ps(-2.4187564849853515625e-4f), x);
	x = _mm512_fmadd_ps(y, _mm512_set1_ps(-3.77489497744594108e-8f), x);

	__m512 z = _mm512_mul_ps(x, x);

	__m512 yc = _mm512_set1_ps(2.443315711809948e-5f);
	yc = _mm512_fmadd_ps(yc, z, _mm512_set1_ps(-1.388731625493765e-3f));
	yc = _mm512_fmadd_ps(yc, z, _mm512_set1_ps( 4.166664568298827e-2f));
	yc = _mm512_mul_ps(yc, _mm512_mul_ps(z, z));
	yc = _mm512_sub_ps(yc, _mm512_mul_ps(z, _mm512_set1_ps(0.5f)));
	yc = _mm512_add_ps(yc, _mm512_set1_ps(1.f));

	__m512 ys = _mm512_set1_ps(-1.9515295891e-4f);
	ys = _mm512_fmadd_ps(ys, z, _mm512_set1_ps( 8.3321608736e-3f));
	ys = _mm512_fmadd_ps(ys, z, _mm512_set1_ps(-1.6666654611e-1f));
	ys = _mm512_mul_ps(ys, _mm512_mul_ps(z, x));
	ys = _mm512_add_ps(ys, x);

	__m512 sin_v = vsel(poly_mask, ys, yc);
	__m512 cos_v = vsel(poly_mask, yc, ys);

	*sin_out = _mm512_xor_ps(sin_v, sign_bit_sin);
	*cos_out = _mm512_xor_ps(cos_v, sign_bit_cos);
}

static inline __m512 atan2_packed(__m512 y, __m512 x)
{
	__m512 ay = vabs(y), ax = vabs(x);
	__m512 a = _mm512_div_ps(_mm512_min_ps(ax, ay),
	                          _mm512_max_ps(_mm512_max_ps(ax, ay), _mm512_set1_ps(FLT_MIN)));
	__m512 a2 = _mm512_mul_ps(a, a);

	__m512 r = _mm512_set1_ps(-0.0464964749f);
	r = _mm512_fmadd_ps(r, a2, _mm512_set1_ps( 0.15931422f));
	r = _mm512_fmadd_ps(r, a2, _mm512_set1_ps(-0.327622764f));
	r = _mm512_mul_ps(r, _mm512_mul_ps(a2, a));
	r = _mm512_add_ps(r, a);

	__mmask16 swap = _mm512_cmp_ps_mask(ay, ax, _CMP_GT_OS);
	r = vsel(swap, _mm512_sub_ps(_mm512_set1_ps(pi * 0.5f), r), r);

	// 부호 비트 추출 → __mmask16.
	__mmask16 negx = _mm512_cmp_epi32_mask(_mm512_castps_si512(x), _mm512_setzero_si512(),
	                                        _MM_CMPINT_LT);
	__mmask16 negy = _mm512_cmp_epi32_mask(_mm512_castps_si512(y), _mm512_setzero_si512(),
	                                        _MM_CMPINT_LT);

	r = vsel(negx, _mm512_sub_ps(_mm512_set1_ps(pi), r), r);
	r = vsel(negy, _mm512_xor_ps(r, _mm512_set1_ps(-0.f)), r);
	return r;
}

// 색공간 유틸 — packed.
static inline __m512 srgb_transfer_packed(__m512 a)
{
	__m512 lin   = _mm512_mul_ps(a, _mm512_set1_ps(12.92f));
	__m512 gamma = _mm512_fmsub_ps(_mm512_set1_ps(1.055f),
	                                pow_packed(a, _mm512_set1_ps(1.f / 2.4f)),
	                                _mm512_set1_ps(0.055f));
	__mmask16 mask = _mm512_cmp_ps_mask(a, _mm512_set1_ps(0.0031308f), _CMP_LE_OS);
	return vsel(mask, lin, gamma);
}
static inline __m512 srgb_transfer_inv_packed(__m512 a)
{
	__m512 lin   = _mm512_div_ps(a, _mm512_set1_ps(12.92f));
	__m512 gamma = pow_packed(_mm512_div_ps(_mm512_add_ps(a, _mm512_set1_ps(0.055f)),
	                                         _mm512_set1_ps(1.055f)),
	                           _mm512_set1_ps(2.4f));
	__mmask16 mask = _mm512_cmp_ps_mask(a, _mm512_set1_ps(0.04045f), _CMP_LT_OS);
	return vsel(mask, lin, gamma);
}
static inline __m512 toe_packed(__m512 x)
{
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	__m512 u = _mm512_fmsub_ps(_mm512_set1_ps(k3), x, _mm512_set1_ps(k1));
	__m512 inside = _mm512_fmadd_ps(u, u, _mm512_mul_ps(_mm512_set1_ps(4.f * k2 * k3), x));
	return _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_add_ps(u, _mm512_sqrt_ps(inside)));
}
static inline __m512 toe_inv_packed(__m512 x)
{
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	__m512 num = _mm512_fmadd_ps(x, x, _mm512_mul_ps(_mm512_set1_ps(k1), x));
	__m512 den = _mm512_mul_ps(_mm512_set1_ps(k3), _mm512_add_ps(x, _mm512_set1_ps(k2)));
	return _mm512_div_ps(num, den);
}

// 행렬 — 같은 상수.
static constexpr float M_RGB_TO_LMS[3][3] = {
	{ 0.4122214708f, 0.5363325363f, 0.0514459929f },
	{ 0.2119034982f, 0.6806995451f, 0.1073969566f },
	{ 0.0883024619f, 0.2817188376f, 0.6299787005f },
};
static constexpr float M_LMS_TO_LAB[3][3] = {
	{ 0.2104542553f,  0.7936177850f, -0.0040720468f },
	{ 1.9779984951f, -2.4285922050f,  0.4505937099f },
	{ 0.0259040371f,  0.7827717662f, -0.8086757660f },
};
static constexpr float M_LAB_TO_LMS[3][3] = {
	{ 1.f, +0.3963377774f, +0.2158037573f },
	{ 1.f, -0.1055613458f, -0.0638541728f },
	{ 1.f, -0.0894841775f, -1.2914855480f },
};
static constexpr float M_LMS_TO_RGB[3][3] = {
	{ +4.0767416621f, -3.3077115913f, +0.2309699292f },
	{ -1.2684380046f, +2.6097574011f, -0.3413193965f },
	{ -0.0041960863f, -0.7034186147f, +1.7076147010f },
};

template <const float (&M)[3][3]>
static inline __m512 row_dot3(int i, __m512 R, __m512 G, __m512 B)
{
	return _mm512_fmadd_ps(_mm512_set1_ps(M[i][2]), B,
	       _mm512_fmadd_ps(_mm512_set1_ps(M[i][1]), G,
	                       _mm512_mul_ps(_mm512_set1_ps(M[i][0]), R)));
}

} // anon

// =============================================================================
// 색공간 변환.
// =============================================================================

PackedLab PackedRGB::to_oklab() const
{
	const __m512 R = this->r, G = this->g, B = this->b;
	__m512 l = row_dot3<M_RGB_TO_LMS>(0, R, G, B);
	__m512 m = row_dot3<M_RGB_TO_LMS>(1, R, G, B);
	__m512 s = row_dot3<M_RGB_TO_LMS>(2, R, G, B);
	l = cbrt_packed(l); m = cbrt_packed(m); s = cbrt_packed(s);
	return PackedLab{
		row_dot3<M_LMS_TO_LAB>(0, l, m, s),
		row_dot3<M_LMS_TO_LAB>(1, l, m, s),
		row_dot3<M_LMS_TO_LAB>(2, l, m, s),
	};
}

PackedRGB PackedLab::to_linear_srgb() const
{
	const __m512 L = this->L, A = this->a, B = this->b;
	__m512 l = row_dot3<M_LAB_TO_LMS>(0, L, A, B);
	__m512 m = row_dot3<M_LAB_TO_LMS>(1, L, A, B);
	__m512 s = row_dot3<M_LAB_TO_LMS>(2, L, A, B);
	l = _mm512_mul_ps(_mm512_mul_ps(l, l), l);
	m = _mm512_mul_ps(_mm512_mul_ps(m, m), m);
	s = _mm512_mul_ps(_mm512_mul_ps(s, s), s);
	return PackedRGB{
		row_dot3<M_LMS_TO_RGB>(0, l, m, s),
		row_dot3<M_LMS_TO_RGB>(1, l, m, s),
		row_dot3<M_LMS_TO_RGB>(2, l, m, s),
	};
}

namespace {

// gamut 분석 — SSE 버전과 동일 식.
static __m512 compute_max_saturation(__m512 a, __m512 b)
{
	const __m512 one = _mm512_set1_ps(1.f);
	__mmask16 maskR = _mm512_cmp_ps_mask(_mm512_fmadd_ps(_mm512_set1_ps(-1.88170328f), a,
	                                                      _mm512_mul_ps(_mm512_set1_ps(-0.80936493f), b)),
	                                      one, _CMP_GT_OS);
	__mmask16 maskG_raw = _mm512_cmp_ps_mask(_mm512_fmadd_ps(_mm512_set1_ps( 1.81444104f), a,
	                                                          _mm512_mul_ps(_mm512_set1_ps(-1.19445276f), b)),
	                                          one, _CMP_GT_OS);
	__mmask16 maskG = (__mmask16)(~maskR & maskG_raw);

	auto pick3 = [&](float r, float g, float b_) {
		__m512 v = _mm512_set1_ps(b_);
		v = vsel(maskG, _mm512_set1_ps(g), v);
		v = vsel(maskR, _mm512_set1_ps(r), v);
		return v;
	};
	__m512 k0 = pick3(+1.19086277f, +0.73956515f, +1.35733652f);
	__m512 k1 = pick3(+1.76576728f, -0.45954404f, -0.00915799f);
	__m512 k2 = pick3(+0.59662641f, +0.08285427f, -1.15130210f);
	__m512 k3 = pick3(+0.75515197f, +0.12541070f, -0.50559606f);
	__m512 k4 = pick3(+0.56771245f, +0.14503204f, +0.00692167f);
	__m512 wl = pick3(+4.0767416621f, -1.2684380046f, -0.0041960863f);
	__m512 wm = pick3(-3.3077115913f, +2.6097574011f, -0.7034186147f);
	__m512 ws = pick3(+0.2309699292f, -0.3413193965f, +1.7076147010f);

	__m512 S = _mm512_add_ps(k0,
	            _mm512_add_ps(_mm512_mul_ps(k1, a),
	            _mm512_add_ps(_mm512_mul_ps(k2, b),
	            _mm512_add_ps(_mm512_mul_ps(_mm512_mul_ps(k3, a), a),
	                          _mm512_mul_ps(_mm512_mul_ps(k4, a), b)))));

	__m512 k_l = _mm512_fmadd_ps(_mm512_set1_ps(+0.3963377774f), a, _mm512_mul_ps(_mm512_set1_ps(+0.2158037573f), b));
	__m512 k_m = _mm512_fmadd_ps(_mm512_set1_ps(-0.1055613458f), a, _mm512_mul_ps(_mm512_set1_ps(-0.0638541728f), b));
	__m512 k_s = _mm512_fmadd_ps(_mm512_set1_ps(-0.0894841775f), a, _mm512_mul_ps(_mm512_set1_ps(-1.2914855480f), b));

	__m512 l_ = _mm512_fmadd_ps(S, k_l, _mm512_set1_ps(1.f));
	__m512 m_ = _mm512_fmadd_ps(S, k_m, _mm512_set1_ps(1.f));
	__m512 s_ = _mm512_fmadd_ps(S, k_s, _mm512_set1_ps(1.f));
	__m512 lp = _mm512_mul_ps(_mm512_mul_ps(l_, l_), l_);
	__m512 mp = _mm512_mul_ps(_mm512_mul_ps(m_, m_), m_);
	__m512 sp = _mm512_mul_ps(_mm512_mul_ps(s_, s_), s_);
	__m512 ldS  = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(3.f), k_l), _mm512_mul_ps(l_, l_));
	__m512 mdS  = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(3.f), k_m), _mm512_mul_ps(m_, m_));
	__m512 sdS  = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(3.f), k_s), _mm512_mul_ps(s_, s_));
	__m512 ldS2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(6.f), _mm512_mul_ps(k_l, k_l)), l_);
	__m512 mdS2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(6.f), _mm512_mul_ps(k_m, k_m)), m_);
	__m512 sdS2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(6.f), _mm512_mul_ps(k_s, k_s)), s_);

	__m512 f  = _mm512_fmadd_ps(wl, lp,  _mm512_fmadd_ps(wm, mp,  _mm512_mul_ps(ws, sp)));
	__m512 f1 = _mm512_fmadd_ps(wl, ldS, _mm512_fmadd_ps(wm, mdS, _mm512_mul_ps(ws, sdS)));
	__m512 f2 = _mm512_fmadd_ps(wl, ldS2,_mm512_fmadd_ps(wm, mdS2,_mm512_mul_ps(ws, sdS2)));

	__m512 num = _mm512_mul_ps(f, f1);
	__m512 den = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(f, f2), _mm512_mul_ps(f1, f1));
	return _mm512_sub_ps(S, _mm512_div_ps(num, den));
}

struct PackedLC512 { __m512 L, C; };
struct PackedST512 { __m512 S, T; };
struct PackedCs512 { __m512 C_0, C_mid, C_max; };

static PackedLC512 find_cusp(__m512 a, __m512 b)
{
	__m512 S_cusp = compute_max_saturation(a, b);
	PackedLab labp{ _mm512_set1_ps(1.f), _mm512_mul_ps(S_cusp, a), _mm512_mul_ps(S_cusp, b) };
	PackedRGB rgbp = labp.to_linear_srgb();
	__m512 mx = _mm512_max_ps(_mm512_max_ps(rgbp.r, rgbp.g), rgbp.b);
	__m512 L_cusp = cbrt_packed(_mm512_div_ps(_mm512_set1_ps(1.f), mx));
	return { L_cusp, _mm512_mul_ps(L_cusp, S_cusp) };
}

static PackedST512 to_ST(PackedLC512 cusp)
{
	__m512 S = _mm512_div_ps(cusp.C, cusp.L);
	__m512 T = _mm512_div_ps(cusp.C, _mm512_sub_ps(_mm512_set1_ps(1.f), cusp.L));
	return { S, T };
}

static __m512 find_gamut_intersection(__m512 a, __m512 b, __m512 L1, __m512 C1, __m512 L0,
                                       PackedLC512 cusp)
{
	__m512 dL = _mm512_sub_ps(L1, L0);
	__m512 dC = C1;
	__mmask16 lower_mask = _mm512_cmp_ps_mask(
		_mm512_fmsub_ps(dL, cusp.C, _mm512_mul_ps(_mm512_sub_ps(cusp.L, L0), C1)),
		_mm512_setzero_ps(), _CMP_LE_OS);

	__m512 t_lower = _mm512_div_ps(
		_mm512_mul_ps(cusp.C, L0),
		_mm512_fmadd_ps(C1, cusp.L, _mm512_mul_ps(cusp.C, _mm512_sub_ps(L0, L1))));

	__m512 t = _mm512_div_ps(
		_mm512_mul_ps(cusp.C, _mm512_sub_ps(L0, _mm512_set1_ps(1.f))),
		_mm512_fmadd_ps(C1, _mm512_sub_ps(cusp.L, _mm512_set1_ps(1.f)),
		                _mm512_mul_ps(cusp.C, _mm512_sub_ps(L0, L1))));

	__m512 k_l = _mm512_fmadd_ps(_mm512_set1_ps(+0.3963377774f), a, _mm512_mul_ps(_mm512_set1_ps(+0.2158037573f), b));
	__m512 k_m = _mm512_fmadd_ps(_mm512_set1_ps(-0.1055613458f), a, _mm512_mul_ps(_mm512_set1_ps(-0.0638541728f), b));
	__m512 k_s = _mm512_fmadd_ps(_mm512_set1_ps(-0.0894841775f), a, _mm512_mul_ps(_mm512_set1_ps(-1.2914855480f), b));

	__m512 l_dt = _mm512_fmadd_ps(dC, k_l, dL);
	__m512 m_dt = _mm512_fmadd_ps(dC, k_m, dL);
	__m512 s_dt = _mm512_fmadd_ps(dC, k_s, dL);

	__m512 L = _mm512_fmadd_ps(L0, _mm512_sub_ps(_mm512_set1_ps(1.f), t), _mm512_mul_ps(t, L1));
	__m512 C = _mm512_mul_ps(t, C1);

	__m512 l_ = _mm512_fmadd_ps(C, k_l, L);
	__m512 m_ = _mm512_fmadd_ps(C, k_m, L);
	__m512 s_ = _mm512_fmadd_ps(C, k_s, L);
	__m512 lp = _mm512_mul_ps(_mm512_mul_ps(l_, l_), l_);
	__m512 mp = _mm512_mul_ps(_mm512_mul_ps(m_, m_), m_);
	__m512 sp = _mm512_mul_ps(_mm512_mul_ps(s_, s_), s_);
	__m512 ldt  = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(3.f), l_dt), _mm512_mul_ps(l_, l_));
	__m512 mdt  = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(3.f), m_dt), _mm512_mul_ps(m_, m_));
	__m512 sdt  = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(3.f), s_dt), _mm512_mul_ps(s_, s_));
	__m512 ldt2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(6.f), _mm512_mul_ps(l_dt, l_dt)), l_);
	__m512 mdt2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(6.f), _mm512_mul_ps(m_dt, m_dt)), m_);
	__m512 sdt2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(6.f), _mm512_mul_ps(s_dt, s_dt)), s_);

	const __m512 FLTMAX = _mm512_set1_ps(FLT_MAX);
	auto channel_t = [&](float wl, float wm, float ws) {
		__m512 wlv = _mm512_set1_ps(wl), wmv = _mm512_set1_ps(wm), wsv = _mm512_set1_ps(ws);
		__m512 r  = _mm512_sub_ps(_mm512_fmadd_ps(wlv, lp, _mm512_fmadd_ps(wmv, mp, _mm512_mul_ps(wsv, sp))),
		                          _mm512_set1_ps(1.f));
		__m512 r1 = _mm512_fmadd_ps(wlv, ldt, _mm512_fmadd_ps(wmv, mdt, _mm512_mul_ps(wsv, sdt)));
		__m512 r2 = _mm512_fmadd_ps(wlv, ldt2,_mm512_fmadd_ps(wmv, mdt2,_mm512_mul_ps(wsv, sdt2)));
		__m512 u  = _mm512_div_ps(r1, _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), _mm512_mul_ps(r, r2),
		                                                _mm512_mul_ps(r1, r1)));
		__m512 candidate = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), r), u);
		__mmask16 valid = _mm512_cmp_ps_mask(u, _mm512_setzero_ps(), _CMP_GE_OS);
		return vsel(valid, candidate, FLTMAX);
	};
	__m512 t_r = channel_t(+4.0767416621f, -3.3077115913f, +0.2309699292f);
	__m512 t_g = channel_t(-1.2684380046f, +2.6097574011f, -0.3413193965f);
	__m512 t_b = channel_t(-0.0041960863f, -0.7034186147f, +1.7076147010f);
	__m512 dt_min = _mm512_min_ps(t_r, _mm512_min_ps(t_g, t_b));
	__m512 t_upper = _mm512_add_ps(t, dt_min);

	return vsel(lower_mask, t_lower, t_upper);
}

static PackedST512 get_ST_mid(__m512 a_, __m512 b_)
{
	auto mk_S = [&]() {
		__m512 inner = _mm512_fmadd_ps(_mm512_set1_ps(4.69891013f), a_,
		                _mm512_fmadd_ps(_mm512_set1_ps(5.38770819f), b_, _mm512_set1_ps(-4.24894561f)));
		inner = _mm512_fmadd_ps(inner, a_, _mm512_fmadd_ps(_mm512_set1_ps(-10.02301043f), b_, _mm512_set1_ps(-2.13704948f)));
		inner = _mm512_fmadd_ps(inner, a_, _mm512_fmadd_ps(_mm512_set1_ps( 1.75198401f), b_, _mm512_set1_ps(-2.19557347f)));
		inner = _mm512_fmadd_ps(inner, a_, _mm512_fmadd_ps(_mm512_set1_ps( 4.15901240f), b_, _mm512_set1_ps( 7.44778970f)));
		return _mm512_add_ps(_mm512_set1_ps(0.11516993f), _mm512_div_ps(_mm512_set1_ps(1.f), inner));
	};
	auto mk_T = [&]() {
		__m512 inner = _mm512_fmadd_ps(_mm512_set1_ps(-0.14661872f), a_,
		                _mm512_fmadd_ps(_mm512_set1_ps(-0.45399568f), b_, _mm512_set1_ps( 0.00299215f)));
		inner = _mm512_fmadd_ps(inner, a_, _mm512_fmadd_ps(_mm512_set1_ps( 0.61223990f), b_, _mm512_set1_ps(-0.27087943f)));
		inner = _mm512_fmadd_ps(inner, a_, _mm512_fmadd_ps(_mm512_set1_ps( 0.90148123f), b_, _mm512_set1_ps( 0.40370612f)));
		inner = _mm512_fmadd_ps(inner, a_, _mm512_fmadd_ps(_mm512_set1_ps(-0.68124379f), b_, _mm512_set1_ps( 1.61320320f)));
		return _mm512_add_ps(_mm512_set1_ps(0.11239642f), _mm512_div_ps(_mm512_set1_ps(1.f), inner));
	};
	return { mk_S(), mk_T() };
}

static PackedCs512 get_Cs(__m512 L, __m512 a_, __m512 b_)
{
	PackedLC512 cusp = find_cusp(a_, b_);
	__m512 C_max = find_gamut_intersection(a_, b_, L, _mm512_set1_ps(1.f), L, cusp);
	PackedST512 ST_max = to_ST(cusp);
	__m512 LSm = _mm512_mul_ps(L, ST_max.S);
	__m512 oneL = _mm512_sub_ps(_mm512_set1_ps(1.f), L);
	__m512 oTm = _mm512_mul_ps(oneL, ST_max.T);
	__m512 k = _mm512_div_ps(C_max, _mm512_min_ps(LSm, oTm));

	PackedST512 STmid = get_ST_mid(a_, b_);
	__m512 Ca_m = _mm512_mul_ps(L, STmid.S);
	__m512 Cb_m = _mm512_mul_ps(oneL, STmid.T);
	__m512 Ca4 = _mm512_mul_ps(_mm512_mul_ps(Ca_m, Ca_m), _mm512_mul_ps(Ca_m, Ca_m));
	__m512 Cb4 = _mm512_mul_ps(_mm512_mul_ps(Cb_m, Cb_m), _mm512_mul_ps(Cb_m, Cb_m));
	__m512 inv = _mm512_add_ps(_mm512_div_ps(_mm512_set1_ps(1.f), Ca4),
	                            _mm512_div_ps(_mm512_set1_ps(1.f), Cb4));
	__m512 C_mid = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.9f), k),
	                              _mm512_sqrt_ps(_mm512_sqrt_ps(_mm512_div_ps(_mm512_set1_ps(1.f), inv))));

	__m512 Ca0 = _mm512_mul_ps(L, _mm512_set1_ps(0.4f));
	__m512 Cb0 = _mm512_mul_ps(oneL, _mm512_set1_ps(0.8f));
	__m512 inv0 = _mm512_add_ps(_mm512_div_ps(_mm512_set1_ps(1.f), _mm512_mul_ps(Ca0, Ca0)),
	                             _mm512_div_ps(_mm512_set1_ps(1.f), _mm512_mul_ps(Cb0, Cb0)));
	__m512 C_0 = _mm512_sqrt_ps(_mm512_div_ps(_mm512_set1_ps(1.f), inv0));

	return { C_0, C_mid, C_max };
}

} // anon

PackedRGB PackedHSL::to_srgb() const
{
	const __m512 hh = this->h, ss = this->s, ll = this->l;
	const __m512 zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.f);

	__mmask16 is_one  = _mm512_cmp_ps_mask(ll, one, _CMP_EQ_OQ);
	__mmask16 is_zero = _mm512_cmp_ps_mask(ll, zero, _CMP_EQ_OQ);

	__m512 sin_v, cos_v;
	sincos_packed(_mm512_mul_ps(_mm512_set1_ps(2.f * pi), hh), &sin_v, &cos_v);
	__m512 a_ = cos_v, b_ = sin_v;
	__m512 L = toe_inv_packed(ll);

	PackedCs512 cs = get_Cs(L, a_, b_);
	__m512 C_0 = cs.C_0, C_mid = cs.C_mid, C_max = cs.C_max;

	const __m512 mid     = _mm512_set1_ps(0.8f);
	const __m512 mid_inv = _mm512_set1_ps(1.25f);

	__m512 t1 = _mm512_mul_ps(mid_inv, ss);
	__m512 k1a = _mm512_mul_ps(mid, C_0);
	__m512 k2a = _mm512_sub_ps(one, _mm512_div_ps(k1a, C_mid));
	__m512 C_a = _mm512_div_ps(_mm512_mul_ps(t1, k1a), _mm512_sub_ps(one, _mm512_mul_ps(k2a, t1)));

	__m512 t2 = _mm512_div_ps(_mm512_sub_ps(ss, mid), _mm512_sub_ps(one, mid));
	__m512 k0b = C_mid;
	__m512 k1b = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(one, mid),
	                                                        _mm512_mul_ps(C_mid, C_mid)),
	                                          _mm512_mul_ps(mid_inv, mid_inv)), C_0);
	__m512 k2b = _mm512_sub_ps(one, _mm512_div_ps(k1b, _mm512_sub_ps(C_max, C_mid)));
	__m512 C_b = _mm512_add_ps(k0b, _mm512_div_ps(_mm512_mul_ps(t2, k1b),
	                                               _mm512_sub_ps(one, _mm512_mul_ps(k2b, t2))));

	__mmask16 sm = _mm512_cmp_ps_mask(ss, mid, _CMP_LT_OS);
	__m512 C = vsel(sm, C_a, C_b);

	PackedLab labp{ L, _mm512_mul_ps(C, a_), _mm512_mul_ps(C, b_) };
	PackedRGB lin = labp.to_linear_srgb();

	__m512 R = srgb_transfer_packed(lin.r);
	__m512 G = srgb_transfer_packed(lin.g);
	__m512 B = srgb_transfer_packed(lin.b);

	R = vsel(is_one, one, vsel(is_zero, zero, R));
	G = vsel(is_one, one, vsel(is_zero, zero, G));
	B = vsel(is_one, one, vsel(is_zero, zero, B));
	return { R, G, B };
}

PackedHSL PackedRGB::to_okhsl() const
{
	PackedRGB lin{
		srgb_transfer_inv_packed(this->r),
		srgb_transfer_inv_packed(this->g),
		srgb_transfer_inv_packed(this->b),
	};
	PackedLab lab = lin.to_oklab();

	__m512 C = _mm512_sqrt_ps(_mm512_fmadd_ps(lab.a, lab.a, _mm512_mul_ps(lab.b, lab.b)));
	__m512 a_ = _mm512_div_ps(lab.a, C);
	__m512 b_ = _mm512_div_ps(lab.b, C);
	__m512 L = lab.L;
	const __m512 signbit = _mm512_set1_ps(-0.f);
	__m512 ang = atan2_packed(_mm512_xor_ps(lab.b, signbit), _mm512_xor_ps(lab.a, signbit));
	__m512 h_ = _mm512_fmadd_ps(_mm512_set1_ps(0.5f / pi), ang, _mm512_set1_ps(0.5f));

	PackedCs512 cs = get_Cs(L, a_, b_);
	__m512 C_0 = cs.C_0, C_mid = cs.C_mid, C_max = cs.C_max;

	const __m512 one = _mm512_set1_ps(1.f);
	const __m512 mid     = _mm512_set1_ps(0.8f);
	const __m512 mid_inv = _mm512_set1_ps(1.25f);

	__m512 k1a = _mm512_mul_ps(mid, C_0);
	__m512 k2a = _mm512_sub_ps(one, _mm512_div_ps(k1a, C_mid));
	__m512 ta  = _mm512_div_ps(C, _mm512_fmadd_ps(k2a, C, k1a));
	__m512 sa  = _mm512_mul_ps(ta, mid);

	__m512 k0b = C_mid;
	__m512 k1b = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(one, mid),
	                                                        _mm512_mul_ps(C_mid, C_mid)),
	                                          _mm512_mul_ps(mid_inv, mid_inv)), C_0);
	__m512 k2b = _mm512_sub_ps(one, _mm512_div_ps(k1b, _mm512_sub_ps(C_max, C_mid)));
	__m512 dC  = _mm512_sub_ps(C, k0b);
	__m512 tb  = _mm512_div_ps(dC, _mm512_fmadd_ps(k2b, dC, k1b));
	__m512 sb  = _mm512_add_ps(mid, _mm512_mul_ps(_mm512_sub_ps(one, mid), tb));

	__mmask16 use_a = _mm512_cmp_ps_mask(C, C_mid, _CMP_LT_OS);
	__m512 s_ = vsel(use_a, sa, sb);
	__m512 l_ = toe_packed(L);
	return { h_, s_, l_ };
}

PackedRGB PackedHSV::to_srgb() const
{
	const __m512 hh = this->h, ss = this->s, vv = this->v;
	__m512 sin_v, cos_v;
	sincos_packed(_mm512_mul_ps(_mm512_set1_ps(2.f * pi), hh), &sin_v, &cos_v);
	__m512 a_ = cos_v, b_ = sin_v;

	PackedLC512 cusp = find_cusp(a_, b_);
	PackedST512 ST_max = to_ST(cusp);
	__m512 S_max = ST_max.S, T_max = ST_max.T;
	const __m512 S0 = _mm512_set1_ps(0.5f);
	__m512 k = _mm512_sub_ps(_mm512_set1_ps(1.f), _mm512_div_ps(S0, S_max));

	__m512 denom = _mm512_fnmadd_ps(_mm512_mul_ps(T_max, k), ss, _mm512_add_ps(S0, T_max));
	__m512 L_v = _mm512_sub_ps(_mm512_set1_ps(1.f), _mm512_div_ps(_mm512_mul_ps(ss, S0), denom));
	__m512 C_v = _mm512_div_ps(_mm512_mul_ps(_mm512_mul_ps(ss, T_max), S0), denom);

	__m512 L = _mm512_mul_ps(vv, L_v);
	__m512 C = _mm512_mul_ps(vv, C_v);

	__m512 L_vt = toe_inv_packed(L_v);
	__m512 C_vt = _mm512_mul_ps(C_v, _mm512_div_ps(L_vt, L_v));
	__m512 L_new = toe_inv_packed(L);
	C = _mm512_mul_ps(C, _mm512_div_ps(L_new, L));
	L = L_new;

	PackedLab labp{ L_vt, _mm512_mul_ps(a_, C_vt), _mm512_mul_ps(b_, C_vt) };
	PackedRGB rgb_scale = labp.to_linear_srgb();
	__m512 mx = _mm512_max_ps(_mm512_max_ps(rgb_scale.r, rgb_scale.g),
	            _mm512_max_ps(rgb_scale.b, _mm512_setzero_ps()));
	__m512 scale_L = cbrt_packed(_mm512_div_ps(_mm512_set1_ps(1.f), mx));
	L = _mm512_mul_ps(L, scale_L);
	C = _mm512_mul_ps(C, scale_L);

	PackedLab final_lab{ L, _mm512_mul_ps(C, a_), _mm512_mul_ps(C, b_) };
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

	__m512 C = _mm512_sqrt_ps(_mm512_fmadd_ps(lab.a, lab.a, _mm512_mul_ps(lab.b, lab.b)));
	__m512 a_ = _mm512_div_ps(lab.a, C);
	__m512 b_ = _mm512_div_ps(lab.b, C);
	__m512 L = lab.L;
	const __m512 signbit = _mm512_set1_ps(-0.f);
	__m512 ang = atan2_packed(_mm512_xor_ps(lab.b, signbit), _mm512_xor_ps(lab.a, signbit));
	__m512 h_ = _mm512_fmadd_ps(_mm512_set1_ps(0.5f / pi), ang, _mm512_set1_ps(0.5f));

	PackedLC512 cusp = find_cusp(a_, b_);
	PackedST512 ST_max = to_ST(cusp);
	__m512 S_max = ST_max.S, T_max = ST_max.T;
	const __m512 S0 = _mm512_set1_ps(0.5f);
	__m512 k = _mm512_sub_ps(_mm512_set1_ps(1.f), _mm512_div_ps(S0, S_max));

	__m512 t = _mm512_div_ps(T_max, _mm512_fmadd_ps(L, T_max, C));
	__m512 L_v = _mm512_mul_ps(t, L);
	__m512 C_v = _mm512_mul_ps(t, C);
	__m512 L_vt = toe_inv_packed(L_v);
	__m512 C_vt = _mm512_mul_ps(C_v, _mm512_div_ps(L_vt, L_v));

	PackedLab labp{ L_vt, _mm512_mul_ps(a_, C_vt), _mm512_mul_ps(b_, C_vt) };
	PackedRGB rgb_scale = labp.to_linear_srgb();
	__m512 mx = _mm512_max_ps(_mm512_max_ps(rgb_scale.r, rgb_scale.g),
	            _mm512_max_ps(rgb_scale.b, _mm512_setzero_ps()));
	__m512 scale_L = cbrt_packed(_mm512_div_ps(_mm512_set1_ps(1.f), mx));
	L = _mm512_div_ps(L, scale_L);
	C = _mm512_div_ps(C, scale_L);

	C = _mm512_mul_ps(C, _mm512_div_ps(toe_packed(L), L));
	L = toe_packed(L);

	__m512 v = _mm512_div_ps(L, L_v);
	__m512 num = _mm512_mul_ps(_mm512_add_ps(S0, T_max), C_v);
	__m512 den = _mm512_fmadd_ps(_mm512_mul_ps(T_max, k), C_v, _mm512_mul_ps(T_max, S0));
	__m512 s = _mm512_div_ps(num, den);
	return { h_, s, v };
}

// =============================================================================
// AoS 16-pixel batch wrapper.  gather/scatter 사용.
// =============================================================================

namespace {

// 16개의 RGB4 (= 16 × 4 floats) 에서 R/G/B 각각을 SoA __m512 로 추출.
static inline PackedRGB load_AoS_RGB(const RGB4* in)
{
	const float* base = &in[0].r;
	const __m512i idx_r = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_g = _mm512_add_epi32(idx_r, _mm512_set1_epi32(1));
	const __m512i idx_b = _mm512_add_epi32(idx_r, _mm512_set1_epi32(2));
	__m512 R = _mm512_i32gather_ps(idx_r, base, 4);
	__m512 G = _mm512_i32gather_ps(idx_g, base, 4);
	__m512 B = _mm512_i32gather_ps(idx_b, base, 4);
	return { R, G, B };
}
static inline void store_AoS_RGB(PackedRGB p, RGB4* out)
{
	float* base = &out[0].r;
	const __m512i idx_r = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_g = _mm512_add_epi32(idx_r, _mm512_set1_epi32(1));
	const __m512i idx_b = _mm512_add_epi32(idx_r, _mm512_set1_epi32(2));
	const __m512i idx_p = _mm512_add_epi32(idx_r, _mm512_set1_epi32(3));
	_mm512_i32scatter_ps(base, idx_r, p.r, 4);
	_mm512_i32scatter_ps(base, idx_g, p.g, 4);
	_mm512_i32scatter_ps(base, idx_b, p.b, 4);
	_mm512_i32scatter_ps(base, idx_p, _mm512_setzero_ps(), 4);
}
static inline PackedLab load_AoS_Lab(const Lab4* in)
{
	const float* base = &in[0].L;
	const __m512i idx_L = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_a = _mm512_add_epi32(idx_L, _mm512_set1_epi32(1));
	const __m512i idx_b = _mm512_add_epi32(idx_L, _mm512_set1_epi32(2));
	return { _mm512_i32gather_ps(idx_L, base, 4),
	         _mm512_i32gather_ps(idx_a, base, 4),
	         _mm512_i32gather_ps(idx_b, base, 4) };
}
static inline void store_AoS_Lab(PackedLab p, Lab4* out)
{
	float* base = &out[0].L;
	const __m512i idx_L = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_a = _mm512_add_epi32(idx_L, _mm512_set1_epi32(1));
	const __m512i idx_b = _mm512_add_epi32(idx_L, _mm512_set1_epi32(2));
	const __m512i idx_p = _mm512_add_epi32(idx_L, _mm512_set1_epi32(3));
	_mm512_i32scatter_ps(base, idx_L, p.L, 4);
	_mm512_i32scatter_ps(base, idx_a, p.a, 4);
	_mm512_i32scatter_ps(base, idx_b, p.b, 4);
	_mm512_i32scatter_ps(base, idx_p, _mm512_setzero_ps(), 4);
}
static inline PackedHSL load_AoS_HSL(const HSL4* in)
{
	const float* base = &in[0].h;
	const __m512i idx_h = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_s = _mm512_add_epi32(idx_h, _mm512_set1_epi32(1));
	const __m512i idx_l = _mm512_add_epi32(idx_h, _mm512_set1_epi32(2));
	return { _mm512_i32gather_ps(idx_h, base, 4),
	         _mm512_i32gather_ps(idx_s, base, 4),
	         _mm512_i32gather_ps(idx_l, base, 4) };
}
static inline void store_AoS_HSL(PackedHSL p, HSL4* out)
{
	float* base = &out[0].h;
	const __m512i idx_h = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_s = _mm512_add_epi32(idx_h, _mm512_set1_epi32(1));
	const __m512i idx_l = _mm512_add_epi32(idx_h, _mm512_set1_epi32(2));
	const __m512i idx_p = _mm512_add_epi32(idx_h, _mm512_set1_epi32(3));
	_mm512_i32scatter_ps(base, idx_h, p.h, 4);
	_mm512_i32scatter_ps(base, idx_s, p.s, 4);
	_mm512_i32scatter_ps(base, idx_l, p.l, 4);
	_mm512_i32scatter_ps(base, idx_p, _mm512_setzero_ps(), 4);
}
static inline PackedHSV load_AoS_HSV(const HSV4* in)
{
	const float* base = &in[0].h;
	const __m512i idx_h = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_s = _mm512_add_epi32(idx_h, _mm512_set1_epi32(1));
	const __m512i idx_v = _mm512_add_epi32(idx_h, _mm512_set1_epi32(2));
	return { _mm512_i32gather_ps(idx_h, base, 4),
	         _mm512_i32gather_ps(idx_s, base, 4),
	         _mm512_i32gather_ps(idx_v, base, 4) };
}
static inline void store_AoS_HSV(PackedHSV p, HSV4* out)
{
	float* base = &out[0].h;
	const __m512i idx_h = _mm512_setr_epi32(0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60);
	const __m512i idx_s = _mm512_add_epi32(idx_h, _mm512_set1_epi32(1));
	const __m512i idx_v = _mm512_add_epi32(idx_h, _mm512_set1_epi32(2));
	const __m512i idx_p = _mm512_add_epi32(idx_h, _mm512_set1_epi32(3));
	_mm512_i32scatter_ps(base, idx_h, p.h, 4);
	_mm512_i32scatter_ps(base, idx_s, p.s, 4);
	_mm512_i32scatter_ps(base, idx_v, p.v, 4);
	_mm512_i32scatter_ps(base, idx_p, _mm512_setzero_ps(), 4);
}

} // anon

void linear_srgb_to_oklab_batch16(const RGB4* in, Lab4* out)
{ store_AoS_Lab(load_AoS_RGB(in).to_oklab(), out); }
void oklab_to_linear_srgb_batch16(const Lab4* in, RGB4* out)
{ store_AoS_RGB(load_AoS_Lab(in).to_linear_srgb(), out); }
void okhsl_to_srgb_batch16(const HSL4* in, RGB4* out)
{ store_AoS_RGB(load_AoS_HSL(in).to_srgb(), out); }
void srgb_to_okhsl_batch16(const RGB4* in, HSL4* out)
{ store_AoS_HSL(load_AoS_RGB(in).to_okhsl(), out); }
void okhsv_to_srgb_batch16(const HSV4* in, RGB4* out)
{ store_AoS_RGB(load_AoS_HSV(in).to_srgb(), out); }
void srgb_to_okhsv_batch16(const RGB4* in, HSV4* out)
{ store_AoS_HSV(load_AoS_RGB(in).to_okhsv(), out); }

} // namespace
