// bench_handsimd.cpp — 핸드 SIMD 단일-색상 변환이 scalar(현재 vec3)를 이길 수 있나?
//
//   (A) scalar (vec3, 현재 코드)
//   (B) hand SIMD: column-major broadcast, scalar cbrt 사용
//   (C) hand SIMD: column-major broadcast + packed cbrt (bit-hack + 2 Newton)
//
// 빌드:
//   g++ -std=c++17 -O3 -mavx2 -mfma -msse4.1 -flto -DNDEBUG \
//       ok_color.cpp bench_handsimd.cpp -o bench_handsimd
#include "ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace ok = ok_color;
using clock_t_ = std::chrono::steady_clock;

struct Sink { volatile float r=0,g=0,b=0; };
static Sink g_sink;

// =============================================================================
// 행렬: column-major. 각 행은 출력 4-lane(첫 3 lane 만 의미 있음, 4번째 = 0).
// =============================================================================
alignas(16) static const float M_LAB_TO_LMS_cols[3][4] = {
	// col0(L): (l_=1, m_=1, s_=1, 0)
	{ 1.f, 1.f, 1.f, 0.f },
	// col1(a): (+0.396, -0.105, -0.089, 0)
	{ +0.3963377774f, -0.1055613458f, -0.0894841775f, 0.f },
	// col2(b): (+0.215, -0.063, -1.291, 0)
	{ +0.2158037573f, -0.0638541728f, -1.2914855480f, 0.f },
};
alignas(16) static const float M_LMS_TO_RGB_cols[3][4] = {
	// col0(l): (r=+4.077, g=-1.268, b=-0.004, 0)
	{ +4.0767416621f, -1.2684380046f, -0.0041960863f, 0.f },
	// col1(m): (r=-3.308, g=+2.610, b=-0.703, 0)
	{ -3.3077115913f, +2.6097574011f, -0.7034186147f, 0.f },
	// col2(s): (r=+0.231, g=-0.341, b=+1.708, 0)
	{ +0.2309699292f, -0.3413193965f, +1.7076147010f, 0.f },
};
alignas(16) static const float M_RGB_TO_LMS_cols[3][4] = {
	// col0(r): (l=+0.412, m=+0.212, s=+0.088, 0)
	{ 0.4122214708f, 0.2119034982f, 0.0883024619f, 0.f },
	{ 0.5363325363f, 0.6806995451f, 0.2817188376f, 0.f },
	{ 0.0514459929f, 0.1073969566f, 0.6299787005f, 0.f },
};
alignas(16) static const float M_LMS_TO_LAB_cols[3][4] = {
	{ +0.2104542553f, +1.9779984951f, +0.0259040371f, 0.f },
	{ +0.7936177850f, -2.4285922050f, +0.7827717662f, 0.f },
	{ -0.0040720468f, +0.4505937099f, -0.8086757660f, 0.f },
};

// =============================================================================
// (B) Lab → linear sRGB, hand SIMD (cbrt 불필요).
// =============================================================================
static inline ok::RGB to_linear_srgb_simd(ok::Lab lab)
{
	__m128 col0 = _mm_load_ps(M_LAB_TO_LMS_cols[0]);
	__m128 col1 = _mm_load_ps(M_LAB_TO_LMS_cols[1]);
	__m128 col2 = _mm_load_ps(M_LAB_TO_LMS_cols[2]);

	__m128 L = _mm_set1_ps(lab.L);
	__m128 a = _mm_set1_ps(lab.a);
	__m128 b = _mm_set1_ps(lab.b);

	// lms_ = col0*L + col1*a + col2*b  (3 dot 동시).
	__m128 lms_ = _mm_fmadd_ps(col2, b, _mm_fmadd_ps(col1, a, _mm_mul_ps(col0, L)));

	// cube: x^3 = x*x*x.
	__m128 lms2 = _mm_mul_ps(lms_, lms_);
	__m128 lms  = _mm_mul_ps(lms2, lms_);

	// LMS → RGB: column-major broadcast.
	__m128 q0 = _mm_load_ps(M_LMS_TO_RGB_cols[0]);
	__m128 q1 = _mm_load_ps(M_LMS_TO_RGB_cols[1]);
	__m128 q2 = _mm_load_ps(M_LMS_TO_RGB_cols[2]);
	__m128 l_b = _mm_shuffle_ps(lms, lms, _MM_SHUFFLE(0,0,0,0));
	__m128 m_b = _mm_shuffle_ps(lms, lms, _MM_SHUFFLE(1,1,1,1));
	__m128 s_b = _mm_shuffle_ps(lms, lms, _MM_SHUFFLE(2,2,2,2));
	__m128 rgb = _mm_fmadd_ps(q2, s_b, _mm_fmadd_ps(q1, m_b, _mm_mul_ps(q0, l_b)));

	alignas(16) float out[4];
	_mm_store_ps(out, rgb);
	return ok::RGB{ out[0], out[1], out[2] };
}

// =============================================================================
// packed cbrt — bit-hack 초기치 + Newton 2회.  x ∈ [0, ~2] 범위에서 1e-6 정확도.
//   Newton:  y_{n+1} = (2*y + a/y²) / 3,  a = 입력.
//   bit hack: float i / 3 + 0x2A4137A8  (cbrt 의 알려진 근사 magic).
// =============================================================================
static inline __m128 cbrt_packed(__m128 x)
{
	__m128i ix = _mm_castps_si128(x);

	// integer divide by 3 — SSE 정수 div 가 없으니 (i * 0x55555556) >> 32 곱셈으로 대체.
	// 부호 비트가 없는(양수) 입력이라 가정 (cbrt 입력은 LMS = positive).
	__m128i mul_lo = _mm_mullo_epi32(ix, _mm_set1_epi32(0x55555556));
	(void)mul_lo;
	// _mm_mul_epi32 / _mm_mulhi_epi32 가 32-bit signed mul-high 를 직접 안 줘서,
	// 간단히 shift 로 근사: i/3 ≈ (i * 0x2AAA_AAAB) high32. 정확하지 않지만 근사.
	// 더 간단한 방법: bit-hack 의 "원래 형태" 인 i = i / 3 + bias 를 부동소수점으로.
	//   y0 = exp2( log2(x) / 3 ) 를 IEEE 비트 패턴 직접 조작으로 근사.
	//   float bits: (sign:1)(exp:8)(mantissa:23). exp = E + 127.
	//   cbrt 후의 exp ≈ E/3 + 127 = (i - 127*MANT) / 3 + 127*MANT = (i + 2*127*MANT) / 3
	// MANT_SHIFT = 1<<23.
	const int M = 1 << 23;
	const int bias = 127 * M;
	// i_y = (i_x + 2*bias) / 3 ≈ (i_x + 2*bias) * 0x55555556 >> 32  (양수 입력만)
	// SSE 에 32-bit signed mul-high 가 없으니 수동:
	//   t = i_x + 2*bias;  결과는 t * (1/3) 의 상위 32-bit.
	//   x86 에는 _mm_mul_epi32 (64-bit 결과 짝수 lane) 와 _mm_mul_epu32 (unsigned, 짝수 lane).
	__m128i t = _mm_add_epi32(ix, _mm_set1_epi32(2 * bias));

	// 4 lane 의 32-bit i / 3 을 64-bit 곱셈 두 번으로 처리 (짝수/홀수 lane 따로).
	__m128i mlo = _mm_mul_epu32(t, _mm_set1_epi32(0x55555556u));  // lane 0, 2
	__m128i thi = _mm_srli_epi64(t, 32);
	__m128i mhi = _mm_mul_epu32(thi, _mm_set1_epi32(0x55555556u)); // lane 1, 3

	// 결과 high32 만 취함.
	__m128i rlo = _mm_srli_epi64(mlo, 32);
	__m128i rhi = mhi;  // 이미 high 위치
	__m128i iy = _mm_or_si128(rlo, _mm_slli_epi64(rhi, 32));

	__m128 y = _mm_castsi128_ps(iy);

	// Newton 2회: y ← (2*y + x/y²) / 3.
	const __m128 third = _mm_set1_ps(1.f / 3.f);
	const __m128 two   = _mm_set1_ps(2.f);

	for (int n = 0; n < 2; ++n) {
		__m128 y2 = _mm_mul_ps(y, y);
		__m128 xy2 = _mm_div_ps(x, y2);
		y = _mm_mul_ps(_mm_fmadd_ps(two, y, xy2), third);
	}
	return y;
}

// =============================================================================
// (B) RGB → Oklab, hand SIMD with scalar cbrt.
// =============================================================================
static inline ok::Lab to_oklab_simd_scalarcbrt(ok::RGB rgb)
{
	__m128 c0 = _mm_load_ps(M_RGB_TO_LMS_cols[0]);
	__m128 c1 = _mm_load_ps(M_RGB_TO_LMS_cols[1]);
	__m128 c2 = _mm_load_ps(M_RGB_TO_LMS_cols[2]);

	__m128 r = _mm_set1_ps(rgb.r);
	__m128 g = _mm_set1_ps(rgb.g);
	__m128 b = _mm_set1_ps(rgb.b);

	__m128 lms = _mm_fmadd_ps(c2, b, _mm_fmadd_ps(c1, g, _mm_mul_ps(c0, r)));

	alignas(16) float a[4];
	_mm_store_ps(a, lms);
	a[0] = std::cbrt(a[0]);
	a[1] = std::cbrt(a[1]);
	a[2] = std::cbrt(a[2]);
	a[3] = 0.f;
	__m128 lms_ = _mm_load_ps(a);

	__m128 q0 = _mm_load_ps(M_LMS_TO_LAB_cols[0]);
	__m128 q1 = _mm_load_ps(M_LMS_TO_LAB_cols[1]);
	__m128 q2 = _mm_load_ps(M_LMS_TO_LAB_cols[2]);
	__m128 l_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(0,0,0,0));
	__m128 m_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(1,1,1,1));
	__m128 s_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(2,2,2,2));
	__m128 lab = _mm_fmadd_ps(q2, s_b, _mm_fmadd_ps(q1, m_b, _mm_mul_ps(q0, l_b)));

	alignas(16) float out[4];
	_mm_store_ps(out, lab);
	return ok::Lab{ out[0], out[1], out[2] };
}

// =============================================================================
// (C) RGB → Oklab, full SIMD with packed cbrt.
// =============================================================================
static inline ok::Lab to_oklab_simd_packed(ok::RGB rgb)
{
	__m128 c0 = _mm_load_ps(M_RGB_TO_LMS_cols[0]);
	__m128 c1 = _mm_load_ps(M_RGB_TO_LMS_cols[1]);
	__m128 c2 = _mm_load_ps(M_RGB_TO_LMS_cols[2]);
	__m128 r = _mm_set1_ps(rgb.r);
	__m128 g = _mm_set1_ps(rgb.g);
	__m128 b = _mm_set1_ps(rgb.b);
	__m128 lms = _mm_fmadd_ps(c2, b, _mm_fmadd_ps(c1, g, _mm_mul_ps(c0, r)));

	__m128 lms_ = cbrt_packed(lms);

	__m128 q0 = _mm_load_ps(M_LMS_TO_LAB_cols[0]);
	__m128 q1 = _mm_load_ps(M_LMS_TO_LAB_cols[1]);
	__m128 q2 = _mm_load_ps(M_LMS_TO_LAB_cols[2]);
	__m128 l_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(0,0,0,0));
	__m128 m_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(1,1,1,1));
	__m128 s_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(2,2,2,2));
	__m128 lab = _mm_fmadd_ps(q2, s_b, _mm_fmadd_ps(q1, m_b, _mm_mul_ps(q0, l_b)));

	alignas(16) float out[4];
	_mm_store_ps(out, lab);
	return ok::Lab{ out[0], out[1], out[2] };
}

// =============================================================================

template <typename F>
static double bench_min(int rounds, int iters, F&& f)
{
	double best = 1e18;
	for (int r = 0; r < rounds; ++r) {
		auto t0 = clock_t_::now();
		f(iters);
		auto t1 = clock_t_::now();
		double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
		best = std::min(best, ns / iters);
	}
	return best;
}

int main()
{
	constexpr int N = 4096;
	constexpr int ROUNDS = 9;
	constexpr int ITERS = 4'000'000;

	std::vector<ok::RGB> rgb_in;
	std::vector<ok::Lab> lab_in;
	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	for (int i = 0; i < N; ++i) {
		ok::RGB rgb{U(rng), U(rng), U(rng)};
		rgb_in.push_back(rgb);
		lab_in.push_back(rgb.to_oklab());
	}

	// 정확성 sanity check.
	{
		ok::RGB t = rgb_in[0];
		ok::Lab a_scalar = t.to_oklab();
		ok::Lab a_simdsc = to_oklab_simd_scalarcbrt(t);
		ok::Lab a_simdpk = to_oklab_simd_packed(t);
		ok::Lab b_scalar = a_scalar;
		ok::RGB r_scalar = b_scalar.to_linear_srgb();
		ok::RGB r_simd   = to_linear_srgb_simd(b_scalar);
		std::printf("정합성 (idx 0):\n");
		std::printf("  to_oklab     scalar=(%.6f,%.6f,%.6f) simd_sc=(%.6f,%.6f,%.6f) simd_pk=(%.6f,%.6f,%.6f)\n",
			a_scalar.L, a_scalar.a, a_scalar.b,
			a_simdsc.L, a_simdsc.a, a_simdsc.b,
			a_simdpk.L, a_simdpk.a, a_simdpk.b);
		std::printf("  to_lin_srgb  scalar=(%.6f,%.6f,%.6f) simd=(%.6f,%.6f,%.6f)\n\n",
			r_scalar.r, r_scalar.g, r_scalar.b,
			r_simd.r, r_simd.g, r_simd.b);
	}

	std::printf("입력풀=%d, 라운드=%d, 반복=%d (min ns/op)\n\n", N, ROUNDS, ITERS);
	std::printf("%-45s | %12s | %10s\n", "variant", "ns/op", "vs scalar");
	std::printf("---------------------------------------------+--------------+-----------\n");

	// to_linear_srgb (cbrt 없음 — 순수 SIMD vs 순수 scalar 직접 비교)
	double t_scalar_lab = bench_min(ROUNDS, ITERS, [&](int it){
		float rs=0,gs=0,bs=0;
		for (int i = 0; i < it; ++i) {
			ok::RGB R = lab_in[i & (N-1)].to_linear_srgb();
			rs+=R.r; gs+=R.g; bs+=R.b;
		}
		g_sink.r=rs; g_sink.g=gs; g_sink.b=bs;
	});
	std::printf("%-45s | %12.3f | %9.3fx\n",
		"oklab→linear_srgb scalar (vec3)", t_scalar_lab, 1.0);

	double t_simd_lab = bench_min(ROUNDS, ITERS, [&](int it){
		float rs=0,gs=0,bs=0;
		for (int i = 0; i < it; ++i) {
			ok::RGB R = to_linear_srgb_simd(lab_in[i & (N-1)]);
			rs+=R.r; gs+=R.g; bs+=R.b;
		}
		g_sink.r=rs; g_sink.g=gs; g_sink.b=bs;
	});
	std::printf("%-45s | %12.3f | %9.3fx\n",
		"oklab→linear_srgb hand SIMD", t_simd_lab, t_scalar_lab / t_simd_lab);

	// to_oklab (cbrt 있음)
	double t_scalar_rgb = bench_min(ROUNDS, ITERS, [&](int it){
		float rs=0,gs=0,bs=0;
		for (int i = 0; i < it; ++i) {
			ok::Lab L = rgb_in[i & (N-1)].to_oklab();
			rs+=L.L; gs+=L.a; bs+=L.b;
		}
		g_sink.r=rs; g_sink.g=gs; g_sink.b=bs;
	});
	std::printf("\n%-45s | %12.3f | %9.3fx\n",
		"linear_srgb→oklab scalar (vec3)", t_scalar_rgb, 1.0);

	double t_simdscbrt = bench_min(ROUNDS, ITERS, [&](int it){
		float rs=0,gs=0,bs=0;
		for (int i = 0; i < it; ++i) {
			ok::Lab L = to_oklab_simd_scalarcbrt(rgb_in[i & (N-1)]);
			rs+=L.L; gs+=L.a; bs+=L.b;
		}
		g_sink.r=rs; g_sink.g=gs; g_sink.b=bs;
	});
	std::printf("%-45s | %12.3f | %9.3fx\n",
		"linear_srgb→oklab SIMD (scalar cbrt)",
		t_simdscbrt, t_scalar_rgb / t_simdscbrt);

	double t_simdpk = bench_min(ROUNDS, ITERS, [&](int it){
		float rs=0,gs=0,bs=0;
		for (int i = 0; i < it; ++i) {
			ok::Lab L = to_oklab_simd_packed(rgb_in[i & (N-1)]);
			rs+=L.L; gs+=L.a; bs+=L.b;
		}
		g_sink.r=rs; g_sink.g=gs; g_sink.b=bs;
	});
	std::printf("%-45s | %12.3f | %9.3fx\n",
		"linear_srgb→oklab SIMD (packed cbrt)",
		t_simdpk, t_scalar_rgb / t_simdpk);

	std::printf("\n[sink] r=%g g=%g b=%g\n",
		(double)g_sink.r, (double)g_sink.g, (double)g_sink.b);
	return 0;
}
