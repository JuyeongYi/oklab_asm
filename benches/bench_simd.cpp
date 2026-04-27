// -----------------------------------------------------------------------------
// bench_simd.cpp — 진짜 SIMD dot 이 단일 색상 변환에서 더 빠른가?
//
// 실험: linear_srgb → Oklab 의 첫 단계 (RGB→LMS, 3-dot 행렬-벡터곱) 을
//   (A) 현재 scalar dot 구현
//   (B) 16-byte 정렬 padded vec4 + _mm_dp_ps (DPPS)
//   (C) padded vec4 + mulps + horizontal sum (DPPS 는 latency 가 길다)
//   (D) padded vec4 + 3개 mulps + hadd (한 번에 3 dot 동시)
// 으로 각각 짜서 비교.
//
// 빌드:
//   g++ -std=c++17 -O3 -mavx2 -mfma -msse4.1 -flto -DNDEBUG \
//       ok_color.cpp bench_simd.cpp -o bench_simd
// -----------------------------------------------------------------------------

#include "ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <immintrin.h>

namespace ok = ok_color;
using clock_t_ = std::chrono::steady_clock;

struct Sink { volatile float L = 0, a = 0, b = 0; };
static Sink g_sink;

// =============================================================================
// (A) 현재 scalar dot 구현 — 비교 기준선.
// =============================================================================
static ok::Lab convert_A_scalar(ok::RGB rgb)
{
	return rgb.to_oklab();
}

// =============================================================================
// (B) padded vec4 + _mm_dp_ps (DPPS, SSE4.1).
//   각 행을 4-wide(마지막 0 padding) 로 만들고, DPPS 한 번으로 dot.
//   DPPS 의 imm8 = 0x71 → 처음 3 lane mul, 결과를 첫 lane 에 합산.
// =============================================================================

alignas(16) static const float M_RGB_TO_LMS_padded[3][4] = {
	{ 0.4122214708f, 0.5363325363f, 0.0514459929f, 0.f },
	{ 0.2119034982f, 0.6806995451f, 0.1073969566f, 0.f },
	{ 0.0883024619f, 0.2817188376f, 0.6299787005f, 0.f },
};
alignas(16) static const float M_LMS_TO_LAB_padded[3][4] = {
	{ 0.2104542553f,  0.7936177850f, -0.0040720468f, 0.f },
	{ 1.9779984951f, -2.4285922050f,  0.4505937099f, 0.f },
	{ 0.0259040371f,  0.7827717662f, -0.8086757660f, 0.f },
};

static ok::Lab convert_B_dpps(ok::RGB rgb)
{
	__m128 v = _mm_set_ps(0.f, rgb.b, rgb.g, rgb.r);

	__m128 r0 = _mm_load_ps(M_RGB_TO_LMS_padded[0]);
	__m128 r1 = _mm_load_ps(M_RGB_TO_LMS_padded[1]);
	__m128 r2 = _mm_load_ps(M_RGB_TO_LMS_padded[2]);

	float l = _mm_cvtss_f32(_mm_dp_ps(r0, v, 0x71));
	float m = _mm_cvtss_f32(_mm_dp_ps(r1, v, 0x71));
	float s = _mm_cvtss_f32(_mm_dp_ps(r2, v, 0x71));

	float l_ = std::cbrt(l), m_ = std::cbrt(m), s_ = std::cbrt(s);
	__m128 v2 = _mm_set_ps(0.f, s_, m_, l_);

	__m128 q0 = _mm_load_ps(M_LMS_TO_LAB_padded[0]);
	__m128 q1 = _mm_load_ps(M_LMS_TO_LAB_padded[1]);
	__m128 q2 = _mm_load_ps(M_LMS_TO_LAB_padded[2]);

	float L = _mm_cvtss_f32(_mm_dp_ps(q0, v2, 0x71));
	float a = _mm_cvtss_f32(_mm_dp_ps(q1, v2, 0x71));
	float b = _mm_cvtss_f32(_mm_dp_ps(q2, v2, 0x71));
	return { L, a, b };
}

// =============================================================================
// (C) padded vec4 + mulps + horizontal sum (3 mul, 2 hadd 형식).
//   DPPS 는 latency 11~14 cycle 이라 mulps + hadd_ps 로 풀어쓰는 게 빠를 수 있다.
// =============================================================================

static inline float horiz_add3(__m128 v)
{
	// v = (x, y, z, 0).
	// shuffle + add 두 번으로 첫 lane 에 x+y+z.
	__m128 t = _mm_movehdup_ps(v);             // (y, y, w, w)
	__m128 sum = _mm_add_ps(v, t);             // (x+y, .., z+w, ..)
	t = _mm_movehl_ps(sum, sum);               // (z+w, .., .., ..)
	sum = _mm_add_ss(sum, t);                  // (x+y + z+w, ..) — w==0 이라 z 만 더해짐
	return _mm_cvtss_f32(sum);
}

static ok::Lab convert_C_mulps_hadd(ok::RGB rgb)
{
	__m128 v = _mm_set_ps(0.f, rgb.b, rgb.g, rgb.r);

	__m128 r0 = _mm_load_ps(M_RGB_TO_LMS_padded[0]);
	__m128 r1 = _mm_load_ps(M_RGB_TO_LMS_padded[1]);
	__m128 r2 = _mm_load_ps(M_RGB_TO_LMS_padded[2]);

	float l = horiz_add3(_mm_mul_ps(r0, v));
	float m = horiz_add3(_mm_mul_ps(r1, v));
	float s = horiz_add3(_mm_mul_ps(r2, v));

	float l_ = std::cbrt(l), m_ = std::cbrt(m), s_ = std::cbrt(s);
	__m128 v2 = _mm_set_ps(0.f, s_, m_, l_);

	__m128 q0 = _mm_load_ps(M_LMS_TO_LAB_padded[0]);
	__m128 q1 = _mm_load_ps(M_LMS_TO_LAB_padded[1]);
	__m128 q2 = _mm_load_ps(M_LMS_TO_LAB_padded[2]);

	float L = horiz_add3(_mm_mul_ps(q0, v2));
	float a = horiz_add3(_mm_mul_ps(q1, v2));
	float b = horiz_add3(_mm_mul_ps(q2, v2));
	return { L, a, b };
}

// =============================================================================
// (D) "한 번에 3 dot 동시" — column-major 트릭.
//   M*v 를 3개 컬럼 벡터의 가중합으로 분해:
//     M*v = col0 * v.r + col1 * v.g + col2 * v.b
//   각 col 이 4-wide 라 mulps × 3 + addps × 2 로 끝.
//   3 dot 이 정말로 packed 로 실행됨.
// =============================================================================

alignas(16) static const float M_RGB_TO_LMS_cols[3][4] = {
	// col0: l_col_r, m_col_r, s_col_r, 0
	{ 0.4122214708f, 0.2119034982f, 0.0883024619f, 0.f },
	// col1: l_col_g, m_col_g, s_col_g, 0
	{ 0.5363325363f, 0.6806995451f, 0.2817188376f, 0.f },
	// col2: l_col_b, m_col_b, s_col_b, 0
	{ 0.0514459929f, 0.1073969566f, 0.6299787005f, 0.f },
};
alignas(16) static const float M_LMS_TO_LAB_cols[3][4] = {
	{ 0.2104542553f,  1.9779984951f,  0.0259040371f, 0.f },
	{ 0.7936177850f, -2.4285922050f,  0.7827717662f, 0.f },
	{-0.0040720468f,  0.4505937099f, -0.8086757660f, 0.f },
};

static ok::Lab convert_D_packed(ok::RGB rgb)
{
	__m128 c0 = _mm_load_ps(M_RGB_TO_LMS_cols[0]);
	__m128 c1 = _mm_load_ps(M_RGB_TO_LMS_cols[1]);
	__m128 c2 = _mm_load_ps(M_RGB_TO_LMS_cols[2]);

	__m128 r = _mm_set1_ps(rgb.r);
	__m128 g = _mm_set1_ps(rgb.g);
	__m128 b = _mm_set1_ps(rgb.b);

	// lms = c0*r + c1*g + c2*b — 3 dot 이 packed lane 으로 동시 실행.
	__m128 lms = _mm_fmadd_ps(c2, b, _mm_fmadd_ps(c1, g, _mm_mul_ps(c0, r)));

	// cbrt 는 4-wide 로 해줄 게 마땅찮아 scalar.
	alignas(16) float lms_arr[4];
	_mm_store_ps(lms_arr, lms);
	__m128 lms_ = _mm_set_ps(0.f,
		std::cbrt(lms_arr[2]), std::cbrt(lms_arr[1]), std::cbrt(lms_arr[0]));

	__m128 q0 = _mm_load_ps(M_LMS_TO_LAB_cols[0]);
	__m128 q1 = _mm_load_ps(M_LMS_TO_LAB_cols[1]);
	__m128 q2 = _mm_load_ps(M_LMS_TO_LAB_cols[2]);

	__m128 l_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(0,0,0,0));
	__m128 m_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(1,1,1,1));
	__m128 s_b = _mm_shuffle_ps(lms_, lms_, _MM_SHUFFLE(2,2,2,2));

	__m128 lab = _mm_fmadd_ps(q2, s_b, _mm_fmadd_ps(q1, m_b, _mm_mul_ps(q0, l_b)));

	alignas(16) float out[4];
	_mm_store_ps(out, lab);
	return { out[0], out[1], out[2] };
}

// =============================================================================

template <typename F>
double bench_min_ns(int rounds, int iters, F&& f)
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
	constexpr int N      = 4096;
	constexpr int ROUNDS = 9;
	constexpr int ITERS  = 2'000'000;

	std::vector<ok::RGB> rgb_in;
	{
		std::mt19937 rng(0xBEEF);
		std::uniform_real_distribution<float> U(0.f, 1.f);
		for (int i = 0; i < N; ++i) rgb_in.push_back({U(rng), U(rng), U(rng)});
	}

	std::printf("입력=%d, 라운드=%d, 반복=%d (min ns/op)\n\n", N, ROUNDS, ITERS);
	std::printf("%-30s | %12s\n", "variant", "ns/op");
	std::printf("------------------------------+--------------\n");

	auto run = [&](const char* name, ok::Lab(*fn)(ok::RGB)) {
		double t = bench_min_ns(ROUNDS, ITERS, [&](int it){
			float Ls=0, as=0, bs=0;
			for (int i = 0; i < it; ++i) {
				ok::Lab l = fn(rgb_in[i & (N-1)]);
				Ls += l.L; as += l.a; bs += l.b;
			}
			g_sink.L = Ls; g_sink.a = as; g_sink.b = bs;
		});
		std::printf("%-30s | %12.3f\n", name, t);
	};

	run("(A) scalar dot (현재 코드)",     convert_A_scalar);
	run("(B) DPPS  (3× _mm_dp_ps)",         convert_B_dpps);
	run("(C) mulps + horizontal add",       convert_C_mulps_hadd);
	run("(D) 3× col mulps  (한 번에 3dot)", convert_D_packed);

	std::printf("\n[sink] L=%g a=%g b=%g\n",
		(double)g_sink.L, (double)g_sink.a, (double)g_sink.b);
	return 0;
}
