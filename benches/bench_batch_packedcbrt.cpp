// bench_batch_packedcbrt.cpp — 4-pixel batch SIMD + packed cbrt.
//
//   (A) scalar baseline (4 픽셀 순서대로 ok::RGB::to_oklab)
//   (E) SSE4 batch, scalar cbrt   (이전에 측정: ~22 ns/pixel)
//   (H) SSE4 batch, packed cbrt   ← 새로 측정.  cbrt 병목 분쇄 시도.
//
// 빌드:
//   g++ -std=c++17 -O3 -mavx2 -mfma -msse4.1 -flto -DNDEBUG \
//       ok_color.cpp bench_batch_packedcbrt.cpp -o bench_batch_packedcbrt
#include "ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace ok = ok_color;
using clock_t_ = std::chrono::steady_clock;

struct alignas(16) RGB4 { float r, g, b, _pad; };
struct alignas(16) Lab4 { float L, a, b, _pad; };

static volatile float g_sink_L = 0, g_sink_a = 0, g_sink_b = 0;

// =============================================================================
// packed_cbrt — 4 lane 동시 cbrt.  bit-hack 초기치 + Newton 2회.
// 입력 가정: x > 0 (모두 양수). LMS = positive RGB·matrix coefs (모두 양수) 이므로 OK.
// 정확도: ~1e-6 (float 한계 근처).
// =============================================================================
static inline __m128i div_by_3_u32(__m128i x)
{
	// x / 3 (unsigned, lane-wise) via multiply-high trick.
	//   x/3 = (x * 0xAAAAAAAB) >> 33   (since 0xAAAAAAAB = ceil(2^33/3))
	// _mm_mul_epu32 는 32×32→64 곱을 짝수 lane (0, 2) 에 대해서만 수행.
	const __m128i magic = _mm_set1_epi32(0xAAAAAAABu);

	__m128i even = _mm_mul_epu32(x, magic);                 // lane 0, 2 → 64-bit prod
	__m128i odd  = _mm_mul_epu32(_mm_srli_epi64(x, 32), magic); // lane 1, 3 → 64-bit prod

	// 각각 33 비트 shift (high 32 + 1 추가) 후 lane 재구성.
	__m128i even_q = _mm_srli_epi64(even, 33);   // (q0, _, q2, _) — q in low 32
	__m128i odd_q  = _mm_srli_epi64(odd,  33);   // (q1, _, q3, _)

	// odd_q 를 32 bit 위로 올려 even_q 와 OR.
	odd_q = _mm_slli_epi64(odd_q, 32);
	return _mm_or_si128(even_q, odd_q);
}

static inline __m128 cbrt_packed(__m128 x)
{
	// 1) bit-hack 초기치.
	//   i_y = (i_x + 2*bias) / 3,   bias = 127 * 2^23 = 0x3F800000
	//   2*bias = 0x7F000000.
	__m128i ix = _mm_castps_si128(x);
	__m128i t  = _mm_add_epi32(ix, _mm_set1_epi32(0x7F000000));
	__m128i iy = div_by_3_u32(t);
	__m128 y   = _mm_castsi128_ps(iy);

	// 2) Newton 2회.  y ← (2y + x/y²) / 3.
	const __m128 third = _mm_set1_ps(1.f / 3.f);
	const __m128 two   = _mm_set1_ps(2.f);
	for (int n = 0; n < 2; ++n) {
		__m128 y2  = _mm_mul_ps(y, y);
		__m128 xy2 = _mm_div_ps(x, y2);
		y = _mm_mul_ps(_mm_fmadd_ps(two, y, xy2), third);
	}
	return y;
}

// =============================================================================
// (A) scalar 4-at-a-time.
// =============================================================================
static inline void convert4_scalar(const RGB4* in, Lab4* out)
{
	for (int i = 0; i < 4; ++i) {
		ok::Lab L = ok::RGB{in[i].r, in[i].g, in[i].b}.to_oklab();
		out[i] = { L.L, L.a, L.b, 0.f };
	}
}

// =============================================================================
// (E) SSE4 4-batch with scalar cbrt — 비교 기준.
// =============================================================================
static inline void convert4_batch_scalarcbrt(const RGB4* in, Lab4* out)
{
	__m128 v0 = _mm_load_ps(&in[0].r), v1 = _mm_load_ps(&in[1].r);
	__m128 v2 = _mm_load_ps(&in[2].r), v3 = _mm_load_ps(&in[3].r);
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	__m128 R = v0, G = v1, B = v2;

	__m128 Ll = _mm_fmadd_ps(_mm_set1_ps(0.0514459929f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.5363325363f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.4122214708f), R)));
	__m128 Ml = _mm_fmadd_ps(_mm_set1_ps(0.1073969566f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.6806995451f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.2119034982f), R)));
	__m128 Sl = _mm_fmadd_ps(_mm_set1_ps(0.6299787005f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.2817188376f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.0883024619f), R)));

	alignas(16) float La[4], Ma[4], Sa[4];
	_mm_store_ps(La, Ll); _mm_store_ps(Ma, Ml); _mm_store_ps(Sa, Sl);
	for (int i = 0; i < 4; ++i) {
		La[i] = std::cbrt(La[i]);
		Ma[i] = std::cbrt(Ma[i]);
		Sa[i] = std::cbrt(Sa[i]);
	}
	__m128 Lp = _mm_load_ps(La), Mp = _mm_load_ps(Ma), Sp = _mm_load_ps(Sa);

	__m128 Lab_L = _mm_fmadd_ps(_mm_set1_ps(-0.0040720468f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps( 0.7936177850f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(0.2104542553f), Lp)));
	__m128 Lab_a = _mm_fmadd_ps(_mm_set1_ps( 0.4505937099f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps(-2.4285922050f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(1.9779984951f), Lp)));
	__m128 Lab_b = _mm_fmadd_ps(_mm_set1_ps(-0.8086757660f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps( 0.7827717662f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(0.0259040371f), Lp)));

	__m128 z = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(Lab_L, Lab_a, Lab_b, z);
	_mm_store_ps(&out[0].L, Lab_L);
	_mm_store_ps(&out[1].L, Lab_a);
	_mm_store_ps(&out[2].L, Lab_b);
	_mm_store_ps(&out[3].L, z);
}

// =============================================================================
// (H) SSE4 4-batch with packed cbrt — 핵심 실험.
// =============================================================================
static inline void convert4_batch_packedcbrt(const RGB4* in, Lab4* out)
{
	__m128 v0 = _mm_load_ps(&in[0].r), v1 = _mm_load_ps(&in[1].r);
	__m128 v2 = _mm_load_ps(&in[2].r), v3 = _mm_load_ps(&in[3].r);
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	__m128 R = v0, G = v1, B = v2;

	__m128 Ll = _mm_fmadd_ps(_mm_set1_ps(0.0514459929f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.5363325363f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.4122214708f), R)));
	__m128 Ml = _mm_fmadd_ps(_mm_set1_ps(0.1073969566f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.6806995451f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.2119034982f), R)));
	__m128 Sl = _mm_fmadd_ps(_mm_set1_ps(0.6299787005f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.2817188376f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.0883024619f), R)));

	// ★ packed cbrt: 4 픽셀 × 3 성분 = 3 packed call.
	__m128 Lp = cbrt_packed(Ll);
	__m128 Mp = cbrt_packed(Ml);
	__m128 Sp = cbrt_packed(Sl);

	__m128 Lab_L = _mm_fmadd_ps(_mm_set1_ps(-0.0040720468f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps( 0.7936177850f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(0.2104542553f), Lp)));
	__m128 Lab_a = _mm_fmadd_ps(_mm_set1_ps( 0.4505937099f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps(-2.4285922050f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(1.9779984951f), Lp)));
	__m128 Lab_b = _mm_fmadd_ps(_mm_set1_ps(-0.8086757660f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps( 0.7827717662f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(0.0259040371f), Lp)));

	__m128 z = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(Lab_L, Lab_a, Lab_b, z);
	_mm_store_ps(&out[0].L, Lab_L);
	_mm_store_ps(&out[1].L, Lab_a);
	_mm_store_ps(&out[2].L, Lab_b);
	_mm_store_ps(&out[3].L, z);
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

	std::vector<RGB4, std::allocator<RGB4>> rgb_in(N);
	std::vector<Lab4> lab_out(N);
	{
		std::mt19937 rng(0xBEEF);
		std::uniform_real_distribution<float> U(0.f, 1.f);
		for (int i = 0; i < N; ++i) rgb_in[i] = { U(rng), U(rng), U(rng), 0.f };
	}

	// 정합성 검사 — 한 batch 의 4 픽셀을 세 방식으로 비교.
	{
		Lab4 sa[4], sb[4], sc[4];
		convert4_scalar(&rgb_in[0], sa);
		convert4_batch_scalarcbrt(&rgb_in[0], sb);
		convert4_batch_packedcbrt(&rgb_in[0], sc);
		std::printf("정합성 (각 4 픽셀):\n");
		for (int i = 0; i < 4; ++i) {
			std::printf("  px[%d]  scalar=(%.6f,%.6f,%.6f)\n", i, sa[i].L, sa[i].a, sa[i].b);
			std::printf("         scbrt =(%.6f,%.6f,%.6f)  Δ=(%.1e,%.1e,%.1e)\n",
				sb[i].L, sb[i].a, sb[i].b,
				sb[i].L - sa[i].L, sb[i].a - sa[i].a, sb[i].b - sa[i].b);
			std::printf("         pcbrt =(%.6f,%.6f,%.6f)  Δ=(%.1e,%.1e,%.1e)\n",
				sc[i].L, sc[i].a, sc[i].b,
				sc[i].L - sa[i].L, sc[i].a - sa[i].a, sc[i].b - sa[i].b);
		}
	}

	std::printf("\n입력풀=%d 픽셀, 라운드=%d, 처리 픽셀수=%d (min ns/픽셀)\n\n",
		N, ROUNDS, ITERS);
	std::printf("%-40s | %12s | %10s\n", "variant", "ns/pixel", "speedup");
	std::printf("-----------------------------------------+--------------+-----------\n");

	double t_scalar = bench_min(ROUNDS, ITERS / 4, [&](int it){
		float Ls=0, as=0, bs=0;
		const int mask = (N/4) - 1;
		for (int i = 0; i < it; ++i) {
			int base = (i & mask) * 4;
			convert4_scalar(&rgb_in[base], &lab_out[base]);
			Ls += lab_out[base].L; as += lab_out[base].a; bs += lab_out[base].b;
		}
		g_sink_L = Ls; g_sink_a = as; g_sink_b = bs;
	}) / 4.0;
	std::printf("%-40s | %12.3f | %9.3fx\n",
		"(A) scalar 4-at-a-time", t_scalar, 1.0);

	double t_batch_sc = bench_min(ROUNDS, ITERS / 4, [&](int it){
		float Ls=0, as=0, bs=0;
		const int mask = (N/4) - 1;
		for (int i = 0; i < it; ++i) {
			int base = (i & mask) * 4;
			convert4_batch_scalarcbrt(&rgb_in[base], &lab_out[base]);
			Ls += lab_out[base].L; as += lab_out[base].a; bs += lab_out[base].b;
		}
		g_sink_L = Ls; g_sink_a = as; g_sink_b = bs;
	}) / 4.0;
	std::printf("%-40s | %12.3f | %9.3fx\n",
		"(E) SSE4 batch + scalar cbrt", t_batch_sc, t_scalar / t_batch_sc);

	double t_batch_pk = bench_min(ROUNDS, ITERS / 4, [&](int it){
		float Ls=0, as=0, bs=0;
		const int mask = (N/4) - 1;
		for (int i = 0; i < it; ++i) {
			int base = (i & mask) * 4;
			convert4_batch_packedcbrt(&rgb_in[base], &lab_out[base]);
			Ls += lab_out[base].L; as += lab_out[base].a; bs += lab_out[base].b;
		}
		g_sink_L = Ls; g_sink_a = as; g_sink_b = bs;
	}) / 4.0;
	std::printf("%-40s | %12.3f | %9.3fx  ★\n",
		"(H) SSE4 batch + packed cbrt", t_batch_pk, t_scalar / t_batch_pk);

	std::printf("\n[sink] L=%g a=%g b=%g\n",
		(double)g_sink_L, (double)g_sink_a, (double)g_sink_b);
	return 0;
}
