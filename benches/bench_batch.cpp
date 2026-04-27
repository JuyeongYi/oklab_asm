// -----------------------------------------------------------------------------
// bench_batch.cpp — single-color 변환 vs 4-pixel batch SIMD 변환 비교.
//
//   (A) scalar — 현재 ok_color::RGB::to_oklab() 를 4번 호출.
//   (E) SSE4   — 4 픽셀을 SoA 로 묶어 한 번에 RGB→Oklab.
//   (F) AVX2   — 8 픽셀을 한 번에. (cbrt 만 scalar.)
//
// 입력은 16-byte 정렬된 padded RGB 배열 (r,g,b,0).
//
// 빌드:
//   g++ -std=c++17 -O3 -mavx2 -mfma -msse4.1 -flto -DNDEBUG \
//       ok_color.cpp bench_batch.cpp -o bench_batch
// -----------------------------------------------------------------------------

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
// (A) scalar baseline — 4 RGB → 4 Lab 를 순서대로 처리.
// =============================================================================
static inline void convert4_scalar(const RGB4* in, Lab4* out)
{
	for (int i = 0; i < 4; ++i) {
		ok::Lab L = ok::RGB{in[i].r, in[i].g, in[i].b}.to_oklab();
		out[i] = { L.L, L.a, L.b, 0.f };
	}
}

// =============================================================================
// (E) SSE4 batch — 4 픽셀을 SoA 로 변환.
// =============================================================================
static inline void convert4_sse(const RGB4* in, Lab4* out)
{
	// 1) AoS → SoA 전치.
	__m128 v0 = _mm_load_ps(&in[0].r);  // (r0, g0, b0, _)
	__m128 v1 = _mm_load_ps(&in[1].r);
	__m128 v2 = _mm_load_ps(&in[2].r);
	__m128 v3 = _mm_load_ps(&in[3].r);
	_MM_TRANSPOSE4_PS(v0, v1, v2, v3);
	__m128 R = v0;  // (r0, r1, r2, r3)
	__m128 G = v1;
	__m128 B = v2;

	// 2) RGB → LMS — 한 행마다 1 mul + 2 fma (4 픽셀 동시).
	__m128 Ll = _mm_fmadd_ps(_mm_set1_ps(0.0514459929f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.5363325363f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.4122214708f), R)));
	__m128 Ml = _mm_fmadd_ps(_mm_set1_ps(0.1073969566f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.6806995451f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.2119034982f), R)));
	__m128 Sl = _mm_fmadd_ps(_mm_set1_ps(0.6299787005f), B,
		        _mm_fmadd_ps(_mm_set1_ps(0.2817188376f), G,
		                     _mm_mul_ps(_mm_set1_ps(0.0883024619f), R)));

	// 3) cbrt — SIMD 명령 없음. scalar 12회.
	alignas(16) float Larr[4], Marr[4], Sarr[4];
	_mm_store_ps(Larr, Ll);
	_mm_store_ps(Marr, Ml);
	_mm_store_ps(Sarr, Sl);
	for (int i = 0; i < 4; ++i) {
		Larr[i] = std::cbrt(Larr[i]);
		Marr[i] = std::cbrt(Marr[i]);
		Sarr[i] = std::cbrt(Sarr[i]);
	}
	__m128 Lp = _mm_load_ps(Larr);
	__m128 Mp = _mm_load_ps(Marr);
	__m128 Sp = _mm_load_ps(Sarr);

	// 4) LMS' → Lab — 다시 SoA SIMD.
	__m128 Lab_L = _mm_fmadd_ps(_mm_set1_ps(-0.0040720468f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps( 0.7936177850f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(0.2104542553f), Lp)));
	__m128 Lab_a = _mm_fmadd_ps(_mm_set1_ps( 0.4505937099f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps(-2.4285922050f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(1.9779984951f), Lp)));
	__m128 Lab_b = _mm_fmadd_ps(_mm_set1_ps(-0.8086757660f), Sp,
		           _mm_fmadd_ps(_mm_set1_ps( 0.7827717662f), Mp,
		                        _mm_mul_ps( _mm_set1_ps(0.0259040371f), Lp)));

	// 5) SoA → AoS 전치 (출력은 padded Lab4).
	__m128 z = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(Lab_L, Lab_a, Lab_b, z);
	_mm_store_ps(&out[0].L, Lab_L);
	_mm_store_ps(&out[1].L, Lab_a);
	_mm_store_ps(&out[2].L, Lab_b);
	_mm_store_ps(&out[3].L, z);
}

// =============================================================================
// (F) AVX2 batch — 8 픽셀을 한 번에 (256-bit ymm).
// =============================================================================
static inline void convert8_avx2(const RGB4* in, Lab4* out)
{
	// 8 RGB 를 4+4 두 묶음으로 로드 후 SoA 전치, 그리고 lo/hi 를 ymm 으로 묶는다.
	__m128 a0 = _mm_load_ps(&in[0].r), a1 = _mm_load_ps(&in[1].r),
	       a2 = _mm_load_ps(&in[2].r), a3 = _mm_load_ps(&in[3].r);
	__m128 b0 = _mm_load_ps(&in[4].r), b1 = _mm_load_ps(&in[5].r),
	       b2 = _mm_load_ps(&in[6].r), b3 = _mm_load_ps(&in[7].r);
	_MM_TRANSPOSE4_PS(a0, a1, a2, a3);
	_MM_TRANSPOSE4_PS(b0, b1, b2, b3);
	__m256 R = _mm256_setr_m128(a0, b0);  // (r0..r3, r4..r7)
	__m256 G = _mm256_setr_m128(a1, b1);
	__m256 B = _mm256_setr_m128(a2, b2);

	auto fma3 = [](float c0, float c1, float c2,
		           __m256 R, __m256 G, __m256 B) {
		return _mm256_fmadd_ps(_mm256_set1_ps(c2), B,
			   _mm256_fmadd_ps(_mm256_set1_ps(c1), G,
			                   _mm256_mul_ps( _mm256_set1_ps(c0), R)));
	};

	__m256 Ll = fma3(0.4122214708f, 0.5363325363f, 0.0514459929f, R, G, B);
	__m256 Ml = fma3(0.2119034982f, 0.6806995451f, 0.1073969566f, R, G, B);
	__m256 Sl = fma3(0.0883024619f, 0.2817188376f, 0.6299787005f, R, G, B);

	alignas(32) float Larr[8], Marr[8], Sarr[8];
	_mm256_store_ps(Larr, Ll);
	_mm256_store_ps(Marr, Ml);
	_mm256_store_ps(Sarr, Sl);
	for (int i = 0; i < 8; ++i) {
		Larr[i] = std::cbrt(Larr[i]);
		Marr[i] = std::cbrt(Marr[i]);
		Sarr[i] = std::cbrt(Sarr[i]);
	}
	__m256 Lp = _mm256_load_ps(Larr);
	__m256 Mp = _mm256_load_ps(Marr);
	__m256 Sp = _mm256_load_ps(Sarr);

	__m256 Lab_L = fma3( 0.2104542553f,  0.7936177850f, -0.0040720468f, Lp, Mp, Sp);
	__m256 Lab_a = fma3( 1.9779984951f, -2.4285922050f,  0.4505937099f, Lp, Mp, Sp);
	__m256 Lab_b = fma3( 0.0259040371f,  0.7827717662f, -0.8086757660f, Lp, Mp, Sp);

	// SoA → AoS for 8 pixels: lo/hi 각각 4-wide 전치 후 store.
	__m128 LL_lo = _mm256_castps256_ps128(Lab_L), LL_hi = _mm256_extractf128_ps(Lab_L, 1);
	__m128 La_lo = _mm256_castps256_ps128(Lab_a), La_hi = _mm256_extractf128_ps(Lab_a, 1);
	__m128 Lb_lo = _mm256_castps256_ps128(Lab_b), Lb_hi = _mm256_extractf128_ps(Lab_b, 1);
	__m128 z = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(LL_lo, La_lo, Lb_lo, z);
	_mm_store_ps(&out[0].L, LL_lo);
	_mm_store_ps(&out[1].L, La_lo);
	_mm_store_ps(&out[2].L, Lb_lo);
	_mm_store_ps(&out[3].L, z);
	z = _mm_setzero_ps();
	_MM_TRANSPOSE4_PS(LL_hi, La_hi, Lb_hi, z);
	_mm_store_ps(&out[4].L, LL_hi);
	_mm_store_ps(&out[5].L, La_hi);
	_mm_store_ps(&out[6].L, Lb_hi);
	_mm_store_ps(&out[7].L, z);
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
	constexpr int N      = 4096;          // 입력풀 (8 의 배수, 2의 거듭제곱)
	constexpr int ROUNDS = 9;
	constexpr int ITERS  = 4'000'000;     // 처리할 픽셀 수

	// 입력풀: padded RGB4, 16-byte 정렬.
	std::vector<RGB4, /*aligned*/std::allocator<RGB4>> rgb_in(N);
	{
		std::mt19937 rng(0xBEEF);
		std::uniform_real_distribution<float> U(0.f, 1.f);
		for (int i = 0; i < N; ++i) rgb_in[i] = { U(rng), U(rng), U(rng), 0.f };
	}
	std::vector<Lab4> lab_out(N);

	std::printf("입력풀=%d 픽셀, 라운드=%d, 처리 픽셀수=%d (min ns/픽셀)\n\n",
		N, ROUNDS, ITERS);
	std::printf("%-30s | %14s | %8s\n",
		"variant", "ns/pixel", "speedup");
	std::printf("------------------------------+----------------+---------\n");

	// (A) scalar 4픽셀씩 (batch 와 같은 단위로 측정).
	double t_scalar = bench_min_ns(ROUNDS, ITERS / 4, [&](int it){
		float Ls=0, as=0, bs=0;
		const int mask = (N/4) - 1;
		for (int i = 0; i < it; ++i) {
			int base = (i & mask) * 4;
			convert4_scalar(&rgb_in[base], &lab_out[base]);
			Ls += lab_out[base].L; as += lab_out[base].a; bs += lab_out[base].b;
		}
		g_sink_L = Ls; g_sink_a = as; g_sink_b = bs;
	}) / 4.0;  // ns per pixel

	std::printf("%-30s | %14.3f | %7.3fx\n",
		"(A) scalar 4-at-a-time", t_scalar, 1.0);

	// (E) SSE4 4-batch.
	double t_sse = bench_min_ns(ROUNDS, ITERS / 4, [&](int it){
		float Ls=0, as=0, bs=0;
		const int mask = (N/4) - 1;
		for (int i = 0; i < it; ++i) {
			int base = (i & mask) * 4;
			convert4_sse(&rgb_in[base], &lab_out[base]);
			Ls += lab_out[base].L; as += lab_out[base].a; bs += lab_out[base].b;
		}
		g_sink_L = Ls; g_sink_a = as; g_sink_b = bs;
	}) / 4.0;
	std::printf("%-30s | %14.3f | %7.3fx\n",
		"(E) SSE4  4-pixel batch", t_sse, t_scalar / t_sse);

	// (G) cbrt 를 빼고 math 만 측정 → 진짜 SIMD 상한.
	auto convert4_sse_nocbrt = [](const RGB4* in, Lab4* out) {
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
		// cbrt 없이 그대로 사용 (정확하지 않지만 SIMD 상한 측정용).
		__m128 Lp = Ll, Mp = Ml, Sp = Sl;
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
	};
	double t_sse_nocbrt = bench_min_ns(ROUNDS, ITERS / 4, [&](int it){
		float Ls=0, as=0, bs=0;
		const int mask = (N/4) - 1;
		for (int i = 0; i < it; ++i) {
			int base = (i & mask) * 4;
			convert4_sse_nocbrt(&rgb_in[base], &lab_out[base]);
			Ls += lab_out[base].L; as += lab_out[base].a; bs += lab_out[base].b;
		}
		g_sink_L = Ls; g_sink_a = as; g_sink_b = bs;
	}) / 4.0;
	std::printf("%-30s | %14.3f | %7.3fx  (※ cbrt 제거, 상한 측정용)\n",
		"(G) SSE4  no-cbrt (math only)", t_sse_nocbrt, t_scalar / t_sse_nocbrt);

	// (F) AVX2 8-batch.
	double t_avx = bench_min_ns(ROUNDS, ITERS / 8, [&](int it){
		float Ls=0, as=0, bs=0;
		const int mask = (N/8) - 1;
		for (int i = 0; i < it; ++i) {
			int base = (i & mask) * 8;
			convert8_avx2(&rgb_in[base], &lab_out[base]);
			Ls += lab_out[base].L; as += lab_out[base].a; bs += lab_out[base].b;
		}
		g_sink_L = Ls; g_sink_a = as; g_sink_b = bs;
	}) / 8.0;
	std::printf("%-30s | %14.3f | %7.3fx\n",
		"(F) AVX2  8-pixel batch", t_avx, t_scalar / t_avx);

	std::printf("\n[sink] L=%g a=%g b=%g\n",
		(double)g_sink_L, (double)g_sink_a, (double)g_sink_b);

	// 결과 정합성 검증: scalar 와 batch 출력이 일치하는지 한 픽셀 비교.
	{
		Lab4 sa[4], sb[4];
		convert4_scalar(&rgb_in[0], sa);
		convert4_sse(&rgb_in[0], sb);
		std::printf("\n정합성 (idx 0):  scalar L=%.6f a=%.6f b=%.6f\n",
			sa[0].L, sa[0].a, sa[0].b);
		std::printf("                 SSE    L=%.6f a=%.6f b=%.6f\n",
			sb[0].L, sb[0].a, sb[0].b);
	}
	return 0;
}
