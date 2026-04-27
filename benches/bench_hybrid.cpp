// bench_hybrid.cpp — 함수별 최적 SIMD 변형을 골라쓰는 hybrid 라이브러리.
//
//   라우팅 결정 (앞 측정 결과 기반):
//     linear_srgb→oklab  →  SSE  (gather/scatter overhead > 본체)
//     oklab→linear_srgb  →  SSE  (본체가 매우 가벼워 SSE 가 압승)
//     okhsl↔srgb         →  AVX512  (transcendental 무거워 16-wide 효과 큼)
//     okhsv↔srgb         →  AVX512  (동일)
//
//   16-pixel batch 단위로 받고, SSE-optimal 함수는 4번 batch4 호출, AVX512-optimal
//   은 한 번 batch16 호출.  LTO 로 함수 호출 비용은 거의 사라짐.

#include "ok_color_v4.h"
#include "ok_color_simd.h"
#include "ok_color_avx512.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

namespace cv4 = ok_color_v4;
namespace s4  = ok_color_simd;
namespace s16 = ok_color_avx512;
using clock_t_ = std::chrono::steady_clock;

static volatile float g_sink = 0.f;

// =============================================================================
// hybrid wrapper — 16-pixel batch, 함수별 라우팅.
// =============================================================================
namespace hybrid {

static inline void linear_srgb_to_oklab_batch16(const s4::RGB4* in, s4::Lab4* out) {
	// SSE 가 더 빠름 → batch4 × 4.
	s4::linear_srgb_to_oklab_batch4(in,    out);
	s4::linear_srgb_to_oklab_batch4(in+4,  out+4);
	s4::linear_srgb_to_oklab_batch4(in+8,  out+8);
	s4::linear_srgb_to_oklab_batch4(in+12, out+12);
}
static inline void oklab_to_linear_srgb_batch16(const s4::Lab4* in, s4::RGB4* out) {
	s4::oklab_to_linear_srgb_batch4(in,    out);
	s4::oklab_to_linear_srgb_batch4(in+4,  out+4);
	s4::oklab_to_linear_srgb_batch4(in+8,  out+8);
	s4::oklab_to_linear_srgb_batch4(in+12, out+12);
}
static inline void okhsl_to_srgb_batch16(const s4::HSL4* in, s4::RGB4* out) {
	// AVX512 가 더 빠름 → batch16.
	s16::okhsl_to_srgb_batch16(reinterpret_cast<const s16::HSL4*>(in),
	                           reinterpret_cast<s16::RGB4*>(out));
}
static inline void srgb_to_okhsl_batch16(const s4::RGB4* in, s4::HSL4* out) {
	s16::srgb_to_okhsl_batch16(reinterpret_cast<const s16::RGB4*>(in),
	                            reinterpret_cast<s16::HSL4*>(out));
}
static inline void okhsv_to_srgb_batch16(const s4::HSV4* in, s4::RGB4* out) {
	s16::okhsv_to_srgb_batch16(reinterpret_cast<const s16::HSV4*>(in),
	                           reinterpret_cast<s16::RGB4*>(out));
}
static inline void srgb_to_okhsv_batch16(const s4::RGB4* in, s4::HSV4* out) {
	s16::srgb_to_okhsv_batch16(reinterpret_cast<const s16::RGB4*>(in),
	                            reinterpret_cast<s16::HSV4*>(out));
}

} // namespace hybrid

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
	constexpr int ITERS_16 = 62'500;     // 1M pixels / 16

	std::vector<s4::RGB4> rgb(N);
	std::vector<s4::Lab4> lab(N);
	std::vector<s4::HSL4> hsl(N);
	std::vector<s4::HSV4> hsv(N);
	std::vector<s4::RGB4> rgb_out(N);
	std::vector<s4::Lab4> lab_out(N);
	std::vector<s4::HSL4> hsl_out(N);
	std::vector<s4::HSV4> hsv_out(N);

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);
	for (int i = 0; i < N; ++i) {
		rgb[i] = { U(rng), U(rng), U(rng), 0 };
		cv4::Lab L = cv4::RGB{rgb[i].r, rgb[i].g, rgb[i].b}.to_oklab();
		lab[i] = { L.L, L.a, L.b, 0 };
		hsl[i] = { U(rng), U(rng), U(rng), 0 };
		hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}

	std::printf("입력풀=%d, 라운드=%d, 16-pixel batch (min ns/pixel)\n\n", N, ROUNDS);
	std::printf("%-30s | %10s | %10s | %10s | %s\n",
		"function", "SSE", "AVX512", "Hybrid", "winner");
	std::printf("-------------------------------+------------+------------+------------+--------\n");

	auto run3 = [&](const char* name,
	                auto sse_fn, auto avx_fn, auto hyb_fn) {
		double t_sse = bench_min(ROUNDS, ITERS_16, [&](int it){
			float acc = 0;
			const int mask = (N/16) - 1;
			for (int i = 0; i < it; ++i) acc += sse_fn(i & mask);
			g_sink = acc;
		}) / 16.0;
		double t_avx = bench_min(ROUNDS, ITERS_16, [&](int it){
			float acc = 0;
			const int mask = (N/16) - 1;
			for (int i = 0; i < it; ++i) acc += avx_fn(i & mask);
			g_sink = acc;
		}) / 16.0;
		double t_hyb = bench_min(ROUNDS, ITERS_16, [&](int it){
			float acc = 0;
			const int mask = (N/16) - 1;
			for (int i = 0; i < it; ++i) acc += hyb_fn(i & mask);
			g_sink = acc;
		}) / 16.0;
		const char* w = (t_hyb < t_sse * 0.95 && t_hyb < t_avx * 0.95) ? "Hybrid"
		              : (t_sse < t_avx) ? "SSE  " : "AVX  ";
		std::printf("%-30s | %10.3f | %10.3f | %10.3f | %s\n", name, t_sse, t_avx, t_hyb, w);
	};

	run3("linear_srgb→oklab",
		[&](int b) {  // SSE: 4 × batch4
			s4::linear_srgb_to_oklab_batch4(&rgb[b*16],    &lab_out[b*16]);
			s4::linear_srgb_to_oklab_batch4(&rgb[b*16+4],  &lab_out[b*16+4]);
			s4::linear_srgb_to_oklab_batch4(&rgb[b*16+8],  &lab_out[b*16+8]);
			s4::linear_srgb_to_oklab_batch4(&rgb[b*16+12], &lab_out[b*16+12]);
			return lab_out[b*16].L;
		},
		[&](int b) {
			s16::linear_srgb_to_oklab_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]),
			                                   reinterpret_cast<s16::Lab4*>(&lab_out[b*16]));
			return lab_out[b*16].L;
		},
		[&](int b) {
			hybrid::linear_srgb_to_oklab_batch16(&rgb[b*16], &lab_out[b*16]);
			return lab_out[b*16].L;
		});

	run3("oklab→linear_srgb",
		[&](int b) {
			s4::oklab_to_linear_srgb_batch4(&lab[b*16],    &rgb_out[b*16]);
			s4::oklab_to_linear_srgb_batch4(&lab[b*16+4],  &rgb_out[b*16+4]);
			s4::oklab_to_linear_srgb_batch4(&lab[b*16+8],  &rgb_out[b*16+8]);
			s4::oklab_to_linear_srgb_batch4(&lab[b*16+12], &rgb_out[b*16+12]);
			return rgb_out[b*16].r;
		},
		[&](int b) {
			s16::oklab_to_linear_srgb_batch16(reinterpret_cast<const s16::Lab4*>(&lab[b*16]),
			                                   reinterpret_cast<s16::RGB4*>(&rgb_out[b*16]));
			return rgb_out[b*16].r;
		},
		[&](int b) {
			hybrid::oklab_to_linear_srgb_batch16(&lab[b*16], &rgb_out[b*16]);
			return rgb_out[b*16].r;
		});

	run3("okhsl→srgb",
		[&](int b) {
			s4::okhsl_to_srgb_batch4(&hsl[b*16],    &rgb_out[b*16]);
			s4::okhsl_to_srgb_batch4(&hsl[b*16+4],  &rgb_out[b*16+4]);
			s4::okhsl_to_srgb_batch4(&hsl[b*16+8],  &rgb_out[b*16+8]);
			s4::okhsl_to_srgb_batch4(&hsl[b*16+12], &rgb_out[b*16+12]);
			return rgb_out[b*16].r;
		},
		[&](int b) {
			s16::okhsl_to_srgb_batch16(reinterpret_cast<const s16::HSL4*>(&hsl[b*16]),
			                            reinterpret_cast<s16::RGB4*>(&rgb_out[b*16]));
			return rgb_out[b*16].r;
		},
		[&](int b) {
			hybrid::okhsl_to_srgb_batch16(&hsl[b*16], &rgb_out[b*16]);
			return rgb_out[b*16].r;
		});

	run3("srgb→okhsl",
		[&](int b) {
			s4::srgb_to_okhsl_batch4(&rgb[b*16],    &hsl_out[b*16]);
			s4::srgb_to_okhsl_batch4(&rgb[b*16+4],  &hsl_out[b*16+4]);
			s4::srgb_to_okhsl_batch4(&rgb[b*16+8],  &hsl_out[b*16+8]);
			s4::srgb_to_okhsl_batch4(&rgb[b*16+12], &hsl_out[b*16+12]);
			return hsl_out[b*16].h;
		},
		[&](int b) {
			s16::srgb_to_okhsl_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]),
			                            reinterpret_cast<s16::HSL4*>(&hsl_out[b*16]));
			return hsl_out[b*16].h;
		},
		[&](int b) {
			hybrid::srgb_to_okhsl_batch16(&rgb[b*16], &hsl_out[b*16]);
			return hsl_out[b*16].h;
		});

	run3("okhsv→srgb",
		[&](int b) {
			s4::okhsv_to_srgb_batch4(&hsv[b*16],    &rgb_out[b*16]);
			s4::okhsv_to_srgb_batch4(&hsv[b*16+4],  &rgb_out[b*16+4]);
			s4::okhsv_to_srgb_batch4(&hsv[b*16+8],  &rgb_out[b*16+8]);
			s4::okhsv_to_srgb_batch4(&hsv[b*16+12], &rgb_out[b*16+12]);
			return rgb_out[b*16].r;
		},
		[&](int b) {
			s16::okhsv_to_srgb_batch16(reinterpret_cast<const s16::HSV4*>(&hsv[b*16]),
			                            reinterpret_cast<s16::RGB4*>(&rgb_out[b*16]));
			return rgb_out[b*16].r;
		},
		[&](int b) {
			hybrid::okhsv_to_srgb_batch16(&hsv[b*16], &rgb_out[b*16]);
			return rgb_out[b*16].r;
		});

	run3("srgb→okhsv",
		[&](int b) {
			s4::srgb_to_okhsv_batch4(&rgb[b*16],    &hsv_out[b*16]);
			s4::srgb_to_okhsv_batch4(&rgb[b*16+4],  &hsv_out[b*16+4]);
			s4::srgb_to_okhsv_batch4(&rgb[b*16+8],  &hsv_out[b*16+8]);
			s4::srgb_to_okhsv_batch4(&rgb[b*16+12], &hsv_out[b*16+12]);
			return hsv_out[b*16].h;
		},
		[&](int b) {
			s16::srgb_to_okhsv_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]),
			                            reinterpret_cast<s16::HSV4*>(&hsv_out[b*16]));
			return hsv_out[b*16].h;
		},
		[&](int b) {
			hybrid::srgb_to_okhsv_batch16(&rgb[b*16], &hsv_out[b*16]);
			return hsv_out[b*16].h;
		});

	std::printf("\n[sink] %g\n", (double)g_sink);
	return 0;
}
