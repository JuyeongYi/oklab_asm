// bench_cpp_sse_avx512.cpp — vec4 기준 CPP / SSE / AVX512 셋 비교.
//   CPP    = ok_color_v4 (16-byte float4, scalar 코드, 한 픽셀씩)
//   SSE    = ok_color_simd (4-pixel SoA batch)
//   AVX512 = ok_color_avx512 (16-pixel SoA batch)
//
// 빌드:
//   g++ -std=c++17 -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1 \
//       -flto -DNDEBUG ok_color_v4.cpp ok_color_simd.cpp ok_color_avx512.cpp \
//       bench_cpp_sse_avx512.cpp -o bench_cpp_sse_avx512

#include "ok_color_v4.h"
#include "ok_color_simd.h"
#include "ok_color_avx512.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

namespace cv4 = ok_color_v4;
namespace s4  = ok_color_simd;
namespace s16 = ok_color_avx512;
using clock_t_ = std::chrono::steady_clock;

static volatile float g_sink = 0.f;

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

// 정확성 sanity check — 하나의 pixel 에 대해 CPP / SSE / AVX512 결과 비교.
static void sanity_check(const s4::RGB4* rgb_pool)
{
	std::printf("정합성 검사 (idx 0 ~ 3):\n");
	for (int i = 0; i < 4; ++i) {
		// CPP
		cv4::RGB cpp_rgb{rgb_pool[i].r, rgb_pool[i].g, rgb_pool[i].b};
		cv4::Lab cpp_lab = cpp_rgb.to_oklab();

		// SSE: 4 픽셀 batch.  i 가 lane 0.
		s4::Lab4 sse_out[4];
		s4::linear_srgb_to_oklab_batch4(&rgb_pool[i & ~3], sse_out);
		s4::Lab4& sse_lab = sse_out[i & 3];

		// AVX512: 16 픽셀 batch.  i 가 lane 0.
		s16::Lab4 avx_out[16];
		s16::linear_srgb_to_oklab_batch16(reinterpret_cast<const s16::RGB4*>(&rgb_pool[i & ~15]), avx_out);
		s16::Lab4& avx_lab = avx_out[i & 15];

		std::printf("  [%d]  cpp =(%.6f,%.6f,%.6f)\n", i, cpp_lab.L, cpp_lab.a, cpp_lab.b);
		std::printf("       sse =(%.6f,%.6f,%.6f)\n",  sse_lab.L, sse_lab.a, sse_lab.b);
		std::printf("       avx =(%.6f,%.6f,%.6f)\n",  avx_lab.L, avx_lab.a, avx_lab.b);
	}
	std::printf("\n");
}

int main()
{
	constexpr int N = 4096;          // 입력풀 (16의 배수)
	constexpr int ROUNDS = 9;
	constexpr int ITERS_CPP = 1'000'000;
	constexpr int ITERS_SSE = 250'000;       // 4 픽셀 단위
	constexpr int ITERS_AVX = 62'500;        // 16 픽셀 단위 (총 동일하게 1M 픽셀)

	std::vector<s4::RGB4> rgb(N);
	std::vector<s4::Lab4> lab(N);
	std::vector<s4::HSL4> hsl(N);
	std::vector<s4::HSV4> hsv(N);
	std::vector<s4::RGB4> rgb_out(N);
	std::vector<s4::Lab4> lab_out(N);
	std::vector<s4::HSL4> hsl_out(N);
	std::vector<s4::HSV4> hsv_out(N);

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);  // 회색 회피
	for (int i = 0; i < N; ++i) {
		rgb[i] = { U(rng), U(rng), U(rng), 0 };
		cv4::Lab L = cv4::RGB{rgb[i].r, rgb[i].g, rgb[i].b}.to_oklab();
		lab[i] = { L.L, L.a, L.b, 0 };
		hsl[i] = { U(rng), U(rng), U(rng), 0 };
		hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}

	sanity_check(rgb.data());

	std::printf("입력풀=%d, 라운드=%d, 처리 픽셀≈1M (min ns/픽셀)\n\n", N, ROUNDS);
	std::printf("%-30s | %10s | %10s | %10s | %10s | %10s\n",
		"function", "CPP (vec4)", "SSE (×4)", "AVX512 (×16)", "SSE/CPP", "AVX/CPP");
	std::printf("-------------------------------+------------+------------+--------------+------------+-----------\n");

	auto run = [&](const char* name,
	               auto cpp_fn, auto sse_fn, auto avx_fn) {
		double t_cpp = bench_min(ROUNDS, ITERS_CPP, [&](int it){
			float acc = 0;
			const int mask = N - 1;
			for (int i = 0; i < it; ++i) acc += cpp_fn(i & mask);
			g_sink = acc;
		});
		double t_sse = bench_min(ROUNDS, ITERS_SSE, [&](int it){
			float acc = 0;
			const int mask = (N/4) - 1;
			for (int i = 0; i < it; ++i) acc += sse_fn(i & mask);
			g_sink = acc;
		}) / 4.0;
		double t_avx = bench_min(ROUNDS, ITERS_AVX, [&](int it){
			float acc = 0;
			const int mask = (N/16) - 1;
			for (int i = 0; i < it; ++i) acc += avx_fn(i & mask);
			g_sink = acc;
		}) / 16.0;
		std::printf("%-30s | %10.3f | %10.3f | %12.3f | %9.3fx | %9.3fx\n",
			name, t_cpp, t_sse, t_avx, t_cpp/t_sse, t_cpp/t_avx);
	};

	run("linear_srgb→oklab",
		[&](int i) {
			cv4::Lab L = cv4::RGB{rgb[i].r, rgb[i].g, rgb[i].b}.to_oklab();
			return L.L + L.a + L.b;
		},
		[&](int b) {
			s4::linear_srgb_to_oklab_batch4(&rgb[b*4], &lab_out[b*4]);
			return lab_out[b*4].L + lab_out[b*4].a + lab_out[b*4].b;
		},
		[&](int b) {
			s16::linear_srgb_to_oklab_batch16(
				reinterpret_cast<const s16::RGB4*>(&rgb[b*16]),
				reinterpret_cast<s16::Lab4*>(&lab_out[b*16]));
			return lab_out[b*16].L + lab_out[b*16].a + lab_out[b*16].b;
		});

	run("oklab→linear_srgb",
		[&](int i) {
			cv4::RGB R = cv4::Lab{lab[i].L, lab[i].a, lab[i].b}.to_linear_srgb();
			return R.r + R.g + R.b;
		},
		[&](int b) {
			s4::oklab_to_linear_srgb_batch4(&lab[b*4], &rgb_out[b*4]);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		},
		[&](int b) {
			s16::oklab_to_linear_srgb_batch16(
				reinterpret_cast<const s16::Lab4*>(&lab[b*16]),
				reinterpret_cast<s16::RGB4*>(&rgb_out[b*16]));
			return rgb_out[b*16].r + rgb_out[b*16].g + rgb_out[b*16].b;
		});

	run("okhsl→srgb",
		[&](int i) {
			cv4::RGB R = cv4::HSL{hsl[i].h, hsl[i].s, hsl[i].l}.to_srgb();
			return R.r + R.g + R.b;
		},
		[&](int b) {
			s4::okhsl_to_srgb_batch4(&hsl[b*4], &rgb_out[b*4]);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		},
		[&](int b) {
			s16::okhsl_to_srgb_batch16(
				reinterpret_cast<const s16::HSL4*>(&hsl[b*16]),
				reinterpret_cast<s16::RGB4*>(&rgb_out[b*16]));
			return rgb_out[b*16].r + rgb_out[b*16].g + rgb_out[b*16].b;
		});

	run("srgb→okhsl",
		[&](int i) {
			cv4::HSL H = cv4::RGB{rgb[i].r, rgb[i].g, rgb[i].b}.to_okhsl();
			return H.h + H.s + H.l;
		},
		[&](int b) {
			s4::srgb_to_okhsl_batch4(&rgb[b*4], &hsl_out[b*4]);
			return hsl_out[b*4].h + hsl_out[b*4].s + hsl_out[b*4].l;
		},
		[&](int b) {
			s16::srgb_to_okhsl_batch16(
				reinterpret_cast<const s16::RGB4*>(&rgb[b*16]),
				reinterpret_cast<s16::HSL4*>(&hsl_out[b*16]));
			return hsl_out[b*16].h + hsl_out[b*16].s + hsl_out[b*16].l;
		});

	run("okhsv→srgb",
		[&](int i) {
			cv4::RGB R = cv4::HSV{hsv[i].h, hsv[i].s, hsv[i].v}.to_srgb();
			return R.r + R.g + R.b;
		},
		[&](int b) {
			s4::okhsv_to_srgb_batch4(&hsv[b*4], &rgb_out[b*4]);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		},
		[&](int b) {
			s16::okhsv_to_srgb_batch16(
				reinterpret_cast<const s16::HSV4*>(&hsv[b*16]),
				reinterpret_cast<s16::RGB4*>(&rgb_out[b*16]));
			return rgb_out[b*16].r + rgb_out[b*16].g + rgb_out[b*16].b;
		});

	run("srgb→okhsv",
		[&](int i) {
			cv4::HSV H = cv4::RGB{rgb[i].r, rgb[i].g, rgb[i].b}.to_okhsv();
			return H.h + H.s + H.v;
		},
		[&](int b) {
			s4::srgb_to_okhsv_batch4(&rgb[b*4], &hsv_out[b*4]);
			return hsv_out[b*4].h + hsv_out[b*4].s + hsv_out[b*4].v;
		},
		[&](int b) {
			s16::srgb_to_okhsv_batch16(
				reinterpret_cast<const s16::RGB4*>(&rgb[b*16]),
				reinterpret_cast<s16::HSV4*>(&hsv_out[b*16]));
			return hsv_out[b*16].h + hsv_out[b*16].s + hsv_out[b*16].v;
		});

	std::printf("\n[sink] %g\n", (double)g_sink);
	return 0;
}
