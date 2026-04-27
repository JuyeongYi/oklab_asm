// bench_four.cpp — 네 가지 변형 비교.
//   CPP                = ok_color_v4 (vec4 scalar, 한 픽셀씩)
//   SSE  (4-wide)      = ok_color_simd     (VEX 인코딩, -msse4.1)
//   AVX512-VL (4-wide) = ok_color_avx512_4 (같은 코드, EVEX 가능 빌드)
//   AVX512 (16-wide)   = ok_color_avx512   (16 픽셀 SoA)
//
// ok_color_simd 와 ok_color_avx512_4 는 *코드가 동일* — 빌드 시 사용 가능 ISA 만 다름.
// SSE TU 는 SSE-only 옵션, AVX512_4 TU 는 AVX512-VL 켠 옵션으로 분리 빌드.
//
// 빌드 (분리 빌드가 핵심):
//   g++ -std=c++17 -O3 -msse4.1 -mfma -DNDEBUG -c ok_color_simd.cpp -o ok_color_simd.o
//   g++ -std=c++17 -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -DNDEBUG \
//       -c ok_color_avx512_4.cpp -o ok_color_avx512_4.o
//   g++ -std=c++17 -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -DNDEBUG \
//       -c ok_color_avx512.cpp   -o ok_color_avx512.o
//   g++ -std=c++17 -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1 \
//       -DNDEBUG ok_color_v4.cpp ok_color_simd.o ok_color_avx512_4.o \
//       ok_color_avx512.o bench_four.cpp -o bench_four

#include "ok_color_v4.h"
#include "ok_color_simd.h"
#include "ok_color_avx512_4.h"
#include "ok_color_avx512.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

namespace cv4 = ok_color_v4;
namespace s4  = ok_color_simd;
namespace v4e = ok_color_avx512_4;
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

int main()
{
	constexpr int N = 4096;
	constexpr int ROUNDS = 9;
	constexpr int ITERS_CPP = 1'000'000;
	constexpr int ITERS_4   = 250'000;
	constexpr int ITERS_16  = 62'500;

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

	std::printf("입력풀=%d, 라운드=%d, ≈1M pixel (min ns/pixel)\n\n", N, ROUNDS);
	std::printf("%-25s | %8s | %9s | %11s | %11s | %s\n",
		"function", "CPP", "SSE×4", "AVX512vl×4", "AVX512×16", "best");
	std::printf("--------------------------+----------+-----------+-------------+-------------+---------\n");

	auto run4 = [&](const char* name,
	                auto cpp_fn, auto sse_fn, auto avx4_fn, auto avx16_fn) {
		double t_cpp = bench_min(ROUNDS, ITERS_CPP, [&](int it){
			float acc = 0; const int mask = N - 1;
			for (int i = 0; i < it; ++i) acc += cpp_fn(i & mask);
			g_sink = acc;
		});
		double t_sse = bench_min(ROUNDS, ITERS_4, [&](int it){
			float acc = 0; const int mask = (N/4) - 1;
			for (int i = 0; i < it; ++i) acc += sse_fn(i & mask);
			g_sink = acc;
		}) / 4.0;
		double t_avx4 = bench_min(ROUNDS, ITERS_4, [&](int it){
			float acc = 0; const int mask = (N/4) - 1;
			for (int i = 0; i < it; ++i) acc += avx4_fn(i & mask);
			g_sink = acc;
		}) / 4.0;
		double t_avx16 = bench_min(ROUNDS, ITERS_16, [&](int it){
			float acc = 0; const int mask = (N/16) - 1;
			for (int i = 0; i < it; ++i) acc += avx16_fn(i & mask);
			g_sink = acc;
		}) / 16.0;

		double best = std::min({t_sse, t_avx4, t_avx16});
		const char* w = best == t_sse ? "SSE×4"
		              : best == t_avx4 ? "AVX-VL×4" : "AVX×16";

		std::printf("%-25s | %8.3f | %9.3f | %11.3f | %11.3f | %s (%.2fx vs CPP)\n",
			name, t_cpp, t_sse, t_avx4, t_avx16, w, t_cpp / best);
	};

	run4("linear_srgb→oklab",
		[&](int i) { auto L = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_oklab(); return L.L+L.a+L.b; },
		[&](int b) { s4 ::linear_srgb_to_oklab_batch4(&rgb[b*4], &lab_out[b*4]); return lab_out[b*4].L; },
		[&](int b) { v4e::linear_srgb_to_oklab_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]), reinterpret_cast<v4e::Lab4*>(&lab_out[b*4])); return lab_out[b*4].L; },
		[&](int b) { s16::linear_srgb_to_oklab_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]), reinterpret_cast<s16::Lab4*>(&lab_out[b*16])); return lab_out[b*16].L; });

	run4("oklab→linear_srgb",
		[&](int i) { auto R = cv4::Lab{lab[i].L,lab[i].a,lab[i].b}.to_linear_srgb(); return R.r+R.g+R.b; },
		[&](int b) { s4 ::oklab_to_linear_srgb_batch4(&lab[b*4], &rgb_out[b*4]); return rgb_out[b*4].r; },
		[&](int b) { v4e::oklab_to_linear_srgb_batch4(reinterpret_cast<const v4e::Lab4*>(&lab[b*4]), reinterpret_cast<v4e::RGB4*>(&rgb_out[b*4])); return rgb_out[b*4].r; },
		[&](int b) { s16::oklab_to_linear_srgb_batch16(reinterpret_cast<const s16::Lab4*>(&lab[b*16]), reinterpret_cast<s16::RGB4*>(&rgb_out[b*16])); return rgb_out[b*16].r; });

	run4("okhsl→srgb",
		[&](int i) { auto R = cv4::HSL{hsl[i].h,hsl[i].s,hsl[i].l}.to_srgb(); return R.r+R.g+R.b; },
		[&](int b) { s4 ::okhsl_to_srgb_batch4(&hsl[b*4], &rgb_out[b*4]); return rgb_out[b*4].r; },
		[&](int b) { v4e::okhsl_to_srgb_batch4(reinterpret_cast<const v4e::HSL4*>(&hsl[b*4]), reinterpret_cast<v4e::RGB4*>(&rgb_out[b*4])); return rgb_out[b*4].r; },
		[&](int b) { s16::okhsl_to_srgb_batch16(reinterpret_cast<const s16::HSL4*>(&hsl[b*16]), reinterpret_cast<s16::RGB4*>(&rgb_out[b*16])); return rgb_out[b*16].r; });

	run4("srgb→okhsl",
		[&](int i) { auto H = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsl(); return H.h+H.s+H.l; },
		[&](int b) { s4 ::srgb_to_okhsl_batch4(&rgb[b*4], &hsl_out[b*4]); return hsl_out[b*4].h; },
		[&](int b) { v4e::srgb_to_okhsl_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]), reinterpret_cast<v4e::HSL4*>(&hsl_out[b*4])); return hsl_out[b*4].h; },
		[&](int b) { s16::srgb_to_okhsl_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]), reinterpret_cast<s16::HSL4*>(&hsl_out[b*16])); return hsl_out[b*16].h; });

	run4("okhsv→srgb",
		[&](int i) { auto R = cv4::HSV{hsv[i].h,hsv[i].s,hsv[i].v}.to_srgb(); return R.r+R.g+R.b; },
		[&](int b) { s4 ::okhsv_to_srgb_batch4(&hsv[b*4], &rgb_out[b*4]); return rgb_out[b*4].r; },
		[&](int b) { v4e::okhsv_to_srgb_batch4(reinterpret_cast<const v4e::HSV4*>(&hsv[b*4]), reinterpret_cast<v4e::RGB4*>(&rgb_out[b*4])); return rgb_out[b*4].r; },
		[&](int b) { s16::okhsv_to_srgb_batch16(reinterpret_cast<const s16::HSV4*>(&hsv[b*16]), reinterpret_cast<s16::RGB4*>(&rgb_out[b*16])); return rgb_out[b*16].r; });

	run4("srgb→okhsv",
		[&](int i) { auto H = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsv(); return H.h+H.s+H.v; },
		[&](int b) { s4 ::srgb_to_okhsv_batch4(&rgb[b*4], &hsv_out[b*4]); return hsv_out[b*4].h; },
		[&](int b) { v4e::srgb_to_okhsv_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]), reinterpret_cast<v4e::HSV4*>(&hsv_out[b*4])); return hsv_out[b*4].h; },
		[&](int b) { s16::srgb_to_okhsv_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]), reinterpret_cast<s16::HSV4*>(&hsv_out[b*16])); return hsv_out[b*16].h; });

	std::printf("\n[sink] %g\n", (double)g_sink);
	return 0;
}
