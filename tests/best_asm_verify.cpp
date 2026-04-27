// best_asm_verify.cpp — ok_color_best_asm 의 실제 동작·정확성·성능 검증.
//   1) 6개 변환을 ok_color_best_asm 으로 호출, 출력이 scalar 와 일치하는지 확인.
//   2) ns/pixel 측정해 hybrid 가 정말 best-of-breed 인지 확인.

#include "ok_color_best_asm.h"
#include "ok_color_v4.h"
#include "orig_ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

namespace best = ok_color_best;
using clock_t_ = std::chrono::steady_clock;

static volatile float g_sink = 0.f;
static int total_failed = 0;

template <typename F>
static double bench_min(int rounds, int iters, F&& f) {
	double best_ns = 1e18;
	for (int r = 0; r < rounds; ++r) {
		auto t0 = clock_t_::now();
		f(iters);
		auto t1 = clock_t_::now();
		double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
		best_ns = std::min(best_ns, ns / iters);
	}
	return best_ns;
}

int main()
{
	constexpr int N = 4096;
	constexpr int ROUNDS = 9;
	constexpr int ITERS = 62'500;  // 1M pixels

	std::vector<best::RGB4> rgb(N);
	std::vector<best::Lab4> lab(N);
	std::vector<best::HSL4> hsl(N);
	std::vector<best::HSV4> hsv(N);
	std::vector<best::RGB4> rgb_out(N);
	std::vector<best::Lab4> lab_out(N);
	std::vector<best::HSL4> hsl_out(N);
	std::vector<best::HSV4> hsv_out(N);

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);
	for (int i = 0; i < N; ++i) {
		rgb[i] = { U(rng), U(rng), U(rng), 0 };
		orig::Lab L = orig::linear_srgb_to_oklab({rgb[i].r, rgb[i].g, rgb[i].b});
		lab[i] = { L.L, L.a, L.b, 0 };
		hsl[i] = { U(rng), U(rng), U(rng), 0 };
		hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}

	// =========================================================================
	// 1) 정확성 — 6개 함수 모두 호출, scalar 와 비교.
	// =========================================================================
	std::printf("=== 정확성 검증 (16 픽셀 × 4 batch = 64 픽셀, scalar 와 비교) ===\n");

	auto check = [&](const char* name, float max_abs, float tol) {
		bool pass = max_abs <= tol;
		std::printf("%-6s %-30s max_abs=%.3e (tol=%.0e)\n",
			pass ? "ok" : "FAIL", name, max_abs, (double)tol);
		if (!pass) ++total_failed;
	};

	// 64 픽셀 = 4 batch 처리.
	for (int b = 0; b < 4; ++b) {
		best::linear_srgb_to_oklab_batch16(&rgb[b*16], &lab_out[b*16]);
	}
	float ma = 0;
	for (int i = 0; i < 64; ++i) {
		orig::Lab L = orig::linear_srgb_to_oklab({rgb[i].r, rgb[i].g, rgb[i].b});
		ma = std::fmax(ma, std::fabs(lab_out[i].L - L.L));
		ma = std::fmax(ma, std::fabs(lab_out[i].a - L.a));
		ma = std::fmax(ma, std::fabs(lab_out[i].b - L.b));
	}
	check("linear_srgb→oklab", ma, 5e-5f);

	for (int b = 0; b < 4; ++b)
		best::oklab_to_linear_srgb_batch16(&lab[b*16], &rgb_out[b*16]);
	ma = 0;
	for (int i = 0; i < 64; ++i) {
		orig::RGB R = orig::oklab_to_linear_srgb({lab[i].L, lab[i].a, lab[i].b});
		ma = std::fmax(ma, std::fabs(rgb_out[i].r - R.r));
		ma = std::fmax(ma, std::fabs(rgb_out[i].g - R.g));
		ma = std::fmax(ma, std::fabs(rgb_out[i].b - R.b));
	}
	check("oklab→linear_srgb", ma, 1e-5f);

	for (int b = 0; b < 4; ++b)
		best::okhsl_to_srgb_batch16(&hsl[b*16], &rgb_out[b*16]);
	ma = 0;
	for (int i = 0; i < 64; ++i) {
		orig::RGB R = orig::okhsl_to_srgb({hsl[i].h, hsl[i].s, hsl[i].l});
		ma = std::fmax(ma, std::fabs(rgb_out[i].r - R.r));
		ma = std::fmax(ma, std::fabs(rgb_out[i].g - R.g));
		ma = std::fmax(ma, std::fabs(rgb_out[i].b - R.b));
	}
	check("okhsl→srgb", ma, 1e-3f);

	for (int b = 0; b < 4; ++b)
		best::srgb_to_okhsl_batch16(&rgb[b*16], &hsl_out[b*16]);
	ma = 0;
	for (int i = 0; i < 64; ++i) {
		orig::HSL H = orig::srgb_to_okhsl({rgb[i].r, rgb[i].g, rgb[i].b});
		ma = std::fmax(ma, std::fabs(hsl_out[i].h - H.h));
		ma = std::fmax(ma, std::fabs(hsl_out[i].s - H.s));
		ma = std::fmax(ma, std::fabs(hsl_out[i].l - H.l));
	}
	check("srgb→okhsl", ma, 1e-3f);

	for (int b = 0; b < 4; ++b)
		best::okhsv_to_srgb_batch16(&hsv[b*16], &rgb_out[b*16]);
	ma = 0;
	for (int i = 0; i < 64; ++i) {
		orig::RGB R = orig::okhsv_to_srgb({hsv[i].h, hsv[i].s, hsv[i].v});
		ma = std::fmax(ma, std::fabs(rgb_out[i].r - R.r));
		ma = std::fmax(ma, std::fabs(rgb_out[i].g - R.g));
		ma = std::fmax(ma, std::fabs(rgb_out[i].b - R.b));
	}
	check("okhsv→srgb", ma, 1e-3f);

	for (int b = 0; b < 4; ++b)
		best::srgb_to_okhsv_batch16(&rgb[b*16], &hsv_out[b*16]);
	ma = 0;
	for (int i = 0; i < 64; ++i) {
		orig::HSV H = orig::srgb_to_okhsv({rgb[i].r, rgb[i].g, rgb[i].b});
		ma = std::fmax(ma, std::fabs(hsv_out[i].h - H.h));
		ma = std::fmax(ma, std::fabs(hsv_out[i].s - H.s));
		ma = std::fmax(ma, std::fabs(hsv_out[i].v - H.v));
	}
	check("srgb→okhsv", ma, 1e-3f);

	std::printf("\n총 실패: %d\n", total_failed);

	// =========================================================================
	// 2) 성능 — best_asm 단일 라이브러리만 호출했을 때의 ns/pixel.
	// =========================================================================
	std::printf("\n=== 성능 측정 — ok_color_best_asm 단일 호출 ===\n\n");

	auto run = [&](const char* name, auto fn) {
		double t = bench_min(ROUNDS, ITERS, [&](int it){
			float acc = 0;
			const int mask = (N/16) - 1;
			for (int i = 0; i < it; ++i) acc += fn(i & mask);
			g_sink = acc;
		}) / 16.0;
		std::printf("  %-30s  %8.3f ns/px\n", name, t);
	};

	run("linear_srgb→oklab", [&](int b){
		best::linear_srgb_to_oklab_batch16(&rgb[b*16], &lab_out[b*16]);
		return lab_out[b*16].L;
	});
	run("oklab→linear_srgb", [&](int b){
		best::oklab_to_linear_srgb_batch16(&lab[b*16], &rgb_out[b*16]);
		return rgb_out[b*16].r;
	});
	run("okhsl→srgb", [&](int b){
		best::okhsl_to_srgb_batch16(&hsl[b*16], &rgb_out[b*16]);
		return rgb_out[b*16].r;
	});
	run("srgb→okhsl", [&](int b){
		best::srgb_to_okhsl_batch16(&rgb[b*16], &hsl_out[b*16]);
		return hsl_out[b*16].h;
	});
	run("okhsv→srgb", [&](int b){
		best::okhsv_to_srgb_batch16(&hsv[b*16], &rgb_out[b*16]);
		return rgb_out[b*16].r;
	});
	run("srgb→okhsv", [&](int b){
		best::srgb_to_okhsv_batch16(&rgb[b*16], &hsv_out[b*16]);
		return hsv_out[b*16].h;
	});

	std::printf("\n[sink] %g\n", (double)g_sink);
	return total_failed == 0 ? 0 : 1;
}
