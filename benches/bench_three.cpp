// bench_three.cpp — orig vs vec3 (현재) vs float4 (v4).
#include "ok_color.h"
#include "ok_color_v4.h"
#include "orig_ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

namespace ok = ok_color;
namespace ok4 = ok_color_v4;
using clock_t_ = std::chrono::steady_clock;

struct Sink { volatile float r=0,g=0,b=0; };
static Sink g_sink;

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

template <typename A, typename B, typename C>
static void bench3(const char* name, int rounds, int iters, A a, B b, C c)
{
	double t_o = bench_min(rounds, iters, a);
	double t_3 = bench_min(rounds, iters, b);
	double t_4 = bench_min(rounds, iters, c);
	std::printf("%-25s | %10.3f | %10.3f | %10.3f | %8.3fx | %8.3fx\n",
		name, t_o, t_3, t_4, t_3 / t_o, t_4 / t_3);
}

static std::vector<ok::RGB> g_rgb_in;
static std::vector<ok::Lab> g_lab_in;
static std::vector<ok::HSL> g_hsl_in;
static std::vector<ok::HSV> g_hsv_in;

int main()
{
	constexpr int N = 4096;
	constexpr int ROUNDS = 9;
	constexpr int ITERS = 2'000'000;

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	for (int i = 0; i < N; ++i) {
		ok::RGB rgb{U(rng), U(rng), U(rng)};
		g_rgb_in.push_back(rgb);
		g_lab_in.push_back(rgb.to_oklab());
		g_hsl_in.push_back(ok::HSL{U(rng), U(rng), U(rng)});
		g_hsv_in.push_back(ok::HSV{U(rng), U(rng), U(rng)});
	}

	std::printf("입력풀=%d, 라운드=%d, 반복=%d (min ns/op)\n\n", N, ROUNDS, ITERS);
	std::printf("%-25s | %10s | %10s | %10s | %9s | %9s\n",
		"function", "orig", "vec3", "float4", "vec3/orig", "v4/vec3");
	std::printf("--------------------------+------------+------------+------------+-----------+-----------\n");

	bench3("linear_srgb→oklab", ROUNDS, ITERS,
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				orig::Lab L = orig::linear_srgb_to_oklab({x.r,x.g,x.b});
				rs+=L.L; gs+=L.a; bs+=L.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				ok::Lab L = x.to_oklab();
				rs+=L.L; gs+=L.a; bs+=L.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				ok4::Lab L = ok4::RGB{x.r,x.g,x.b}.to_oklab();
				rs+=L.L; gs+=L.a; bs+=L.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; });

	bench3("oklab→linear_srgb", ROUNDS, ITERS,
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_lab_in[i & (N-1)];
				orig::RGB R = orig::oklab_to_linear_srgb({x.L,x.a,x.b});
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_lab_in[i & (N-1)];
				ok::RGB R = x.to_linear_srgb();
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_lab_in[i & (N-1)];
				ok4::RGB R = ok4::Lab{x.L,x.a,x.b}.to_linear_srgb();
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; });

	bench3("okhsl→srgb", ROUNDS, ITERS,
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_hsl_in[i & (N-1)];
				orig::RGB R = orig::okhsl_to_srgb({x.h,x.s,x.l});
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_hsl_in[i & (N-1)];
				ok::RGB R = x.to_srgb();
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_hsl_in[i & (N-1)];
				ok4::RGB R = ok4::HSL{x.h,x.s,x.l}.to_srgb();
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; });

	bench3("srgb→okhsl", ROUNDS, ITERS,
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				orig::HSL H = orig::srgb_to_okhsl({x.r,x.g,x.b});
				rs+=H.h; gs+=H.s; bs+=H.l;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				ok::HSL H = x.to_okhsl();
				rs+=H.h; gs+=H.s; bs+=H.l;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				ok4::HSL H = ok4::RGB{x.r,x.g,x.b}.to_okhsl();
				rs+=H.h; gs+=H.s; bs+=H.l;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; });

	bench3("okhsv→srgb", ROUNDS, ITERS,
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_hsv_in[i & (N-1)];
				orig::RGB R = orig::okhsv_to_srgb({x.h,x.s,x.v});
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_hsv_in[i & (N-1)];
				ok::RGB R = x.to_srgb();
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_hsv_in[i & (N-1)];
				ok4::RGB R = ok4::HSV{x.h,x.s,x.v}.to_srgb();
				rs+=R.r; gs+=R.g; bs+=R.b;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; });

	bench3("srgb→okhsv", ROUNDS, ITERS,
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				orig::HSV H = orig::srgb_to_okhsv({x.r,x.g,x.b});
				rs+=H.h; gs+=H.s; bs+=H.v;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				ok::HSV H = x.to_okhsv();
				rs+=H.h; gs+=H.s; bs+=H.v;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; },
		[&](int it){ float rs=0,gs=0,bs=0;
			for (int i = 0; i < it; ++i) {
				auto& x = g_rgb_in[i & (N-1)];
				ok4::HSV H = ok4::RGB{x.r,x.g,x.b}.to_okhsv();
				rs+=H.h; gs+=H.s; bs+=H.v;
			} g_sink.r=rs; g_sink.g=gs; g_sink.b=bs; });

	std::printf("\n[sink] r=%g g=%g b=%g\n",
		(double)g_sink.r, (double)g_sink.g, (double)g_sink.b);
	return 0;
}
