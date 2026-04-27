// simd_verify.cpp — ok_color_simd 의 정확성 + 성능 통합 검증.
//   원본(orig namespace) 과 batch SIMD 가 같은 입력에 대해
//     - 출력 차이 (max_abs / max_rel) 가 허용오차 안인지
//     - batch SIMD 가 scalar 대비 몇 배 빠른지
// 동시에 보고.
//
// 빌드:
//   g++ -std=c++17 -O3 -mavx2 -mfma -msse4.1 -flto -DNDEBUG \
//       ok_color_simd.cpp simd_verify.cpp -o simd_verify

#include "ok_color_simd.h"
#include "orig_ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

namespace ok4 = ok_color_simd;
using clock_t_ = std::chrono::steady_clock;

struct DiffStat {
	float max_abs = 0.f, max_rel = 0.f;
	int worst_idx = -1;
	void update(int idx, float a, float b) {
		float d = std::fabs(a - b);
		if (d > max_abs) { max_abs = d; worst_idx = idx; }
		float den = std::fmax(std::fabs(a), std::fabs(b));
		if (den > 0.f) max_rel = std::fmax(max_rel, d / den);
	}
};

static int total_failed = 0;
static void report(const char* name, int n, const DiffStat& d, float tol)
{
	bool pass = d.max_abs <= tol;
	std::printf("%-6s %-32s n=%3d  max_abs=%.3e  max_rel=%.3e  tol=%.0e\n",
		pass ? "ok" : "FAIL", name, n, d.max_abs, d.max_rel, (double)tol);
	if (!pass) ++total_failed;
}

// 입력 샘플러.  회색(R=G=B) 은 hue 가 정의되지 않아 inverse 비교가 무의미하므로 제외.
static std::vector<ok4::RGB4> samples_rgb(int n, uint32_t seed)
{
	std::vector<ok4::RGB4> out;
	const float corners[][3] = {
		{1,0,0},{0,1,0},{0,0,1},{1,1,0},{1,0,1},{0,1,1},
		{.25f,.5f,.75f},{.9f,.1f,.5f},{.1f,.9f,.5f},
	};
	for (auto& c : corners) out.push_back({c[0], c[1], c[2], 0});
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	while ((int)out.size() < n) {
		float r = U(rng), g = U(rng), b = U(rng);
		// 회색 근사 입력은 hue 계산이 noise 가 되므로 건너뜀.
		float mx = std::fmax(std::fmax(r, g), b), mn = std::fmin(std::fmin(r, g), b);
		if (mx - mn < 0.05f) continue;
		out.push_back({ r, g, b, 0 });
	}
	return out;
}
static std::vector<ok4::HSL4> samples_hsl(int n, uint32_t seed)
{
	std::vector<ok4::HSL4> out;
	const float corners[][3] = {
		{0,0,0},{0,0,1},{0,1,.5f},{.25f,1,.5f},{.5f,1,.5f},{.75f,1,.5f},
		{.5f,0,.3f},{.999f,.999f,.999f},{.001f,.001f,.001f}
	};
	for (auto& c : corners) out.push_back({c[0],c[1],c[2],0});
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	while ((int)out.size() < n) out.push_back({ U(rng), U(rng), U(rng), 0 });
	return out;
}
static std::vector<ok4::HSV4> samples_hsv(int n, uint32_t seed)
{
	std::vector<ok4::HSV4> out;
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	for (int i = 0; i < n; ++i) out.push_back({ U(rng), U(rng), U(rng), 0 });
	return out;
}
static std::vector<ok4::RGB4> samples_clip(int n, uint32_t seed)
{
	std::vector<ok4::RGB4> out;
	const float oog[][3] = {
		{1.5f,-0.2f,0.8f},{1.2f,1.2f,-0.1f},{2.0f,0.0f,0.0f},
		{0.5f,0.5f,0.5f},{0.95f,0.5f,0.05f}
	};
	for (auto& c : oog) out.push_back({c[0],c[1],c[2],0});
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> U(-0.3f, 1.3f);
	while ((int)out.size() < n) out.push_back({ U(rng), U(rng), U(rng), 0 });
	return out;
}

// =============================================================================
// 정확성 비교 — 한 그룹당 100개 입력.  4개씩 batch SIMD 호출.
// =============================================================================

static void verify_linear_srgb_to_oklab()
{
	auto in = samples_rgb(100, 0xC0DE0001u);
	while (in.size() % 4) in.push_back({0,0,0,0});
	std::vector<ok4::Lab4> out_simd(in.size());
	for (size_t i = 0; i < in.size(); i += 4)
		ok4::linear_srgb_to_oklab_batch4(&in[i], &out_simd[i]);
	DiffStat d;
	for (int i = 0; i < 100; ++i) {
		orig::Lab L = orig::linear_srgb_to_oklab({in[i].r, in[i].g, in[i].b});
		d.update(i, out_simd[i].L, L.L);
		d.update(i, out_simd[i].a, L.a);
		d.update(i, out_simd[i].b, L.b);
	}
	report("linear_srgb→oklab", 100, d, 5e-5f);  // packed cbrt 근사 한계
}
static void verify_oklab_to_linear_srgb()
{
	auto rgb_in = samples_rgb(100, 0xC0DE0002u);
	std::vector<ok4::Lab4> in;
	for (auto& x : rgb_in) {
		orig::Lab L = orig::linear_srgb_to_oklab({x.r, x.g, x.b});
		in.push_back({L.L, L.a, L.b, 0});
	}
	while (in.size() % 4) in.push_back({0,0,0,0});
	std::vector<ok4::RGB4> out_simd(in.size());
	for (size_t i = 0; i < in.size(); i += 4)
		ok4::oklab_to_linear_srgb_batch4(&in[i], &out_simd[i]);
	DiffStat d;
	for (int i = 0; i < 100; ++i) {
		orig::RGB R = orig::oklab_to_linear_srgb({in[i].L, in[i].a, in[i].b});
		d.update(i, out_simd[i].r, R.r);
		d.update(i, out_simd[i].g, R.g);
		d.update(i, out_simd[i].b, R.b);
	}
	report("oklab→linear_srgb", 100, d, 1e-5f);
}
static void verify_okhsl_to_srgb()
{
	auto in = samples_hsl(100, 0xC0DE0003u);
	while (in.size() % 4) in.push_back({0,0,0,0});
	std::vector<ok4::RGB4> out_simd(in.size());
	for (size_t i = 0; i < in.size(); i += 4)
		ok4::okhsl_to_srgb_batch4(&in[i], &out_simd[i]);
	DiffStat d;
	for (int i = 0; i < 100; ++i) {
		orig::RGB R = orig::okhsl_to_srgb({in[i].h, in[i].s, in[i].l});
		d.update(i, out_simd[i].r, R.r);
		d.update(i, out_simd[i].g, R.g);
		d.update(i, out_simd[i].b, R.b);
	}
	report("okhsl→srgb", 100, d, 1e-3f);  // pow + sincos 누적
}
static void verify_srgb_to_okhsl()
{
	auto in = samples_rgb(100, 0xC0DE0004u);
	while (in.size() % 4) in.push_back({0,0,0,0});
	std::vector<ok4::HSL4> out_simd(in.size());
	for (size_t i = 0; i < in.size(); i += 4)
		ok4::srgb_to_okhsl_batch4(&in[i], &out_simd[i]);
	DiffStat d;
	for (int i = 0; i < 100; ++i) {
		orig::HSL H = orig::srgb_to_okhsl({in[i].r, in[i].g, in[i].b});
		d.update(i, out_simd[i].h, H.h);
		d.update(i, out_simd[i].s, H.s);
		d.update(i, out_simd[i].l, H.l);
	}
	report("srgb→okhsl", 100, d, 1e-3f);
	// 디버그: worst index 의 입력/출력 출력.
	if (d.worst_idx >= 0) {
		int i = d.worst_idx;
		orig::HSL H = orig::srgb_to_okhsl({in[i].r, in[i].g, in[i].b});
		std::printf("    worst@idx=%d  in=(%.4f, %.4f, %.4f)\n",
			i, in[i].r, in[i].g, in[i].b);
		std::printf("    scalar=(h=%.6f, s=%.6f, l=%.6f)\n", H.h, H.s, H.l);
		std::printf("    simd  =(h=%.6f, s=%.6f, l=%.6f)\n",
			out_simd[i].h, out_simd[i].s, out_simd[i].l);
	}
}
static void verify_okhsv_to_srgb()
{
	auto in = samples_hsv(100, 0xC0DE0005u);
	while (in.size() % 4) in.push_back({0,0,0,0});
	std::vector<ok4::RGB4> out_simd(in.size());
	for (size_t i = 0; i < in.size(); i += 4)
		ok4::okhsv_to_srgb_batch4(&in[i], &out_simd[i]);
	DiffStat d;
	for (int i = 0; i < 100; ++i) {
		orig::RGB R = orig::okhsv_to_srgb({in[i].h, in[i].s, in[i].v});
		d.update(i, out_simd[i].r, R.r);
		d.update(i, out_simd[i].g, R.g);
		d.update(i, out_simd[i].b, R.b);
	}
	report("okhsv→srgb", 100, d, 1e-3f);
}
static void verify_srgb_to_okhsv()
{
	auto in = samples_rgb(100, 0xC0DE0006u);
	while (in.size() % 4) in.push_back({0,0,0,0});
	std::vector<ok4::HSV4> out_simd(in.size());
	for (size_t i = 0; i < in.size(); i += 4)
		ok4::srgb_to_okhsv_batch4(&in[i], &out_simd[i]);
	DiffStat d;
	for (int i = 0; i < 100; ++i) {
		orig::HSV H = orig::srgb_to_okhsv({in[i].r, in[i].g, in[i].b});
		d.update(i, out_simd[i].h, H.h);
		d.update(i, out_simd[i].s, H.s);
		d.update(i, out_simd[i].v, H.v);
	}
	report("srgb→okhsv", 100, d, 1e-3f);
}

template <typename ClipFn, typename OrigClipFn>
static void verify_clip(const char* name, ClipFn simd_fn, OrigClipFn orig_fn, uint32_t seed)
{
	auto in = samples_clip(100, seed);
	while (in.size() % 4) in.push_back({0,0,0,0});
	std::vector<ok4::RGB4> out_simd(in.size());
	for (size_t i = 0; i < in.size(); i += 4)
		simd_fn(&in[i], &out_simd[i]);
	DiffStat d;
	for (int i = 0; i < 100; ++i) {
		orig::RGB R = orig_fn({in[i].r, in[i].g, in[i].b});
		d.update(i, out_simd[i].r, R.r);
		d.update(i, out_simd[i].g, R.g);
		d.update(i, out_simd[i].b, R.b);
	}
	report(name, 100, d, 1e-4f);
}

// =============================================================================
// 성능 측정.
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

static volatile float g_sink = 0.f;

int main()
{
	std::printf("=== ok_color_simd : 정확성 검증 ===\n");
	verify_linear_srgb_to_oklab();
	verify_oklab_to_linear_srgb();
	verify_okhsl_to_srgb();
	verify_srgb_to_okhsl();
	verify_okhsv_to_srgb();
	verify_srgb_to_okhsv();
	verify_clip("clip:preserve_chroma",
		ok4::gamut_clip_preserve_chroma_batch4, orig::gamut_clip_preserve_chroma, 0xC0DE0007u);
	verify_clip("clip:project_to_0.5",
		ok4::gamut_clip_project_to_0_5_batch4,  orig::gamut_clip_project_to_0_5,  0xC0DE0008u);
	verify_clip("clip:project_to_L_cusp",
		ok4::gamut_clip_project_to_L_cusp_batch4, orig::gamut_clip_project_to_L_cusp, 0xC0DE0009u);
	verify_clip("clip:adaptive_L0_0.5",
		[](const ok4::RGB4* in, ok4::RGB4* out){ ok4::gamut_clip_adaptive_L0_0_5_batch4(in, out, 0.05f); },
		[](orig::RGB rgb){ return orig::gamut_clip_adaptive_L0_0_5(rgb, 0.05f); },
		0xC0DE000Au);
	verify_clip("clip:adaptive_L0_L_cusp",
		[](const ok4::RGB4* in, ok4::RGB4* out){ ok4::gamut_clip_adaptive_L0_L_cusp_batch4(in, out, 0.05f); },
		[](orig::RGB rgb){ return orig::gamut_clip_adaptive_L0_L_cusp(rgb, 0.05f); },
		0xC0DE000Bu);

	std::printf("\n총 실패: %d\n", total_failed);

	// =========================================================================
	std::printf("\n=== 성능 비교 (min ns/pixel) ===\n\n");

	constexpr int N = 4096;
	constexpr int ROUNDS = 9;
	constexpr int ITERS_S = 1'000'000;
	constexpr int ITERS_B = 250'000;     // batch 1 회 = 4 픽셀

	std::vector<ok4::RGB4> rgb(N);
	std::vector<ok4::Lab4> lab(N);
	std::vector<ok4::HSL4> hsl(N);
	std::vector<ok4::HSV4> hsv(N);
	std::vector<ok4::RGB4> rgb_out(N);
	std::vector<ok4::Lab4> lab_out(N);
	std::vector<ok4::HSL4> hsl_out(N);
	std::vector<ok4::HSV4> hsv_out(N);

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	for (int i = 0; i < N; ++i) {
		rgb[i] = { U(rng), U(rng), U(rng), 0 };
		orig::Lab L = orig::linear_srgb_to_oklab({rgb[i].r, rgb[i].g, rgb[i].b});
		lab[i] = { L.L, L.a, L.b, 0 };
		hsl[i] = { U(rng), U(rng), U(rng), 0 };
		hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}

	std::printf("%-30s | %12s | %12s | %10s\n",
		"function", "scalar", "batch SIMD", "speedup");
	std::printf("-------------------------------+--------------+--------------+-----------\n");

	auto run = [&](const char* name, auto scalar_fn, auto simd_fn) {
		double t_s = bench_min(ROUNDS, ITERS_S, [&](int it){
			float acc = 0;
			const int mask = N - 1;
			for (int i = 0; i < it; ++i) acc += scalar_fn(i & mask);
			g_sink = acc;
		});
		double t_b = bench_min(ROUNDS, ITERS_B, [&](int it){
			float acc = 0;
			const int mask = (N/4) - 1;
			for (int i = 0; i < it; ++i) acc += simd_fn(i & mask);
			g_sink = acc;
		}) / 4.0;
		std::printf("%-30s | %12.3f | %12.3f | %9.3fx\n",
			name, t_s, t_b, t_s / t_b);
	};

	run("linear_srgb→oklab",
		[&](int i) {
			orig::Lab L = orig::linear_srgb_to_oklab({rgb[i].r, rgb[i].g, rgb[i].b});
			return L.L + L.a + L.b;
		},
		[&](int b) {
			ok4::linear_srgb_to_oklab_batch4(&rgb[b*4], &lab_out[b*4]);
			return lab_out[b*4].L + lab_out[b*4].a + lab_out[b*4].b;
		});
	run("oklab→linear_srgb",
		[&](int i) {
			orig::RGB R = orig::oklab_to_linear_srgb({lab[i].L, lab[i].a, lab[i].b});
			return R.r + R.g + R.b;
		},
		[&](int b) {
			ok4::oklab_to_linear_srgb_batch4(&lab[b*4], &rgb_out[b*4]);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		});
	run("okhsl→srgb",
		[&](int i) {
			orig::RGB R = orig::okhsl_to_srgb({hsl[i].h, hsl[i].s, hsl[i].l});
			return R.r + R.g + R.b;
		},
		[&](int b) {
			ok4::okhsl_to_srgb_batch4(&hsl[b*4], &rgb_out[b*4]);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		});
	run("srgb→okhsl",
		[&](int i) {
			orig::HSL H = orig::srgb_to_okhsl({rgb[i].r, rgb[i].g, rgb[i].b});
			return H.h + H.s + H.l;
		},
		[&](int b) {
			ok4::srgb_to_okhsl_batch4(&rgb[b*4], &hsl_out[b*4]);
			return hsl_out[b*4].h + hsl_out[b*4].s + hsl_out[b*4].l;
		});
	run("okhsv→srgb",
		[&](int i) {
			orig::RGB R = orig::okhsv_to_srgb({hsv[i].h, hsv[i].s, hsv[i].v});
			return R.r + R.g + R.b;
		},
		[&](int b) {
			ok4::okhsv_to_srgb_batch4(&hsv[b*4], &rgb_out[b*4]);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		});
	run("srgb→okhsv",
		[&](int i) {
			orig::HSV H = orig::srgb_to_okhsv({rgb[i].r, rgb[i].g, rgb[i].b});
			return H.h + H.s + H.v;
		},
		[&](int b) {
			ok4::srgb_to_okhsv_batch4(&rgb[b*4], &hsv_out[b*4]);
			return hsv_out[b*4].h + hsv_out[b*4].s + hsv_out[b*4].v;
		});

	// gamut clip 들 — 색역 밖 입력으로 측정.
	std::vector<ok4::RGB4> oog(N);
	std::uniform_real_distribution<float> U2(-0.3f, 1.3f);
	std::mt19937 rng2(0xF00D);
	for (int i = 0; i < N; ++i) oog[i] = { U2(rng2), U2(rng2), U2(rng2), 0 };

	run("clip:preserve_chroma",
		[&](int i) {
			orig::RGB R = orig::gamut_clip_preserve_chroma({oog[i].r, oog[i].g, oog[i].b});
			return R.r + R.g + R.b;
		},
		[&](int b) {
			ok4::gamut_clip_preserve_chroma_batch4(&oog[b*4], &rgb_out[b*4]);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		});
	run("clip:adaptive_L0_L_cusp",
		[&](int i) {
			orig::RGB R = orig::gamut_clip_adaptive_L0_L_cusp({oog[i].r, oog[i].g, oog[i].b}, 0.05f);
			return R.r + R.g + R.b;
		},
		[&](int b) {
			ok4::gamut_clip_adaptive_L0_L_cusp_batch4(&oog[b*4], &rgb_out[b*4], 0.05f);
			return rgb_out[b*4].r + rgb_out[b*4].g + rgb_out[b*4].b;
		});

	std::printf("\n[sink] %g\n", (double)g_sink);
	return total_failed == 0 ? 0 : 1;
}
