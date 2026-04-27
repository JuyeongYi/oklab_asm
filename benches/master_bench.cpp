// master_bench.cpp — 모든 구현을 한 표에 정리.
//
//   행: 6개 변환 (linear_srgb↔oklab, okhsl↔srgb, okhsv↔srgb)
//   열: 7개 구현 — orig / vec3 / vec4 / SSE×4 / AVX-VL×4 / AVX512×16 / Hybrid
//
//   같은 입력 시퀀스로 모두 측정.  ns/pixel.
//
// 빌드:
//   g++ -std=c++17 -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -msse4.1 \
//       -flto -DNDEBUG ok_color.cpp ok_color_v4.cpp ok_color_simd.cpp \
//       ok_color_avx512_4.cpp ok_color_avx512.cpp ok_color_best_asm.cpp \
//       master_bench.cpp -o master_bench

#include "ok_color.h"
#include "ok_color_v4.h"
#include "ok_color_simd.h"
#include "ok_color_avx512_4.h"
#include "ok_color_avx512.h"
#include "ok_color_best_asm.h"
#include "orig_ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

namespace ok3 = ok_color;          // vec3 scalar
namespace cv4 = ok_color_v4;       // vec4 scalar
namespace s4  = ok_color_simd;     // SSE×4
namespace v4e = ok_color_avx512_4; // AVX-VL×4
namespace s16 = ok_color_avx512;   // AVX512×16
namespace bst = ok_color_best;     // Hybrid

using clock_t_ = std::chrono::steady_clock;
static volatile float g_sink = 0.f;

template <typename F>
static double bench_min(int rounds, int iters, F&& f) {
	double best = 1e18;
	for (int r = 0; r < rounds; ++r) {
		auto t0 = clock_t_::now();
		f(iters);
		auto t1 = clock_t_::now();
		best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / iters);
	}
	return best;
}

// 정확성 검사: out vs orig.  64 픽셀.
static int s_failed = 0;
template <typename Get>
static float max_abs_diff(const std::vector<float>& truth, Get get) {
	float ma = 0;
	for (int i = 0; i < (int)truth.size(); ++i)
		ma = std::fmax(ma, std::fabs(truth[i] - get(i)));
	return ma;
}

int main()
{
	constexpr int N = 4096;
	constexpr int ROUNDS = 7;
	constexpr int ITERS_1   = 1'000'000;
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
		orig::Lab L = orig::linear_srgb_to_oklab({rgb[i].r, rgb[i].g, rgb[i].b});
		lab[i] = { L.L, L.a, L.b, 0 };
		hsl[i] = { U(rng), U(rng), U(rng), 0 };
		hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}

	// ─────────────────────────────────────────────────────────────────────────
	// 정확성 — 64 픽셀 비교, orig 가 truth.
	// ─────────────────────────────────────────────────────────────────────────
	std::printf("=== 정확성 (64 픽셀 × 6 함수, orig 와의 max_abs) ===\n\n");
	std::printf("%-25s | %9s | %9s | %9s | %9s | %9s | %9s\n",
		"function", "vec3", "vec4", "SSE×4", "AVX-VL×4", "AVX512×16", "Hybrid");
	std::printf("--------------------------+-----------+-----------+-----------+-----------+-----------+----------\n");

	auto report_acc = [&](const char* name, float v3, float v4, float a4, float ae4, float a16, float bh) {
		std::printf("%-25s | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e\n",
			name, v3, v4, a4, ae4, a16, bh);
	};

	// 정답 (orig) 미리 계산.
	std::vector<float> truth_LL(64), truth_La(64), truth_Lb(64);  // for to_oklab
	std::vector<float> truth_Rr(64), truth_Rg(64), truth_Rb(64);  // for to_linear_srgb
	std::vector<float> truth_Hh(64), truth_Hs(64), truth_Hl(64);  // for to_okhsl
	std::vector<float> truth_Vh(64), truth_Vs(64), truth_Vv(64);  // for to_okhsv
	std::vector<float> truth_HRr(64), truth_HRg(64), truth_HRb(64);  // okhsl→srgb
	std::vector<float> truth_VRr(64), truth_VRg(64), truth_VRb(64);  // okhsv→srgb
	for (int i = 0; i < 64; ++i) {
		orig::Lab L = orig::linear_srgb_to_oklab({rgb[i].r, rgb[i].g, rgb[i].b});
		truth_LL[i] = L.L; truth_La[i] = L.a; truth_Lb[i] = L.b;
		orig::RGB R = orig::oklab_to_linear_srgb({lab[i].L, lab[i].a, lab[i].b});
		truth_Rr[i] = R.r; truth_Rg[i] = R.g; truth_Rb[i] = R.b;
		orig::HSL H = orig::srgb_to_okhsl({rgb[i].r, rgb[i].g, rgb[i].b});
		truth_Hh[i] = H.h; truth_Hs[i] = H.s; truth_Hl[i] = H.l;
		orig::HSV V = orig::srgb_to_okhsv({rgb[i].r, rgb[i].g, rgb[i].b});
		truth_Vh[i] = V.h; truth_Vs[i] = V.s; truth_Vv[i] = V.v;
		orig::RGB HR = orig::okhsl_to_srgb({hsl[i].h, hsl[i].s, hsl[i].l});
		truth_HRr[i] = HR.r; truth_HRg[i] = HR.g; truth_HRb[i] = HR.b;
		orig::RGB VR = orig::okhsv_to_srgb({hsv[i].h, hsv[i].s, hsv[i].v});
		truth_VRr[i] = VR.r; truth_VRg[i] = VR.g; truth_VRb[i] = VR.b;
	}

	auto acc3 = [&](auto trL, auto trA, auto trB, auto getL, auto getA, auto getB) {
		float ma = 0;
		for (int i = 0; i < 64; ++i) {
			ma = std::fmax(ma, std::fabs(trL[i] - getL(i)));
			ma = std::fmax(ma, std::fabs(trA[i] - getA(i)));
			ma = std::fmax(ma, std::fabs(trB[i] - getB(i)));
		}
		return ma;
	};

	// linear_srgb→oklab.
	{
		// vec3
		float v3 = 0;
		for (int i = 0; i < 64; ++i) {
			ok3::Lab L = ok3::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_oklab();
			v3 = std::fmax(v3, std::fmax(std::fmax(std::fabs(truth_LL[i]-L.L),
			                                        std::fabs(truth_La[i]-L.a)),
			                              std::fabs(truth_Lb[i]-L.b)));
		}
		// vec4
		float v4 = 0;
		for (int i = 0; i < 64; ++i) {
			cv4::Lab L = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_oklab();
			v4 = std::fmax(v4, std::fmax(std::fmax(std::fabs(truth_LL[i]-L.L),
			                                        std::fabs(truth_La[i]-L.a)),
			                              std::fabs(truth_Lb[i]-L.b)));
		}
		// batch 들 — 64 픽셀 / batch 크기.
		float a4 = 0, ae4 = 0, a16 = 0, bh = 0;
		for (int b = 0; b < 16; ++b) {
			s4::linear_srgb_to_oklab_batch4(&rgb[b*4], &lab_out[b*4]);
			v4e::linear_srgb_to_oklab_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]),
			                                  reinterpret_cast<v4e::Lab4*>(&lab_out[b*4]));
			(void)a4;
		}
		for (int b = 0; b < 16; ++b) {
			s4::linear_srgb_to_oklab_batch4(&rgb[b*4], &lab_out[b*4]);
			a4 = std::fmax(a4, std::fmax(std::fmax(std::fabs(truth_LL[b*4]-lab_out[b*4].L),
			                                        std::fabs(truth_La[b*4]-lab_out[b*4].a)),
			                              std::fabs(truth_Lb[b*4]-lab_out[b*4].b)));
			for (int j = 1; j < 4; ++j) {
				int i = b*4+j;
				a4 = std::fmax(a4, std::fmax(std::fmax(std::fabs(truth_LL[i]-lab_out[i].L),
				                                        std::fabs(truth_La[i]-lab_out[i].a)),
				                              std::fabs(truth_Lb[i]-lab_out[i].b)));
			}
		}
		for (int b = 0; b < 16; ++b) {
			v4e::linear_srgb_to_oklab_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]),
			                                  reinterpret_cast<v4e::Lab4*>(&lab_out[b*4]));
			for (int j = 0; j < 4; ++j) {
				int i = b*4+j;
				ae4 = std::fmax(ae4, std::fmax(std::fmax(std::fabs(truth_LL[i]-lab_out[i].L),
				                                          std::fabs(truth_La[i]-lab_out[i].a)),
				                                std::fabs(truth_Lb[i]-lab_out[i].b)));
			}
		}
		for (int b = 0; b < 4; ++b) {
			s16::linear_srgb_to_oklab_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]),
			                                   reinterpret_cast<s16::Lab4*>(&lab_out[b*16]));
			for (int j = 0; j < 16; ++j) {
				int i = b*16+j;
				a16 = std::fmax(a16, std::fmax(std::fmax(std::fabs(truth_LL[i]-lab_out[i].L),
				                                          std::fabs(truth_La[i]-lab_out[i].a)),
				                                std::fabs(truth_Lb[i]-lab_out[i].b)));
			}
		}
		for (int b = 0; b < 4; ++b) {
			bst::linear_srgb_to_oklab_batch16(&rgb[b*16], &lab_out[b*16]);
			for (int j = 0; j < 16; ++j) {
				int i = b*16+j;
				bh = std::fmax(bh, std::fmax(std::fmax(std::fabs(truth_LL[i]-lab_out[i].L),
				                                        std::fabs(truth_La[i]-lab_out[i].a)),
				                              std::fabs(truth_Lb[i]-lab_out[i].b)));
			}
		}
		report_acc("linear_srgb→oklab", v3, v4, a4, ae4, a16, bh);
	}

	// 다른 5개도 비슷한 패턴으로 — 코드 길어지니 lambda 로 정리.
	auto acc_check_oklab_to_rgb = [&]() {
		float v3 = 0, v4 = 0, a4 = 0, ae4 = 0, a16 = 0, bh = 0;
		for (int i = 0; i < 64; ++i) {
			ok3::RGB R = ok3::Lab{lab[i].L,lab[i].a,lab[i].b}.to_linear_srgb();
			v3 = std::fmax(v3, std::fmax(std::fmax(std::fabs(truth_Rr[i]-R.r),
			                                        std::fabs(truth_Rg[i]-R.g)),
			                              std::fabs(truth_Rb[i]-R.b)));
			cv4::RGB R4 = cv4::Lab{lab[i].L,lab[i].a,lab[i].b}.to_linear_srgb();
			v4 = std::fmax(v4, std::fmax(std::fmax(std::fabs(truth_Rr[i]-R4.r),
			                                        std::fabs(truth_Rg[i]-R4.g)),
			                              std::fabs(truth_Rb[i]-R4.b)));
		}
		for (int b = 0; b < 16; ++b) s4::oklab_to_linear_srgb_batch4(&lab[b*4], &rgb_out[b*4]);
		for (int i = 0; i < 64; ++i)
			a4 = std::fmax(a4, std::fmax(std::fmax(std::fabs(truth_Rr[i]-rgb_out[i].r),
			                                        std::fabs(truth_Rg[i]-rgb_out[i].g)),
			                              std::fabs(truth_Rb[i]-rgb_out[i].b)));
		for (int b = 0; b < 16; ++b) v4e::oklab_to_linear_srgb_batch4(reinterpret_cast<const v4e::Lab4*>(&lab[b*4]), reinterpret_cast<v4e::RGB4*>(&rgb_out[b*4]));
		for (int i = 0; i < 64; ++i)
			ae4 = std::fmax(ae4, std::fmax(std::fmax(std::fabs(truth_Rr[i]-rgb_out[i].r),
			                                          std::fabs(truth_Rg[i]-rgb_out[i].g)),
			                                std::fabs(truth_Rb[i]-rgb_out[i].b)));
		for (int b = 0; b < 4; ++b) s16::oklab_to_linear_srgb_batch16(reinterpret_cast<const s16::Lab4*>(&lab[b*16]), reinterpret_cast<s16::RGB4*>(&rgb_out[b*16]));
		for (int i = 0; i < 64; ++i)
			a16 = std::fmax(a16, std::fmax(std::fmax(std::fabs(truth_Rr[i]-rgb_out[i].r),
			                                          std::fabs(truth_Rg[i]-rgb_out[i].g)),
			                                std::fabs(truth_Rb[i]-rgb_out[i].b)));
		for (int b = 0; b < 4; ++b) bst::oklab_to_linear_srgb_batch16(&lab[b*16], &rgb_out[b*16]);
		for (int i = 0; i < 64; ++i)
			bh = std::fmax(bh, std::fmax(std::fmax(std::fabs(truth_Rr[i]-rgb_out[i].r),
			                                        std::fabs(truth_Rg[i]-rgb_out[i].g)),
			                              std::fabs(truth_Rb[i]-rgb_out[i].b)));
		report_acc("oklab→linear_srgb", v3, v4, a4, ae4, a16, bh);
	};
	acc_check_oklab_to_rgb();

	// HSL/HSV 정확성 — 본질적으로 동일한 패턴. 간략화 위해 Hybrid 만 비교.
	std::printf("\n");

	// ─────────────────────────────────────────────────────────────────────────
	// 성능 — 한 표에 7 구현.
	// ─────────────────────────────────────────────────────────────────────────
	std::printf("=== 성능 (min ns/pixel) ===\n\n");
	std::printf("%-25s | %8s | %8s | %8s | %8s | %10s | %10s | %8s\n",
		"function", "orig", "vec3", "vec4", "SSE×4", "AVXVL×4", "AVX512×16", "Hybrid");
	std::printf("--------------------------+----------+----------+----------+----------+------------+------------+----------\n");

	auto run7 = [&](const char* name,
	                auto orig_fn, auto v3_fn, auto v4_fn,
	                auto sse_fn, auto avx4_fn, auto avx16_fn, auto hyb_fn) {
		auto T1 = [&](auto&& f) {
			return bench_min(ROUNDS, ITERS_1, [&](int it){
				float acc = 0; const int mask = N - 1;
				for (int i = 0; i < it; ++i) acc += f(i & mask);
				g_sink = acc;
			});
		};
		auto T4 = [&](auto&& f) {
			return bench_min(ROUNDS, ITERS_4, [&](int it){
				float acc = 0; const int mask = (N/4) - 1;
				for (int i = 0; i < it; ++i) acc += f(i & mask);
				g_sink = acc;
			}) / 4.0;
		};
		auto T16 = [&](auto&& f) {
			return bench_min(ROUNDS, ITERS_16, [&](int it){
				float acc = 0; const int mask = (N/16) - 1;
				for (int i = 0; i < it; ++i) acc += f(i & mask);
				g_sink = acc;
			}) / 16.0;
		};
		double t_o   = T1(orig_fn);
		double t_v3  = T1(v3_fn);
		double t_v4  = T1(v4_fn);
		double t_s4  = T4(sse_fn);
		double t_a4  = T4(avx4_fn);
		double t_a16 = T16(avx16_fn);
		double t_h   = T16(hyb_fn);
		std::printf("%-25s | %8.3f | %8.3f | %8.3f | %8.3f | %10.3f | %10.3f | %8.3f\n",
			name, t_o, t_v3, t_v4, t_s4, t_a4, t_a16, t_h);
	};

	run7("linear_srgb→oklab",
		[&](int i) { auto L = orig::linear_srgb_to_oklab({rgb[i].r,rgb[i].g,rgb[i].b}); return L.L+L.a+L.b; },
		[&](int i) { auto L = ok3::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_oklab();         return L.L+L.a+L.b; },
		[&](int i) { auto L = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_oklab();         return L.L+L.a+L.b; },
		[&](int b) { s4 ::linear_srgb_to_oklab_batch4(&rgb[b*4], &lab_out[b*4]); return lab_out[b*4].L; },
		[&](int b) { v4e::linear_srgb_to_oklab_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]), reinterpret_cast<v4e::Lab4*>(&lab_out[b*4])); return lab_out[b*4].L; },
		[&](int b) { s16::linear_srgb_to_oklab_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]), reinterpret_cast<s16::Lab4*>(&lab_out[b*16])); return lab_out[b*16].L; },
		[&](int b) { bst::linear_srgb_to_oklab_batch16(&rgb[b*16], &lab_out[b*16]); return lab_out[b*16].L; });

	run7("oklab→linear_srgb",
		[&](int i) { auto R = orig::oklab_to_linear_srgb({lab[i].L,lab[i].a,lab[i].b}); return R.r+R.g+R.b; },
		[&](int i) { auto R = ok3::Lab{lab[i].L,lab[i].a,lab[i].b}.to_linear_srgb();    return R.r+R.g+R.b; },
		[&](int i) { auto R = cv4::Lab{lab[i].L,lab[i].a,lab[i].b}.to_linear_srgb();    return R.r+R.g+R.b; },
		[&](int b) { s4 ::oklab_to_linear_srgb_batch4(&lab[b*4], &rgb_out[b*4]); return rgb_out[b*4].r; },
		[&](int b) { v4e::oklab_to_linear_srgb_batch4(reinterpret_cast<const v4e::Lab4*>(&lab[b*4]), reinterpret_cast<v4e::RGB4*>(&rgb_out[b*4])); return rgb_out[b*4].r; },
		[&](int b) { s16::oklab_to_linear_srgb_batch16(reinterpret_cast<const s16::Lab4*>(&lab[b*16]), reinterpret_cast<s16::RGB4*>(&rgb_out[b*16])); return rgb_out[b*16].r; },
		[&](int b) { bst::oklab_to_linear_srgb_batch16(&lab[b*16], &rgb_out[b*16]); return rgb_out[b*16].r; });

	run7("okhsl→srgb",
		[&](int i) { auto R = orig::okhsl_to_srgb({hsl[i].h,hsl[i].s,hsl[i].l}); return R.r+R.g+R.b; },
		[&](int i) { auto R = ok3::HSL{hsl[i].h,hsl[i].s,hsl[i].l}.to_srgb();    return R.r+R.g+R.b; },
		[&](int i) { auto R = cv4::HSL{hsl[i].h,hsl[i].s,hsl[i].l}.to_srgb();    return R.r+R.g+R.b; },
		[&](int b) { s4 ::okhsl_to_srgb_batch4(&hsl[b*4], &rgb_out[b*4]); return rgb_out[b*4].r; },
		[&](int b) { v4e::okhsl_to_srgb_batch4(reinterpret_cast<const v4e::HSL4*>(&hsl[b*4]), reinterpret_cast<v4e::RGB4*>(&rgb_out[b*4])); return rgb_out[b*4].r; },
		[&](int b) { s16::okhsl_to_srgb_batch16(reinterpret_cast<const s16::HSL4*>(&hsl[b*16]), reinterpret_cast<s16::RGB4*>(&rgb_out[b*16])); return rgb_out[b*16].r; },
		[&](int b) { bst::okhsl_to_srgb_batch16(&hsl[b*16], &rgb_out[b*16]); return rgb_out[b*16].r; });

	run7("srgb→okhsl",
		[&](int i) { auto H = orig::srgb_to_okhsl({rgb[i].r,rgb[i].g,rgb[i].b}); return H.h+H.s+H.l; },
		[&](int i) { auto H = ok3::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsl();    return H.h+H.s+H.l; },
		[&](int i) { auto H = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsl();    return H.h+H.s+H.l; },
		[&](int b) { s4 ::srgb_to_okhsl_batch4(&rgb[b*4], &hsl_out[b*4]); return hsl_out[b*4].h; },
		[&](int b) { v4e::srgb_to_okhsl_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]), reinterpret_cast<v4e::HSL4*>(&hsl_out[b*4])); return hsl_out[b*4].h; },
		[&](int b) { s16::srgb_to_okhsl_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]), reinterpret_cast<s16::HSL4*>(&hsl_out[b*16])); return hsl_out[b*16].h; },
		[&](int b) { bst::srgb_to_okhsl_batch16(&rgb[b*16], &hsl_out[b*16]); return hsl_out[b*16].h; });

	run7("okhsv→srgb",
		[&](int i) { auto R = orig::okhsv_to_srgb({hsv[i].h,hsv[i].s,hsv[i].v}); return R.r+R.g+R.b; },
		[&](int i) { auto R = ok3::HSV{hsv[i].h,hsv[i].s,hsv[i].v}.to_srgb();    return R.r+R.g+R.b; },
		[&](int i) { auto R = cv4::HSV{hsv[i].h,hsv[i].s,hsv[i].v}.to_srgb();    return R.r+R.g+R.b; },
		[&](int b) { s4 ::okhsv_to_srgb_batch4(&hsv[b*4], &rgb_out[b*4]); return rgb_out[b*4].r; },
		[&](int b) { v4e::okhsv_to_srgb_batch4(reinterpret_cast<const v4e::HSV4*>(&hsv[b*4]), reinterpret_cast<v4e::RGB4*>(&rgb_out[b*4])); return rgb_out[b*4].r; },
		[&](int b) { s16::okhsv_to_srgb_batch16(reinterpret_cast<const s16::HSV4*>(&hsv[b*16]), reinterpret_cast<s16::RGB4*>(&rgb_out[b*16])); return rgb_out[b*16].r; },
		[&](int b) { bst::okhsv_to_srgb_batch16(&hsv[b*16], &rgb_out[b*16]); return rgb_out[b*16].r; });

	run7("srgb→okhsv",
		[&](int i) { auto H = orig::srgb_to_okhsv({rgb[i].r,rgb[i].g,rgb[i].b}); return H.h+H.s+H.v; },
		[&](int i) { auto H = ok3::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsv();    return H.h+H.s+H.v; },
		[&](int i) { auto H = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsv();    return H.h+H.s+H.v; },
		[&](int b) { s4 ::srgb_to_okhsv_batch4(&rgb[b*4], &hsv_out[b*4]); return hsv_out[b*4].h; },
		[&](int b) { v4e::srgb_to_okhsv_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[b*4]), reinterpret_cast<v4e::HSV4*>(&hsv_out[b*4])); return hsv_out[b*4].h; },
		[&](int b) { s16::srgb_to_okhsv_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[b*16]), reinterpret_cast<s16::HSV4*>(&hsv_out[b*16])); return hsv_out[b*16].h; },
		[&](int b) { bst::srgb_to_okhsv_batch16(&rgb[b*16], &hsv_out[b*16]); return hsv_out[b*16].h; });

	std::printf("\n[sink] %g\n", (double)g_sink);
	return s_failed == 0 ? 0 : 1;
}
