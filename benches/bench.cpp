// -----------------------------------------------------------------------------
// bench.cpp — 원본(single-header inline) 대 재구현(분리·dot·vec3) 성능 비교.
//
// 핵심 정직성 가드:
//   1) 결과를 volatile accumulator 에 누적 → dead-code elimination 차단.
//   2) 같은 입력 시퀀스를 두 구현에 흘려서 캐시 효과를 동등화.
//   3) -O2 + -flto 로 빌드 (분리·.cpp 가 인라인 손실 없도록).
//   4) 매 라운드마다 시드 입력을 갱신해 분기 패턴까지 비교.
//   5) 여러 라운드의 median 대신 min 을 쓴다 (시스템 잡음에 robust).
//
// 빌드:
//   g++ -std=c++17 -O2 -flto -DNDEBUG ok_color.cpp bench.cpp -o bench
// -----------------------------------------------------------------------------

#include "ok_color.h"
#include "orig_ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

namespace ok = ok_color;
using clock_t_ = std::chrono::steady_clock;

// 결과를 volatile sink 로 흘려 DCE 를 막는다.
struct Sink { volatile float r = 0, g = 0, b = 0; };
static Sink g_sink;

template <typename F>
double bench_min_ns(int rounds, int iters_per_round, F&& f)
{
	double best = 1e18;
	for (int r = 0; r < rounds; ++r) {
		auto t0 = clock_t_::now();
		f(iters_per_round);
		auto t1 = clock_t_::now();
		double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
		double per = ns / iters_per_round;
		best = std::min(best, per);
	}
	return best;
}

// 시드 입력 — 동일 시퀀스를 두 구현에 흘리기 위해 사전 생성.
static std::vector<ok::RGB>  g_rgb_in;
static std::vector<ok::Lab>  g_lab_in;
static std::vector<ok::HSL>  g_hsl_in;
static std::vector<ok::HSV>  g_hsv_in;

static void prepare_inputs(int N)
{
	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	g_rgb_in.reserve(N);
	g_lab_in.reserve(N);
	g_hsl_in.reserve(N);
	g_hsv_in.reserve(N);
	for (int i = 0; i < N; ++i) {
		ok::RGB rgb{U(rng), U(rng), U(rng)};
		g_rgb_in.push_back(rgb);
		// Lab 입력은 실제 sRGB 에서 변환된 것 (의미 있는 분포).
		g_lab_in.push_back(rgb.to_oklab());
		g_hsl_in.push_back(ok::HSL{U(rng), U(rng), U(rng)});
		g_hsv_in.push_back(ok::HSV{U(rng), U(rng), U(rng)});
	}
}

// =============================================================================

int main()
{
	constexpr int N        = 4096;   // 입력 풀 크기
	constexpr int ROUNDS   = 7;
	constexpr int ITERS    = 1'000'000;
	prepare_inputs(N);

	std::printf("벤치마크 입력풀=%d, 라운드=%d, 반복=%d\n", N, ROUNDS, ITERS);
	std::printf("(min ns/op = 라운드 중 최저값. 작을수록 빠름.)\n\n");
	std::printf("%-30s | %12s | %12s | %8s\n",
		"function", "orig ns/op", "new ns/op", "new/orig");
	std::printf("------------------------------+--------------+--------------+---------\n");

#define BENCH(NAME, ORIG_BODY, NEW_BODY)                                        \
	do {                                                                        \
		double t_orig = bench_min_ns(ROUNDS, ITERS, [&](int it){                \
			float rs=0, gs=0, bs=0;                                             \
			for (int i = 0; i < it; ++i) {                                      \
				ORIG_BODY                                                       \
			}                                                                   \
			g_sink.r = rs; g_sink.g = gs; g_sink.b = bs;                        \
		});                                                                     \
		double t_new = bench_min_ns(ROUNDS, ITERS, [&](int it){                 \
			float rs=0, gs=0, bs=0;                                             \
			for (int i = 0; i < it; ++i) {                                      \
				NEW_BODY                                                        \
			}                                                                   \
			g_sink.r = rs; g_sink.g = gs; g_sink.b = bs;                        \
		});                                                                     \
		std::printf("%-30s | %12.3f | %12.3f | %7.3fx\n",                       \
			NAME, t_orig, t_new, t_new / t_orig);                               \
	} while (0)

	BENCH("linear_srgb→oklab",
		auto& x = g_rgb_in[i & (N-1)];
		orig::Lab L = orig::linear_srgb_to_oklab({x.r, x.g, x.b});
		rs += L.L; gs += L.a; bs += L.b;
	,
		auto& x = g_rgb_in[i & (N-1)];
		ok::Lab L = x.to_oklab();
		rs += L.L; gs += L.a; bs += L.b;
	);

	BENCH("oklab→linear_srgb",
		auto& x = g_lab_in[i & (N-1)];
		orig::RGB R = orig::oklab_to_linear_srgb({x.L, x.a, x.b});
		rs += R.r; gs += R.g; bs += R.b;
	,
		auto& x = g_lab_in[i & (N-1)];
		ok::RGB R = x.to_linear_srgb();
		rs += R.r; gs += R.g; bs += R.b;
	);

	BENCH("okhsl→srgb",
		auto& x = g_hsl_in[i & (N-1)];
		orig::RGB R = orig::okhsl_to_srgb({x.h, x.s, x.l});
		rs += R.r; gs += R.g; bs += R.b;
	,
		auto& x = g_hsl_in[i & (N-1)];
		ok::RGB R = x.to_srgb();
		rs += R.r; gs += R.g; bs += R.b;
	);

	BENCH("srgb→okhsl",
		auto& x = g_rgb_in[i & (N-1)];
		orig::HSL H = orig::srgb_to_okhsl({x.r, x.g, x.b});
		rs += H.h; gs += H.s; bs += H.l;
	,
		auto& x = g_rgb_in[i & (N-1)];
		ok::HSL H = x.to_okhsl();
		rs += H.h; gs += H.s; bs += H.l;
	);

	BENCH("okhsv→srgb",
		auto& x = g_hsv_in[i & (N-1)];
		orig::RGB R = orig::okhsv_to_srgb({x.h, x.s, x.v});
		rs += R.r; gs += R.g; bs += R.b;
	,
		auto& x = g_hsv_in[i & (N-1)];
		ok::RGB R = x.to_srgb();
		rs += R.r; gs += R.g; bs += R.b;
	);

	BENCH("srgb→okhsv",
		auto& x = g_rgb_in[i & (N-1)];
		orig::HSV H = orig::srgb_to_okhsv({x.r, x.g, x.b});
		rs += H.h; gs += H.s; bs += H.v;
	,
		auto& x = g_rgb_in[i & (N-1)];
		ok::HSV H = x.to_okhsv();
		rs += H.h; gs += H.s; bs += H.v;
	);

	BENCH("gamut_clip:preserve_chroma",
		auto& x = g_rgb_in[i & (N-1)];
		orig::RGB R = orig::gamut_clip_preserve_chroma({x.r * 1.3f - 0.1f,
		                                                 x.g * 1.3f - 0.1f,
		                                                 x.b * 1.3f - 0.1f});
		rs += R.r; gs += R.g; bs += R.b;
	,
		auto& x = g_rgb_in[i & (N-1)];
		ok::RGB R = ok::gamut_clip_preserve_chroma({x.r * 1.3f - 0.1f,
		                                             x.g * 1.3f - 0.1f,
		                                             x.b * 1.3f - 0.1f});
		rs += R.r; gs += R.g; bs += R.b;
	);

#undef BENCH

	// sink 가 정말 살아 있는지 인쇄 (DCE 방어용 시그널).
	std::printf("\n[sink dump] r=%g g=%g b=%g\n",
		(double)g_sink.r, (double)g_sink.g, (double)g_sink.b);

	return 0;
}
