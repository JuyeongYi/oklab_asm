// -----------------------------------------------------------------------------
// compare_test.cpp — 원본 single-header 와 분리·재구현된 ok_color 가 같은 결과를
// 내는지 확인하는 회귀 테스트.
//
// 빌드:
//   sed 's/namespace ok_color/namespace orig/g' ok_color.original.utf8.h > orig_ok_color.h
//   g++ -std=c++17 -O2 ok_color.cpp compare_test.cpp -o compare_test
//
// 각 변환 함수마다 100개의 다양한 입력 (무작위 균일 + 코너 + 경계값)을 만들고,
// 두 구현의 출력 차이의 최댓값(absolute / relative)을 보고한다.
// -----------------------------------------------------------------------------

#include "ok_color.h"
#include "orig_ok_color.h"

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <string>

namespace ok = ok_color;

// =============================================================================
// 입력 샘플러
//   각 변환마다 100개. 구성:
//     - 코너/경계: 약 16개 (검정/흰색/RGB primaries/CMY/h=0,0.25,0.5,0.75 …)
//     - 무작위 균일: 약 84개
// =============================================================================

struct vec3f { float x, y, z; };

static std::vector<vec3f> samples_unit_cube(int n, uint32_t seed)
{
	std::vector<vec3f> out;
	out.reserve(n);

	// 코너/경계 — 색역 안. 16개.
	const vec3f corners[] = {
		{0,0,0}, {1,1,1},
		{1,0,0}, {0,1,0}, {0,0,1},
		{1,1,0}, {1,0,1}, {0,1,1},
		{0.5f,0.5f,0.5f}, {0.25f,0.25f,0.25f}, {0.75f,0.75f,0.75f},
		{1,0.5f,0}, {0,0.5f,1}, {0.5f,0,1}, {0.5f,1,0}, {1,0.25f,0.75f},
	};
	for (auto c : corners) out.push_back(c);

	// 무작위.
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	while ((int)out.size() < n) out.push_back({ U(rng), U(rng), U(rng) });

	return out;
}

// HSV/HSL 입력: hue 는 [0,1), s/v(또는 l) 은 [0,1].
// l=0, l=1, s=0, s=1 같은 경계도 일부러 섞는다.
static std::vector<vec3f> samples_hsx(int n, uint32_t seed)
{
	std::vector<vec3f> out;
	out.reserve(n);

	const vec3f boundary[] = {
		{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},  // 검정/흰색 (s=0)
		{0.0f, 1.0f, 0.5f}, {0.25f, 1.0f, 0.5f}, // 풀채도 hue 0/π/2
		{0.5f, 1.0f, 0.5f}, {0.75f, 1.0f, 0.5f}, // 풀채도 hue π/3π/2
		{0.5f, 0.0f, 0.3f}, {0.5f, 1.0f, 0.0f},  // 무채색/검정 가까이
		{0.999f, 0.999f, 0.999f}, {0.001f, 0.001f, 0.001f},
		{0.5f, 0.5f, 0.5f}, {0.1f, 0.9f, 0.9f},
	};
	for (auto b : boundary) out.push_back(b);

	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> U(0.f, 1.f);
	while ((int)out.size() < n) out.push_back({ U(rng), U(rng), U(rng) });

	return out;
}

// gamut clip 용 입력: 일부는 색역 밖.
static std::vector<vec3f> samples_clip(int n, uint32_t seed)
{
	std::vector<vec3f> out;
	out.reserve(n);

	// 색역 밖 코너들.
	const vec3f oog[] = {
		{1.5f, -0.2f, 0.8f}, {1.2f, 1.2f, -0.1f}, {-0.3f, 0.5f, 0.7f},
		{2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f}, {0.0f, 0.0f, 2.0f},
		{1.1f, 1.1f, 1.1f}, {-0.1f, -0.1f, -0.1f},
	};
	for (auto x : oog) out.push_back(x);

	// 색역 안 (no-op 분기 검증용).
	const vec3f inside[] = {
		{0.5f, 0.5f, 0.5f}, {0.1f, 0.2f, 0.3f},
		{0.95f, 0.5f, 0.05f}, {0.5f, 0.95f, 0.05f},
	};
	for (auto x : inside) out.push_back(x);

	// 무작위 — 50% 색역 밖, 50% 안.
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> Uin(0.f, 1.f);
	std::uniform_real_distribution<float> Uout(-0.3f, 1.3f);
	while ((int)out.size() < n) {
		bool oog_pick = ((int)out.size() & 1) != 0;
		out.push_back(oog_pick ? vec3f{Uout(rng), Uout(rng), Uout(rng)}
		                       : vec3f{Uin(rng), Uin(rng), Uin(rng)});
	}

	return out;
}

// =============================================================================
// 비교 헬퍼
// =============================================================================

struct DiffStat
{
	float max_abs = 0.f;
	float max_rel = 0.f;
	int   worst_idx = -1;

	void update(int idx, float a, float b)
	{
		const float d = std::fabs(a - b);
		if (d > max_abs) { max_abs = d; worst_idx = idx; }
		const float denom = std::fmax(std::fabs(a), std::fabs(b));
		if (denom > 0.f) {
			const float r = d / denom;
			if (r > max_rel) max_rel = r;
		}
	}
};

static int total_failed = 0;
static int total_cases  = 0;
static int total_tests  = 0;

// 0 에 가까운 차이는 부동소수점 라운드 오차이므로 1e-6 이내를 동등으로 본다.
// (정확히 같은 식·같은 순서로 계산하면 0 이 나오지만, dot 으로 묶는 과정에서 미묘한
//  곱셈/덧셈 순서가 달라질 수 있어 약간의 여유를 둔다.)
static constexpr float TOL_ABS = 1e-6f;

static void report(const char* name, int n, const DiffStat& d, float tol = TOL_ABS)
{
	const bool pass = d.max_abs <= tol;
	std::printf("%-6s %-30s n=%3d  max_abs=%.3e  max_rel=%.3e  worst@idx=%d\n",
		pass ? "ok" : "FAIL", name, n, d.max_abs, d.max_rel, d.worst_idx);
	if (!pass) ++total_failed;
	total_cases += n;
	++total_tests;
}

// =============================================================================
// 변환별 비교
// =============================================================================

static void test_linear_srgb_to_oklab()
{
	const auto in = samples_unit_cube(100, 0xC0DE0001u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB   nrgb{in[i].x, in[i].y, in[i].z};
		orig::RGB orgb{in[i].x, in[i].y, in[i].z};
		ok::Lab   nl = nrgb.to_oklab();
		orig::Lab ol = orig::linear_srgb_to_oklab(orgb);
		d.update(i, nl.L, ol.L);
		d.update(i, nl.a, ol.a);
		d.update(i, nl.b, ol.b);
	}
	report("linear_srgb→oklab", (int)in.size(), d);
}

static void test_oklab_to_linear_srgb()
{
	// Oklab 입력은 sRGB 입력에서 변환한 값 + 일부 합성된 값.
	auto rgb_in = samples_unit_cube(100, 0xC0DE0002u);
	std::vector<vec3f> in;
	in.reserve(100);
	for (auto& s : rgb_in) {
		orig::Lab L = orig::linear_srgb_to_oklab({s.x, s.y, s.z});
		in.push_back({L.L, L.a, L.b});
	}
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::Lab   nl{in[i].x, in[i].y, in[i].z};
		orig::Lab ol{in[i].x, in[i].y, in[i].z};
		ok::RGB   nr = nl.to_linear_srgb();
		orig::RGB orr = orig::oklab_to_linear_srgb(ol);
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("oklab→linear_srgb", (int)in.size(), d);
}

static void test_okhsl_to_srgb()
{
	const auto in = samples_hsx(100, 0xC0DE0003u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::HSL   nhsl{in[i].x, in[i].y, in[i].z};
		orig::HSL ohsl{in[i].x, in[i].y, in[i].z};
		ok::RGB   nr = nhsl.to_srgb();
		orig::RGB orr = orig::okhsl_to_srgb(ohsl);
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("okhsl→srgb", (int)in.size(), d);
}

static void test_srgb_to_okhsl()
{
	const auto in = samples_unit_cube(100, 0xC0DE0004u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB   nr{in[i].x, in[i].y, in[i].z};
		orig::RGB orr{in[i].x, in[i].y, in[i].z};
		ok::HSL   nhsl = nr.to_okhsl();
		orig::HSL ohsl = orig::srgb_to_okhsl(orr);
		d.update(i, nhsl.h, ohsl.h);
		d.update(i, nhsl.s, ohsl.s);
		d.update(i, nhsl.l, ohsl.l);
	}
	report("srgb→okhsl", (int)in.size(), d);
}

static void test_okhsv_to_srgb()
{
	const auto in = samples_hsx(100, 0xC0DE0005u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::HSV   nhsv{in[i].x, in[i].y, in[i].z};
		orig::HSV ohsv{in[i].x, in[i].y, in[i].z};
		ok::RGB   nr = nhsv.to_srgb();
		orig::RGB orr = orig::okhsv_to_srgb(ohsv);
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("okhsv→srgb", (int)in.size(), d);
}

static void test_srgb_to_okhsv()
{
	const auto in = samples_unit_cube(100, 0xC0DE0006u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB   nr{in[i].x, in[i].y, in[i].z};
		orig::RGB orr{in[i].x, in[i].y, in[i].z};
		ok::HSV   nhsv = nr.to_okhsv();
		orig::HSV ohsv = orig::srgb_to_okhsv(orr);
		d.update(i, nhsv.h, ohsv.h);
		d.update(i, nhsv.s, ohsv.s);
		d.update(i, nhsv.v, ohsv.v);
	}
	report("srgb→okhsv", (int)in.size(), d);
}

// 보너스: gamut clip 5종도 100개씩 비교.
static void test_gamut_clip_preserve_chroma()
{
	const auto in = samples_clip(100, 0xC0DE0007u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB nr = ok::gamut_clip_preserve_chroma({in[i].x, in[i].y, in[i].z});
		orig::RGB orr = orig::gamut_clip_preserve_chroma({in[i].x, in[i].y, in[i].z});
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("clip:preserve_chroma", (int)in.size(), d);
}

static void test_gamut_clip_project_to_0_5()
{
	const auto in = samples_clip(100, 0xC0DE0008u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB nr = ok::gamut_clip_project_to_0_5({in[i].x, in[i].y, in[i].z});
		orig::RGB orr = orig::gamut_clip_project_to_0_5({in[i].x, in[i].y, in[i].z});
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("clip:project_to_0.5", (int)in.size(), d);
}

static void test_gamut_clip_project_to_L_cusp()
{
	const auto in = samples_clip(100, 0xC0DE0009u);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB nr = ok::gamut_clip_project_to_L_cusp({in[i].x, in[i].y, in[i].z});
		orig::RGB orr = orig::gamut_clip_project_to_L_cusp({in[i].x, in[i].y, in[i].z});
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("clip:project_to_L_cusp", (int)in.size(), d);
}

static void test_gamut_clip_adaptive_L0_0_5()
{
	const auto in = samples_clip(100, 0xC0DE000Au);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB nr = ok::gamut_clip_adaptive_L0_0_5({in[i].x, in[i].y, in[i].z});
		orig::RGB orr = orig::gamut_clip_adaptive_L0_0_5({in[i].x, in[i].y, in[i].z});
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("clip:adaptive_L0_0.5", (int)in.size(), d);
}

static void test_gamut_clip_adaptive_L0_L_cusp()
{
	const auto in = samples_clip(100, 0xC0DE000Bu);
	DiffStat d;
	for (int i = 0; i < (int)in.size(); ++i) {
		ok::RGB nr = ok::gamut_clip_adaptive_L0_L_cusp({in[i].x, in[i].y, in[i].z});
		orig::RGB orr = orig::gamut_clip_adaptive_L0_L_cusp({in[i].x, in[i].y, in[i].z});
		d.update(i, nr.r, orr.r);
		d.update(i, nr.g, orr.g);
		d.update(i, nr.b, orr.b);
	}
	report("clip:adaptive_L0_L_cusp", (int)in.size(), d);
}

// =============================================================================

int main()
{
	std::printf("=== ok_color : original vs refactored 비교 ===\n");
	std::printf("입력 100개 / 변환, 허용오차 %.0e\n\n", (double)TOL_ABS);

	std::printf("[색공간 변환]\n");
	test_linear_srgb_to_oklab();
	test_oklab_to_linear_srgb();
	test_okhsl_to_srgb();
	test_srgb_to_okhsl();
	test_okhsv_to_srgb();
	test_srgb_to_okhsv();

	std::printf("\n[gamut clipping]\n");
	test_gamut_clip_preserve_chroma();
	test_gamut_clip_project_to_0_5();
	test_gamut_clip_project_to_L_cusp();
	test_gamut_clip_adaptive_L0_0_5();
	test_gamut_clip_adaptive_L0_L_cusp();

	std::printf("\n");
	std::printf("총 테스트: %d개,  총 입력 케이스: %d개\n", total_tests, total_cases);
	if (total_failed == 0) {
		std::printf("  *** PASS — 모든 변환이 원본과 1e-6 이내로 일치 ***\n");
		return 0;
	} else {
		std::printf("  *** %d 변환이 허용오차를 초과 ***\n", total_failed);
		return 1;
	}
}
