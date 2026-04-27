// 분리된 ok_color.h / ok_color.cpp 가 원본과 같은 결과를 내는지 빠르게 확인하는
// 스모크 테스트. 빌드:
//   g++ -std=c++17 -O2 ok_color.cpp smoke_test.cpp -o smoke_test
#include "ok_color.h"
#include <cstdio>
#include <cmath>
#include <string>

using namespace ok_color;

static int failed = 0;

static void check(const char* tag, float got, float want, float tol = 1e-4f)
{
	const float d = std::fabs(got - want);
	if (d > tol) {
		printf("FAIL  %-40s got=%.6f want=%.6f diff=%.2e\n", tag, got, want, d);
		++failed;
	} else {
		printf("ok    %-40s got=%.6f want=%.6f\n", tag, got, want);
	}
}

static void check_rgb(const char* tag, RGB got, RGB want, float tol = 1e-4f)
{
	check((std::string(tag) + ".r").c_str(), got.r, want.r, tol);
	check((std::string(tag) + ".g").c_str(), got.g, want.g, tol);
	check((std::string(tag) + ".b").c_str(), got.b, want.b, tol);
}

int main()
{
	// ---- 1. union 메모리 별칭 검증 ---------------------------------------
	{
		Lab lab(0.5f, 0.1f, -0.2f);
		check("union.Lab.L=vector.x", lab.vector.x, 0.5f, 0.f);
		check("union.Lab.a=vector.y", lab.vector.y, 0.1f, 0.f);
		check("union.Lab.b=vector.z", lab.vector.z, -0.2f, 0.f);

		// vec3 ctor 경로
		Lab lab2(vec3{0.7f, 0.05f, 0.05f});
		check("union.Lab(vec3).L", lab2.L, 0.7f, 0.f);
		check("union.Lab(vec3).a", lab2.a, 0.05f, 0.f);
	}

	// ---- 2. dot / 산술 sanity --------------------------------------------
	{
		vec3 a{1, 2, 3};
		vec3 b{4, -1, 0.5f};
		check("dot",          a.dot(b),       1*4 + 2*-1 + 3*0.5f);
		check("free_dot",     dot(a, b),      a.dot(b));
		check("scalar_left",  (2.f * a).x,    2.f);
		check("length",       vec3{3,4,0}.length(), 5.f);
	}

	// ---- 3. Lab ↔ linear sRGB round-trip ---------------------------------
	{
		const RGB inputs[] = {
			{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f},
			{0.5f, 0.7f, 0.2f}, {1.f, 1.f, 1.f}, {0.f, 0.f, 0.f},
		};
		for (RGB in : inputs) {
			Lab lab = in.to_oklab();
			RGB out = lab.to_linear_srgb();
			char tag[64];
			std::snprintf(tag, sizeof tag, "Lab_rt(%.2f,%.2f,%.2f)", in.r, in.g, in.b);
			check_rgb(tag, out, in);
		}
	}

	// ---- 4. OkHSL round-trip --------------------------------------------
	{
		const RGB inputs[] = {
			{1.f, 0.f, 0.f},
			{0.2f, 0.5f, 0.9f},
			{0.95f, 0.85f, 0.1f},
		};
		for (RGB in : inputs) {
			HSL hsl = in.to_okhsl();
			RGB out = hsl.to_srgb();
			char tag[64];
			std::snprintf(tag, sizeof tag, "HSL_rt(%.2f,%.2f,%.2f)", in.r, in.g, in.b);
			check_rgb(tag, out, in, 1e-3f);  // toe + transfer 함수 누적 오차
		}
	}

	// ---- 5. OkHSV round-trip --------------------------------------------
	{
		const RGB inputs[] = {
			{1.f, 0.f, 0.f},
			{0.2f, 0.5f, 0.9f},
			{0.95f, 0.85f, 0.1f},
		};
		for (RGB in : inputs) {
			HSV hsv = in.to_okhsv();
			RGB out = hsv.to_srgb();
			char tag[64];
			std::snprintf(tag, sizeof tag, "HSV_rt(%.2f,%.2f,%.2f)", in.r, in.g, in.b);
			check_rgb(tag, out, in, 1e-3f);
		}
	}

	// ---- 6. gamut clip — 색역 안 색은 변하지 않음 ------------------------
	{
		RGB in{0.3f, 0.4f, 0.5f};
		check_rgb("clip_preserve_inside", gamut_clip_preserve_chroma(in), in, 0.f);
		check_rgb("clip_0.5_inside",      gamut_clip_project_to_0_5(in),  in, 0.f);
	}

	// ---- 7. gamut clip — 밖의 색은 [0,1]^3 로 ----------------------------
	{
		RGB out_of_gamut{1.5f, -0.2f, 0.8f};
		RGB clipped = gamut_clip_preserve_chroma(out_of_gamut);
		printf("clip(out_of_gamut) = (%.3f, %.3f, %.3f)\n",
		       clipped.r, clipped.g, clipped.b);
	}

	if (failed == 0) {
		printf("\n  *** all checks passed ***\n");
		return 0;
	} else {
		printf("\n  *** %d checks FAILED ***\n", failed);
		return 1;
	}
}
