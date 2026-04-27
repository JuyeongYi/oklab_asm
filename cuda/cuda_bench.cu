// cuda_bench.cu — CUDA 커널 정확성·성능 검증.
//   원본 (CPU scalar) 와 CUDA 결과를 비교, throughput 측정.
//   메모리는 explicit cudaMalloc/cudaMemcpy/cudaFree (unified memory 안 씀).

#include "ok_color_cuda.cuh"
#include "../original/orig_ok_color.h"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

namespace cu = ok_color_cuda;
using clock_t_ = std::chrono::steady_clock;

#define CUDA_CHECK(x) do { cudaError_t _ce = (x); if (_ce != cudaSuccess) { \
	std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_ce), __FILE__, __LINE__); \
	std::exit(1); }} while(0)

// 런타임 cudaEvent 기반 GPU 시간 측정.
static float bench_kernel_ms(int rounds, std::function<void()> launch)
{
	cudaEvent_t s, e;
	CUDA_CHECK(cudaEventCreate(&s));
	CUDA_CHECK(cudaEventCreate(&e));
	float best = 1e9f;
	for (int r = 0; r < rounds; ++r) {
		CUDA_CHECK(cudaEventRecord(s));
		launch();
		CUDA_CHECK(cudaEventRecord(e));
		CUDA_CHECK(cudaEventSynchronize(e));
		float ms;
		CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
		best = std::min(best, ms);
	}
	CUDA_CHECK(cudaEventDestroy(s));
	CUDA_CHECK(cudaEventDestroy(e));
	return best;
}

int main()
{
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	std::printf("GPU: %s, compute %d.%d, %.1f GB, %d SMs\n\n",
		prop.name, prop.major, prop.minor,
		prop.totalGlobalMem / (1024.0*1024*1024),
		prop.multiProcessorCount);

	// ─────────────────────────────────────────────────────────────────────────
	// 입력 준비: 호스트 측 4M 픽셀 (16-byte 정렬).
	// ─────────────────────────────────────────────────────────────────────────
	constexpr int N = 4 * 1024 * 1024;  // 4M pixels
	std::vector<cu::RGB4> h_rgb(N);
	std::vector<cu::Lab4> h_lab(N);
	std::vector<cu::HSL4> h_hsl(N);
	std::vector<cu::HSV4> h_hsv(N);
	std::vector<cu::RGB4> h_rgb_out(N);
	std::vector<cu::Lab4> h_lab_out(N);
	std::vector<cu::HSL4> h_hsl_out(N);
	std::vector<cu::HSV4> h_hsv_out(N);

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);
	for (int i = 0; i < N; ++i) {
		h_rgb[i] = { U(rng), U(rng), U(rng), 0 };
		orig::Lab L = orig::linear_srgb_to_oklab({h_rgb[i].r, h_rgb[i].g, h_rgb[i].b});
		h_lab[i] = { L.L, L.a, L.b, 0 };
		h_hsl[i] = { U(rng), U(rng), U(rng), 0 };
		h_hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}

	// ─────────────────────────────────────────────────────────────────────────
	// 디바이스 메모리 — explicit cudaMalloc.
	// ─────────────────────────────────────────────────────────────────────────
	cu::RGB4 *d_rgb, *d_rgb_out;
	cu::Lab4 *d_lab, *d_lab_out;
	cu::HSL4 *d_hsl, *d_hsl_out;
	cu::HSV4 *d_hsv, *d_hsv_out;
	CUDA_CHECK(cudaMalloc(&d_rgb,     N * sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_rgb_out, N * sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_lab,     N * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_lab_out, N * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_hsl,     N * sizeof(cu::HSL4)));
	CUDA_CHECK(cudaMalloc(&d_hsl_out, N * sizeof(cu::HSL4)));
	CUDA_CHECK(cudaMalloc(&d_hsv,     N * sizeof(cu::HSV4)));
	CUDA_CHECK(cudaMalloc(&d_hsv_out, N * sizeof(cu::HSV4)));

	// H2D — explicit cudaMemcpy.
	CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), N * sizeof(cu::RGB4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_lab, h_lab.data(), N * sizeof(cu::Lab4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_hsl, h_hsl.data(), N * sizeof(cu::HSL4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_hsv, h_hsv.data(), N * sizeof(cu::HSV4), cudaMemcpyHostToDevice));

	// ─────────────────────────────────────────────────────────────────────────
	// 1) 정확성 — 64 픽셀 D2H 후 scalar 와 비교.
	// ─────────────────────────────────────────────────────────────────────────
	std::printf("=== 정확성 (CUDA vs orig scalar, max_abs over 64 pixels) ===\n\n");

	auto check_diff = [&](const char* name, std::function<float()> fn, float tol) {
		float ma = fn();
		bool pass = ma <= tol;
		std::printf("%-6s %-25s max_abs=%.3e  tol=%.0e\n",
			pass ? "ok" : "FAIL", name, ma, (double)tol);
	};

	cu::linear_srgb_to_oklab(d_rgb, d_lab_out, N);
	CUDA_CHECK(cudaMemcpy(h_lab_out.data(), d_lab_out, 64 * sizeof(cu::Lab4), cudaMemcpyDeviceToHost));
	check_diff("linear_srgb→oklab", [&](){
		float ma = 0;
		for (int i = 0; i < 64; ++i) {
			orig::Lab L = orig::linear_srgb_to_oklab({h_rgb[i].r, h_rgb[i].g, h_rgb[i].b});
			ma = std::fmax(ma, std::fabs(h_lab_out[i].L - L.L));
			ma = std::fmax(ma, std::fabs(h_lab_out[i].a - L.a));
			ma = std::fmax(ma, std::fabs(h_lab_out[i].b - L.b));
		}
		return ma;
	}, 1e-5f);

	cu::oklab_to_linear_srgb(d_lab, d_rgb_out, N);
	CUDA_CHECK(cudaMemcpy(h_rgb_out.data(), d_rgb_out, 64 * sizeof(cu::RGB4), cudaMemcpyDeviceToHost));
	check_diff("oklab→linear_srgb", [&](){
		float ma = 0;
		for (int i = 0; i < 64; ++i) {
			orig::RGB R = orig::oklab_to_linear_srgb({h_lab[i].L, h_lab[i].a, h_lab[i].b});
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].r - R.r));
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].g - R.g));
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].b - R.b));
		}
		return ma;
	}, 1e-5f);

	cu::okhsl_to_srgb(d_hsl, d_rgb_out, N);
	CUDA_CHECK(cudaMemcpy(h_rgb_out.data(), d_rgb_out, 64 * sizeof(cu::RGB4), cudaMemcpyDeviceToHost));
	check_diff("okhsl→srgb", [&](){
		float ma = 0;
		for (int i = 0; i < 64; ++i) {
			orig::RGB R = orig::okhsl_to_srgb({h_hsl[i].h, h_hsl[i].s, h_hsl[i].l});
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].r - R.r));
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].g - R.g));
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].b - R.b));
		}
		return ma;
	}, 1e-4f);

	cu::srgb_to_okhsl(d_rgb, d_hsl_out, N);
	CUDA_CHECK(cudaMemcpy(h_hsl_out.data(), d_hsl_out, 64 * sizeof(cu::HSL4), cudaMemcpyDeviceToHost));
	check_diff("srgb→okhsl", [&](){
		float ma = 0;
		for (int i = 0; i < 64; ++i) {
			orig::HSL H = orig::srgb_to_okhsl({h_rgb[i].r, h_rgb[i].g, h_rgb[i].b});
			ma = std::fmax(ma, std::fabs(h_hsl_out[i].h - H.h));
			ma = std::fmax(ma, std::fabs(h_hsl_out[i].s - H.s));
			ma = std::fmax(ma, std::fabs(h_hsl_out[i].l - H.l));
		}
		return ma;
	}, 1e-4f);

	cu::okhsv_to_srgb(d_hsv, d_rgb_out, N);
	CUDA_CHECK(cudaMemcpy(h_rgb_out.data(), d_rgb_out, 64 * sizeof(cu::RGB4), cudaMemcpyDeviceToHost));
	check_diff("okhsv→srgb", [&](){
		float ma = 0;
		for (int i = 0; i < 64; ++i) {
			orig::RGB R = orig::okhsv_to_srgb({h_hsv[i].h, h_hsv[i].s, h_hsv[i].v});
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].r - R.r));
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].g - R.g));
			ma = std::fmax(ma, std::fabs(h_rgb_out[i].b - R.b));
		}
		return ma;
	}, 1e-4f);

	cu::srgb_to_okhsv(d_rgb, d_hsv_out, N);
	CUDA_CHECK(cudaMemcpy(h_hsv_out.data(), d_hsv_out, 64 * sizeof(cu::HSV4), cudaMemcpyDeviceToHost));
	check_diff("srgb→okhsv", [&](){
		float ma = 0;
		for (int i = 0; i < 64; ++i) {
			orig::HSV H = orig::srgb_to_okhsv({h_rgb[i].r, h_rgb[i].g, h_rgb[i].b});
			ma = std::fmax(ma, std::fabs(h_hsv_out[i].h - H.h));
			ma = std::fmax(ma, std::fabs(h_hsv_out[i].s - H.s));
			ma = std::fmax(ma, std::fabs(h_hsv_out[i].v - H.v));
		}
		return ma;
	}, 1e-4f);

	// ─────────────────────────────────────────────────────────────────────────
	// 2) 성능 — kernel-only 시간 (cudaEvent), 입력은 이미 디바이스에 있음.
	// ─────────────────────────────────────────────────────────────────────────
	std::printf("\n=== 성능 (kernel only, %d pixels = %.1f MB I/O) ===\n",
		N, (N * sizeof(cu::RGB4) * 2) / (1024.0*1024));
	std::printf("%-25s | %10s | %10s | %12s\n", "function", "ms", "ns/pixel", "Gpx/s");
	std::printf("--------------------------+------------+------------+-------------\n");

	auto run = [&](const char* name, std::function<void()> launch) {
		// warmup
		for (int i = 0; i < 3; ++i) launch();
		float ms = bench_kernel_ms(7, launch);
		double ns_per = ms * 1e6 / N;
		double gpx = N / (ms * 1e6);
		std::printf("%-25s | %10.3f | %10.3f | %12.3f\n", name, ms, ns_per, gpx);
	};

	run("linear_srgb→oklab",   [&](){ cu::linear_srgb_to_oklab(d_rgb, d_lab_out, N); });
	run("oklab→linear_srgb",   [&](){ cu::oklab_to_linear_srgb(d_lab, d_rgb_out, N); });
	run("okhsl→srgb",          [&](){ cu::okhsl_to_srgb(d_hsl, d_rgb_out, N); });
	run("srgb→okhsl",          [&](){ cu::srgb_to_okhsl(d_rgb, d_hsl_out, N); });
	run("okhsv→srgb",          [&](){ cu::okhsv_to_srgb(d_hsv, d_rgb_out, N); });
	run("srgb→okhsv",          [&](){ cu::srgb_to_okhsv(d_rgb, d_hsv_out, N); });
	run("clip:preserve_chroma",[&](){ cu::gamut_clip_preserve_chroma(d_rgb, d_rgb_out, N); });

	// H2D + 커널 + D2H 풀 round-trip 시간도 별도로.
	std::printf("\n=== 성능 (full round-trip: H2D + kernel + D2H) ===\n");
	std::printf("%-25s | %10s | %10s\n", "function", "ms", "ns/pixel");
	std::printf("--------------------------+------------+------------\n");
	auto run_rt = [&](const char* name, std::function<void()> launch_with_copies) {
		for (int i = 0; i < 3; ++i) launch_with_copies();
		float ms = bench_kernel_ms(7, launch_with_copies);
		double ns_per = ms * 1e6 / N;
		std::printf("%-25s | %10.3f | %10.3f\n", name, ms, ns_per);
	};
	run_rt("linear_srgb→oklab", [&](){
		cudaMemcpy(d_rgb, h_rgb.data(), N * sizeof(cu::RGB4), cudaMemcpyHostToDevice);
		cu::linear_srgb_to_oklab(d_rgb, d_lab_out, N);
		cudaMemcpy(h_lab_out.data(), d_lab_out, N * sizeof(cu::Lab4), cudaMemcpyDeviceToHost);
	});
	run_rt("srgb→okhsv", [&](){
		cudaMemcpy(d_rgb, h_rgb.data(), N * sizeof(cu::RGB4), cudaMemcpyHostToDevice);
		cu::srgb_to_okhsv(d_rgb, d_hsv_out, N);
		cudaMemcpy(h_hsv_out.data(), d_hsv_out, N * sizeof(cu::HSV4), cudaMemcpyDeviceToHost);
	});

	// ─────────────────────────────────────────────────────────────────────────
	// 정리.
	// ─────────────────────────────────────────────────────────────────────────
	CUDA_CHECK(cudaFree(d_rgb));
	CUDA_CHECK(cudaFree(d_rgb_out));
	CUDA_CHECK(cudaFree(d_lab));
	CUDA_CHECK(cudaFree(d_lab_out));
	CUDA_CHECK(cudaFree(d_hsl));
	CUDA_CHECK(cudaFree(d_hsl_out));
	CUDA_CHECK(cudaFree(d_hsv));
	CUDA_CHECK(cudaFree(d_hsv_out));

	return 0;
}
