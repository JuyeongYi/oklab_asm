// master_all.cu — 100K 픽셀 한 방 시나리오, CPU 타임아웃 지원.
//
//   CPU 는 각 round 가 timeout_ms 를 넘기면 그 시점까지 처리한 픽셀 수와 함께 조기 종료.
//   GPU 는 단일 kernel launch (timeout 무관).
//
//   메모리: explicit cudaMalloc / cudaMemcpy / cudaFree (unified 안 씀).

#include "../scalar_vec3/ok_color.h"
#include "../scalar_vec4/ok_color_v4.h"
#include "../simd_sse/ok_color_simd.h"
#include "../simd_avx512_vl/ok_color_avx512_4.h"
#include "../simd_avx512/ok_color_avx512.h"
#include "../hybrid/ok_color_best_asm.h"
#include "../original/orig_ok_color.h"
#include "../cuda/ok_color_cuda.cuh"

#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

namespace ok3 = ok_color;
namespace cv4 = ok_color_v4;
namespace s4  = ok_color_simd;
namespace v4e = ok_color_avx512_4;
namespace s16 = ok_color_avx512;
namespace bst = ok_color_best;
namespace cu  = ok_color_cuda;

#define CUDA_CHECK(x) do { cudaError_t _ce = (x); if (_ce != cudaSuccess) { \
	std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_ce), __FILE__, __LINE__); \
	std::exit(1); }} while(0)

using clock_t_ = std::chrono::steady_clock;
static volatile float g_sink = 0.f;

constexpr double TIMEOUT_MS = 100.0;
constexpr int    ROUNDS     = 9;

// CPU 측 결과: ms 와 처리한 픽셀.  pixels < total 이면 timeout.
struct CpuResult { double ms; int pixels; bool timed_out; };

template <typename F>
static CpuResult bench_cpu(int total_pixels, F&& work)
{
	CpuResult best = { 1e18, total_pixels, false };
	for (int r = 0; r < ROUNDS; ++r) {
		auto deadline = clock_t_::now() + std::chrono::microseconds((int64_t)(TIMEOUT_MS * 1000));
		auto t0 = clock_t_::now();
		int pixels = work(deadline);
		auto t1 = clock_t_::now();
		double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (pixels < total_pixels) {
			// timed out — return immediately, don't average.
			return { ms, pixels, true };
		}
		if (ms < best.ms) best.ms = ms;
	}
	return best;
}

static float bench_gpu_min_ms(std::function<void()> launch) {
	cudaEvent_t s, e;
	CUDA_CHECK(cudaEventCreate(&s));
	CUDA_CHECK(cudaEventCreate(&e));
	float best = 1e9f;
	for (int r = 0; r < ROUNDS; ++r) {
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

// 출력 헬퍼: 시간 ms 또는 "TO X.XK" (timeout 시 처리한 픽셀 수).
static std::string fmt(const CpuResult& r) {
	char buf[32];
	if (r.timed_out) {
		std::snprintf(buf, sizeof buf, "TO%5.1fK", r.pixels / 1000.0);
	} else {
		std::snprintf(buf, sizeof buf, "%7.3f", r.ms);
	}
	return buf;
}

int main()
{
	constexpr int K   = 100'000;
	constexpr int K16 = (K + 15) & ~15;

	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	std::printf("=== 100K-픽셀 한 방 시나리오 (timeout=%g ms / round) ===\n", TIMEOUT_MS);
	std::printf("입력: %d 픽셀, %.2f MB I/O 한 방향\n", K,
		K * sizeof(s4::RGB4) / (1024.0 * 1024));
	std::printf("CPU: %d round 의 min ms.  TO 표기 = timeout, 처리된 픽셀(K) 단위\n",
		ROUNDS);
	std::printf("GPU: %s — single kernel launch + (옵션) H2D/D2H\n\n",
		prop.name);

	// 입력 준비.
	std::vector<s4::RGB4> rgb(K16);
	std::vector<s4::Lab4> lab(K16);
	std::vector<s4::HSL4> hsl(K16);
	std::vector<s4::HSV4> hsv(K16);
	std::vector<s4::RGB4> rgb_out(K16);
	std::vector<s4::Lab4> lab_out(K16);
	std::vector<s4::HSL4> hsl_out(K16);
	std::vector<s4::HSV4> hsv_out(K16);

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);
	for (int i = 0; i < K16; ++i) {
		rgb[i] = { U(rng), U(rng), U(rng), 0 };
		orig::Lab L = orig::linear_srgb_to_oklab({rgb[i].r, rgb[i].g, rgb[i].b});
		lab[i] = { L.L, L.a, L.b, 0 };
		hsl[i] = { U(rng), U(rng), U(rng), 0 };
		hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}

	// GPU 메모리.
	cu::RGB4 *d_rgb, *d_rgb_out;
	cu::Lab4 *d_lab, *d_lab_out;
	cu::HSL4 *d_hsl, *d_hsl_out;
	cu::HSV4 *d_hsv, *d_hsv_out;
	CUDA_CHECK(cudaMalloc(&d_rgb,     K16 * sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_rgb_out, K16 * sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_lab,     K16 * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_lab_out, K16 * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_hsl,     K16 * sizeof(cu::HSL4)));
	CUDA_CHECK(cudaMalloc(&d_hsl_out, K16 * sizeof(cu::HSL4)));
	CUDA_CHECK(cudaMalloc(&d_hsv,     K16 * sizeof(cu::HSV4)));
	CUDA_CHECK(cudaMalloc(&d_hsv_out, K16 * sizeof(cu::HSV4)));
	CUDA_CHECK(cudaMemcpy(d_rgb, rgb.data(), K * sizeof(cu::RGB4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_lab, lab.data(), K * sizeof(cu::Lab4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_hsl, hsl.data(), K * sizeof(cu::HSL4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_hsv, hsv.data(), K * sizeof(cu::HSV4), cudaMemcpyHostToDevice));

	// 헤더.
	std::printf("%-22s | %7s | %7s | %7s | %7s | %7s | %7s | %7s | %8s | %8s\n",
		"function", "orig", "vec3", "vec4", "SSE×4", "AVXVL×4", "AVX×16", "Hybrid", "GPU(K)", "GPU(RT)");
	std::printf("-----------------------+---------+---------+---------+---------+---------+---------+---------+----------+----------\n");

	using TP = std::chrono::steady_clock::time_point;

	// scalar 한 픽셀씩 처리, 256 픽셀마다 timeout 체크.
	auto loop_scalar = [&](auto&& body) {
		return [&, body](TP deadline) {
			float a = 0;
			for (int i = 0; i < K; ++i) {
				a += body(i);
				if ((i & 0xFF) == 0xFF && clock_t_::now() > deadline) {
					g_sink = a;
					return i + 1;
				}
			}
			g_sink = a;
			return K;
		};
	};
	// SIMD batch 4-pixel — 1024 픽셀마다 (256 batch) 체크.
	auto loop_b4 = [&](auto&& body) {
		return [&, body](TP deadline) {
			for (int i = 0; i < K16; i += 4) {
				body(i);
				if ((i & 0x3FF) == 0x3FC && clock_t_::now() > deadline) {
					g_sink = lab_out[0].L;
					return std::min(K, i + 4);
				}
			}
			g_sink = lab_out[0].L;
			return K;
		};
	};
	// SIMD batch 16-pixel — 4096 픽셀마다 체크.
	auto loop_b16 = [&](auto&& body) {
		return [&, body](TP deadline) {
			for (int i = 0; i < K16; i += 16) {
				body(i);
				if ((i & 0xFFF) == 0xFF0 && clock_t_::now() > deadline) {
					g_sink = lab_out[0].L;
					return std::min(K, i + 16);
				}
			}
			g_sink = lab_out[0].L;
			return K;
		};
	};

	auto run_row = [&](const char* name,
	                   auto orig_body, auto v3_body, auto v4_body,
	                   auto sse_body, auto avx4_body, auto avx16_body, auto hyb_body,
	                   std::function<void()> gpu_kernel,
	                   std::function<void()> gpu_round_trip) {
		auto t_o   = bench_cpu(K, loop_scalar(orig_body));
		auto t_v3  = bench_cpu(K, loop_scalar(v3_body));
		auto t_v4  = bench_cpu(K, loop_scalar(v4_body));
		auto t_s   = bench_cpu(K, loop_b4 (sse_body));
		auto t_a4  = bench_cpu(K, loop_b4 (avx4_body));
		auto t_a16 = bench_cpu(K, loop_b16(avx16_body));
		auto t_h   = bench_cpu(K, loop_b16(hyb_body));
		for (int i = 0; i < 3; ++i) gpu_kernel();
		float t_gk  = bench_gpu_min_ms(gpu_kernel);
		for (int i = 0; i < 3; ++i) gpu_round_trip();
		float t_grt = bench_gpu_min_ms(gpu_round_trip);
		std::printf("%-22s | %s | %s | %s | %s | %s | %s | %s | %8.3f | %8.3f\n",
			name,
			fmt(t_o).c_str(), fmt(t_v3).c_str(), fmt(t_v4).c_str(),
			fmt(t_s).c_str(), fmt(t_a4).c_str(), fmt(t_a16).c_str(), fmt(t_h).c_str(),
			t_gk, t_grt);
	};

	run_row("linear_srgb→oklab",
		[&](int i){ auto L = orig::linear_srgb_to_oklab({rgb[i].r,rgb[i].g,rgb[i].b}); return L.L; },
		[&](int i){ auto L = ok3::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_oklab(); return L.L; },
		[&](int i){ auto L = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_oklab(); return L.L; },
		[&](int i){ s4 ::linear_srgb_to_oklab_batch4(&rgb[i], &lab_out[i]); },
		[&](int i){ v4e::linear_srgb_to_oklab_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[i]), reinterpret_cast<v4e::Lab4*>(&lab_out[i])); },
		[&](int i){ s16::linear_srgb_to_oklab_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[i]), reinterpret_cast<s16::Lab4*>(&lab_out[i])); },
		[&](int i){ bst::linear_srgb_to_oklab_batch16(&rgb[i], &lab_out[i]); },
		[&](){ cu::linear_srgb_to_oklab(d_rgb, d_lab_out, K); },
		[&](){ cudaMemcpy(d_rgb, rgb.data(), K * sizeof(cu::RGB4), cudaMemcpyHostToDevice);
		       cu::linear_srgb_to_oklab(d_rgb, d_lab_out, K);
		       cudaMemcpy(lab.data(), d_lab_out, K * sizeof(cu::Lab4), cudaMemcpyDeviceToHost); });

	run_row("oklab→linear_srgb",
		[&](int i){ auto R = orig::oklab_to_linear_srgb({lab[i].L,lab[i].a,lab[i].b}); return R.r; },
		[&](int i){ auto R = ok3::Lab{lab[i].L,lab[i].a,lab[i].b}.to_linear_srgb(); return R.r; },
		[&](int i){ auto R = cv4::Lab{lab[i].L,lab[i].a,lab[i].b}.to_linear_srgb(); return R.r; },
		[&](int i){ s4 ::oklab_to_linear_srgb_batch4(&lab[i], &rgb_out[i]); },
		[&](int i){ v4e::oklab_to_linear_srgb_batch4(reinterpret_cast<const v4e::Lab4*>(&lab[i]), reinterpret_cast<v4e::RGB4*>(&rgb_out[i])); },
		[&](int i){ s16::oklab_to_linear_srgb_batch16(reinterpret_cast<const s16::Lab4*>(&lab[i]), reinterpret_cast<s16::RGB4*>(&rgb_out[i])); },
		[&](int i){ bst::oklab_to_linear_srgb_batch16(&lab[i], &rgb_out[i]); },
		[&](){ cu::oklab_to_linear_srgb(d_lab, d_rgb_out, K); },
		[&](){ cudaMemcpy(d_lab, lab.data(), K * sizeof(cu::Lab4), cudaMemcpyHostToDevice);
		       cu::oklab_to_linear_srgb(d_lab, d_rgb_out, K);
		       cudaMemcpy(rgb.data(), d_rgb_out, K * sizeof(cu::RGB4), cudaMemcpyDeviceToHost); });

	run_row("okhsl→srgb",
		[&](int i){ auto R = orig::okhsl_to_srgb({hsl[i].h,hsl[i].s,hsl[i].l}); return R.r; },
		[&](int i){ auto R = ok3::HSL{hsl[i].h,hsl[i].s,hsl[i].l}.to_srgb(); return R.r; },
		[&](int i){ auto R = cv4::HSL{hsl[i].h,hsl[i].s,hsl[i].l}.to_srgb(); return R.r; },
		[&](int i){ s4 ::okhsl_to_srgb_batch4(&hsl[i], &rgb_out[i]); },
		[&](int i){ v4e::okhsl_to_srgb_batch4(reinterpret_cast<const v4e::HSL4*>(&hsl[i]), reinterpret_cast<v4e::RGB4*>(&rgb_out[i])); },
		[&](int i){ s16::okhsl_to_srgb_batch16(reinterpret_cast<const s16::HSL4*>(&hsl[i]), reinterpret_cast<s16::RGB4*>(&rgb_out[i])); },
		[&](int i){ bst::okhsl_to_srgb_batch16(&hsl[i], &rgb_out[i]); },
		[&](){ cu::okhsl_to_srgb(d_hsl, d_rgb_out, K); },
		[&](){ cudaMemcpy(d_hsl, hsl.data(), K * sizeof(cu::HSL4), cudaMemcpyHostToDevice);
		       cu::okhsl_to_srgb(d_hsl, d_rgb_out, K);
		       cudaMemcpy(rgb.data(), d_rgb_out, K * sizeof(cu::RGB4), cudaMemcpyDeviceToHost); });

	run_row("srgb→okhsl",
		[&](int i){ auto H = orig::srgb_to_okhsl({rgb[i].r,rgb[i].g,rgb[i].b}); return H.h; },
		[&](int i){ auto H = ok3::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsl(); return H.h; },
		[&](int i){ auto H = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsl(); return H.h; },
		[&](int i){ s4 ::srgb_to_okhsl_batch4(&rgb[i], &hsl_out[i]); },
		[&](int i){ v4e::srgb_to_okhsl_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[i]), reinterpret_cast<v4e::HSL4*>(&hsl_out[i])); },
		[&](int i){ s16::srgb_to_okhsl_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[i]), reinterpret_cast<s16::HSL4*>(&hsl_out[i])); },
		[&](int i){ bst::srgb_to_okhsl_batch16(&rgb[i], &hsl_out[i]); },
		[&](){ cu::srgb_to_okhsl(d_rgb, d_hsl_out, K); },
		[&](){ cudaMemcpy(d_rgb, rgb.data(), K * sizeof(cu::RGB4), cudaMemcpyHostToDevice);
		       cu::srgb_to_okhsl(d_rgb, d_hsl_out, K);
		       cudaMemcpy(hsl.data(), d_hsl_out, K * sizeof(cu::HSL4), cudaMemcpyDeviceToHost); });

	run_row("okhsv→srgb",
		[&](int i){ auto R = orig::okhsv_to_srgb({hsv[i].h,hsv[i].s,hsv[i].v}); return R.r; },
		[&](int i){ auto R = ok3::HSV{hsv[i].h,hsv[i].s,hsv[i].v}.to_srgb(); return R.r; },
		[&](int i){ auto R = cv4::HSV{hsv[i].h,hsv[i].s,hsv[i].v}.to_srgb(); return R.r; },
		[&](int i){ s4 ::okhsv_to_srgb_batch4(&hsv[i], &rgb_out[i]); },
		[&](int i){ v4e::okhsv_to_srgb_batch4(reinterpret_cast<const v4e::HSV4*>(&hsv[i]), reinterpret_cast<v4e::RGB4*>(&rgb_out[i])); },
		[&](int i){ s16::okhsv_to_srgb_batch16(reinterpret_cast<const s16::HSV4*>(&hsv[i]), reinterpret_cast<s16::RGB4*>(&rgb_out[i])); },
		[&](int i){ bst::okhsv_to_srgb_batch16(&hsv[i], &rgb_out[i]); },
		[&](){ cu::okhsv_to_srgb(d_hsv, d_rgb_out, K); },
		[&](){ cudaMemcpy(d_hsv, hsv.data(), K * sizeof(cu::HSV4), cudaMemcpyHostToDevice);
		       cu::okhsv_to_srgb(d_hsv, d_rgb_out, K);
		       cudaMemcpy(rgb.data(), d_rgb_out, K * sizeof(cu::RGB4), cudaMemcpyDeviceToHost); });

	run_row("srgb→okhsv",
		[&](int i){ auto H = orig::srgb_to_okhsv({rgb[i].r,rgb[i].g,rgb[i].b}); return H.h; },
		[&](int i){ auto H = ok3::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsv(); return H.h; },
		[&](int i){ auto H = cv4::RGB{rgb[i].r,rgb[i].g,rgb[i].b}.to_okhsv(); return H.h; },
		[&](int i){ s4 ::srgb_to_okhsv_batch4(&rgb[i], &hsv_out[i]); },
		[&](int i){ v4e::srgb_to_okhsv_batch4(reinterpret_cast<const v4e::RGB4*>(&rgb[i]), reinterpret_cast<v4e::HSV4*>(&hsv_out[i])); },
		[&](int i){ s16::srgb_to_okhsv_batch16(reinterpret_cast<const s16::RGB4*>(&rgb[i]), reinterpret_cast<s16::HSV4*>(&hsv_out[i])); },
		[&](int i){ bst::srgb_to_okhsv_batch16(&rgb[i], &hsv_out[i]); },
		[&](){ cu::srgb_to_okhsv(d_rgb, d_hsv_out, K); },
		[&](){ cudaMemcpy(d_rgb, rgb.data(), K * sizeof(cu::RGB4), cudaMemcpyHostToDevice);
		       cu::srgb_to_okhsv(d_rgb, d_hsv_out, K);
		       cudaMemcpy(hsv.data(), d_hsv_out, K * sizeof(cu::HSV4), cudaMemcpyDeviceToHost); });

	std::printf("\n[sink] %g\n", (double)g_sink);

	CUDA_CHECK(cudaFree(d_rgb)); CUDA_CHECK(cudaFree(d_rgb_out));
	CUDA_CHECK(cudaFree(d_lab)); CUDA_CHECK(cudaFree(d_lab_out));
	CUDA_CHECK(cudaFree(d_hsl)); CUDA_CHECK(cudaFree(d_hsl_out));
	CUDA_CHECK(cudaFree(d_hsv)); CUDA_CHECK(cudaFree(d_hsv_out));
	return 0;
}
