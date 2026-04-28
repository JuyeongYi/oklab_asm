// ok_color_cuda_ptx.cu — 인라인 PTX 로 cbrt 를 직접 짜본 시연 변형.
//
//   비교 대상:
//     (A) cbrtf       — correctly-rounded.  ~16 cycle.
//     (B) __cbrtf     — fast intrinsic.    ~8 cycle.   (nvcc --use_fast_math 와 같음)
//     (C) inline PTX  — lg2.approx + mul + ex2.approx.  ~6 cycle, 22-bit 정확도.
//
//   `__cbrtf` 는 헤더에 노출돼있지 않고 PTX 직접 호출이 안 되므로 (B) 는 사실
//   `__powf(x, 1.0f/3.0f)` 와 동등.
//
//   이 파일은 시연용. 실제 ok_color_cuda 에는 보수적인 `cbrtf` 사용.

#include "ok_color_cuda.cuh"
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

namespace ok_color_cuda { namespace ptx_demo {

// ─── 변형 (A): correctly-rounded cbrtf ────────────────────────────────────────
__device__ __forceinline__ float cbrt_correct(float x) {
	return cbrtf(x);
}

// ─── 변형 (B): fast intrinsic powf(x, 1/3) ──────────────────────────────────
__device__ __forceinline__ float cbrt_fast_intrinsic(float x) {
	return __powf(x, 1.0f / 3.0f);
}

// ─── 변형 (C): 직접 인라인 PTX ──────────────────────────────────────────────
//
//   cbrt(x) = exp2(log2(x) / 3).
//   PTX:   lg2.approx.f32  t, x;
//          mul.f32         t, t, 0.33333333;
//          ex2.approx.f32  result, t;
//
//   approx.f32 명령은 IEEE 정확하지 않지만 색상 변환 (8-bit 출력) 에는 충분 (1 ULP 안).
//   x ≤ 0 처리는 호출자 책임 (color 변환에서 LMS 가 항상 양수라 안전).
__device__ __forceinline__ float cbrt_ptx(float x) {
	float t, result;
	asm volatile("lg2.approx.f32 %0, %1;" : "=f"(t) : "f"(x));
	t *= 0.33333333333f;
	asm volatile("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(t));
	return result;
}

// 위 세 cbrt 변형으로 to_oklab 을 만들어내는 매크로.
#define MAKE_KERNEL(NAME, CBRT_FN) \
__global__ void k_##NAME(const RGB4* __restrict__ in, Lab4* __restrict__ out, int N) { \
	int idx = blockIdx.x * blockDim.x + threadIdx.x; \
	if (idx >= N) return; \
	RGB4 c = in[idx]; \
	float l = 0.4122214708f*c.r + 0.5363325363f*c.g + 0.0514459929f*c.b; \
	float m = 0.2119034982f*c.r + 0.6806995451f*c.g + 0.1073969566f*c.b; \
	float s = 0.0883024619f*c.r + 0.2817188376f*c.g + 0.6299787005f*c.b; \
	l = CBRT_FN(l); m = CBRT_FN(m); s = CBRT_FN(s); \
	Lab4 o; \
	o.L = 0.2104542553f*l + 0.7936177850f*m - 0.0040720468f*s; \
	o.a = 1.9779984951f*l - 2.4285922050f*m + 0.4505937099f*s; \
	o.b = 0.0259040371f*l + 0.7827717662f*m - 0.8086757660f*s; \
	o._pad = 0.f; \
	out[idx] = o; \
}

MAKE_KERNEL(correct,         cbrt_correct)
MAKE_KERNEL(fast_intrinsic,  cbrt_fast_intrinsic)
MAKE_KERNEL(inline_ptx,      cbrt_ptx)

}} // namespace

// =============================================================================

#define CUDA_CHECK(x) do { cudaError_t _ce = (x); if (_ce != cudaSuccess) { \
	std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_ce), __FILE__, __LINE__); \
	std::exit(1); }} while(0)

namespace cu  = ok_color_cuda;
using namespace ok_color_cuda::ptx_demo;
using clock_t_ = std::chrono::steady_clock;

static float bench_kernel_ms(int rounds, std::function<void()> launch) {
	cudaEvent_t s, e;
	CUDA_CHECK(cudaEventCreate(&s));
	CUDA_CHECK(cudaEventCreate(&e));
	float best = 1e9f;
	for (int r = 0; r < rounds; ++r) {
		CUDA_CHECK(cudaEventRecord(s));
		launch();
		CUDA_CHECK(cudaEventRecord(e));
		CUDA_CHECK(cudaEventSynchronize(e));
		float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
		best = std::min(best, ms);
	}
	CUDA_CHECK(cudaEventDestroy(s)); CUDA_CHECK(cudaEventDestroy(e));
	return best;
}

int main()
{
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	std::printf("GPU: %s, compute %d.%d\n\n", prop.name, prop.major, prop.minor);

	constexpr int N = 4 * 1024 * 1024;  // 4M pixels
	constexpr int ROUNDS = 11;

	std::vector<cu::RGB4> h_rgb(N);
	std::vector<cu::Lab4> h_lab_correct(N), h_lab_fast(N), h_lab_ptx(N);

	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);
	for (int i = 0; i < N; ++i) h_rgb[i] = { U(rng), U(rng), U(rng), 0 };

	cu::RGB4* d_rgb;
	cu::Lab4 *d_correct, *d_fast, *d_ptx;
	CUDA_CHECK(cudaMalloc(&d_rgb,     N * sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_correct, N * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_fast,    N * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_ptx,     N * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), N * sizeof(cu::RGB4), cudaMemcpyHostToDevice));

	int b = 256;
	dim3 grid((N + b - 1) / b);

	auto launch_correct = [&](){ cu::ptx_demo::k_correct<<<grid, b>>>(d_rgb, d_correct, N); };
	auto launch_fast    = [&](){ cu::ptx_demo::k_fast_intrinsic<<<grid, b>>>(d_rgb, d_fast, N); };
	auto launch_ptx     = [&](){ cu::ptx_demo::k_inline_ptx<<<grid, b>>>(d_rgb, d_ptx, N); };

	// 워밍업.
	for (int i = 0; i < 3; ++i) { launch_correct(); launch_fast(); launch_ptx(); }
	CUDA_CHECK(cudaDeviceSynchronize());

	float ms_correct = bench_kernel_ms(ROUNDS, launch_correct);
	float ms_fast    = bench_kernel_ms(ROUNDS, launch_fast);
	float ms_ptx     = bench_kernel_ms(ROUNDS, launch_ptx);

	// 결과 D2H.
	CUDA_CHECK(cudaMemcpy(h_lab_correct.data(), d_correct, N * sizeof(cu::Lab4), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_lab_fast.data(),    d_fast,    N * sizeof(cu::Lab4), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_lab_ptx.data(),     d_ptx,     N * sizeof(cu::Lab4), cudaMemcpyDeviceToHost));

	// correct 대비 fast/ptx 의 max_abs.
	float max_abs_fast = 0, max_abs_ptx = 0;
	for (int i = 0; i < N; ++i) {
		max_abs_fast = std::fmax(max_abs_fast, std::fmax(std::fmax(
			std::fabs(h_lab_correct[i].L - h_lab_fast[i].L),
			std::fabs(h_lab_correct[i].a - h_lab_fast[i].a)),
			std::fabs(h_lab_correct[i].b - h_lab_fast[i].b)));
		max_abs_ptx = std::fmax(max_abs_ptx, std::fmax(std::fmax(
			std::fabs(h_lab_correct[i].L - h_lab_ptx[i].L),
			std::fabs(h_lab_correct[i].a - h_lab_ptx[i].a)),
			std::fabs(h_lab_correct[i].b - h_lab_ptx[i].b)));
	}

	std::printf("=== linear_srgb→oklab @ 4M pixels — cbrt 변형 비교 ===\n\n");
	std::printf("%-30s | %10s | %12s | %14s\n", "variant", "ms", "Gpx/s", "max_abs vs cbrtf");
	std::printf("-------------------------------+------------+--------------+----------------\n");
	std::printf("%-30s | %10.3f | %12.2f | %14s\n", "(A) cbrtf (correctly-rounded)",
		ms_correct, N / (ms_correct * 1e6), "—");
	std::printf("%-30s | %10.3f | %12.2f | %14.3e\n", "(B) __powf(x, 1/3) intrinsic",
		ms_fast, N / (ms_fast * 1e6), max_abs_fast);
	std::printf("%-30s | %10.3f | %12.2f | %14.3e\n", "(C) inline PTX lg2/mul/ex2",
		ms_ptx, N / (ms_ptx * 1e6), max_abs_ptx);

	CUDA_CHECK(cudaFree(d_rgb));
	CUDA_CHECK(cudaFree(d_correct));
	CUDA_CHECK(cudaFree(d_fast));
	CUDA_CHECK(cudaFree(d_ptx));
	return 0;
}
