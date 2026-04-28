// ok_color_tensor.cu — Tensor Core MMA 로 linear_srgb_to_oklab 재구현.
//
//   설계:
//     - 한 warp (32 threads) = 16 픽셀 처리 (= MMA m16 타일).
//     - TF32 fragment (FP32 range, 10-bit mantissa) 로 정밀도 유지.
//     - 행렬곱 두 개를 wmma::mma_sync 두 번으로 처리 (RGB→LMS, LMS'→Lab).
//     - cbrt 는 여전히 scalar (Tensor Core 가 처리 불가).
//
//   Fragment 차원 (m16n16k8 TF32):
//     a_frag: 16×8     row_major  (input pixels, 첫 3 col 만 의미)
//     b_frag: 8×16     col_major  (matrix^T padded, 첫 3 row × 3 col 만 의미)
//     c_frag: 16×16   (output, 첫 3 col 만 의미)
//
//   Padding: 3×3 행렬 → 16×16 → 9/256 = 3.5% utilization.  MMA throughput 이
//   초과해서 절대 시간엔 영향 작음.

#include "../cuda/ok_color_cuda.cuh"
#include <mma.h>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

using namespace nvcuda;

namespace ok_color_cuda { namespace tensor {

// ─── M_RGB_TO_LMS^T  (3×3 행렬을 16×16 col-major 로 padding) ──────────────────
// 16×16 col-major: index = col * 16 + row.
// 호스트에서 cudaMemcpyToSymbol 로 초기화.
__device__ __constant__ float c_M_RGB_TO_LMS_T[16 * 16];
__device__ __constant__ float c_M_LMS_TO_LAB_T[16 * 16];

static void init_const_matrices_host()
{
	float h_M1[16*16] = {0};
	const float M1[3][3] = {
		{0.4122214708f, 0.5363325363f, 0.0514459929f},
		{0.2119034982f, 0.6806995451f, 0.1073969566f},
		{0.0883024619f, 0.2817188376f, 0.6299787005f},
	};
	for (int n = 0; n < 3; ++n)
		for (int k = 0; k < 3; ++k)
			h_M1[n * 16 + k] = M1[n][k];
	cudaMemcpyToSymbol(c_M_RGB_TO_LMS_T, h_M1, sizeof(h_M1));

	float h_M2[16*16] = {0};
	const float M2[3][3] = {
		{ 0.2104542553f,  0.7936177850f, -0.0040720468f},
		{ 1.9779984951f, -2.4285922050f,  0.4505937099f},
		{ 0.0259040371f,  0.7827717662f, -0.8086757660f},
	};
	for (int n = 0; n < 3; ++n)
		for (int k = 0; k < 3; ++k)
			h_M2[n * 16 + k] = M2[n][k];
	cudaMemcpyToSymbol(c_M_LMS_TO_LAB_T, h_M2, sizeof(h_M2));
}

// ─── 메인 커널 ────────────────────────────────────────────────────────────────
//
//   block 당 8 warp (256 threads), warp 당 16 픽셀.
//   block 처리량 = 8 warp × 16 픽셀 = 128 픽셀.
//
//   shared mem 레이아웃 (per warp):
//     in_tile:  16×8 row-major float  (fragment A 용)
//     out_tile: 16×16 row-major float (fragment C 출력 용)
//
__global__ void k_to_oklab_tensor(const RGB4* __restrict__ in,
                                   Lab4* __restrict__ out, int N)
{
	constexpr int WARPS_PER_BLOCK = 8;
	constexpr int M = 16, N_DIM = 16, K = 8;

	__shared__ float s_in [WARPS_PER_BLOCK][M * K];
	__shared__ float s_lms[WARPS_PER_BLOCK][M * N_DIM];
	__shared__ float s_lms_pow[WARPS_PER_BLOCK][M * K];
	__shared__ float s_lab[WARPS_PER_BLOCK][M * N_DIM];

	const int warp_in_block = threadIdx.x / 32;
	const int lane          = threadIdx.x % 32;
	const int global_warp   = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
	const int pixel_base    = global_warp * 16;

	if (pixel_base >= N) return;

	// (1) 16 픽셀의 RGB 를 SoA-like 로 shared mem 에 로드.
	//     a_frag (16 row × 8 col, row_major) → row r 의 col 0..2 가 (R,G,B).
	if (lane < 16) {
		int idx = pixel_base + lane;
		idx = (idx < N) ? idx : 0;
		RGB4 rgb = in[idx];
		float* row = s_in[warp_in_block] + lane * K;
		row[0] = rgb.r; row[1] = rgb.g; row[2] = rgb.b;
		row[3] = 0.f; row[4] = 0.f; row[5] = 0.f; row[6] = 0.f; row[7] = 0.f;
	}
	__syncwarp();

	// (2) MMA 1: RGB → LMS.
	//     A = 16×8 (input)  row_major, B = 8×16 (M^T)  col_major, C = 16×16.
	wmma::fragment<wmma::matrix_a, M, N_DIM, K, wmma::precision::tf32, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N_DIM, K, wmma::precision::tf32, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, M, N_DIM, K, float> c_frag;

	wmma::load_matrix_sync(a_frag, s_in[warp_in_block], K);
	wmma::load_matrix_sync(b_frag, c_M_RGB_TO_LMS_T, M);
	// TF32 변환 (FP32 → TF32 은 fragment element 별로 처리).
	for (int i = 0; i < a_frag.num_elements; ++i)
		a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
	for (int i = 0; i < b_frag.num_elements; ++i)
		b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);

	wmma::fill_fragment(c_frag, 0.0f);
	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	wmma::store_matrix_sync(s_lms[warp_in_block], c_frag, N_DIM, wmma::mem_row_major);
	__syncwarp();

	// (3) cbrt — scalar.
	if (lane < 16) {
		float* in_row  = s_lms[warp_in_block] + lane * N_DIM;
		float* out_row = s_lms_pow[warp_in_block] + lane * K;
		out_row[0] = cbrtf(in_row[0]);
		out_row[1] = cbrtf(in_row[1]);
		out_row[2] = cbrtf(in_row[2]);
		out_row[3] = 0.f; out_row[4] = 0.f; out_row[5] = 0.f; out_row[6] = 0.f; out_row[7] = 0.f;
	}
	__syncwarp();

	// (4) MMA 2: LMS' → Lab.
	wmma::load_matrix_sync(a_frag, s_lms_pow[warp_in_block], K);
	wmma::load_matrix_sync(b_frag, c_M_LMS_TO_LAB_T, M);
	for (int i = 0; i < a_frag.num_elements; ++i)
		a_frag.x[i] = wmma::__float_to_tf32(a_frag.x[i]);
	for (int i = 0; i < b_frag.num_elements; ++i)
		b_frag.x[i] = wmma::__float_to_tf32(b_frag.x[i]);

	wmma::fill_fragment(c_frag, 0.0f);
	wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
	wmma::store_matrix_sync(s_lab[warp_in_block], c_frag, N_DIM, wmma::mem_row_major);
	__syncwarp();

	// (5) Lab 출력.
	if (lane < 16) {
		int idx = pixel_base + lane;
		if (idx < N) {
			float* row = s_lab[warp_in_block] + lane * N_DIM;
			Lab4 o; o.L = row[0]; o.a = row[1]; o.b = row[2]; o._pad = 0.f;
			out[idx] = o;
		}
	}
}

}} // namespace ok_color_cuda::tensor

// =============================================================================
// 벤치 main.
// =============================================================================

#define CUDA_CHECK(x) do { cudaError_t _ce = (x); if (_ce != cudaSuccess) { \
	std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_ce), __FILE__, __LINE__); \
	std::exit(1); }} while(0)

namespace cu = ok_color_cuda;

static float bench_kernel_ms(int rounds, std::function<void()> launch) {
	cudaEvent_t s, e;
	CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e));
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
	cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	std::printf("GPU: %s, compute %d.%d, %d SMs\n\n",
		prop.name, prop.major, prop.minor, prop.multiProcessorCount);
	if (prop.major < 8) {
		std::fprintf(stderr, "Tensor Core TF32 needs sm_80+. Aborting.\n");
		return 1;
	}

	// constant matrices 초기화 (host → device).
	cu::tensor::init_const_matrices_host();
	CUDA_CHECK(cudaDeviceSynchronize());

	constexpr int N = 4 * 1024 * 1024;
	constexpr int ROUNDS = 11;

	std::vector<cu::RGB4> h_rgb(N);
	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);
	for (int i = 0; i < N; ++i) h_rgb[i] = { U(rng), U(rng), U(rng), 0 };

	cu::RGB4* d_rgb;
	cu::Lab4 *d_lab_cuda, *d_lab_tensor;
	CUDA_CHECK(cudaMalloc(&d_rgb,        N * sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_lab_cuda,   N * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_lab_tensor, N * sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), N * sizeof(cu::RGB4), cudaMemcpyHostToDevice));

	auto launch_cuda = [&](){ cu::linear_srgb_to_oklab(d_rgb, d_lab_cuda, N); };

	// Tensor 커널: 16 픽셀 / warp, 8 warp / block = 128 픽셀 / block.
	int b = 256;  // 8 warps
	dim3 g((N + 127) / 128);
	auto launch_tensor = [&](){
		cu::tensor::k_to_oklab_tensor<<<g, b>>>(d_rgb, d_lab_tensor, N);
	};

	for (int i = 0; i < 3; ++i) { launch_cuda(); launch_tensor(); }
	CUDA_CHECK(cudaDeviceSynchronize());

	float t_cuda   = bench_kernel_ms(ROUNDS, launch_cuda);
	float t_tensor = bench_kernel_ms(ROUNDS, launch_tensor);

	std::printf("=== linear_srgb→oklab @ 4M pixels ===\n\n");
	std::printf("%-30s | %10s | %10s\n", "variant", "ms", "Gpx/s");
	std::printf("-------------------------------+------------+-----------\n");
	std::printf("%-30s | %10.3f | %10.2f\n",
		"CUDA (regular, scalar fma)", t_cuda, N / (t_cuda * 1e6));
	std::printf("%-30s | %10.3f | %10.2f\n",
		"Tensor Core (wmma TF32)", t_tensor, N / (t_tensor * 1e6));
	std::printf("\nspeedup: %.3fx\n", t_cuda / t_tensor);

	// 정확성.
	std::vector<cu::Lab4> h_cuda(N), h_tensor(N);
	CUDA_CHECK(cudaMemcpy(h_cuda.data(),   d_lab_cuda,   N * sizeof(cu::Lab4), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_tensor.data(), d_lab_tensor, N * sizeof(cu::Lab4), cudaMemcpyDeviceToHost));
	float ma = 0;
	for (int i = 0; i < 4096; ++i) {
		ma = std::fmax(ma, std::fabs(h_cuda[i].L - h_tensor[i].L));
		ma = std::fmax(ma, std::fabs(h_cuda[i].a - h_tensor[i].a));
		ma = std::fmax(ma, std::fabs(h_cuda[i].b - h_tensor[i].b));
	}
	std::printf("정확성 (Tensor vs CUDA, 4096 pixels): max_abs = %.3e\n", ma);

	CUDA_CHECK(cudaFree(d_rgb));
	CUDA_CHECK(cudaFree(d_lab_cuda));
	CUDA_CHECK(cudaFree(d_lab_tensor));
	return 0;
}
