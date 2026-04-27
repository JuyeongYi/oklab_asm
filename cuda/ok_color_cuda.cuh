#pragma once
// -----------------------------------------------------------------------------
// ok_color_cuda — CUDA 커널.
//   설계: SIMT 모델, thread 당 1 픽셀.  AoS (RGB4) 가 자연스럽게 coalesced.
//
//   transcendental 은 GPU HW (`cbrtf`, `sinf`, `cosf`, `atan2f`, `powf`) 사용 —
//   CPU 의 bit-hack/Newton/polynomial 트릭 모두 불필요.
//
//   각 함수는 host-side launch wrapper.  device 포인터를 받아 kernel 실행.
//   gamut clipping 5종 모두 포함.
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>

namespace ok_color_cuda
{

struct alignas(16) RGB4 { float r, g, b, _pad; };
struct alignas(16) Lab4 { float L, a, b, _pad; };
struct alignas(16) HSL4 { float h, s, l, _pad; };
struct alignas(16) HSV4 { float h, s, v, _pad; };

// 색공간 변환 — N 픽셀 한 번에 처리.
void linear_srgb_to_oklab(const RGB4* d_in, Lab4* d_out, int N, cudaStream_t s = 0);
void oklab_to_linear_srgb(const Lab4* d_in, RGB4* d_out, int N, cudaStream_t s = 0);
void okhsl_to_srgb        (const HSL4* d_in, RGB4* d_out, int N, cudaStream_t s = 0);
void srgb_to_okhsl        (const RGB4* d_in, HSL4* d_out, int N, cudaStream_t s = 0);
void okhsv_to_srgb        (const HSV4* d_in, RGB4* d_out, int N, cudaStream_t s = 0);
void srgb_to_okhsv        (const RGB4* d_in, HSV4* d_out, int N, cudaStream_t s = 0);

// 색역 클리핑.
void gamut_clip_preserve_chroma   (const RGB4* d_in, RGB4* d_out, int N, cudaStream_t s = 0);
void gamut_clip_project_to_0_5    (const RGB4* d_in, RGB4* d_out, int N, cudaStream_t s = 0);
void gamut_clip_project_to_L_cusp (const RGB4* d_in, RGB4* d_out, int N, cudaStream_t s = 0);
void gamut_clip_adaptive_L0_0_5   (const RGB4* d_in, RGB4* d_out, int N, float alpha = 0.05f, cudaStream_t s = 0);
void gamut_clip_adaptive_L0_L_cusp(const RGB4* d_in, RGB4* d_out, int N, float alpha = 0.05f, cudaStream_t s = 0);

} // namespace ok_color_cuda
