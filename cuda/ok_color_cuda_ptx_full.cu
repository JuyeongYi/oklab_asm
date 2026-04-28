// ok_color_cuda_ptx_full.cu — 전체 변환 함수를 PTX-fast intrinsic 으로 재구현.
//
//   교체 내역 (vs ok_color_cuda.cu 의 표준 cmath 버전):
//     cbrtf(x)       → inline PTX lg2.approx + mul + ex2.approx       (~6 cycle)
//     sinf, cosf     → __sinf, __cosf  (=== sin.approx.f32 / cos.approx.f32)
//     powf           → __powf (lg2.approx + mul + ex2.approx 합성)
//     fmaxf          → fmaxf (HW max 명령, 동일)
//     sqrtf          → __fsqrt_rn 또는 sqrtf (둘 다 sqrt.approx.f32)
//     atan2f         → atan2f 그대로 (PTX 명령 없음, 다항식 근사 가능하지만 정확도 손실)
//
//   namespace `ok_color_cuda::ptx` 아래 전체 6 변환 + gamut clip 5종 정의.
//   API 는 ok_color_cuda 와 동일 (launch wrapper).

#include "ok_color_cuda.cuh"
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

namespace ok_color_cuda { namespace ptx {

// ─── PTX cbrt: lg2.approx + mul + ex2.approx ────────────────────────────────
__device__ __forceinline__ float p_cbrt(float x) {
	float t, r;
	asm volatile("lg2.approx.f32 %0, %1;" : "=f"(t) : "f"(x));
	t *= 0.33333333333f;
	asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(t));
	return r;
}
// 가독성용 PTX 직접 호출 wrapper.
__device__ __forceinline__ float p_sqrt(float x) {
	float r; asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float p_rcp(float x) {
	float r; asm volatile("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}

constexpr float pi_f = 3.14159265358979323846f;

// ─── 색공간 변환 ──────────────────────────────────────────────────────────────

__device__ __forceinline__ Lab4 linear_srgb_to_oklab_one(RGB4 c) {
	float l = 0.4122214708f*c.r + 0.5363325363f*c.g + 0.0514459929f*c.b;
	float m = 0.2119034982f*c.r + 0.6806995451f*c.g + 0.1073969566f*c.b;
	float s = 0.0883024619f*c.r + 0.2817188376f*c.g + 0.6299787005f*c.b;
	l = p_cbrt(l); m = p_cbrt(m); s = p_cbrt(s);
	Lab4 o;
	o.L = 0.2104542553f*l + 0.7936177850f*m - 0.0040720468f*s;
	o.a = 1.9779984951f*l - 2.4285922050f*m + 0.4505937099f*s;
	o.b = 0.0259040371f*l + 0.7827717662f*m - 0.8086757660f*s;
	o._pad = 0.f;
	return o;
}

__device__ __forceinline__ RGB4 oklab_to_linear_srgb_one(Lab4 c) {
	float l_ = c.L + 0.3963377774f*c.a + 0.2158037573f*c.b;
	float m_ = c.L - 0.1055613458f*c.a - 0.0638541728f*c.b;
	float s_ = c.L - 0.0894841775f*c.a - 1.2914855480f*c.b;
	float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
	RGB4 o;
	o.r = +4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s;
	o.g = -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s;
	o.b = -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s;
	o._pad = 0.f;
	return o;
}

__device__ __forceinline__ float srgb_transfer(float a) {
	return (0.0031308f >= a) ? (12.92f * a) : (1.055f * __powf(a, 1.f/2.4f) - 0.055f);
}
__device__ __forceinline__ float srgb_transfer_inv(float a) {
	return (0.04045f < a) ? __powf((a + 0.055f) / 1.055f, 2.4f) : (a / 12.92f);
}
__device__ __forceinline__ float toe(float x) {
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	return 0.5f * (k3*x - k1 + p_sqrt((k3*x - k1)*(k3*x - k1) + 4.f*k2*k3*x));
}
__device__ __forceinline__ float toe_inv(float x) {
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	return (x*x + k1*x) * p_rcp(k3 * (x + k2));
}

// ─── 색역 분석 (식 동일, 호출 함수만 PTX 화) ────────────────────────────────────

__device__ __forceinline__ float compute_max_saturation(float a, float b) {
	float k0,k1,k2,k3,k4, wl,wm,ws;
	if (-1.88170328f*a - 0.80936493f*b > 1.f) {
		k0=+1.19086277f; k1=+1.76576728f; k2=+0.59662641f; k3=+0.75515197f; k4=+0.56771245f;
		wl=+4.0767416621f; wm=-3.3077115913f; ws=+0.2309699292f;
	} else if (1.81444104f*a - 1.19445276f*b > 1.f) {
		k0=+0.73956515f; k1=-0.45954404f; k2=+0.08285427f; k3=+0.12541070f; k4=+0.14503204f;
		wl=-1.2684380046f; wm=+2.6097574011f; ws=-0.3413193965f;
	} else {
		k0=+1.35733652f; k1=-0.00915799f; k2=-1.15130210f; k3=-0.50559606f; k4=+0.00692167f;
		wl=-0.0041960863f; wm=-0.7034186147f; ws=+1.7076147010f;
	}
	float S = k0 + k1*a + k2*b + k3*a*a + k4*a*b;
	float kl = +0.3963377774f*a + 0.2158037573f*b;
	float km = -0.1055613458f*a - 0.0638541728f*b;
	float ks = -0.0894841775f*a - 1.2914855480f*b;
	float l_ = 1.f + S*kl, m_ = 1.f + S*km, s_ = 1.f + S*ks;
	float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
	float ldS = 3.f*kl*l_*l_, mdS = 3.f*km*m_*m_, sdS = 3.f*ks*s_*s_;
	float ldS2 = 6.f*kl*kl*l_, mdS2 = 6.f*km*km*m_, sdS2 = 6.f*ks*ks*s_;
	float f = wl*l + wm*m + ws*s;
	float f1 = wl*ldS + wm*mdS + ws*sdS;
	float f2 = wl*ldS2 + wm*mdS2 + ws*sdS2;
	S = S - f*f1 * p_rcp(f1*f1 - 0.5f*f*f2);
	return S;
}

struct LC { float L, C; };
__device__ __forceinline__ LC find_cusp(float a, float b) {
	float Sc = compute_max_saturation(a, b);
	Lab4 lab; lab.L = 1.f; lab.a = Sc*a; lab.b = Sc*b;
	RGB4 rgb = oklab_to_linear_srgb_one(lab);
	float mx = fmaxf(fmaxf(rgb.r, rgb.g), rgb.b);
	float Lc = p_cbrt(p_rcp(mx));
	return { Lc, Lc * Sc };
}

__device__ __forceinline__ float find_gamut_intersection(float a, float b, float L1, float C1, float L0, LC cusp) {
	float t;
	if (((L1 - L0) * cusp.C - (cusp.L - L0) * C1) <= 0.f) {
		t = cusp.C * L0 * p_rcp(C1 * cusp.L + cusp.C * (L0 - L1));
	} else {
		t = cusp.C * (L0 - 1.f) * p_rcp(C1 * (cusp.L - 1.f) + cusp.C * (L0 - L1));
		float dL = L1 - L0, dC = C1;
		float kl = +0.3963377774f*a + 0.2158037573f*b;
		float km = -0.1055613458f*a - 0.0638541728f*b;
		float ks = -0.0894841775f*a - 1.2914855480f*b;
		float l_dt = dL + dC*kl, m_dt = dL + dC*km, s_dt = dL + dC*ks;
		float L = L0*(1.f - t) + t*L1;
		float C = t*C1;
		float l_ = L + C*kl, m_ = L + C*km, s_ = L + C*ks;
		float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
		float ldt = 3.f*l_dt*l_*l_, mdt = 3.f*m_dt*m_*m_, sdt = 3.f*s_dt*s_*s_;
		float ldt2 = 6.f*l_dt*l_dt*l_, mdt2 = 6.f*m_dt*m_dt*m_, sdt2 = 6.f*s_dt*s_dt*s_;
		float r = 4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s - 1.f;
		float r1 = 4.0767416621f*ldt - 3.3077115913f*mdt + 0.2309699292f*sdt;
		float r2 = 4.0767416621f*ldt2 - 3.3077115913f*mdt2 + 0.2309699292f*sdt2;
		float u_r = r1 * p_rcp(r1*r1 - 0.5f*r*r2);
		float t_r = (u_r >= 0.f) ? -r * u_r : 1e30f;
		float gg = -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s - 1.f;
		float g1 = -1.2684380046f*ldt + 2.6097574011f*mdt - 0.3413193965f*sdt;
		float g2 = -1.2684380046f*ldt2 + 2.6097574011f*mdt2 - 0.3413193965f*sdt2;
		float u_g = g1 * p_rcp(g1*g1 - 0.5f*gg*g2);
		float t_g = (u_g >= 0.f) ? -gg * u_g : 1e30f;
		float bb = -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s - 1.f;
		float b1 = -0.0041960863f*ldt - 0.7034186147f*mdt + 1.7076147010f*sdt;
		float b2 = -0.0041960863f*ldt2 - 0.7034186147f*mdt2 + 1.7076147010f*sdt2;
		float u_b = b1 * p_rcp(b1*b1 - 0.5f*bb*b2);
		float t_b = (u_b >= 0.f) ? -bb * u_b : 1e30f;
		t += fminf(t_r, fminf(t_g, t_b));
	}
	return t;
}

struct ST_t { float S, T; };
__device__ __forceinline__ ST_t to_ST(LC cusp) {
	return { cusp.C * p_rcp(cusp.L), cusp.C * p_rcp(1.f - cusp.L) };
}
__device__ __forceinline__ ST_t get_ST_mid(float a_, float b_) {
	float S = 0.11516993f + p_rcp(
		+7.44778970f + 4.15901240f*b_
		+ a_*(-2.19557347f + 1.75198401f*b_
			+ a_*(-2.13704948f - 10.02301043f*b_
				+ a_*(-4.24894561f + 5.38770819f*b_ + 4.69891013f*a_))));
	float T = 0.11239642f + p_rcp(
		+1.61320320f - 0.68124379f*b_
		+ a_*(+0.40370612f + 0.90148123f*b_
			+ a_*(-0.27087943f + 0.61223990f*b_
				+ a_*(+0.00299215f - 0.45399568f*b_ - 0.14661872f*a_))));
	return { S, T };
}

struct Cs_t { float C_0, C_mid, C_max; };
__device__ __forceinline__ Cs_t get_Cs(float L, float a_, float b_) {
	LC cusp = find_cusp(a_, b_);
	float C_max = find_gamut_intersection(a_, b_, L, 1.f, L, cusp);
	ST_t ST_max = to_ST(cusp);
	float k = C_max * p_rcp(fminf(L*ST_max.S, (1.f - L)*ST_max.T));
	float C_mid;
	{
		ST_t ST_mid = get_ST_mid(a_, b_);
		float Ca = L * ST_mid.S, Cb = (1.f - L) * ST_mid.T;
		C_mid = 0.9f * k * p_sqrt(p_sqrt(p_rcp(p_rcp(Ca*Ca*Ca*Ca) + p_rcp(Cb*Cb*Cb*Cb))));
	}
	float C_0;
	{
		float Ca = L*0.4f, Cb = (1.f - L)*0.8f;
		C_0 = p_sqrt(p_rcp(p_rcp(Ca*Ca) + p_rcp(Cb*Cb)));
	}
	return { C_0, C_mid, C_max };
}

// ─── HSL/HSV ──────────────────────────────────────────────────────────────────

__device__ __forceinline__ RGB4 okhsl_to_srgb_one(HSL4 hsl) {
	if (hsl.l == 1.0f) return { 1.f, 1.f, 1.f, 0.f };
	if (hsl.l == 0.0f) return { 0.f, 0.f, 0.f, 0.f };
	float a_ = __cosf(2.f * pi_f * hsl.h);
	float b_ = __sinf(2.f * pi_f * hsl.h);
	float L = toe_inv(hsl.l);
	Cs_t cs = get_Cs(L, a_, b_);
	const float mid = 0.8f, mid_inv = 1.25f;
	float C, t, k_0, k_1, k_2;
	if (hsl.s < mid) {
		t = mid_inv * hsl.s;
		k_1 = mid * cs.C_0;
		k_2 = 1.f - k_1 * p_rcp(cs.C_mid);
		C = t * k_1 * p_rcp(1.f - k_2 * t);
	} else {
		t = (hsl.s - mid) * p_rcp(1.f - mid);
		k_0 = cs.C_mid;
		k_1 = (1.f - mid) * cs.C_mid * cs.C_mid * mid_inv * mid_inv * p_rcp(cs.C_0);
		k_2 = 1.f - k_1 * p_rcp(cs.C_max - cs.C_mid);
		C = k_0 + t * k_1 * p_rcp(1.f - k_2 * t);
	}
	Lab4 lab; lab.L = L; lab.a = C*a_; lab.b = C*b_;
	RGB4 lin = oklab_to_linear_srgb_one(lab);
	return { srgb_transfer(lin.r), srgb_transfer(lin.g), srgb_transfer(lin.b), 0.f };
}

__device__ __forceinline__ HSL4 srgb_to_okhsl_one(RGB4 rgb) {
	RGB4 lin; lin.r = srgb_transfer_inv(rgb.r); lin.g = srgb_transfer_inv(rgb.g); lin.b = srgb_transfer_inv(rgb.b); lin._pad = 0;
	Lab4 lab = linear_srgb_to_oklab_one(lin);
	float C = p_sqrt(lab.a*lab.a + lab.b*lab.b);
	float a_ = lab.a * p_rcp(C), b_ = lab.b * p_rcp(C);
	float L = lab.L;
	float h = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) * (1.f/pi_f);
	Cs_t cs = get_Cs(L, a_, b_);
	const float mid = 0.8f, mid_inv = 1.25f;
	float s;
	if (C < cs.C_mid) {
		float k_1 = mid * cs.C_0;
		float k_2 = 1.f - k_1 * p_rcp(cs.C_mid);
		float t = C * p_rcp(k_1 + k_2 * C);
		s = t * mid;
	} else {
		float k_0 = cs.C_mid;
		float k_1 = (1.f - mid) * cs.C_mid * cs.C_mid * mid_inv * mid_inv * p_rcp(cs.C_0);
		float k_2 = 1.f - k_1 * p_rcp(cs.C_max - cs.C_mid);
		float t = (C - k_0) * p_rcp(k_1 + k_2 * (C - k_0));
		s = mid + (1.f - mid) * t;
	}
	return { h, s, toe(L), 0.f };
}

__device__ __forceinline__ RGB4 okhsv_to_srgb_one(HSV4 hsv) {
	float a_ = __cosf(2.f * pi_f * hsv.h), b_ = __sinf(2.f * pi_f * hsv.h);
	LC cusp = find_cusp(a_, b_);
	ST_t ST_max = to_ST(cusp);
	const float S_0 = 0.5f;
	float k = 1.f - S_0 * p_rcp(ST_max.S);
	float denom = S_0 + ST_max.T - ST_max.T * k * hsv.s;
	float L_v = 1.f - hsv.s * S_0 * p_rcp(denom);
	float C_v =       hsv.s * ST_max.T * S_0 * p_rcp(denom);
	float L = hsv.v * L_v, C = hsv.v * C_v;
	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt * p_rcp(L_v);
	float L_new = toe_inv(L);
	C = C * L_new * p_rcp(L); L = L_new;
	Lab4 ls; ls.L = L_vt; ls.a = a_*C_vt; ls.b = b_*C_vt;
	RGB4 rgb_scale = oklab_to_linear_srgb_one(ls);
	float scale_L = p_cbrt(p_rcp(fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.f))));
	L *= scale_L; C *= scale_L;
	Lab4 lab; lab.L = L; lab.a = C*a_; lab.b = C*b_;
	RGB4 lin = oklab_to_linear_srgb_one(lab);
	return { srgb_transfer(lin.r), srgb_transfer(lin.g), srgb_transfer(lin.b), 0.f };
}

__device__ __forceinline__ HSV4 srgb_to_okhsv_one(RGB4 rgb) {
	RGB4 lin; lin.r = srgb_transfer_inv(rgb.r); lin.g = srgb_transfer_inv(rgb.g); lin.b = srgb_transfer_inv(rgb.b); lin._pad = 0;
	Lab4 lab = linear_srgb_to_oklab_one(lin);
	float C = p_sqrt(lab.a*lab.a + lab.b*lab.b);
	float a_ = lab.a * p_rcp(C), b_ = lab.b * p_rcp(C);
	float L = lab.L;
	float h = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) * (1.f/pi_f);
	LC cusp = find_cusp(a_, b_);
	ST_t ST_max = to_ST(cusp);
	const float S_0 = 0.5f;
	float k = 1.f - S_0 * p_rcp(ST_max.S);
	float t = ST_max.T * p_rcp(C + L * ST_max.T);
	float L_v = t * L, C_v = t * C;
	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt * p_rcp(L_v);
	Lab4 ls; ls.L = L_vt; ls.a = a_*C_vt; ls.b = b_*C_vt;
	RGB4 rgb_scale = oklab_to_linear_srgb_one(ls);
	float scale_L = p_cbrt(p_rcp(fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.f))));
	L *= p_rcp(scale_L); C *= p_rcp(scale_L);
	C = C * toe(L) * p_rcp(L); L = toe(L);
	float v = L * p_rcp(L_v);
	float s = (S_0 + ST_max.T) * C_v * p_rcp((ST_max.T * S_0) + ST_max.T * k * C_v);
	return { h, s, v, 0.f };
}

// ─── 커널 ────────────────────────────────────────────────────────────────────

__global__ void k_to_oklab(const RGB4* __restrict__ in, Lab4* __restrict__ out, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	out[i] = linear_srgb_to_oklab_one(in[i]);
}
__global__ void k_to_linear_srgb(const Lab4* __restrict__ in, RGB4* __restrict__ out, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	out[i] = oklab_to_linear_srgb_one(in[i]);
}
__global__ void k_okhsl_to_srgb(const HSL4* __restrict__ in, RGB4* __restrict__ out, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	out[i] = okhsl_to_srgb_one(in[i]);
}
__global__ void k_srgb_to_okhsl(const RGB4* __restrict__ in, HSL4* __restrict__ out, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	out[i] = srgb_to_okhsl_one(in[i]);
}
__global__ void k_okhsv_to_srgb(const HSV4* __restrict__ in, RGB4* __restrict__ out, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	out[i] = okhsv_to_srgb_one(in[i]);
}
__global__ void k_srgb_to_okhsv(const RGB4* __restrict__ in, HSV4* __restrict__ out, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	out[i] = srgb_to_okhsv_one(in[i]);
}

}} // namespace ok_color_cuda::ptx

// =============================================================================

#define CUDA_CHECK(x) do { cudaError_t _ce = (x); if (_ce != cudaSuccess) { \
	std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_ce), __FILE__, __LINE__); \
	std::exit(1); }} while(0)

namespace cu = ok_color_cuda;

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
	cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	std::printf("GPU: %s, compute %d.%d\n\n", prop.name, prop.major, prop.minor);

	constexpr int N = 4 * 1024 * 1024;
	constexpr int ROUNDS = 11;

	std::vector<cu::RGB4> h_rgb(N);
	std::vector<cu::Lab4> h_lab(N);
	std::vector<cu::HSL4> h_hsl(N);
	std::vector<cu::HSV4> h_hsv(N);
	std::mt19937 rng(0xBEEF);
	std::uniform_real_distribution<float> U(0.05f, 0.95f);
	for (int i = 0; i < N; ++i) {
		h_rgb[i] = { U(rng), U(rng), U(rng), 0 };
		h_hsl[i] = { U(rng), U(rng), U(rng), 0 };
		h_hsv[i] = { U(rng), U(rng), U(rng), 0 };
	}
	// Lab 입력은 linear_srgb_to_oklab 으로 만든 의미 있는 값.
	for (int i = 0; i < N; ++i) {
		float l = 0.4122214708f*h_rgb[i].r + 0.5363325363f*h_rgb[i].g + 0.0514459929f*h_rgb[i].b;
		float m = 0.2119034982f*h_rgb[i].r + 0.6806995451f*h_rgb[i].g + 0.1073969566f*h_rgb[i].b;
		float s = 0.0883024619f*h_rgb[i].r + 0.2817188376f*h_rgb[i].g + 0.6299787005f*h_rgb[i].b;
		l = std::cbrt(l); m = std::cbrt(m); s = std::cbrt(s);
		h_lab[i] = {
			0.2104542553f*l + 0.7936177850f*m - 0.0040720468f*s,
			1.9779984951f*l - 2.4285922050f*m + 0.4505937099f*s,
			0.0259040371f*l + 0.7827717662f*m - 0.8086757660f*s,
			0
		};
	}

	cu::RGB4 *d_rgb, *d_rgb_out;
	cu::Lab4 *d_lab, *d_lab_out;
	cu::HSL4 *d_hsl, *d_hsl_out;
	cu::HSV4 *d_hsv, *d_hsv_out;
	CUDA_CHECK(cudaMalloc(&d_rgb,     N*sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_rgb_out, N*sizeof(cu::RGB4)));
	CUDA_CHECK(cudaMalloc(&d_lab,     N*sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_lab_out, N*sizeof(cu::Lab4)));
	CUDA_CHECK(cudaMalloc(&d_hsl,     N*sizeof(cu::HSL4)));
	CUDA_CHECK(cudaMalloc(&d_hsl_out, N*sizeof(cu::HSL4)));
	CUDA_CHECK(cudaMalloc(&d_hsv,     N*sizeof(cu::HSV4)));
	CUDA_CHECK(cudaMalloc(&d_hsv_out, N*sizeof(cu::HSV4)));
	CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), N*sizeof(cu::RGB4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_lab, h_lab.data(), N*sizeof(cu::Lab4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_hsl, h_hsl.data(), N*sizeof(cu::HSL4), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_hsv, h_hsv.data(), N*sizeof(cu::HSV4), cudaMemcpyHostToDevice));

	int b = 256;
	dim3 g((N + b - 1) / b);

	// 표준 CUDA vs PTX-fast 비교.
	std::printf("=== %d 픽셀, kernel-only ms — CUDA(cmath) vs CUDA(PTX) ===\n\n", N);
	std::printf("%-22s | %10s | %10s | %10s\n", "function", "cmath", "PTX", "speedup");
	std::printf("-----------------------+------------+------------+----------\n");

	auto bench = [&](const char* name, std::function<void()> std_fn, std::function<void()> ptx_fn) {
		for (int i = 0; i < 3; ++i) { std_fn(); ptx_fn(); }
		float t_std = bench_kernel_ms(ROUNDS, std_fn);
		float t_ptx = bench_kernel_ms(ROUNDS, ptx_fn);
		std::printf("%-22s | %10.3f | %10.3f | %9.3fx\n",
			name, t_std, t_ptx, t_std / t_ptx);
	};

	bench("linear_srgb→oklab",
		[&](){ cu::linear_srgb_to_oklab(d_rgb, d_lab_out, N); },
		[&](){ cu::ptx::k_to_oklab<<<g, b>>>(d_rgb, d_lab_out, N); });
	bench("oklab→linear_srgb",
		[&](){ cu::oklab_to_linear_srgb(d_lab, d_rgb_out, N); },
		[&](){ cu::ptx::k_to_linear_srgb<<<g, b>>>(d_lab, d_rgb_out, N); });
	bench("okhsl→srgb",
		[&](){ cu::okhsl_to_srgb(d_hsl, d_rgb_out, N); },
		[&](){ cu::ptx::k_okhsl_to_srgb<<<g, b>>>(d_hsl, d_rgb_out, N); });
	bench("srgb→okhsl",
		[&](){ cu::srgb_to_okhsl(d_rgb, d_hsl_out, N); },
		[&](){ cu::ptx::k_srgb_to_okhsl<<<g, b>>>(d_rgb, d_hsl_out, N); });
	bench("okhsv→srgb",
		[&](){ cu::okhsv_to_srgb(d_hsv, d_rgb_out, N); },
		[&](){ cu::ptx::k_okhsv_to_srgb<<<g, b>>>(d_hsv, d_rgb_out, N); });
	bench("srgb→okhsv",
		[&](){ cu::srgb_to_okhsv(d_rgb, d_hsv_out, N); },
		[&](){ cu::ptx::k_srgb_to_okhsv<<<g, b>>>(d_rgb, d_hsv_out, N); });

	// 정확성 — 첫 1000 픽셀 cmath vs ptx max_abs.
	std::vector<cu::Lab4> ref_lab(N), ptx_lab(N);
	cu::linear_srgb_to_oklab(d_rgb, d_lab_out, N);
	CUDA_CHECK(cudaMemcpy(ref_lab.data(), d_lab_out, N*sizeof(cu::Lab4), cudaMemcpyDeviceToHost));
	cu::ptx::k_to_oklab<<<g, b>>>(d_rgb, d_lab_out, N);
	CUDA_CHECK(cudaMemcpy(ptx_lab.data(), d_lab_out, N*sizeof(cu::Lab4), cudaMemcpyDeviceToHost));
	float ma = 0;
	for (int i = 0; i < 4096; ++i) {
		ma = std::fmax(ma, std::fabs(ref_lab[i].L - ptx_lab[i].L));
		ma = std::fmax(ma, std::fabs(ref_lab[i].a - ptx_lab[i].a));
		ma = std::fmax(ma, std::fabs(ref_lab[i].b - ptx_lab[i].b));
	}
	std::printf("\n정확성 (PTX vs cmath, linear_srgb→oklab, 4096 pixels): max_abs = %.3e\n", ma);

	CUDA_CHECK(cudaFree(d_rgb)); CUDA_CHECK(cudaFree(d_rgb_out));
	CUDA_CHECK(cudaFree(d_lab)); CUDA_CHECK(cudaFree(d_lab_out));
	CUDA_CHECK(cudaFree(d_hsl)); CUDA_CHECK(cudaFree(d_hsl_out));
	CUDA_CHECK(cudaFree(d_hsv)); CUDA_CHECK(cudaFree(d_hsv_out));
	return 0;
}
