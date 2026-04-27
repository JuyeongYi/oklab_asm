// ok_color_cuda.cu — Björn Ottosson 의 ok_color 알고리즘을 CUDA 커널로.
//
//   알고리즘은 scalar (CPU) 와 동일.  GPU 의 SIMT 모델에서는 thread 당 1 픽셀로
//   처리하는 것이 가장 자연스러움 — AoS 메모리 레이아웃이 그대로 coalesce 됨.

#include "ok_color_cuda.cuh"
#include <cstdio>

namespace ok_color_cuda
{

namespace dev {

constexpr float pi_f = 3.14159265358979323846f;

// ─────────────────────────────────────────────────────────────────────────────
// 색공간 변환 (single-pixel device functions).
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ Lab4 linear_srgb_to_oklab_one(RGB4 c)
{
	float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
	float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
	float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;
	l = cbrtf(l); m = cbrtf(m); s = cbrtf(s);
	Lab4 o;
	o.L = 0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s;
	o.a = 1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s;
	o.b = 0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s;
	o._pad = 0.f;
	return o;
}

__device__ __forceinline__ RGB4 oklab_to_linear_srgb_one(Lab4 c)
{
	float l_ = c.L + 0.3963377774f * c.a + 0.2158037573f * c.b;
	float m_ = c.L - 0.1055613458f * c.a - 0.0638541728f * c.b;
	float s_ = c.L - 0.0894841775f * c.a - 1.2914855480f * c.b;
	float l = l_ * l_ * l_;
	float m = m_ * m_ * m_;
	float s = s_ * s_ * s_;
	RGB4 o;
	o.r = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
	o.g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
	o.b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
	o._pad = 0.f;
	return o;
}

__device__ __forceinline__ float srgb_transfer(float a)
{
	return (0.0031308f >= a) ? (12.92f * a) : (1.055f * powf(a, 1.f / 2.4f) - 0.055f);
}
__device__ __forceinline__ float srgb_transfer_inv(float a)
{
	return (0.04045f < a) ? powf((a + 0.055f) / 1.055f, 2.4f) : (a / 12.92f);
}
__device__ __forceinline__ float toe(float x)
{
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	return 0.5f * (k3 * x - k1 + sqrtf((k3*x - k1)*(k3*x - k1) + 4.f * k2 * k3 * x));
}
__device__ __forceinline__ float toe_inv(float x)
{
	const float k1 = 0.206f, k2 = 0.03f, k3 = (1.f + k1) / (1.f + k2);
	return (x*x + k1*x) / (k3 * (x + k2));
}

// ─────────────────────────────────────────────────────────────────────────────
// 색역 분석.  CPU 버전과 동일 식, scalar 로 (per-thread).
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float compute_max_saturation(float a, float b)
{
	float k0, k1, k2, k3, k4, wl, wm, ws;
	if (-1.88170328f * a - 0.80936493f * b > 1.f) {
		k0=+1.19086277f; k1=+1.76576728f; k2=+0.59662641f; k3=+0.75515197f; k4=+0.56771245f;
		wl=+4.0767416621f; wm=-3.3077115913f; ws=+0.2309699292f;
	} else if (1.81444104f * a - 1.19445276f * b > 1.f) {
		k0=+0.73956515f; k1=-0.45954404f; k2=+0.08285427f; k3=+0.12541070f; k4=+0.14503204f;
		wl=-1.2684380046f; wm=+2.6097574011f; ws=-0.3413193965f;
	} else {
		k0=+1.35733652f; k1=-0.00915799f; k2=-1.15130210f; k3=-0.50559606f; k4=+0.00692167f;
		wl=-0.0041960863f; wm=-0.7034186147f; ws=+1.7076147010f;
	}
	float S = k0 + k1*a + k2*b + k3*a*a + k4*a*b;
	float k_l = +0.3963377774f * a + 0.2158037573f * b;
	float k_m = -0.1055613458f * a - 0.0638541728f * b;
	float k_s = -0.0894841775f * a - 1.2914855480f * b;
	{
		float l_ = 1.f + S*k_l, m_ = 1.f + S*k_m, s_ = 1.f + S*k_s;
		float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
		float lds = 3.f*k_l*l_*l_, mds = 3.f*k_m*m_*m_, sds = 3.f*k_s*s_*s_;
		float lds2 = 6.f*k_l*k_l*l_, mds2 = 6.f*k_m*k_m*m_, sds2 = 6.f*k_s*k_s*s_;
		float f = wl*l + wm*m + ws*s;
		float f1 = wl*lds + wm*mds + ws*sds;
		float f2 = wl*lds2 + wm*mds2 + ws*sds2;
		S = S - f*f1 / (f1*f1 - 0.5f*f*f2);
	}
	return S;
}

struct LC { float L, C; };

__device__ __forceinline__ LC find_cusp(float a, float b)
{
	float Sc = compute_max_saturation(a, b);
	Lab4 lab; lab.L = 1.f; lab.a = Sc * a; lab.b = Sc * b;
	RGB4 rgb = oklab_to_linear_srgb_one(lab);
	float mx = fmaxf(fmaxf(rgb.r, rgb.g), rgb.b);
	float Lc = cbrtf(1.f / mx);
	return { Lc, Lc * Sc };
}

__device__ __forceinline__ float find_gamut_intersection(float a, float b, float L1, float C1, float L0, LC cusp)
{
	float t;
	if (((L1 - L0) * cusp.C - (cusp.L - L0) * C1) <= 0.f) {
		t = cusp.C * L0 / (C1 * cusp.L + cusp.C * (L0 - L1));
	} else {
		t = cusp.C * (L0 - 1.f) / (C1 * (cusp.L - 1.f) + cusp.C * (L0 - L1));
		float dL = L1 - L0, dC = C1;
		float k_l = +0.3963377774f * a + 0.2158037573f * b;
		float k_m = -0.1055613458f * a - 0.0638541728f * b;
		float k_s = -0.0894841775f * a - 1.2914855480f * b;
		float l_dt = dL + dC*k_l, m_dt = dL + dC*k_m, s_dt = dL + dC*k_s;
		float L = L0 * (1.f - t) + t * L1;
		float C = t * C1;
		float l_ = L + C*k_l, m_ = L + C*k_m, s_ = L + C*k_s;
		float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
		float ldt = 3.f*l_dt*l_*l_, mdt = 3.f*m_dt*m_*m_, sdt = 3.f*s_dt*s_*s_;
		float ldt2 = 6.f*l_dt*l_dt*l_, mdt2 = 6.f*m_dt*m_dt*m_, sdt2 = 6.f*s_dt*s_dt*s_;
		float r = 4.0767416621f*l - 3.3077115913f*m + 0.2309699292f*s - 1.f;
		float r1 = 4.0767416621f*ldt - 3.3077115913f*mdt + 0.2309699292f*sdt;
		float r2 = 4.0767416621f*ldt2 - 3.3077115913f*mdt2 + 0.2309699292f*sdt2;
		float u_r = r1 / (r1*r1 - 0.5f*r*r2);
		float t_r = (u_r >= 0.f) ? -r * u_r : 1e30f;
		float gg = -1.2684380046f*l + 2.6097574011f*m - 0.3413193965f*s - 1.f;
		float g1 = -1.2684380046f*ldt + 2.6097574011f*mdt - 0.3413193965f*sdt;
		float g2 = -1.2684380046f*ldt2 + 2.6097574011f*mdt2 - 0.3413193965f*sdt2;
		float u_g = g1 / (g1*g1 - 0.5f*gg*g2);
		float t_g = (u_g >= 0.f) ? -gg * u_g : 1e30f;
		float bb = -0.0041960863f*l - 0.7034186147f*m + 1.7076147010f*s - 1.f;
		float b1 = -0.0041960863f*ldt - 0.7034186147f*mdt + 1.7076147010f*sdt;
		float b2 = -0.0041960863f*ldt2 - 0.7034186147f*mdt2 + 1.7076147010f*sdt2;
		float u_b = b1 / (b1*b1 - 0.5f*bb*b2);
		float t_b = (u_b >= 0.f) ? -bb * u_b : 1e30f;
		t += fminf(t_r, fminf(t_g, t_b));
	}
	return t;
}

struct ST_t { float S, T; };
__device__ __forceinline__ ST_t to_ST(LC cusp)
{
	return { cusp.C / cusp.L, cusp.C / (1.f - cusp.L) };
}

__device__ __forceinline__ ST_t get_ST_mid(float a_, float b_)
{
	float S = 0.11516993f + 1.f / (
		+7.44778970f + 4.15901240f * b_
		+ a_ * (-2.19557347f + 1.75198401f * b_
			+ a_ * (-2.13704948f - 10.02301043f * b_
				+ a_ * (-4.24894561f + 5.38770819f * b_ + 4.69891013f * a_))));
	float T = 0.11239642f + 1.f / (
		+1.61320320f - 0.68124379f * b_
		+ a_ * (+0.40370612f + 0.90148123f * b_
			+ a_ * (-0.27087943f + 0.61223990f * b_
				+ a_ * (+0.00299215f - 0.45399568f * b_ - 0.14661872f * a_))));
	return { S, T };
}

struct Cs_t { float C_0, C_mid, C_max; };
__device__ __forceinline__ Cs_t get_Cs(float L, float a_, float b_)
{
	LC cusp = find_cusp(a_, b_);
	float C_max = find_gamut_intersection(a_, b_, L, 1.f, L, cusp);
	ST_t ST_max = to_ST(cusp);
	float k = C_max / fminf(L * ST_max.S, (1.f - L) * ST_max.T);
	float C_mid;
	{
		ST_t ST_mid = get_ST_mid(a_, b_);
		float Ca = L * ST_mid.S;
		float Cb = (1.f - L) * ST_mid.T;
		C_mid = 0.9f * k * sqrtf(sqrtf(1.f / (1.f / (Ca*Ca*Ca*Ca) + 1.f / (Cb*Cb*Cb*Cb))));
	}
	float C_0;
	{
		float Ca = L * 0.4f, Cb = (1.f - L) * 0.8f;
		C_0 = sqrtf(1.f / (1.f / (Ca*Ca) + 1.f / (Cb*Cb)));
	}
	return { C_0, C_mid, C_max };
}

// ─────────────────────────────────────────────────────────────────────────────
// HSL/HSV — CPU 와 동일 식.
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ RGB4 okhsl_to_srgb_one(HSL4 hsl)
{
	if (hsl.l == 1.0f) return { 1.f, 1.f, 1.f, 0.f };
	if (hsl.l == 0.0f) return { 0.f, 0.f, 0.f, 0.f };
	float a_ = cosf(2.f * pi_f * hsl.h);
	float b_ = sinf(2.f * pi_f * hsl.h);
	float L = toe_inv(hsl.l);
	Cs_t cs = get_Cs(L, a_, b_);
	const float mid = 0.8f, mid_inv = 1.25f;
	float C, t, k_0, k_1, k_2;
	if (hsl.s < mid) {
		t = mid_inv * hsl.s;
		k_1 = mid * cs.C_0;
		k_2 = 1.f - k_1 / cs.C_mid;
		C = t * k_1 / (1.f - k_2 * t);
	} else {
		t = (hsl.s - mid) / (1.f - mid);
		k_0 = cs.C_mid;
		k_1 = (1.f - mid) * cs.C_mid * cs.C_mid * mid_inv * mid_inv / cs.C_0;
		k_2 = 1.f - k_1 / (cs.C_max - cs.C_mid);
		C = k_0 + t * k_1 / (1.f - k_2 * t);
	}
	Lab4 lab; lab.L = L; lab.a = C * a_; lab.b = C * b_;
	RGB4 lin = oklab_to_linear_srgb_one(lab);
	return { srgb_transfer(lin.r), srgb_transfer(lin.g), srgb_transfer(lin.b), 0.f };
}

__device__ __forceinline__ HSL4 srgb_to_okhsl_one(RGB4 rgb)
{
	RGB4 lin; lin.r = srgb_transfer_inv(rgb.r); lin.g = srgb_transfer_inv(rgb.g); lin.b = srgb_transfer_inv(rgb.b); lin._pad = 0;
	Lab4 lab = linear_srgb_to_oklab_one(lin);
	float C = sqrtf(lab.a * lab.a + lab.b * lab.b);
	float a_ = lab.a / C, b_ = lab.b / C;
	float L = lab.L;
	float h = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi_f;
	Cs_t cs = get_Cs(L, a_, b_);
	const float mid = 0.8f, mid_inv = 1.25f;
	float s;
	if (C < cs.C_mid) {
		float k_1 = mid * cs.C_0;
		float k_2 = 1.f - k_1 / cs.C_mid;
		float t = C / (k_1 + k_2 * C);
		s = t * mid;
	} else {
		float k_0 = cs.C_mid;
		float k_1 = (1.f - mid) * cs.C_mid * cs.C_mid * mid_inv * mid_inv / cs.C_0;
		float k_2 = 1.f - k_1 / (cs.C_max - cs.C_mid);
		float t = (C - k_0) / (k_1 + k_2 * (C - k_0));
		s = mid + (1.f - mid) * t;
	}
	return { h, s, toe(L), 0.f };
}

__device__ __forceinline__ RGB4 okhsv_to_srgb_one(HSV4 hsv)
{
	float a_ = cosf(2.f * pi_f * hsv.h), b_ = sinf(2.f * pi_f * hsv.h);
	LC cusp = find_cusp(a_, b_);
	ST_t ST_max = to_ST(cusp);
	const float S_0 = 0.5f;
	float k = 1.f - S_0 / ST_max.S;
	float L_v = 1.f - hsv.s * S_0 / (S_0 + ST_max.T - ST_max.T * k * hsv.s);
	float C_v =        hsv.s * ST_max.T * S_0 / (S_0 + ST_max.T - ST_max.T * k * hsv.s);
	float L = hsv.v * L_v, C = hsv.v * C_v;
	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;
	float L_new = toe_inv(L);
	C = C * L_new / L; L = L_new;
	Lab4 ls; ls.L = L_vt; ls.a = a_ * C_vt; ls.b = b_ * C_vt;
	RGB4 rgb_scale = oklab_to_linear_srgb_one(ls);
	float scale_L = cbrtf(1.f / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.f)));
	L *= scale_L; C *= scale_L;
	Lab4 lab; lab.L = L; lab.a = C * a_; lab.b = C * b_;
	RGB4 lin = oklab_to_linear_srgb_one(lab);
	return { srgb_transfer(lin.r), srgb_transfer(lin.g), srgb_transfer(lin.b), 0.f };
}

__device__ __forceinline__ HSV4 srgb_to_okhsv_one(RGB4 rgb)
{
	RGB4 lin; lin.r = srgb_transfer_inv(rgb.r); lin.g = srgb_transfer_inv(rgb.g); lin.b = srgb_transfer_inv(rgb.b); lin._pad = 0;
	Lab4 lab = linear_srgb_to_oklab_one(lin);
	float C = sqrtf(lab.a*lab.a + lab.b*lab.b);
	float a_ = lab.a/C, b_ = lab.b/C;
	float L = lab.L;
	float h = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi_f;
	LC cusp = find_cusp(a_, b_);
	ST_t ST_max = to_ST(cusp);
	const float S_0 = 0.5f;
	float k = 1.f - S_0 / ST_max.S;
	float t = ST_max.T / (C + L * ST_max.T);
	float L_v = t * L, C_v = t * C;
	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;
	Lab4 ls; ls.L = L_vt; ls.a = a_*C_vt; ls.b = b_*C_vt;
	RGB4 rgb_scale = oklab_to_linear_srgb_one(ls);
	float scale_L = cbrtf(1.f / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.f)));
	L /= scale_L; C /= scale_L;
	C = C * toe(L) / L; L = toe(L);
	float v = L / L_v;
	float s = (S_0 + ST_max.T) * C_v / ((ST_max.T * S_0) + ST_max.T * k * C_v);
	return { h, s, v, 0.f };
}

// ─────────────────────────────────────────────────────────────────────────────
// gamut clipping — CPU 와 동일.
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float clamp_f(float x, float a, float b) { return fminf(fmaxf(x, a), b); }
__device__ __forceinline__ float sgn_f(float x) { return (x > 0.f) - (x < 0.f); }

__device__ __forceinline__ RGB4 clip_helper(RGB4 rgb, int strategy, float alpha)
{
	if (rgb.r < 1.f && rgb.g < 1.f && rgb.b < 1.f && rgb.r > 0.f && rgb.g > 0.f && rgb.b > 0.f)
		return rgb;
	Lab4 lab = linear_srgb_to_oklab_one(rgb);
	float L = lab.L;
	float eps = 1e-5f;
	float C = fmaxf(eps, sqrtf(lab.a*lab.a + lab.b*lab.b));
	float a_ = lab.a / C, b_ = lab.b / C;

	float L0;
	switch (strategy) {
		case 0: L0 = clamp_f(L, 0.f, 1.f); break;
		case 1: L0 = 0.5f; break;
		case 2: { LC cusp = find_cusp(a_, b_); L0 = cusp.L; break; }
		case 3: {
			float Ld = L - 0.5f;
			float e1 = 0.5f + fabsf(Ld) + alpha * C;
			L0 = 0.5f * (1.f + sgn_f(Ld) * (e1 - sqrtf(e1*e1 - 2.f*fabsf(Ld))));
			break;
		}
		default: {  // 4
			LC cusp = find_cusp(a_, b_);
			float Ld = L - cusp.L;
			float k = 2.f * (Ld > 0.f ? 1.f - cusp.L : cusp.L);
			float e1 = 0.5f * k + fabsf(Ld) + alpha * C / k;
			L0 = cusp.L + 0.5f * (sgn_f(Ld) * (e1 - sqrtf(e1*e1 - 2.f*k*fabsf(Ld))));
			break;
		}
	}

	LC cusp_for_intersection;
	if (strategy == 2) { cusp_for_intersection = find_cusp(a_, b_); }
	else if (strategy == 4) { cusp_for_intersection = find_cusp(a_, b_); }
	else { cusp_for_intersection = find_cusp(a_, b_); }

	float t = find_gamut_intersection(a_, b_, L, C, L0, cusp_for_intersection);
	float Lc = L0 * (1.f - t) + t * L;
	float Cc = t * C;
	Lab4 out; out.L = Lc; out.a = Cc * a_; out.b = Cc * b_;
	return oklab_to_linear_srgb_one(out);
}

} // dev

// =============================================================================
// 커널 정의 + host launch wrapper.
// =============================================================================

__global__ void k_linear_srgb_to_oklab(const RGB4* __restrict__ in, Lab4* __restrict__ out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	out[idx] = dev::linear_srgb_to_oklab_one(in[idx]);
}
__global__ void k_oklab_to_linear_srgb(const Lab4* __restrict__ in, RGB4* __restrict__ out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	out[idx] = dev::oklab_to_linear_srgb_one(in[idx]);
}
__global__ void k_okhsl_to_srgb(const HSL4* __restrict__ in, RGB4* __restrict__ out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	out[idx] = dev::okhsl_to_srgb_one(in[idx]);
}
__global__ void k_srgb_to_okhsl(const RGB4* __restrict__ in, HSL4* __restrict__ out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	out[idx] = dev::srgb_to_okhsl_one(in[idx]);
}
__global__ void k_okhsv_to_srgb(const HSV4* __restrict__ in, RGB4* __restrict__ out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	out[idx] = dev::okhsv_to_srgb_one(in[idx]);
}
__global__ void k_srgb_to_okhsv(const RGB4* __restrict__ in, HSV4* __restrict__ out, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	out[idx] = dev::srgb_to_okhsv_one(in[idx]);
}
__global__ void k_clip(const RGB4* __restrict__ in, RGB4* __restrict__ out, int N, int strategy, float alpha) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) return;
	out[idx] = dev::clip_helper(in[idx], strategy, alpha);
}

// Launch wrapper.
namespace {
	inline dim3 grid_for(int N, int block) { return dim3((N + block - 1) / block); }
}

void linear_srgb_to_oklab(const RGB4* d_in, Lab4* d_out, int N, cudaStream_t s) {
	int b = 256; k_linear_srgb_to_oklab<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N);
}
void oklab_to_linear_srgb(const Lab4* d_in, RGB4* d_out, int N, cudaStream_t s) {
	int b = 256; k_oklab_to_linear_srgb<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N);
}
void okhsl_to_srgb(const HSL4* d_in, RGB4* d_out, int N, cudaStream_t s) {
	int b = 256; k_okhsl_to_srgb<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N);
}
void srgb_to_okhsl(const RGB4* d_in, HSL4* d_out, int N, cudaStream_t s) {
	int b = 256; k_srgb_to_okhsl<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N);
}
void okhsv_to_srgb(const HSV4* d_in, RGB4* d_out, int N, cudaStream_t s) {
	int b = 256; k_okhsv_to_srgb<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N);
}
void srgb_to_okhsv(const RGB4* d_in, HSV4* d_out, int N, cudaStream_t s) {
	int b = 256; k_srgb_to_okhsv<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N);
}
void gamut_clip_preserve_chroma   (const RGB4* d_in, RGB4* d_out, int N, cudaStream_t s) {
	int b = 256; k_clip<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N, 0, 0.05f);
}
void gamut_clip_project_to_0_5    (const RGB4* d_in, RGB4* d_out, int N, cudaStream_t s) {
	int b = 256; k_clip<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N, 1, 0.05f);
}
void gamut_clip_project_to_L_cusp (const RGB4* d_in, RGB4* d_out, int N, cudaStream_t s) {
	int b = 256; k_clip<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N, 2, 0.05f);
}
void gamut_clip_adaptive_L0_0_5   (const RGB4* d_in, RGB4* d_out, int N, float alpha, cudaStream_t s) {
	int b = 256; k_clip<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N, 3, alpha);
}
void gamut_clip_adaptive_L0_L_cusp(const RGB4* d_in, RGB4* d_out, int N, float alpha, cudaStream_t s) {
	int b = 256; k_clip<<<grid_for(N, b), b, 0, s>>>(d_in, d_out, N, 4, alpha);
}

} // namespace ok_color_cuda
