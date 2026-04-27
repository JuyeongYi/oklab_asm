// ok_color_v4.cpp — float4 기반 구현. 행렬은 4x3 (각 행 4-wide, .w=0).
//                   API/알고리즘 모두 ok_color.cpp 와 동일, 자료형만 다름.

#include "ok_color_v4.h"
#include <cmath>
#include <cfloat>

namespace ok_color_v4
{

namespace {

	constexpr float pi =
		3.1415926535897932384626433832795028841971693993751058209749445923078164062f;

	// 4-wide 패딩된 행렬 (마지막 lane = 0 → dot 결과 영향 없음).
	constexpr float4 M_RGB_TO_LMS[3] = {
		{ 0.4122214708f, 0.5363325363f, 0.0514459929f, 0.f },
		{ 0.2119034982f, 0.6806995451f, 0.1073969566f, 0.f },
		{ 0.0883024619f, 0.2817188376f, 0.6299787005f, 0.f },
	};
	constexpr float4 M_LMS_TO_LAB[3] = {
		{ 0.2104542553f,  0.7936177850f, -0.0040720468f, 0.f },
		{ 1.9779984951f, -2.4285922050f,  0.4505937099f, 0.f },
		{ 0.0259040371f,  0.7827717662f, -0.8086757660f, 0.f },
	};
	constexpr float4 M_LAB_TO_LMS[3] = {
		{ 1.f, +0.3963377774f, +0.2158037573f, 0.f },
		{ 1.f, -0.1055613458f, -0.0638541728f, 0.f },
		{ 1.f, -0.0894841775f, -1.2914855480f, 0.f },
	};
	constexpr float4 M_LMS_TO_RGB[3] = {
		{ +4.0767416621f, -3.3077115913f, +0.2309699292f, 0.f },
		{ -1.2684380046f, +2.6097574011f, -0.3413193965f, 0.f },
		{ -0.0041960863f, -0.7034186147f, +1.7076147010f, 0.f },
	};

} // anon

float clamp(float x, float min, float max) { if (x<min)return min; if (x>max)return max; return x; }
float sgn(float x) { return (float)(0.f<x) - (float)(x<0.f); }

float srgb_transfer_function(float a)
{
	return .0031308f >= a ? 12.92f * a : 1.055f * powf(a, .4166666666666667f) - .055f;
}
float srgb_transfer_function_inv(float a)
{
	return .04045f < a ? powf((a + .055f) / 1.055f, 2.4f) : a / 12.92f;
}

float toe(float x)
{
	constexpr float k_1 = 0.206f, k_2 = 0.03f;
	constexpr float k_3 = (1.f + k_1) / (1.f + k_2);
	const float u = k_3 * x - k_1;
	return 0.5f * (u + sqrtf(u*u + 4.f * k_2 * k_3 * x));
}
float toe_inv(float x)
{
	constexpr float k_1 = 0.206f, k_2 = 0.03f;
	constexpr float k_3 = (1.f + k_1) / (1.f + k_2);
	return (x*x + k_1*x) / (k_3 * (x + k_2));
}

Lab RGB::to_oklab() const
{
	const float4 rgb = this->vector;  // .w = 0 보장
	const float4 lms = {
		M_RGB_TO_LMS[0].dot(rgb),
		M_RGB_TO_LMS[1].dot(rgb),
		M_RGB_TO_LMS[2].dot(rgb),
		0.f,
	};
	const float4 lms_ = { cbrtf(lms.x), cbrtf(lms.y), cbrtf(lms.z), 0.f };
	return Lab{
		M_LMS_TO_LAB[0].dot(lms_),
		M_LMS_TO_LAB[1].dot(lms_),
		M_LMS_TO_LAB[2].dot(lms_),
	};
}

RGB Lab::to_linear_srgb() const
{
	const float4 lab = this->vector;
	const float4 lms_ = {
		M_LAB_TO_LMS[0].dot(lab),
		M_LAB_TO_LMS[1].dot(lab),
		M_LAB_TO_LMS[2].dot(lab),
		0.f,
	};
	const float4 lms = { lms_.x*lms_.x*lms_.x, lms_.y*lms_.y*lms_.y, lms_.z*lms_.z*lms_.z, 0.f };
	return RGB{
		M_LMS_TO_RGB[0].dot(lms),
		M_LMS_TO_RGB[1].dot(lms),
		M_LMS_TO_RGB[2].dot(lms),
	};
}

float compute_max_saturation(float a, float b)
{
	float k0, k1, k2, k3, k4;
	float4 w;
	if (-1.88170328f * a - 0.80936493f * b > 1.f) {
		k0=+1.19086277f; k1=+1.76576728f; k2=+0.59662641f; k3=+0.75515197f; k4=+0.56771245f;
		w = M_LMS_TO_RGB[0];
	} else if (1.81444104f * a - 1.19445276f * b > 1.f) {
		k0=+0.73956515f; k1=-0.45954404f; k2=+0.08285427f; k3=+0.12541070f; k4=+0.14503204f;
		w = M_LMS_TO_RGB[1];
	} else {
		k0=+1.35733652f; k1=-0.00915799f; k2=-1.15130210f; k3=-0.50559606f; k4=+0.00692167f;
		w = M_LMS_TO_RGB[2];
	}
	float S = k0 + k1*a + k2*b + k3*a*a + k4*a*b;

	const float4 ab_dir = { 0.f, a, b, 0.f };
	const float k_l = M_LAB_TO_LMS[0].dot(ab_dir);
	const float k_m = M_LAB_TO_LMS[1].dot(ab_dir);
	const float k_s = M_LAB_TO_LMS[2].dot(ab_dir);
	{
		const float l_ = 1.f + S*k_l, m_ = 1.f + S*k_m, s_ = 1.f + S*k_s;
		const float4 lms     = { l_*l_*l_, m_*m_*m_, s_*s_*s_, 0.f };
		const float4 lms_dS  = { 3.f*k_l*l_*l_, 3.f*k_m*m_*m_, 3.f*k_s*s_*s_, 0.f };
		const float4 lms_dS2 = { 6.f*k_l*k_l*l_, 6.f*k_m*k_m*m_, 6.f*k_s*k_s*s_, 0.f };
		const float f  = w.dot(lms);
		const float f1 = w.dot(lms_dS);
		const float f2 = w.dot(lms_dS2);
		S = S - f*f1 / (f1*f1 - 0.5f*f*f2);
	}
	return S;
}

LC find_cusp(float a, float b)
{
	const float S_cusp = compute_max_saturation(a, b);
	const RGB rgb_at_max = Lab{ 1.f, S_cusp*a, S_cusp*b }.to_linear_srgb();
	const float max_rgb = fmaxf(fmaxf(rgb_at_max.r, rgb_at_max.g), rgb_at_max.b);
	const float L_cusp = cbrtf(1.f / max_rgb);
	const float C_cusp = L_cusp * S_cusp;
	return { L_cusp, C_cusp };
}

float find_gamut_intersection(float a, float b, float L1, float C1, float L0, LC cusp)
{
	float t;
	if (((L1 - L0) * cusp.C - (cusp.L - L0) * C1) <= 0.f) {
		t = cusp.C * L0 / (C1 * cusp.L + cusp.C * (L0 - L1));
	} else {
		t = cusp.C * (L0 - 1.f) / (C1 * (cusp.L - 1.f) + cusp.C * (L0 - L1));
		{
			const float dL = L1 - L0;
			const float dC = C1;
			const float4 ab_dir = { 0.f, a, b, 0.f };
			const float k_l = M_LAB_TO_LMS[0].dot(ab_dir);
			const float k_m = M_LAB_TO_LMS[1].dot(ab_dir);
			const float k_s = M_LAB_TO_LMS[2].dot(ab_dir);
			const float l_dt = dL + dC*k_l, m_dt = dL + dC*k_m, s_dt = dL + dC*k_s;
			const float L = L0 * (1.f - t) + t * L1;
			const float C = t * C1;
			const float l_ = L + C*k_l, m_ = L + C*k_m, s_ = L + C*k_s;
			const float4 lms     = { l_*l_*l_, m_*m_*m_, s_*s_*s_, 0.f };
			const float4 lms_dt  = { 3.f*l_dt*l_*l_, 3.f*m_dt*m_*m_, 3.f*s_dt*s_*s_, 0.f };
			const float4 lms_dt2 = { 6.f*l_dt*l_dt*l_, 6.f*m_dt*m_dt*m_, 6.f*s_dt*s_dt*s_, 0.f };
			float t_r, t_g, t_b;
			{
				const float4& wr = M_LMS_TO_RGB[0];
				const float r = wr.dot(lms) - 1.f, r1 = wr.dot(lms_dt), r2 = wr.dot(lms_dt2);
				const float u_r = r1 / (r1*r1 - 0.5f*r*r2);
				t_r = u_r >= 0.f ? -r * u_r : FLT_MAX;
			}
			{
				const float4& wg = M_LMS_TO_RGB[1];
				const float g = wg.dot(lms) - 1.f, g1 = wg.dot(lms_dt), g2 = wg.dot(lms_dt2);
				const float u_g = g1 / (g1*g1 - 0.5f*g*g2);
				t_g = u_g >= 0.f ? -g * u_g : FLT_MAX;
			}
			{
				const float4& wb = M_LMS_TO_RGB[2];
				const float bb = wb.dot(lms) - 1.f, b1 = wb.dot(lms_dt), b2 = wb.dot(lms_dt2);
				const float u_b = b1 / (b1*b1 - 0.5f*bb*b2);
				t_b = u_b >= 0.f ? -bb * u_b : FLT_MAX;
			}
			t += fminf(t_r, fminf(t_g, t_b));
		}
	}
	return t;
}

float find_gamut_intersection(float a, float b, float L1, float C1, float L0)
{
	const LC cusp = find_cusp(a, b);
	return find_gamut_intersection(a, b, L1, C1, L0, cusp);
}

namespace {
	struct HueSplit { float L; float C; float a_; float b_; };
	HueSplit split_hue(const Lab& lab)
	{
		constexpr float eps = 0.00001f;
		const float C = fmaxf(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
		return { lab.L, C, lab.a / C, lab.b / C };
	}
	RGB project_along(const HueSplit& h, float L0)
	{
		const float t = find_gamut_intersection(h.a_, h.b_, h.L, h.C, L0);
		const float L_clipped = L0 * (1.f - t) + t * h.L;
		const float C_clipped = t * h.C;
		return Lab{ L_clipped, C_clipped * h.a_, C_clipped * h.b_ }.to_linear_srgb();
	}
	bool inside_unit_cube(RGB rgb)
	{
		return rgb.r > 0 && rgb.g > 0 && rgb.b > 0
		    && rgb.r < 1 && rgb.g < 1 && rgb.b < 1;
	}
}

RGB gamut_clip_preserve_chroma(RGB rgb)
{
	if (inside_unit_cube(rgb)) return rgb;
	const HueSplit h = split_hue(rgb.to_oklab());
	return project_along(h, clamp(h.L, 0.f, 1.f));
}
RGB gamut_clip_project_to_0_5(RGB rgb)
{
	if (inside_unit_cube(rgb)) return rgb;
	return project_along(split_hue(rgb.to_oklab()), 0.5f);
}
RGB gamut_clip_project_to_L_cusp(RGB rgb)
{
	if (inside_unit_cube(rgb)) return rgb;
	const HueSplit h = split_hue(rgb.to_oklab());
	const LC cusp = find_cusp(h.a_, h.b_);
	return project_along(h, cusp.L);
}
RGB gamut_clip_adaptive_L0_0_5(RGB rgb, float alpha)
{
	if (inside_unit_cube(rgb)) return rgb;
	const HueSplit h = split_hue(rgb.to_oklab());
	const float Ld = h.L - 0.5f;
	const float e1 = 0.5f + fabsf(Ld) + alpha * h.C;
	const float L0 = 0.5f * (1.f + sgn(Ld) * (e1 - sqrtf(e1*e1 - 2.f * fabsf(Ld))));
	return project_along(h, L0);
}
RGB gamut_clip_adaptive_L0_L_cusp(RGB rgb, float alpha)
{
	if (inside_unit_cube(rgb)) return rgb;
	const HueSplit h = split_hue(rgb.to_oklab());
	const LC cusp = find_cusp(h.a_, h.b_);
	const float Ld = h.L - cusp.L;
	const float k = 2.f * (Ld > 0 ? 1.f - cusp.L : cusp.L);
	const float e1 = 0.5f * k + fabsf(Ld) + alpha * h.C / k;
	const float L0 = cusp.L + 0.5f * (sgn(Ld) * (e1 - sqrtf(e1*e1 - 2.f * k * fabsf(Ld))));
	return project_along(h, L0);
}

ST LC::to_ST() const
{
	return { this->C / this->L, this->C / (1.f - this->L) };
}

ST get_ST_mid(float a_, float b_)
{
	const float S = 0.11516993f + 1.f / (
		+7.44778970f + 4.15901240f * b_
		+ a_ * (-2.19557347f + 1.75198401f * b_
			+ a_ * (-2.13704948f - 10.02301043f * b_
				+ a_ * (-4.24894561f + 5.38770819f * b_ + 4.69891013f * a_))));
	const float T = 0.11239642f + 1.f / (
		+1.61320320f - 0.68124379f * b_
		+ a_ * (+0.40370612f + 0.90148123f * b_
			+ a_ * (-0.27087943f + 0.61223990f * b_
				+ a_ * (+0.00299215f - 0.45399568f * b_ - 0.14661872f * a_))));
	return { S, T };
}

Cs get_Cs(float L, float a_, float b_)
{
	const LC cusp = find_cusp(a_, b_);
	const float C_max = find_gamut_intersection(a_, b_, L, 1.f, L, cusp);
	const ST ST_max = cusp.to_ST();
	const float k = C_max / fminf(L * ST_max.S, (1.f - L) * ST_max.T);
	float C_mid;
	{
		const ST ST_mid = get_ST_mid(a_, b_);
		const float C_a = L * ST_mid.S;
		const float C_b = (1.f - L) * ST_mid.T;
		C_mid = 0.9f * k * sqrtf(sqrtf(1.f / (
			1.f / (C_a*C_a*C_a*C_a) + 1.f / (C_b*C_b*C_b*C_b))));
	}
	float C_0;
	{
		const float C_a = L * 0.4f;
		const float C_b = (1.f - L) * 0.8f;
		C_0 = sqrtf(1.f / (1.f / (C_a*C_a) + 1.f / (C_b*C_b)));
	}
	return { C_0, C_mid, C_max };
}

RGB HSL::to_srgb() const
{
	if (this->l == 1.0f) return { 1.f, 1.f, 1.f };
	if (this->l == 0.0f) return { 0.f, 0.f, 0.f };
	const float a_ = cosf(2.f * pi * this->h);
	const float b_ = sinf(2.f * pi * this->h);
	const float L  = toe_inv(this->l);
	const Cs cs = get_Cs(L, a_, b_);
	const float C_0 = cs.C_0, C_mid = cs.C_mid, C_max = cs.C_max;
	constexpr float mid = 0.8f, mid_inv = 1.25f;
	float C, t, k_0, k_1, k_2;
	if (this->s < mid) {
		t = mid_inv * this->s;
		k_1 = mid * C_0;
		k_2 = 1.f - k_1 / C_mid;
		C = t * k_1 / (1.f - k_2 * t);
	} else {
		t = (this->s - mid) / (1.f - mid);
		k_0 = C_mid;
		k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		k_2 = 1.f - k_1 / (C_max - C_mid);
		C = k_0 + t * k_1 / (1.f - k_2 * t);
	}
	const RGB rgb_lin = Lab{ L, C * a_, C * b_ }.to_linear_srgb();
	return RGB{
		srgb_transfer_function(rgb_lin.r),
		srgb_transfer_function(rgb_lin.g),
		srgb_transfer_function(rgb_lin.b),
	};
}

HSL RGB::to_okhsl() const
{
	const Lab lab = RGB{
		srgb_transfer_function_inv(this->r),
		srgb_transfer_function_inv(this->g),
		srgb_transfer_function_inv(this->b),
	}.to_oklab();
	const float C  = sqrtf(lab.a*lab.a + lab.b*lab.b);
	const float a_ = lab.a / C, b_ = lab.b / C, L = lab.L;
	const float h_ = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi;
	const Cs cs = get_Cs(L, a_, b_);
	const float C_0 = cs.C_0, C_mid = cs.C_mid, C_max = cs.C_max;
	constexpr float mid = 0.8f, mid_inv = 1.25f;
	float s_;
	if (C < C_mid) {
		const float k_1 = mid * C_0;
		const float k_2 = 1.f - k_1 / C_mid;
		const float t = C / (k_1 + k_2 * C);
		s_ = t * mid;
	} else {
		const float k_0 = C_mid;
		const float k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		const float k_2 = 1.f - k_1 / (C_max - C_mid);
		const float t = (C - k_0) / (k_1 + k_2 * (C - k_0));
		s_ = mid + (1.f - mid) * t;
	}
	const float l_ = toe(L);
	return { h_, s_, l_ };
}

RGB HSV::to_srgb() const
{
	const float a_ = cosf(2.f * pi * this->h);
	const float b_ = sinf(2.f * pi * this->h);
	const LC cusp = find_cusp(a_, b_);
	const ST ST_max = cusp.to_ST();
	const float S_max = ST_max.S, T_max = ST_max.T;
	constexpr float S_0 = 0.5f;
	const float k = 1.f - S_0 / S_max;
	const float L_v = 1.f - this->s * S_0 / (S_0 + T_max - T_max * k * this->s);
	const float C_v =        this->s * T_max * S_0 / (S_0 + T_max - T_max * k * this->s);
	float L = this->v * L_v;
	float C = this->v * C_v;
	const float L_vt = toe_inv(L_v);
	const float C_vt = C_v * L_vt / L_v;
	const float L_new = toe_inv(L);
	C = C * L_new / L; L = L_new;
	const RGB rgb_scale = Lab{ L_vt, a_*C_vt, b_*C_vt }.to_linear_srgb();
	const float scale_L = cbrtf(1.f / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.f)));
	L *= scale_L; C *= scale_L;
	const RGB rgb_lin = Lab{ L, C*a_, C*b_ }.to_linear_srgb();
	return RGB{
		srgb_transfer_function(rgb_lin.r),
		srgb_transfer_function(rgb_lin.g),
		srgb_transfer_function(rgb_lin.b),
	};
}

HSV RGB::to_okhsv() const
{
	const Lab lab = RGB{
		srgb_transfer_function_inv(this->r),
		srgb_transfer_function_inv(this->g),
		srgb_transfer_function_inv(this->b),
	}.to_oklab();
	float C = sqrtf(lab.a*lab.a + lab.b*lab.b);
	const float a_ = lab.a / C, b_ = lab.b / C;
	float L = lab.L;
	const float h_ = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi;
	const LC cusp = find_cusp(a_, b_);
	const ST ST_max = cusp.to_ST();
	const float S_max = ST_max.S, T_max = ST_max.T;
	constexpr float S_0 = 0.5f;
	const float k = 1.f - S_0 / S_max;
	const float t = T_max / (C + L * T_max);
	const float L_v = t * L;
	const float C_v = t * C;
	const float L_vt = toe_inv(L_v);
	const float C_vt = C_v * L_vt / L_v;
	const RGB rgb_scale = Lab{ L_vt, a_*C_vt, b_*C_vt }.to_linear_srgb();
	const float scale_L = cbrtf(1.f / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g), fmaxf(rgb_scale.b, 0.f)));
	L /= scale_L; C /= scale_L;
	C = C * toe(L) / L; L = toe(L);
	const float v = L / L_v;
	const float s = (S_0 + T_max) * C_v / (T_max * S_0 + T_max * k * C_v);
	return { h_, s, v };
}

} // namespace ok_color_v4
