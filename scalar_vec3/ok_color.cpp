// -----------------------------------------------------------------------------
// ok_color.cpp — ok_color.h 의 구현 파일.
//
// Copyright(c) 2021 Björn Ottosson — MIT License.
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the "Software"),
//   to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.
//
// 이 파일은 원본 ok_color.h 의 구현부를 옮긴 뒤,
//   - 행렬-벡터 곱을 vec3::dot 으로 재표현하고,
//   - 변환 함수를 멤버 메서드로 옮긴 것이다.
// 알고리즘과 수치 결과는 원본과 동일하다.
// -----------------------------------------------------------------------------

#include "ok_color.h"

#include <cmath>
#include <cfloat>

namespace ok_color
{

// =============================================================================
// 내부 상수
// =============================================================================
namespace {

	// 정밀도 높은 π. OkHSV/OkHSL 에서 hue 와 (a, b) 사이 변환에만 쓰인다.
	constexpr float pi =
		3.1415926535897932384626433832795028841971693993751058209749445923078164062f;


	// -------------------------------------------------------------------------
	// 색공간 변환 행렬.
	//
	//   Oklab 변환은 두 단계 행렬 + 비선형 단계(cube / cube root)로 정의된다:
	//
	//     linear sRGB --[M_RGB_TO_LMS]--> LMS --[cbrt]--> LMS' --[M_LMS_TO_LAB]--> Lab
	//     Lab         --[M_LAB_TO_LMS]--> LMS' --[cube]--> LMS  --[M_LMS_TO_RGB]--> linear sRGB
	//
	//   각 행렬을 행 벡터 3개(vec3)로 표현해두면, 행렬-벡터 곱이 그대로
	//   row.dot(input) 로 읽힌다. 또한 같은 행이 여러 함수 (예: compute_max_saturation,
	//   find_gamut_intersection)에서 재사용되므로, 한 곳에 모아 두면 추후 다른 색역
	//   (P3, Rec.2020 등) 으로 바꿀 때 행렬만 갈아끼우면 된다.
	// -------------------------------------------------------------------------

	// linear sRGB → LMS (cone response).
	constexpr vec3 M_RGB_TO_LMS[3] = {
		{ 0.4122214708f, 0.5363325363f, 0.0514459929f }, // l
		{ 0.2119034982f, 0.6806995451f, 0.1073969566f }, // m
		{ 0.0883024619f, 0.2817188376f, 0.6299787005f }, // s
	};

	// LMS' (= cbrt of LMS) → Lab.
	constexpr vec3 M_LMS_TO_LAB[3] = {
		{ 0.2104542553f,  0.7936177850f, -0.0040720468f }, // L
		{ 1.9779984951f, -2.4285922050f,  0.4505937099f }, // a
		{ 0.0259040371f,  0.7827717662f, -0.8086757660f }, // b
	};

	// Lab → LMS'. M_LMS_TO_LAB 의 (수치적) 역행렬.
	//   첫 열은 모두 1 인데, 이는 L 이 LMS' 에 대등하게 더해진다는 사실을 반영한다.
	constexpr vec3 M_LAB_TO_LMS[3] = {
		{ 1.f, +0.3963377774f, +0.2158037573f }, // l_
		{ 1.f, -0.1055613458f, -0.0638541728f }, // m_
		{ 1.f, -0.0894841775f, -1.2914855480f }, // s_
	};

	// LMS → linear sRGB. M_RGB_TO_LMS 의 역행렬.
	constexpr vec3 M_LMS_TO_RGB[3] = {
		{ +4.0767416621f, -3.3077115913f, +0.2309699292f }, // r
		{ -1.2684380046f, +2.6097574011f, -0.3413193965f }, // g
		{ -0.0041960863f, -0.7034186147f, +1.7076147010f }, // b
	};

} // namespace (internal)


// =============================================================================
// vec3 의 비-인라인 메서드.
//   length 와 normalized 는 sqrt 를 호출하므로 cmath 를 보고 있는 .cpp 에 둔다.
// =============================================================================

float vec3::length() const
{
	return sqrtf(dot(*this));
}

vec3 vec3::normalized() const
{
	// 길이가 정확히 0 이면 그대로 둔다 (NaN 방지). 이 라이브러리 안에서 일어날 일이
	// 거의 없으므로 별도의 epsilon 보정은 두지 않는다.
	float L = length();
	return L > 0.f ? (*this / L) : *this;
}


// =============================================================================
// 작은 유틸
// =============================================================================

float clamp(float x, float min, float max)
{
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

float sgn(float x)
{
	// (0 < x) 와 (x < 0) 두 비교의 차이로 -1 / 0 / +1 을 분기 없이 만들어낸다.
	return (float)(0.f < x) - (float)(x < 0.f);
}


// =============================================================================
// sRGB transfer function.
//   linear sRGB 와 표시용 sRGB 사이의 비선형 매핑.
//   IEC 61966-2-1 표준 공식. 작은 영역은 12.92 배로 선형, 큰 영역은 1.055 * x^(1/2.4)
//   에서 0.055 를 빼는 곡선 — 두 영역이 1차 미분까지 연속이 되도록 임계점이 0.0031308
//   / 0.04045 로 정해져 있다.
// =============================================================================

float srgb_transfer_function(float a)
{
	// linear → sRGB. 매우 어두운 부분(<= 0.0031308)은 직선, 그 외엔 감마 1/2.4 곡선.
	return .0031308f >= a
		? 12.92f * a
		: 1.055f * powf(a, .4166666666666667f) - .055f;  // 1/2.4 ≈ 0.41666...
}

float srgb_transfer_function_inv(float a)
{
	// sRGB → linear. 임계점이 0.04045 인 이유는 0.0031308 * 12.92 = 0.04045 이기 때문.
	return .04045f < a
		? powf((a + .055f) / 1.055f, 2.4f)
		: a / 12.92f;
}


// =============================================================================
// toe / toe_inv — perceptual lightness 응답 곡선.
//
//   Oklab 의 L 은 luminance 와 거의 선형이지만, 어두운 영역의 인지 변별력이 부족하다.
//   OkHSL 에서 사용되는 perceptual l 은 toe(L) 로 정의되며, 어두운 영역을 "위로 끌어
//   올려" 시각적으로 균등한 슬라이더가 만들어지도록 한다.
//
//   곡선은 두 매개변수 (k_1, k_2) 를 갖는다:
//     toe(x) = 0.5 * ( k3*x - k1 + sqrt((k3*x - k1)^2 + 4*k2*k3*x) )
//   k_3 = (1+k_1)/(1+k_2) 는 toe(1) = 1 을 만들어주는 정규화 상수.
//
//   이 형태는 (x*x + k1*x) / (k3*(x + k2)) 의 닫힌 해 역함수를 갖는다는 것이 핵심이다.
// =============================================================================

float toe(float x)
{
	constexpr float k_1 = 0.206f;
	constexpr float k_2 = 0.03f;
	constexpr float k_3 = (1.f + k_1) / (1.f + k_2);  // toe(1) = 1 보장

	// 이차 방정식의 양수 해를 직접 풀어쓴 형태.
	const float u = k_3 * x - k_1;
	return 0.5f * (u + sqrtf(u * u + 4.f * k_2 * k_3 * x));
}

float toe_inv(float x)
{
	constexpr float k_1 = 0.206f;
	constexpr float k_2 = 0.03f;
	constexpr float k_3 = (1.f + k_1) / (1.f + k_2);

	// toe(y) = x 를 y 에 대해 풀면 y = (x^2 + k1*x) / (k3*(x + k2)).
	return (x * x + k_1 * x) / (k_3 * (x + k_2));
}


// =============================================================================
// 색공간 변환: Oklab ↔ linear sRGB
// =============================================================================

// linear sRGB → Oklab.
//
//   1단계  RGB → LMS  (cone-like response). 행렬 곱.
//   2단계  LMS' = cbrt(LMS).  비선형 단계 (perceptual 균일화의 핵심).
//   3단계  LMS' → Lab.  다시 행렬 곱.
Lab RGB::to_oklab() const
{
	// 1) linear RGB → LMS. 세 개의 행 dot 으로 한 줄씩 표현.
	const vec3 rgb = this->vector;
	const vec3 lms = {
		M_RGB_TO_LMS[0].dot(rgb),
		M_RGB_TO_LMS[1].dot(rgb),
		M_RGB_TO_LMS[2].dot(rgb),
	};

	// 2) cube root. Oklab 이 perceptually 균일한 핵심이 이 비선형성에 있다.
	const vec3 lms_ = { cbrtf(lms.x), cbrtf(lms.y), cbrtf(lms.z) };

	// 3) LMS' → Lab.
	return Lab{
		M_LMS_TO_LAB[0].dot(lms_),
		M_LMS_TO_LAB[1].dot(lms_),
		M_LMS_TO_LAB[2].dot(lms_),
	};
}

// Oklab → linear sRGB. RGB::to_oklab() 의 역.
RGB Lab::to_linear_srgb() const
{
	// 1) Lab → LMS' (역행렬 곱).
	const vec3 lab = this->vector;
	const vec3 lms_ = {
		M_LAB_TO_LMS[0].dot(lab),
		M_LAB_TO_LMS[1].dot(lab),
		M_LAB_TO_LMS[2].dot(lab),
	};

	// 2) cube. cbrt 의 역.
	const vec3 lms = {
		lms_.x * lms_.x * lms_.x,
		lms_.y * lms_.y * lms_.y,
		lms_.z * lms_.z * lms_.z,
	};

	// 3) LMS → linear RGB.
	return RGB{
		M_LMS_TO_RGB[0].dot(lms),
		M_LMS_TO_RGB[1].dot(lms),
		M_LMS_TO_RGB[2].dot(lms),
	};
}


// =============================================================================
// compute_max_saturation
//
//   "이 hue 방향으로 sRGB 색역 안에 들어가는 가장 큰 saturation S = C/L 은 얼마인가?"
//
//   사고 흐름:
//     • 어떤 channel(R/G/B) 이 가장 먼저 1 을 넘는지에 따라 답이 결정된다.
//     • 정규화된 (a, b) 가 어느 영역에 있는지를 두 개의 선형 부등식으로 미리 가른다.
//     • 그 channel 에 맞는 다항식으로 S 의 초기치를 만든 뒤,
//     • Halley's method 한 번이면 1e-6 미만 오차로 수렴한다.
//
//   풀이 식:
//     L = 1, a_lab = S*a, b_lab = S*b 인 점에서
//       l_(S) = 1 + S * k_l,   k_l = M_LAB_TO_LMS[0] · (0, a, b)
//       l(S)  = l_(S)^3
//     channel 의 sRGB 값 = wl*l(S) + wm*m(S) + ws*s(S) = 1 인 S 를 찾는 문제.
//     f(S) = w·(l, m, s)(S) − 1 이 0 이 되는 S.
// =============================================================================

float compute_max_saturation(float a, float b)
{
	// 어떤 channel 이 임계인지 분기. 두 부등식의 좌변은 (a,b) 평면을 세 부채꼴로 가른다.
	float k0, k1, k2, k3, k4;
	vec3 w;  // M_LMS_TO_RGB 의 한 행 (선택된 channel).

	if (-1.88170328f * a - 0.80936493f * b > 1.f)
	{
		// Red 가 먼저 1 을 넘는 영역.
		k0 = +1.19086277f; k1 = +1.76576728f; k2 = +0.59662641f;
		k3 = +0.75515197f; k4 = +0.56771245f;
		w  = M_LMS_TO_RGB[0];
	}
	else if (1.81444104f * a - 1.19445276f * b > 1.f)
	{
		// Green.
		k0 = +0.73956515f; k1 = -0.45954404f; k2 = +0.08285427f;
		k3 = +0.12541070f; k4 = +0.14503204f;
		w  = M_LMS_TO_RGB[1];
	}
	else
	{
		// Blue.
		k0 = +1.35733652f; k1 = -0.00915799f; k2 = -1.15130210f;
		k3 = -0.50559606f; k4 = +0.00692167f;
		w  = M_LMS_TO_RGB[2];
	}

	// 1) 다항식 초기치. (a, b) 의 2차식.
	float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

	// 2) k_l, k_m, k_s 계산.
	//    M_LAB_TO_LMS 의 첫 열은 1 이므로, hue 방향벡터 (0, a, b) 와 dot 하면 a, b
	//    성분만 남아 정확히 원본의 0.396*a + 0.215*b 형태가 된다.
	const vec3 ab_dir = { 0.f, a, b };
	const float k_l = M_LAB_TO_LMS[0].dot(ab_dir);
	const float k_m = M_LAB_TO_LMS[1].dot(ab_dir);
	const float k_s = M_LAB_TO_LMS[2].dot(ab_dir);

	// 3) Halley's method 한 번 적용.
	{
		// 현재 S 에서의 l_(S), m_(S), s_(S).
		const float l_ = 1.f + S * k_l;
		const float m_ = 1.f + S * k_m;
		const float s_ = 1.f + S * k_s;

		// l, m, s = (l_)^3 등.
		const vec3 lms = { l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_ };

		// dl/dS = 3 * k_l * l_^2  (체인룰).  d2l/dS2 = 6 * k_l^2 * l_  (체인룰 재적용).
		const vec3 lms_dS  = { 3.f * k_l * l_ * l_,
			                   3.f * k_m * m_ * m_,
			                   3.f * k_s * s_ * s_ };
		const vec3 lms_dS2 = { 6.f * k_l * k_l * l_,
			                   6.f * k_m * k_m * m_,
			                   6.f * k_s * k_s * s_ };

		// f(S) = w · LMS − 1 (constant 항은 −1 인데 어차피 도함수에서 사라짐).
		const float f  = w.dot(lms);
		const float f1 = w.dot(lms_dS);
		const float f2 = w.dot(lms_dS2);

		// Halley step:  S ← S − f * f' / (f'^2 − 0.5 * f * f'').
		// (Newton 의 변형으로 2차 도함수를 함께 이용해 수렴 속도를 1단계 끌어올린다.)
		S = S - f * f1 / (f1 * f1 - 0.5f * f * f2);
	}

	return S;
}


// =============================================================================
// find_cusp — 주어진 hue 에서 sRGB 색역의 cusp(가장 chroma 가 큰 점).
//
//   원리:
//     1) saturation S = C/L 의 최대치를 먼저 구한다 (compute_max_saturation).
//     2) 그러면 Lab 에서 (1, S*a, S*b) 가 sRGB 경계 위 한 점이지만, RGB 가 [0,1] 이 아닌
//        스케일을 갖는다. 한 channel 이 1 이 되도록 전체를 cube-root 비례 축소.
//        (선형 RGB 의 max 에 cbrt 가 붙는 이유는 Lab→RGB 가 LMS' 단계에서 cube 거치기
//        때문 — L 을 1/cbrt(max) 배 하면 RGB 가 정확히 1/max 배 된다.)
// =============================================================================

LC find_cusp(float a, float b)
{
	// 1) 최대 saturation.
	const float S_cusp = compute_max_saturation(a, b);

	// 2) (L=1, a=S*a, b=S*b) 에서의 RGB 를 본다. 가장 큰 채널이 1 이 되도록 L 만 축소.
	const RGB rgb_at_max = Lab{ 1.f, S_cusp * a, S_cusp * b }.to_linear_srgb();
	const float max_rgb  = fmaxf(fmaxf(rgb_at_max.r, rgb_at_max.g), rgb_at_max.b);
	const float L_cusp   = cbrtf(1.f / max_rgb);
	const float C_cusp   = L_cusp * S_cusp;

	return { L_cusp, C_cusp };
}


// =============================================================================
// find_gamut_intersection
//
//   Lab 의 (L0, 0) 점에서 (L1, C1) 점으로 가는 직선이 sRGB 색역 경계와 만나는 t.
//   결과 점은 (L0*(1-t) + t*L1, t*C1).
//
//   알고리즘:
//     • 색역 단면(constant hue 슬라이스)은 (0,0)-(1,0)-(L_cusp, C_cusp) 삼각형으로
//       근사된다 (실제 경계는 약간 곡면).
//     • lower half (cusp 아래쪽)는 곡면 영향이 작아 삼각형 교차로 충분.
//     • upper half (cusp 위쪽)는 곡선이 휘므로, 삼각형 교차를 초기값으로 두고
//       실제 RGB=1 등치선에 대해 Halley step 한 번을 r/g/b 각각 풀어 가장 작은 양의 t 를
//       채택한다.
// =============================================================================

float find_gamut_intersection(float a, float b, float L1, float C1, float L0, LC cusp)
{
	float t;

	// 직선이 cusp 점의 어느 쪽 반평면에 있는지 외적의 부호로 판정.
	if (((L1 - L0) * cusp.C - (cusp.L - L0) * C1) <= 0.f)
	{
		// ---- Lower half: 삼각형 (0,0)-(L_cusp,C_cusp) 변과의 교차로 충분. ----
		t = cusp.C * L0 / (C1 * cusp.L + cusp.C * (L0 - L1));
	}
	else
	{
		// ---- Upper half: 삼각형 (1,0)-(L_cusp,C_cusp) 변과의 교차를 초기값으로. ----
		t = cusp.C * (L0 - 1.f) / (C1 * (cusp.L - 1.f) + cusp.C * (L0 - L1));

		// 이어서 r=1 / g=1 / b=1 등치선에 대해 Halley 한 단계.
		// (필요시 2~3단계까지 반복하면 정확도 ↑, 비용도 비례 증가.)
		{
			const float dL = L1 - L0;
			const float dC = C1;

			// k_l, k_m, k_s 는 hue 방향 (0, a, b) 에 대한 Lab→LMS' 도함수.
			const vec3 ab_dir = { 0.f, a, b };
			const float k_l = M_LAB_TO_LMS[0].dot(ab_dir);
			const float k_m = M_LAB_TO_LMS[1].dot(ab_dir);
			const float k_s = M_LAB_TO_LMS[2].dot(ab_dir);

			// l_(t) 의 t-방향 도함수: dL + dC * k_l. m, s 도 마찬가지.
			const float l_dt = dL + dC * k_l;
			const float m_dt = dL + dC * k_m;
			const float s_dt = dL + dC * k_s;

			// 직선상 점에서의 LMS', LMS, 도함수들.
			const float L = L0 * (1.f - t) + t * L1;
			const float C = t * C1;

			const float l_ = L + C * k_l;
			const float m_ = L + C * k_m;
			const float s_ = L + C * k_s;

			const vec3 lms = { l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_ };

			const vec3 lms_dt = { 3.f * l_dt * l_ * l_,
				                  3.f * m_dt * m_ * m_,
				                  3.f * s_dt * s_ * s_ };
			const vec3 lms_dt2 = { 6.f * l_dt * l_dt * l_,
				                   6.f * m_dt * m_dt * m_,
				                   6.f * s_dt * s_dt * s_ };

			// channel 별로 r(t)=1, g(t)=1, b(t)=1 이 되는 t 를 Halley step 으로 추정.
			//   각 channel: f = w·LMS − 1, f' = w·dLMS, f'' = w·d2LMS.
			//   Halley:  Δt = −f / (f' − 0.5 * f * f''/f') = −f * u,  u = f' / (f'^2 − 0.5*f*f'').
			float t_r, t_g, t_b;
			{
				const vec3& wr = M_LMS_TO_RGB[0];
				const float r  = wr.dot(lms) - 1.f;
				const float r1 = wr.dot(lms_dt);
				const float r2 = wr.dot(lms_dt2);
				const float u_r = r1 / (r1 * r1 - 0.5f * r * r2);
				// u_r < 0 이면 그 channel 은 t 를 줄이는 쪽이므로 후보에서 제외.
				t_r = u_r >= 0.f ? -r * u_r : FLT_MAX;
			}
			{
				const vec3& wg = M_LMS_TO_RGB[1];
				const float g  = wg.dot(lms) - 1.f;
				const float g1 = wg.dot(lms_dt);
				const float g2 = wg.dot(lms_dt2);
				const float u_g = g1 / (g1 * g1 - 0.5f * g * g2);
				t_g = u_g >= 0.f ? -g * u_g : FLT_MAX;
			}
			{
				const vec3& wb = M_LMS_TO_RGB[2];
				const float bb  = wb.dot(lms) - 1.f;
				const float b1  = wb.dot(lms_dt);
				const float b2  = wb.dot(lms_dt2);
				const float u_b = b1 / (b1 * b1 - 0.5f * bb * b2);
				t_b = u_b >= 0.f ? -bb * u_b : FLT_MAX;
			}

			// 셋 중 가장 빨리 1 을 만나는 channel 의 보정량을 채택.
			t += fminf(t_r, fminf(t_g, t_b));
		}
	}

	return t;
}

float find_gamut_intersection(float a, float b, float L1, float C1, float L0)
{
	// cusp 을 모르는 호출자를 위한 편의 오버로드.
	const LC cusp = find_cusp(a, b);
	return find_gamut_intersection(a, b, L1, C1, L0, cusp);
}


// =============================================================================
// gamut clip 들. 공통 패턴은 다음과 같다:
//
//   1) RGB 가 이미 [0,1]^3 안이면 그대로 반환 (no-op).
//   2) Oklab 으로 변환, hue 방향 (a_, b_) 단위 벡터 추출, chroma C 계산.
//   3) 끌어당길 기준점 L0 를 전략별로 고름.
//   4) (L0, 0) → (L, C) 직선이 색역 경계와 만나는 t 를 구해 그 위치로 점을 옮김.
//   5) Oklab → linear sRGB 로 되돌려 반환.
//
// 내부 헬퍼로 묶지 않은 이유는 원본의 5 가지 변형을 1:1 로 추적하기 쉬우라는 것이다.
// =============================================================================

namespace {
	// 안에 헬퍼: 작은 chroma 에서 0 으로 나누는 것을 막기 위한 epsilon 적용 chroma 분해.
	struct HueSplit { float L; float C; float a_; float b_; };

	HueSplit split_hue(const Lab& lab)
	{
		constexpr float eps = 0.00001f;
		const float C = fmaxf(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
		return { lab.L, C, lab.a / C, lab.b / C };
	}

	// 공통 마무리: 직선상 t 위치를 구해 다시 linear sRGB 로 복원.
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
} // namespace

RGB gamut_clip_preserve_chroma(RGB rgb)
{
	if (inside_unit_cube(rgb)) return rgb;

	const HueSplit h = split_hue(rgb.to_oklab());

	// L0 = clamp(L). 색의 lightness 에 가장 가까운 회색을 기준으로 끌어당김.
	const float L0 = clamp(h.L, 0.f, 1.f);
	return project_along(h, L0);
}

RGB gamut_clip_project_to_0_5(RGB rgb)
{
	if (inside_unit_cube(rgb)) return rgb;

	const HueSplit h = split_hue(rgb.to_oklab());

	// L0 = 0.5 고정. 회색 중앙으로 균등하게 끌어당김.
	return project_along(h, 0.5f);
}

RGB gamut_clip_project_to_L_cusp(RGB rgb)
{
	if (inside_unit_cube(rgb)) return rgb;

	const HueSplit h = split_hue(rgb.to_oklab());

	// L0 = L_cusp. cusp 의 lightness 가 hue 마다 다르므로 자연스러운 결과.
	// (find_gamut_intersection 안에서도 cusp 을 다시 계산함 — 최적화하려면 한 번만.)
	const LC cusp = find_cusp(h.a_, h.b_);
	return project_along(h, cusp.L);
}

RGB gamut_clip_adaptive_L0_0_5(RGB rgb, float alpha)
{
	if (inside_unit_cube(rgb)) return rgb;

	const HueSplit h = split_hue(rgb.to_oklab());

	// 0.5 기준 적응형. chroma 가 클수록 L0 가 L 쪽으로 끌려와 lightness 보존이 강해짐.
	//   Ld   = L − 0.5
	//   e1   = 0.5 + |Ld| + alpha*C
	//   L0   = 0.5 + 0.5 * sgn(Ld) * (e1 − sqrt(e1^2 − 2*|Ld|))
	const float Ld = h.L - 0.5f;
	const float e1 = 0.5f + fabsf(Ld) + alpha * h.C;
	const float L0 = 0.5f * (1.f + sgn(Ld) * (e1 - sqrtf(e1 * e1 - 2.f * fabsf(Ld))));

	return project_along(h, L0);
}

RGB gamut_clip_adaptive_L0_L_cusp(RGB rgb, float alpha)
{
	if (inside_unit_cube(rgb)) return rgb;

	const HueSplit h = split_hue(rgb.to_oklab());

	const LC cusp = find_cusp(h.a_, h.b_);

	// 위와 같은 적응형이지만 기준이 cusp.L. 위/아래 비대칭에 맞춰 k 로 보정.
	const float Ld = h.L - cusp.L;
	const float k  = 2.f * (Ld > 0 ? 1.f - cusp.L : cusp.L);

	const float e1 = 0.5f * k + fabsf(Ld) + alpha * h.C / k;
	const float L0 = cusp.L + 0.5f * (sgn(Ld) * (e1 - sqrtf(e1 * e1 - 2.f * k * fabsf(Ld))));

	return project_along(h, L0);
}


// =============================================================================
// LC ↔ ST, get_ST_mid, get_Cs
// =============================================================================

ST LC::to_ST() const
{
	// S = C/L,  T = C/(1-L). cusp 점이 (L, C) 일 때, 같은 hue 에서 일반 점 (L*, C*) 의
	// "삼각형 안" 최대 chroma 는 fmin(S * L*, T * (1-L*)) 로 즉시 구해진다.
	return { this->C / this->L, this->C / (1.f - this->L) };
}

// 매끄럽게 근사된 mid ST.
//   (a_, b_) 를 입력으로 받는 다항식 피팅. Björn 의 최적화 과정에서 나온 계수이며,
//   설계 조건은 항상 S_mid < S_max, T_mid < T_max 가 되는 것.
//   일반 구현이라기보단 데이터 피팅 결과이므로 수식보다는 표 같은 물건이다.
ST get_ST_mid(float a_, float b_)
{
	const float S = 0.11516993f + 1.f / (
		+7.44778970f + 4.15901240f * b_
		+ a_ * (-2.19557347f + 1.75198401f * b_
			+ a_ * (-2.13704948f - 10.02301043f * b_
				+ a_ * (-4.24894561f + 5.38770819f * b_ + 4.69891013f * a_
					)))
		);

	const float T = 0.11239642f + 1.f / (
		+1.61320320f - 0.68124379f * b_
		+ a_ * (+0.40370612f + 0.90148123f * b_
			+ a_ * (-0.27087943f + 0.61223990f * b_
				+ a_ * (+0.00299215f - 0.45399568f * b_ - 0.14661872f * a_
					)))
		);

	return { S, T };
}

// 한 (L, hue) 위치의 (C_0, C_mid, C_max) 묶음.
//   C_max : 색역 경계까지의 실제 최대 chroma (find_gamut_intersection 으로 정확히 계산).
//   C_mid : ST_mid 기반 매끄러운 보간값. cusp 부근의 곡선 효과를 k 로 보정해 0.9 배 적용.
//   C_0   : hue 와 무관한 표준 sketch (S=0.4, T=0.8 가정).
//
//   소프트 minimum: 1/(1/x^p + 1/y^p)^(1/p) 형태. p=4 (mid) / p=2 (0) 로 sharp 한 ‖∧‖
//   대신 둥근 최소를 만든다. 이게 OkHSL 슬라이더가 cusp 부근에서 끊어지지 않게 해주는
//   핵심 트릭이다.
Cs get_Cs(float L, float a_, float b_)
{
	const LC cusp = find_cusp(a_, b_);

	// C_max — 실제 색역 경계 교차.
	const float C_max = find_gamut_intersection(a_, b_, L, 1.f, L, cusp);

	// 곡면 보정 계수 k: 삼각형 가정에서의 한계 vs 실제 한계 비.
	const ST ST_max = cusp.to_ST();
	const float k = C_max / fminf(L * ST_max.S, (1.f - L) * ST_max.T);

	// C_mid — ST_mid 기반의 매끄러운 보간. p=4 의 소프트 min.
	float C_mid;
	{
		const ST ST_mid = get_ST_mid(a_, b_);
		const float C_a = L * ST_mid.S;
		const float C_b = (1.f - L) * ST_mid.T;
		C_mid = 0.9f * k * sqrtf(sqrtf(1.f / (
			1.f / (C_a * C_a * C_a * C_a) + 1.f / (C_b * C_b * C_b * C_b))));
	}

	// C_0 — hue 무관 sketch. 평균치 ST 로 p=2 소프트 min.
	float C_0;
	{
		const float C_a = L * 0.4f;
		const float C_b = (1.f - L) * 0.8f;
		C_0 = sqrtf(1.f / (1.f / (C_a * C_a) + 1.f / (C_b * C_b)));
	}

	return { C_0, C_mid, C_max };
}


// =============================================================================
// HSL → sRGB.
//
//   알고리즘 (요약):
//     1) hue h 를 (a_, b_) 단위 벡터로 풀고, perceptual l 을 toe_inv 로 Oklab L 로.
//     2) 같은 (L, hue) 에서의 (C_0, C_mid, C_max) 를 구함.
//     3) saturation s 를 두 구간으로 나눠 chroma C 로 매핑:
//          s ∈ [0,   0.8]  →  (C_0, C_mid) 사이를 유리식 보간
//          s ∈ [0.8, 1  ]  →  (C_mid, C_max) 사이를 유리식 보간
//        유리식 형태는 미분 연속이 유지되도록 설계됨.
//     4) Lab → linear sRGB → 감마 인코딩.
// =============================================================================

RGB HSL::to_srgb() const
{
	// (a) 극단값 가지치기.
	if (this->l == 1.0f) return { 1.f, 1.f, 1.f };
	if (this->l == 0.0f) return { 0.f, 0.f, 0.f };

	// (b) hue → (a_, b_) 단위 벡터, perceptual l → Oklab L.
	const float a_ = cosf(2.f * pi * this->h);
	const float b_ = sinf(2.f * pi * this->h);
	const float L  = toe_inv(this->l);

	// (c) 기준 chroma 3종.
	const Cs cs = get_Cs(L, a_, b_);
	const float C_0   = cs.C_0;
	const float C_mid = cs.C_mid;
	const float C_max = cs.C_max;

	// (d) 두 구간 유리식 보간. 분기점은 mid = 0.8.
	constexpr float mid     = 0.8f;
	constexpr float mid_inv = 1.25f;  // 1 / 0.8

	float C, t, k_0, k_1, k_2;
	if (this->s < mid)
	{
		// 첫 구간: s ∈ [0, 0.8]. (0, 0) ~ (mid, C_mid) 사이에서 C = t*k1 / (1 − k2*t).
		t   = mid_inv * this->s;
		k_1 = mid * C_0;
		k_2 = 1.f - k_1 / C_mid;
		C   = t * k_1 / (1.f - k_2 * t);
	}
	else
	{
		// 둘째 구간: s ∈ [0.8, 1]. (mid, C_mid) ~ (1, C_max).
		t   = (this->s - mid) / (1.f - mid);
		k_0 = C_mid;
		k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		k_2 = 1.f - k_1 / (C_max - C_mid);
		C   = k_0 + t * k_1 / (1.f - k_2 * t);
	}

	// (e) Lab → linear sRGB → 감마 인코딩.
	const RGB rgb_lin = Lab{ L, C * a_, C * b_ }.to_linear_srgb();
	return RGB{
		srgb_transfer_function(rgb_lin.r),
		srgb_transfer_function(rgb_lin.g),
		srgb_transfer_function(rgb_lin.b),
	};
}


// =============================================================================
// sRGB → HSL. okhsl_to_srgb 의 역연산.
//   유리식 보간을 닫힌 해로 풀 수 있다는 점이 OkHSL 의 큰 장점 — t 도 1차 다항식 비율
//   이라 분리해 풀어내면 끝.
// =============================================================================

HSL RGB::to_okhsl() const
{
	// 1) sRGB → linear → Lab.
	const Lab lab = RGB{
		srgb_transfer_function_inv(this->r),
		srgb_transfer_function_inv(this->g),
		srgb_transfer_function_inv(this->b),
	}.to_oklab();

	// 2) hue, chroma 분해.
	const float C  = sqrtf(lab.a * lab.a + lab.b * lab.b);
	const float a_ = lab.a / C;
	const float b_ = lab.b / C;
	const float L  = lab.L;
	// atan2(-b, -a) / pi 는 hue 를 [-1, 1] 로 풀고, 0.5 + 0.5*… 로 [0, 1] 매핑.
	const float h_ = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi;

	// 3) 같은 (L, hue) 에서의 (C_0, C_mid, C_max).
	const Cs cs = get_Cs(L, a_, b_);
	const float C_0   = cs.C_0;
	const float C_mid = cs.C_mid;
	const float C_max = cs.C_max;

	// 4) HSL→sRGB 의 유리식 보간을 역으로 풀어 s 를 얻는다.
	constexpr float mid     = 0.8f;
	constexpr float mid_inv = 1.25f;

	float s_;
	if (C < C_mid)
	{
		const float k_1 = mid * C_0;
		const float k_2 = 1.f - k_1 / C_mid;
		const float t   = C / (k_1 + k_2 * C);  // C = t*k1/(1−k2*t) 의 역
		s_ = t * mid;
	}
	else
	{
		const float k_0 = C_mid;
		const float k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		const float k_2 = 1.f - k_1 / (C_max - C_mid);
		const float t   = (C - k_0) / (k_1 + k_2 * (C - k_0));
		s_ = mid + (1.f - mid) * t;
	}

	// 5) perceptual l 은 toe(L).
	const float l_ = toe(L);
	return { h_, s_, l_ };
}


// =============================================================================
// HSV → sRGB.
//
//   직각삼각형 가정으로 출발해 두 가지 보정을 더한 형태:
//     A) 곡면 보정: cusp 부근에서 색역이 휘어 있으므로, "cusp ST" 를 기준으로 직선 보간한
//        뒤 RGB 의 max 가 정확히 1 이 되도록 cube-root 비례 축소.
//     B) toe 보정: lightness 슬라이더가 perceptual 하게 균등하도록 toe_inv/toe 를 끼움.
//
//   매개변수 S_0 = 0.5 는 OkHSV 의 saturation 슬라이더 형상을 결정. k = 1 − S_0/S_max
//   는 cusp 부근의 휘어짐 강도.
// =============================================================================

RGB HSV::to_srgb() const
{
	// hue → (a_, b_).
	const float a_ = cosf(2.f * pi * this->h);
	const float b_ = sinf(2.f * pi * this->h);

	// cusp 와 그 ST 표현. T_max, S_max 는 hue 별 색역의 폭.
	const LC cusp     = find_cusp(a_, b_);
	const ST ST_max   = cusp.to_ST();
	const float S_max = ST_max.S;
	const float T_max = ST_max.T;
	constexpr float S_0 = 0.5f;
	const float k = 1.f - S_0 / S_max;

	// 1) 직각삼각형 가정에서 v=1 일 때의 (L_v, C_v).
	const float L_v = 1.f - this->s * S_0 / (S_0 + T_max - T_max * k * this->s);
	const float C_v =        this->s * T_max * S_0 / (S_0 + T_max - T_max * k * this->s);

	// 2) v 만큼 축소.
	float L = this->v * L_v;
	float C = this->v * C_v;

	// 3) toe 보정 + cusp 곡면 보정 준비.
	//    L_vt, C_vt 는 v=1 슬라이스의 "toe-inv 한 좌표". 거기서 RGB 의 max 가 1 이 되게
	//    축소하는 비율 scale_L 을 cube-root 로 잡는다.
	const float L_vt = toe_inv(L_v);
	const float C_vt = C_v * L_vt / L_v;

	// 4) 현재 점에도 toe-inv 보정 적용.
	const float L_new = toe_inv(L);
	C = C * L_new / L;
	L = L_new;

	// 5) cusp 곡면 보정 — RGB max 가 1 이 되는 cube-root 스케일.
	const RGB rgb_scale = Lab{ L_vt, a_ * C_vt, b_ * C_vt }.to_linear_srgb();
	const float scale_L = cbrtf(1.f / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g),
		                                   fmaxf(rgb_scale.b, 0.f)));
	L *= scale_L;
	C *= scale_L;

	// 6) Lab → linear sRGB → 감마 인코딩.
	const RGB rgb_lin = Lab{ L, C * a_, C * b_ }.to_linear_srgb();
	return RGB{
		srgb_transfer_function(rgb_lin.r),
		srgb_transfer_function(rgb_lin.g),
		srgb_transfer_function(rgb_lin.b),
	};
}


// =============================================================================
// sRGB → HSV. okhsv_to_srgb 의 역.
//   곡면 보정과 toe 보정을 차례대로 풀어내면 v 와 s 를 분리해서 얻을 수 있다.
// =============================================================================

HSV RGB::to_okhsv() const
{
	// 1) 감마 디코딩 → Lab.
	const Lab lab = RGB{
		srgb_transfer_function_inv(this->r),
		srgb_transfer_function_inv(this->g),
		srgb_transfer_function_inv(this->b),
	}.to_oklab();

	// 2) hue / chroma 분해.
	float C  = sqrtf(lab.a * lab.a + lab.b * lab.b);
	const float a_ = lab.a / C;
	const float b_ = lab.b / C;
	float L  = lab.L;
	const float h_ = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi;

	// 3) cusp / ST_max / k.
	const LC cusp     = find_cusp(a_, b_);
	const ST ST_max   = cusp.to_ST();
	const float S_max = ST_max.S;
	const float T_max = ST_max.T;
	constexpr float S_0 = 0.5f;
	const float k = 1.f - S_0 / S_max;

	// 4) "v=1" 슬라이스의 (L_v, C_v) 를 역삼각형 식으로 복원.
	const float t   = T_max / (C + L * T_max);
	const float L_v = t * L;
	const float C_v = t * C;

	// 5) cusp 곡면 보정의 역. (L_vt, C_vt) 에서 RGB max 가 1 이 되도록 한 scale_L 의 역.
	const float L_vt = toe_inv(L_v);
	const float C_vt = C_v * L_vt / L_v;

	const RGB rgb_scale = Lab{ L_vt, a_ * C_vt, b_ * C_vt }.to_linear_srgb();
	const float scale_L = cbrtf(1.f / fmaxf(fmaxf(rgb_scale.r, rgb_scale.g),
		                                   fmaxf(rgb_scale.b, 0.f)));
	L /= scale_L;
	C /= scale_L;

	// 6) toe 보정의 역.
	C = C * toe(L) / L;
	L = toe(L);

	// 7) v 와 s.
	const float v = L / L_v;
	const float s = (S_0 + T_max) * C_v / (T_max * S_0 + T_max * k * C_v);

	return { h_, s, v };
}

} // namespace ok_color
