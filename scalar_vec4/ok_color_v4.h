#pragma once
// -----------------------------------------------------------------------------
// ok_color_v4 — float4 + 4x4 padded 행렬로 재구성한 ok_color 사본.
//   목적: 컴파일러가 packed SIMD (mulps/fma_ps) 를 자동으로 쓸 수 있는지 확인.
//   API 는 ok_color 와 동일, 네임스페이스만 ok_color_v4.
// -----------------------------------------------------------------------------

namespace ok_color_v4
{

// =============================================================================
// float4 — 16-byte 정렬, 4-wide. .w 는 padding 으로 항상 0 을 유지.
// =============================================================================
struct alignas(16) float4
{
	float x, y, z, w;

	float4 operator+(float4 o) const { return { this->x + o.x, this->y + o.y, this->z + o.z, this->w + o.w }; }
	float4 operator-(float4 o) const { return { this->x - o.x, this->y - o.y, this->z - o.z, this->w - o.w }; }
	float4 operator-()         const { return { -this->x, -this->y, -this->z, -this->w }; }
	float4 operator*(float s)  const { return { this->x * s, this->y * s, this->z * s, this->w * s }; }
	float4 operator/(float s)  const { return { this->x / s, this->y / s, this->z / s, this->w / s }; }

	// 4-wide dot. .w 가 0 이면 3-wide 와 같은 값.
	float dot(float4 o) const { return this->x * o.x + this->y * o.y + this->z * o.z + this->w * o.w; }
};

inline float4 operator*(float s, float4 v) { return v * s; }
inline float dot(float4 a, float4 b) { return a.dot(b); }


struct Lab; struct RGB; struct HSV; struct HSL;


struct Lab
{
	union {
		struct { float L, a, b; };
		float4 vector;
	};
	Lab() : vector{ 0, 0, 0, 0 } {}
	Lab(float L_, float a_, float b_) : vector{ L_, a_, b_, 0.f } {}
	Lab(float4 v) : vector(v) {}
	operator float4() const { return this->vector; }

	RGB to_linear_srgb() const;
};

struct RGB
{
	union {
		struct { float r, g, b; };
		float4 vector;
	};
	RGB() : vector{ 0, 0, 0, 0 } {}
	RGB(float r_, float g_, float b_) : vector{ r_, g_, b_, 0.f } {}
	RGB(float4 v) : vector(v) {}
	operator float4() const { return this->vector; }

	Lab to_oklab() const;
	HSL to_okhsl() const;
	HSV to_okhsv() const;
};

struct HSV
{
	union {
		struct { float h, s, v; };
		float4 vector;
	};
	HSV() : vector{ 0, 0, 0, 0 } {}
	HSV(float h_, float s_, float v_) : vector{ h_, s_, v_, 0.f } {}
	HSV(float4 vv) : vector(vv) {}
	operator float4() const { return this->vector; }

	RGB to_srgb() const;
};

struct HSL
{
	union {
		struct { float h, s, l; };
		float4 vector;
	};
	HSL() : vector{ 0, 0, 0, 0 } {}
	HSL(float h_, float s_, float l_) : vector{ h_, s_, l_, 0.f } {}
	HSL(float4 v) : vector(v) {}
	operator float4() const { return this->vector; }

	RGB to_srgb() const;
};

struct ST;
struct LC { float L; float C; ST to_ST() const; };
struct ST { float S; float T; };
struct Cs { float C_0; float C_mid; float C_max; };

float clamp(float x, float min, float max);
float sgn(float x);
float srgb_transfer_function(float a);
float srgb_transfer_function_inv(float a);
float toe(float x);
float toe_inv(float x);

float compute_max_saturation(float a, float b);
LC find_cusp(float a, float b);
float find_gamut_intersection(float a, float b, float L1, float C1, float L0, LC cusp);
float find_gamut_intersection(float a, float b, float L1, float C1, float L0);
ST get_ST_mid(float a_, float b_);
Cs get_Cs(float L, float a_, float b_);

RGB gamut_clip_preserve_chroma(RGB rgb);
RGB gamut_clip_project_to_0_5(RGB rgb);
RGB gamut_clip_project_to_L_cusp(RGB rgb);
RGB gamut_clip_adaptive_L0_0_5(RGB rgb, float alpha = 0.05f);
RGB gamut_clip_adaptive_L0_L_cusp(RGB rgb, float alpha = 0.05f);

} // namespace ok_color_v4
