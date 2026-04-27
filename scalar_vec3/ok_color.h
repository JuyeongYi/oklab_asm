#pragma once
// -----------------------------------------------------------------------------
// ok_color — Oklab / OkLCh / OkHSL / OkHSV color space conversions
//            and sRGB gamut mapping.
//
// Original: Björn Ottosson (2021), MIT License.
//   https://bottosson.github.io/misc/ok_color.h
//   https://bottosson.github.io/posts/oklab/
//   https://bottosson.github.io/posts/colorpicker/
//   https://bottosson.github.io/posts/gamutclipping/
//
// 이 파일은 원본 single-header 코드를 (선언) 헤더와 (구현) ok_color.cpp 로 분리하고,
//   - 공통 vec3 자료형을 도입해 색상 타입(Lab/RGB/HSV/HSL)이 익명 union 으로 vec3와
//     같은 메모리를 공유하도록 재구성하고,
//   - <x>_to_<y> 형태의 변환 함수들을 소스 타입의 멤버 메서드로 옮기고,
//   - 행렬-벡터 곱을 dot 연산으로 표현
// 한 것이다. 동작은 원본과 동일하다.
//
// 용어:
//   - "linear sRGB" : 감마 인코딩 적용 전 RGB. 파이프라인 내부 표현.
//   - "sRGB"        : 감마 인코딩 적용 후 RGB. 화면/이미지 파일이 다루는 값.
//   각 함수의 RGB 가 어느 쪽인지 함수별 주석에 명시한다.
//
//   - hue (h) 는 [0,1) 범위 (0 = 빨강 방향).
//   - s/v/l 은 [0,1].
// -----------------------------------------------------------------------------

namespace ok_color
{

// =============================================================================
// vec3 — 공통 3-성분 벡터.
//   - 사용자 정의 생성자가 없는 aggregate. 그래서 brace-init / constexpr 가능.
//   - 색상 타입(Lab/RGB/HSV/HSL)과 익명 union 으로 묶여 같은 메모리를 가리킨다.
//   - 산술 연산자와 dot/length/normalized 를 멤버로 제공.
// =============================================================================
struct vec3
{
	float x, y, z;

	// 산술
	vec3 operator+(vec3 o) const { return { this->x + o.x, this->y + o.y, this->z + o.z }; }
	vec3 operator-(vec3 o) const { return { this->x - o.x, this->y - o.y, this->z - o.z }; }
	vec3 operator-()       const { return { -this->x, -this->y, -this->z }; }
	vec3 operator*(float s) const { return { this->x * s, this->y * s, this->z * s }; }
	vec3 operator/(float s) const { return { this->x / s, this->y / s, this->z / s }; }

	vec3& operator+=(vec3 o) { this->x += o.x; this->y += o.y; this->z += o.z; return *this; }
	vec3& operator-=(vec3 o) { this->x -= o.x; this->y -= o.y; this->z -= o.z; return *this; }
	vec3& operator*=(float s) { this->x *= s; this->y *= s; this->z *= s; return *this; }
	vec3& operator/=(float s) { this->x /= s; this->y /= s; this->z /= s; return *this; }

	// 내적·길이.
	float dot(vec3 o)  const { return this->x * o.x + this->y * o.y + this->z * o.z; }
	float length()     const;
	vec3  normalized() const;

	// 성분별 곱 (Hadamard product). 감마 등에 유용.
	vec3 cwise_mul(vec3 o) const { return { this->x * o.x, this->y * o.y, this->z * o.z }; }
};

// 좌측 스칼라 곱.
inline vec3 operator*(float s, vec3 v) { return v * s; }

// 자유 함수 형태의 dot (호출부 가독성용).
inline float dot(vec3 a, vec3 b) { return a.dot(b); }


// =============================================================================
// 색상 표현 구조체 forward 선언 — 멤버 메서드의 반환 타입에서 서로를 참조한다.
// =============================================================================
struct Lab;
struct RGB;
struct HSV;
struct HSL;


// =============================================================================
// Lab — 지각적으로 균일한 Oklab 좌표.
//   L: lightness (0~1), a: 녹-적, b: 청-황.
// =============================================================================
struct Lab
{
	union {
		struct { float L, a, b; };
		vec3 vector;
	};
	Lab() : L(0), a(0), b(0) {}
	Lab(float L_, float a_, float b_) : L(L_), a(a_), b(b_) {}
	Lab(vec3 v) : vector(v) {}
	operator vec3() const { return this->vector; }

	// Oklab → linear sRGB.
	RGB to_linear_srgb() const;
};


// =============================================================================
// RGB — RGB 트리플. linear sRGB 또는 표시용 sRGB. 메서드별로 의미가 다르다.
// =============================================================================
struct RGB
{
	union {
		struct { float r, g, b; };
		vec3 vector;
	};
	RGB() : r(0), g(0), b(0) {}
	RGB(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}
	RGB(vec3 v) : vector(v) {}
	operator vec3() const { return this->vector; }

	// linear sRGB → Oklab. 호출 시 *this 가 linear sRGB 라고 가정.
	Lab to_oklab() const;

	// 표시용 sRGB → OkHSL. 호출 시 *this 가 표시용 sRGB 라고 가정.
	HSL to_okhsl() const;

	// 표시용 sRGB → OkHSV. 호출 시 *this 가 표시용 sRGB 라고 가정.
	HSV to_okhsv() const;
};


// =============================================================================
// HSV — OkHSV 좌표. (h, s, v).
// =============================================================================
struct HSV
{
	union {
		struct { float h, s, v; };
		vec3 vector;
	};
	HSV() : h(0), s(0), v(0) {}
	HSV(float h_, float s_, float v_) : h(h_), s(s_), v(v_) {}
	HSV(vec3 vv) : vector(vv) {}
	operator vec3() const { return this->vector; }

	// OkHSV → 표시용 sRGB.
	RGB to_srgb() const;
};


// =============================================================================
// HSL — OkHSL 좌표. 여기서의 l 은 toe()를 통과한 perceptual lightness 이다.
// =============================================================================
struct HSL
{
	union {
		struct { float h, s, l; };
		vec3 vector;
	};
	HSL() : h(0), s(0), l(0) {}
	HSL(float h_, float s_, float l_) : h(h_), s(s_), l(l_) {}
	HSL(vec3 v) : vector(v) {}
	operator vec3() const { return this->vector; }

	// OkHSL → 표시용 sRGB.
	RGB to_srgb() const;
};


// =============================================================================
// 부속 타입. (vec3 와 묶지 않음 — 2-성분이거나 의미가 충분히 다름.)
// =============================================================================

// (Lightness, Chroma) 쌍. cusp 위치 등에 사용.
struct ST;
struct LC
{
	float L;
	float C;

	// LC cusp → ST 표현.
	ST to_ST() const;
};

// LC cusp 의 대안 표현.
//   S = C_cusp / L_cusp,  T = C_cusp / (1 - L_cusp).
// 주어진 L 에서 색역 삼각형 안의 최대 chroma 는 fmin(S*L, T*(1-L)) 로 즉시 구해진다.
struct ST { float S; float T; };

// 한 (L, hue) 위치에서의 세 가지 기준 chroma. OkHSL saturation 보간에 사용.
//   C_0   : 직각삼각형 가정 기준
//   C_mid : 매끄러운 중간 chroma (s=0.8 분기점)
//   C_max : sRGB 색역 경계까지의 최대 chroma
struct Cs { float C_0; float C_mid; float C_max; };


// =============================================================================
// 작은 유틸
// =============================================================================

// 값을 [min, max] 로 자른다.
float clamp(float x, float min, float max);

// 부호: x>0 → +1, x<0 → -1, x==0 → 0.
float sgn(float x);

// linear sRGB 성분 → 표시용 sRGB 성분 (감마 인코딩).
float srgb_transfer_function(float a);

// 표시용 sRGB 성분 → linear sRGB 성분 (감마 디코딩).
float srgb_transfer_function_inv(float a);

// 어두운 영역(toe)을 강조하는 perceptual 응답 곡선.
// Oklab 의 L (선형 luminance 기반) 을 OkHSL 의 l 로 매핑할 때 사용한다.
float toe(float x);

// toe() 의 닫힌 해(closed-form) 역함수.
float toe_inv(float x);


// =============================================================================
// 색역(gamut) 분석
//   아래 함수들의 (a, b) 는 a^2 + b^2 == 1 로 정규화된 hue 방향 단위벡터다.
// =============================================================================

// 정규화된 (a, b) hue 방향에서 sRGB 안에 들어가는 최대 saturation S = C/L.
// 다항식 초기치 + Halley's method 1회 = 1e-6 미만 오차.
float compute_max_saturation(float a, float b);

// 주어진 hue 에서 sRGB 색역 cusp(가장 chroma 가 큰 점)의 (L, C).
LC find_cusp(float a, float b);

// (L0, 0) → (L1, C1) 직선이 sRGB 색역 경계와 만나는 t. 결과 점은
//   ( L0*(1-t) + t*L1 , t*C1 ).
// cusp 을 미리 계산해 둔 호출자용 오버로드.
float find_gamut_intersection(float a, float b, float L1, float C1, float L0, LC cusp);

// 위와 동일하되 cusp 을 내부에서 찾는 편의 오버로드.
float find_gamut_intersection(float a, float b, float L1, float C1, float L0);

// hue (a_, b_) 에서 매끄럽게 근사된 "중간" ST. 다항식 피팅으로
//   S_mid < S_max, T_mid < T_max 가 항상 성립.
ST get_ST_mid(float a_, float b_);

// 주어진 (L, hue) 에서의 (C_0, C_mid, C_max) 삼중을 한꺼번에 계산.
Cs get_Cs(float L, float a_, float b_);


// =============================================================================
// 색역 클리핑 (gamut clipping)
//   입력/출력 모두 linear sRGB. 이미 색역 안이면 그대로 반환.
//   다섯 가지 전략의 차이는 "어느 점(L0)을 향해 색을 끌어당기느냐" 이다.
// =============================================================================

// chroma 보존 우선. L0 = clamp(L, 0, 1). lightness 만 살짝 양보.
RGB gamut_clip_preserve_chroma(RGB rgb);

// L0 = 0.5 고정. 진한 색일수록 회색에 가까워진다.
RGB gamut_clip_project_to_0_5(RGB rgb);

// L0 = L_cusp. hue 별로 기준 lightness 가 달라 자연스럽지만 cusp 계산 비용 발생.
RGB gamut_clip_project_to_L_cusp(RGB rgb);

// 0.5 기반 적응형. alpha 가 클수록 chroma 를 더 깎고 lightness 를 더 보존.
RGB gamut_clip_adaptive_L0_0_5(RGB rgb, float alpha = 0.05f);

// L_cusp 기반 적응형.
RGB gamut_clip_adaptive_L0_L_cusp(RGB rgb, float alpha = 0.05f);

} // namespace ok_color
