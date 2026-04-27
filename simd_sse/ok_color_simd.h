#pragma once
// -----------------------------------------------------------------------------
// ok_color_simd — ok_color 의 batch (4-pixel SoA) SIMD 재구현.
//
//   설계 원칙 (ok_color.h 의 규칙을 그대로 따름):
//     - **공용체**: 각 색상 타입은 union 으로 named 멤버 (.r/.g/.b 등) 와 채널 배열
//       (.ch[0..2]) 두 가지 view 를 제공.
//     - **멤버 메서드**: 변환은 소스 타입의 메서드 (rgb.to_oklab() 등). 자유 함수는
//       gamut_clip_* 처럼 한 타입에 묶이지 않는 것들에만.
//     - **this->** 접두사: 멤버 접근에 모두 this-> 사용 (가독성).
//
//   SoA 의 의미:
//     - PackedRGB 의 .r 한 개는 4 픽셀의 r 값을 담은 __m128.
//       → r = (r₀, r₁, r₂, r₃)
//     - 4 픽셀이 lane 별로 *완전 독립* 이라 진정한 4-way ILP 가 펼쳐진다.
//     - cbrt/log/exp/pow/sin/cos 등 transcendental 도 모두 packed 구현.
//     - 분기는 mask + blendv 로 처리 (lane divergence 무관).
//
//   정확도: 1e-5 ~ 1e-6 (display/UI 용도로 충분, 원본과 거의 일치).
// -----------------------------------------------------------------------------

#include <immintrin.h>

namespace ok_color_simd
{

// =============================================================================
// 색상 타입 forward 선언.
// =============================================================================
struct PackedLab;
struct PackedRGB;
struct PackedHSL;
struct PackedHSV;


// =============================================================================
// PackedLab — 4 픽셀의 Oklab 좌표 (SoA).
//   .L = (L₀,L₁,L₂,L₃),  .a = (a₀..a₃),  .b = (b₀..b₃)
// =============================================================================
struct PackedLab
{
	union {
		struct { __m128 L, a, b; };
		__m128 ch[3];
	};
	PackedLab() : L(_mm_setzero_ps()), a(_mm_setzero_ps()), b(_mm_setzero_ps()) {}
	PackedLab(__m128 L_, __m128 a_, __m128 b_) : L(L_), a(a_), b(b_) {}

	__m128&       operator[](int i)       { return this->ch[i]; }
	const __m128& operator[](int i) const { return this->ch[i]; }

	// Oklab → linear sRGB.
	PackedRGB to_linear_srgb() const;
};


// =============================================================================
// PackedRGB — 4 픽셀의 RGB.  의미는 메서드별로 (linear vs 표시용 sRGB).
// =============================================================================
struct PackedRGB
{
	union {
		struct { __m128 r, g, b; };
		__m128 ch[3];
	};
	PackedRGB() : r(_mm_setzero_ps()), g(_mm_setzero_ps()), b(_mm_setzero_ps()) {}
	PackedRGB(__m128 r_, __m128 g_, __m128 b_) : r(r_), g(g_), b(b_) {}

	__m128&       operator[](int i)       { return this->ch[i]; }
	const __m128& operator[](int i) const { return this->ch[i]; }

	// linear sRGB → Oklab (호출 시 *this 가 linear sRGB 라고 가정).
	PackedLab to_oklab() const;
	// 표시용 sRGB → OkHSL.
	PackedHSL to_okhsl() const;
	// 표시용 sRGB → OkHSV.
	PackedHSV to_okhsv() const;
};


// =============================================================================
// PackedHSV — 4 픽셀의 OkHSV.
// =============================================================================
struct PackedHSV
{
	union {
		struct { __m128 h, s, v; };
		__m128 ch[3];
	};
	PackedHSV() : h(_mm_setzero_ps()), s(_mm_setzero_ps()), v(_mm_setzero_ps()) {}
	PackedHSV(__m128 h_, __m128 s_, __m128 v_) : h(h_), s(s_), v(v_) {}

	__m128&       operator[](int i)       { return this->ch[i]; }
	const __m128& operator[](int i) const { return this->ch[i]; }

	// OkHSV → 표시용 sRGB.
	PackedRGB to_srgb() const;
};


// =============================================================================
// PackedHSL — 4 픽셀의 OkHSL.
// =============================================================================
struct PackedHSL
{
	union {
		struct { __m128 h, s, l; };
		__m128 ch[3];
	};
	PackedHSL() : h(_mm_setzero_ps()), s(_mm_setzero_ps()), l(_mm_setzero_ps()) {}
	PackedHSL(__m128 h_, __m128 s_, __m128 l_) : h(h_), s(s_), l(l_) {}

	__m128&       operator[](int i)       { return this->ch[i]; }
	const __m128& operator[](int i) const { return this->ch[i]; }

	// OkHSL → 표시용 sRGB.
	PackedRGB to_srgb() const;
};


// =============================================================================
// 부속 packed 타입.
// =============================================================================
struct PackedST;
struct PackedLC
{
	union {
		struct { __m128 L, C; };
		__m128 ch[2];
	};
	PackedLC() : L(_mm_setzero_ps()), C(_mm_setzero_ps()) {}
	PackedLC(__m128 L_, __m128 C_) : L(L_), C(C_) {}

	// LC cusp → ST 표현.
	PackedST to_ST() const;
};

struct PackedST
{
	union {
		struct { __m128 S, T; };
		__m128 ch[2];
	};
	PackedST() : S(_mm_setzero_ps()), T(_mm_setzero_ps()) {}
	PackedST(__m128 S_, __m128 T_) : S(S_), T(T_) {}
};

struct PackedCs
{
	union {
		struct { __m128 C_0, C_mid, C_max; };
		__m128 ch[3];
	};
	PackedCs() : C_0(_mm_setzero_ps()), C_mid(_mm_setzero_ps()), C_max(_mm_setzero_ps()) {}
	PackedCs(__m128 c0, __m128 cm, __m128 cM) : C_0(c0), C_mid(cm), C_max(cM) {}
};


// =============================================================================
// 색역 분석 (free function — 한 타입에 묶이지 않음).
//   입력 (a, b) 는 packed 정규화 단위벡터 (각 lane 에서 a²+b²=1).
// =============================================================================
__m128   compute_max_saturation(__m128 a, __m128 b);
PackedLC find_cusp(__m128 a, __m128 b);
__m128   find_gamut_intersection(__m128 a, __m128 b, __m128 L1, __m128 C1, __m128 L0,
                                 PackedLC cusp);
__m128   find_gamut_intersection(__m128 a, __m128 b, __m128 L1, __m128 C1, __m128 L0);
PackedST get_ST_mid(__m128 a, __m128 b);
PackedCs get_Cs(__m128 L, __m128 a, __m128 b);


// =============================================================================
// 색역 클리핑 (free function).  in-gamut lane 은 자동으로 원본 통과.
// =============================================================================
PackedRGB gamut_clip_preserve_chroma(PackedRGB rgb);
PackedRGB gamut_clip_project_to_0_5(PackedRGB rgb);
PackedRGB gamut_clip_project_to_L_cusp(PackedRGB rgb);
PackedRGB gamut_clip_adaptive_L0_0_5(PackedRGB rgb, float alpha = 0.05f);
PackedRGB gamut_clip_adaptive_L0_L_cusp(PackedRGB rgb, float alpha = 0.05f);


// =============================================================================
// AoS (RGB4/Lab4/...) 4 개 입출력 편의 wrapper.  내부에서 transpose.
// =============================================================================
struct alignas(16) RGB4 { float r, g, b, _pad; };
struct alignas(16) Lab4 { float L, a, b, _pad; };
struct alignas(16) HSL4 { float h, s, l, _pad; };
struct alignas(16) HSV4 { float h, s, v, _pad; };

void linear_srgb_to_oklab_batch4(const RGB4* in, Lab4* out);
void oklab_to_linear_srgb_batch4(const Lab4* in, RGB4* out);
void okhsl_to_srgb_batch4(const HSL4* in, RGB4* out);
void srgb_to_okhsl_batch4(const RGB4* in, HSL4* out);
void okhsv_to_srgb_batch4(const HSV4* in, RGB4* out);
void srgb_to_okhsv_batch4(const RGB4* in, HSV4* out);

void gamut_clip_preserve_chroma_batch4(const RGB4* in, RGB4* out);
void gamut_clip_project_to_0_5_batch4(const RGB4* in, RGB4* out);
void gamut_clip_project_to_L_cusp_batch4(const RGB4* in, RGB4* out);
void gamut_clip_adaptive_L0_0_5_batch4(const RGB4* in, RGB4* out, float alpha = 0.05f);
void gamut_clip_adaptive_L0_L_cusp_batch4(const RGB4* in, RGB4* out, float alpha = 0.05f);

} // namespace ok_color_simd
