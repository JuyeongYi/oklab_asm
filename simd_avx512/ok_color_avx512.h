#pragma once
// -----------------------------------------------------------------------------
// ok_color_avx512 — ok_color_simd 의 16-wide (AVX512) 변형.
//   설계 원칙은 ok_color_simd 와 동일: SoA, packed transcendental, mask + blend 분기.
//   __m128 → __m512, __m128 mask → __mmask16.
// -----------------------------------------------------------------------------

#include <immintrin.h>

namespace ok_color_avx512
{

struct PackedLab; struct PackedRGB; struct PackedHSL; struct PackedHSV;

struct PackedLab {
	union { struct { __m512 L, a, b; }; __m512 ch[3]; };
	PackedLab() : L(_mm512_setzero_ps()), a(_mm512_setzero_ps()), b(_mm512_setzero_ps()) {}
	PackedLab(__m512 L_, __m512 a_, __m512 b_) : L(L_), a(a_), b(b_) {}
	__m512&       operator[](int i)       { return this->ch[i]; }
	const __m512& operator[](int i) const { return this->ch[i]; }
	PackedRGB to_linear_srgb() const;
};
struct PackedRGB {
	union { struct { __m512 r, g, b; }; __m512 ch[3]; };
	PackedRGB() : r(_mm512_setzero_ps()), g(_mm512_setzero_ps()), b(_mm512_setzero_ps()) {}
	PackedRGB(__m512 r_, __m512 g_, __m512 b_) : r(r_), g(g_), b(b_) {}
	__m512&       operator[](int i)       { return this->ch[i]; }
	const __m512& operator[](int i) const { return this->ch[i]; }
	PackedLab to_oklab() const;
	PackedHSL to_okhsl() const;
	PackedHSV to_okhsv() const;
};
struct PackedHSV {
	union { struct { __m512 h, s, v; }; __m512 ch[3]; };
	PackedHSV() : h(_mm512_setzero_ps()), s(_mm512_setzero_ps()), v(_mm512_setzero_ps()) {}
	PackedHSV(__m512 h_, __m512 s_, __m512 v_) : h(h_), s(s_), v(v_) {}
	__m512&       operator[](int i)       { return this->ch[i]; }
	const __m512& operator[](int i) const { return this->ch[i]; }
	PackedRGB to_srgb() const;
};
struct PackedHSL {
	union { struct { __m512 h, s, l; }; __m512 ch[3]; };
	PackedHSL() : h(_mm512_setzero_ps()), s(_mm512_setzero_ps()), l(_mm512_setzero_ps()) {}
	PackedHSL(__m512 h_, __m512 s_, __m512 l_) : h(h_), s(s_), l(l_) {}
	__m512&       operator[](int i)       { return this->ch[i]; }
	const __m512& operator[](int i) const { return this->ch[i]; }
	PackedRGB to_srgb() const;
};

// AoS 편의 타입 — ok_color_simd 와 동일.
struct alignas(16) RGB4 { float r, g, b, _pad; };
struct alignas(16) Lab4 { float L, a, b, _pad; };
struct alignas(16) HSL4 { float h, s, l, _pad; };
struct alignas(16) HSV4 { float h, s, v, _pad; };

// AoS 16-pixel batch wrapper.
void linear_srgb_to_oklab_batch16(const RGB4* in, Lab4* out);
void oklab_to_linear_srgb_batch16(const Lab4* in, RGB4* out);
void okhsl_to_srgb_batch16(const HSL4* in, RGB4* out);
void srgb_to_okhsl_batch16(const RGB4* in, HSL4* out);
void okhsv_to_srgb_batch16(const HSV4* in, RGB4* out);
void srgb_to_okhsv_batch16(const RGB4* in, HSV4* out);

} // namespace
