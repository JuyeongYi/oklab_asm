# ptx — 인라인 PTX (NVIDIA virtual ISA) 시연

CUDA 의 표준 cmath 변환 (`cbrtf`, `sinf`, `cosf`, `powf`...) 을 PTX 의 `*.approx.f32` 명령으로 직접 교체한 변형. nvcc 의 `--use_fast_math` 가 자동으로 하는 일을 *명시적으로* 수행해서 비교 가능하게 만든 것.

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_ptx_demo.cu` | 시연용. `linear_srgb_to_oklab` 의 cbrt 부분만 (A) `cbrtf` (correctly-rounded), (B) `__powf(x, 1/3)` 인트린식, (C) inline PTX `lg2.approx + mul + ex2.approx` 세 변형 비교. |
| `ok_color_ptx_full.cu` | 6 변환 전부 PTX 화. namespace `ok_color_cuda::ptx`. cmath 변형 (ok_color_cuda) 와 같은 입력으로 ms 비교. |

## 핵심 PTX 명령

| PTX | 의미 | 정확도 | 사용처 |
|---|---|---|---|
| `lg2.approx.f32` | log₂(x) | 22-bit | cbrt = exp2(log2/3), pow |
| `ex2.approx.f32` | 2^x | 22-bit | cbrt 후처리, exp |
| `sin.approx.f32` | sin (range-reduced) | 22-bit | hue → (a, b) |
| `cos.approx.f32` | cos | 22-bit | hue → (a, b) |
| `rcp.approx.f32` | 1/x | 22-bit | division 대체 |
| `sqrt.approx.f32` | √x | 22-bit | chroma |
| `rsqrt.approx.f32` | 1/√x | 22-bit | normalize |

PTX 에 직접 명령이 *없는* 것: `atan2`, `cbrt`, `pow`, `tan`, `asin`, `acos`. 이들은 위 명령들의 합성으로 만든다.

## 구현 패턴

```cuda
__device__ __forceinline__ float p_cbrt(float x) {
    float t, r;
    asm volatile("lg2.approx.f32 %0, %1;" : "=f"(t) : "f"(x));
    t *= 0.33333333333f;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(t));
    return r;
}
```

`asm volatile` 로 PTX 어셈블리 그대로 삽입. `%0` 은 출력, `%1` 은 입력. 제약자 `"=f"` 는 float register 출력, `"f"` 는 float register 입력.

## 측정 결과 (RTX 4080 SUPER, 4M 픽셀)

3회 실행 평균 — speedup 컬럼 = cmath / PTX:

| 함수 | cmath ms | PTX ms | 비율 |
|---|---:|---:|---:|
| linear_srgb→oklab | 0.215 | 0.214 | 1.00x |
| oklab→linear_srgb | 0.213 | 0.214 | 1.00x |
| okhsl→srgb | 0.214 | 0.213 | 1.00x |
| srgb→okhsl | 0.214 | 0.215 | 1.00x |
| okhsv→srgb | 0.205 | 0.215 | 0.95x |
| srgb→okhsv | 0.214 | 0.213 | 1.00x |

**모두 ±5% 측정 노이즈 안. PTX 의 이론적 cycle 절약이 실측에서 안 나타남.**

## 왜 이론과 다른가 — memory bound

4M 픽셀 × 32 byte (RGB4 in + Lab4 out) = 128 MB / 0.21ms = 610 GB/s. RTX 4080 SUPER 의 736 GB/s 대비 83% 활용. 메모리 wait 가 시간의 대부분을 차지하고, SM 들은 그 wait 동안 다른 warp 의 ALU 작업으로 채움 → cbrt 가 16 cycle 이든 6 cycle 이든 보이지 않음.

PTX 가 진짜 의미 있을 영역:
1. **compute-bound**: 작은 batch 또는 GPU 안에서 chain 된 변환 (메모리 트래픽 amortize)
2. **occupancy 낮음**: SM 가 다른 warp 로 latency 못 가릴 때
3. **PTX-only 명령**: warp shuffle, predicated, atomic 등 — 우리 워크로드와 무관

## 정확도

PTX `*.approx.f32` 는 22-bit (≈ 2.4e-7 ULP). 표준 cmath 와 비교해 max_abs 3.87e-7 — 색상 출력 8-bit (1/256 = 0.004) 와 비교하면 **1만 배 더 정확**. 화면에 보이는 차이 0.

## 빌드

```bash
nvcc -O3 -arch=sm_89 -std=c++17 -Icuda \
     cuda/ok_color_cuda.cu ptx/ok_color_ptx_full.cu -o build/ptx_full
```

또는 `./build.sh ptx` 로 통합 빌드.

## 결론

PTX 직접 작성은 흥미로운 **연습**이지만 우리 워크로드에선 의미 있는 가속을 못 줍니다. 실용적 코드는 nvcc 의 `--use_fast_math` 또는 `__sinf`/`__cosf`/`__expf`/`__logf`/`__powf` 같은 fast intrinsic 으로 충분.

PTX 가 진짜 필요한 곳:
- 특정 명령어 패턴 강제 (예: `ld.global.cg` cache-global hint)
- warp-level primitive 의 명시적 컨트롤
- 컴파일러가 못 추론하는 micro-optimization
