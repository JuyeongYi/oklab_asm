# tensor — Tensor Core (wmma TF32) 시도

NVIDIA Tensor Core 의 `mma.sync` 명령으로 행렬곱을 가속해 본 실험. 결과는 **0% 가속** — memory-bound 워크로드에서 Tensor Core 가 의미 없다는 것을 직접 측정으로 입증한 자료.

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_tensor.cu` | `linear_srgb_to_oklab` 을 wmma TF32 m16n16k8 로 재구현. 한 warp = 16 픽셀. cbrt 만 scalar. |

## 동작 원리

1. **block 당 8 warp (256 threads), warp 당 16 픽셀** 처리.
2. 16 RGB 를 shared memory 16×8 row-major 타일로 로드 (첫 3 col 만 의미).
3. **MMA 1**: A(16×8) × B(8×16) → C(16×16). RGB → LMS.
   - B 는 `__constant__` 메모리의 16×16 col-major (3×3 행렬을 zero-pad).
4. **cbrt** 3개 scalar (lane 당 픽셀 하나).
5. **MMA 2**: A(16×8) × B(8×16) → C(16×16). LMS' → Lab.
6. Lab 출력.

PTX 명령 한 줄:
```
mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 d, a, b, c;
```

## 결과 (RTX 4080 SUPER, 4M pixels)

| variant | ms | Gpx/s | 정확도 |
|---|---:|---:|---:|
| CUDA regular (scalar fma) | 0.207 | 20.27 | 1e-6 |
| **Tensor Core (wmma TF32)** | **0.207** | **20.28** | **1.4e-3** |

**Speedup: 1.000x — 동률**.

## 왜 가속이 0인가

세 요소가 동시에 작용:

1. **Memory bandwidth 한계**: 4M × 32 byte I/O = 128 MB.
   - 0.207ms × 619 GB/s = 128MB. 현재 736 GB/s peak 의 84% 활용. 더 빨라질 수 없음.
   - Tensor Core 가 compute 를 0 cycle 에 끝내도 메모리는 그대로.

2. **Padding 손해**: 3×3 행렬을 16×16 타일에 채워서 9/256 = **3.5% utilization**. MMA throughput 이 압도적 (1024 fma / instruction) 이라 절대 시간으론 빠르지만, 어차피 메모리 대기.

3. **cbrt scalar 잔존**: Tensor Core 는 행렬 연산만. transcendental 은 SFU 가 따로. 픽셀당 3 cbrt × ~6 cycle 는 그대로.

## 정확도 — TF32 의 한계

TF32 = 1 sign + 8 exp (FP32 range 그대로) + **10 mantissa** (FP32 의 23-bit 대비 -13 bits).
- 단일 곱셈 정밀도: ~1e-3
- 누적 두 번 (RGB→LMS→Lab) 후: 1.4e-3

색상 표현으로는 8-bit 출력 (1/256 = 0.004) 안에 들어가서 화면엔 안 보이지만, 일반 CUDA 의 1e-6 보다 **1000배** 나쁨. ML 파이프라인 (color → tensor → ...) 처럼 여러 단계 누적 시 위험.

대안:
- **FP16 m16n16k16**: 더 빠르지만 정확도 더 나쁨 (~1e-2)
- **TF32 m16n8k8**: 더 작은 타일, 같은 정밀도
- **double-precision m8n8k4 FP64**: 정밀도 충분, throughput 매우 낮음

## 사용 시점

이 구현체는 **거의 안 쓸 영역**입니다.

Tensor Core 가 진짜 가치 있는 곳:
- **NN inference**: 행렬이 16×16 보다 큼, FP16/BF16 자연 적합
- **compute-bound**: arithmetic intensity 가 높아 메모리 대역 안 비는 워크로드
- **Tile-fused 연산**: 한 타일 안에서 multiple ops 합성 (예: GEMM + bias + activation)

색상 변환은 이 셋 중 어디에도 안 해당. **결론적으로 Tensor Core 의 도메인이 아닙니다.**

## 빌드

```bash
nvcc -O3 -arch=sm_89 -std=c++17 -Icuda \
     cuda/ok_color_cuda.cu tensor/ok_color_tensor.cu -o build/tensor_bench
```

## 의미

이 폴더의 진짜 가치는 **반증 자료**입니다 — "Tensor Core 면 무조건 빠르다" 는 미신을 measurement-driven 으로 깬 데이터. PTX 와 마찬가지로, 도구의 강력함은 워크로드 특성에 맞을 때만 발휘됨.
