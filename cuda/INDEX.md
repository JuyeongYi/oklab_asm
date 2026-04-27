# cuda — NVIDIA GPU 커널

CUDA 13.1 / compute capability 8.9+ (Ada Lovelace 등). thread 당 1 픽셀 처리.

## 핵심 설계

1. **SIMT, AoS 자연 coalesce**: warp 의 32 thread 가 각자 1 RGB4 (16 byte) 를 읽으면 자동으로 1024-byte coalesced load 가 됨. SSE/AVX 의 SoA transpose 가 불필요.
2. **HW transcendental**: `cbrtf`, `sinf`, `cosf`, `atan2f`, `powf` 가 Ada SFU 의 hardware 명령. CPU 의 bit-hack/Newton/polynomial 트릭 *전부 불필요* — 그냥 표준 함수 호출. 정확도 1 ULP 수준.
3. **explicit memory management**: `cudaMallocManaged` (unified memory) 사용 안 함. 명시적 `cudaMalloc` / `cudaMemcpy` / `cudaFree` 만 사용.
4. **gamut clipping 5종 통합**: 단일 `k_clip` 커널 + strategy enum. warp divergence 가능성이 있지만 실측상 영향 작음 (대부분 픽셀이 동일 strategy 분기).

## 파일

| 파일 | 설명 |
|---|---|
| `ok_color_cuda.cuh` | host launch wrapper API |
| `ok_color_cuda.cu` | `__device__` 함수 + `__global__` 커널 + launch wrapper |
| `cuda_bench.cu` | 정확성 검증 + 성능 측정 |

## 정확성

원본 (CPU scalar) 와 비교한 max_abs (4M 픽셀 중 64개 비교):

| 함수 | max_abs |
|---|---:|
| linear_srgb→oklab | 3.0e-7 |
| oklab→linear_srgb | 4.2e-7 |
| okhsl→srgb | 7.2e-7 |
| srgb→okhsl | 1.3e-6 |
| okhsv→srgb | 3.6e-7 |
| srgb→okhsv | 1.8e-6 |

CPU SIMD (1e-5 ~ 1e-3) 보다 한 자릿수 *더* 정확함.

## 성능 (RTX 4080 SUPER, 4M 픽셀)

**Kernel only** (입력이 이미 device 에 있을 때):

| 함수 | ms | ns/pixel | Gpx/s |
|---|---:|---:|---:|
| linear_srgb→oklab | 0.217 | 0.052 | 19.32 |
| oklab→linear_srgb | 0.217 | 0.052 | 19.36 |
| okhsl→srgb | 0.213 | 0.051 | 19.69 |
| srgb→okhsl | 0.214 | 0.051 | 19.60 |
| okhsv→srgb | 0.214 | 0.051 | 19.60 |
| srgb→okhsv | 0.215 | 0.051 | 19.51 |
| clip:preserve_chroma | 0.213 | 0.051 | 19.69 |

CPU Hybrid 의 1.99 ns/px 대비 **38배 빠름**. 모든 함수가 거의 동일 시간 = **memory bandwidth bound**.

**Full round-trip** (H2D + kernel + D2H):

| 함수 | ms | ns/pixel |
|---|---:|---:|
| linear_srgb→oklab | 11.4 | 2.72 |
| srgb→okhsv | 10.9 | 2.60 |

PCIe 4.0 전송 비용이 kernel 의 50배. **CPU Hybrid (1.99 ns/px) 보다 round-trip 으로 가면 *느림***.

## 사용 시점

**적합**:
- 데이터가 이미 GPU 에 있는 파이프라인 (영상 디코더 → 색 변환 → 영상 인코더)
- GPU 에서 이어지는 추가 작업 (필터링, ML inference 등)
- 매우 큰 batch (수백만 픽셀 이상)

**부적합**:
- 단일 픽셀 변환 (PCIe 전송이 100배 비쌈)
- 실시간 GUI color picker (CPU 가 즉시)
- 작은 이미지 (< 1 MP)

## 빌드

```bash
nvcc -O3 -arch=sm_89 -std=c++17 \
     cuda/ok_color_cuda.cu cuda/cuda_bench.cu -o build/cuda_bench
```

`-arch=sm_XX` 는 GPU compute capability 에 맞춰 변경. 주요 값:
- Pascal (GTX 10xx): `sm_60` ~ `sm_61`
- Volta (V100): `sm_70`
- Turing (RTX 20xx): `sm_75`
- Ampere (RTX 30xx, A100): `sm_80` ~ `sm_86`
- Ada (RTX 40xx): `sm_89`
- Hopper (H100): `sm_90`
- Blackwell (RTX 50xx, B200): `sm_100`

## 한계 / 개선 여지

- **memory bound**: shared memory 또는 L1 활용으로 더 빨라지진 않음 (입력당 메모리 한 번만 읽음).
- **PCIe**: pinned memory (`cudaHostAlloc`) + async copy + stream 으로 transfer ~2배 빠르게 가능.
- **larger batches**: 메모리 transfer 가 amortize 되어 round-trip 효율 높아짐.
- **persistent kernel + queue**: 데이터를 GPU 에 두고 여러 변환을 cudaStream 으로 chain.
